"""Ajusta ViTPose (preentrenado en COCO) a los 24 puntos de Ski2DPose: cuerpo, bastones y esquis.

El recorte es el mismo que el de VitPoseImageProcessor (caja con 25 % de relleno,
proporcion 3:4 y transformacion «unbiased»), asi que el modelo guardado se usa
despues con el procesador de transformers sin cambios. Uso:
    python src/entrenar.py data/ski2dpose runs/vitpose24 [--epocas 40] [--lote 32]
    python src/entrenar.py data/ski2dpose runs/prueba --ver 12   # solo mira los aumentos
"""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, VitPoseForPoseEstimation
from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale, get_warp_matrix

from dibujo import dibujar
from evaluar import evaluar, por_punto, resumen
from metricas import pck
from to_coco import PUNTOS, VOLTEO

BASE = "usyd-community/vitpose-base-simple"
ANCHO, ALTO = 192, 256        # entrada del modelo
HM_ANCHO, HM_ALTO = 48, 64    # mapas de calor, a 1/4
SIGMA = 2.0
PESO_OCULTO = 0.5             # los puntos tapados tienen posicion estimada: cuentan menos
# Canal de la cabeza COCO con el que arranca cada uno de los 24 puntos (media si son
# varios), para no tirar lo que el modelo ya sabe del cuerpo. Esquis y bastones
# arrancan desde el tobillo y la muneca de su lado, que es lo que tienen mas cerca.
DESDE_COCO = [[0, 1, 2, 3, 4], [5, 6],
              [6], [8], [10], [10], [5], [7], [9], [9],
              [12], [14], [16], [11], [13], [15],
              [16], [16], [16], [16], [15], [15], [15], [15]]

cv2.setNumThreads(0)  # los hilos los ponen los trabajadores del DataLoader


class Ski2DPose(Dataset):
    def __init__(self, raiz, split, media, desv, aumentar):
        coco = json.loads((raiz / f"coco_{split}.json").read_text())
        anns = {a["image_id"]: a for a in coco["annotations"]}
        self.items = [(raiz / im["file_name"], anns[im["id"]]) for im in coco["images"]]
        self.media, self.desv, self.aumentar = np.float32(media), np.float32(desv), aumentar
        self.yy, self.xx = np.mgrid[0:HM_ALTO, 0:HM_ANCHO].astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, a = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2RGB)
        kps = np.array(a["keypoints"], np.float32).reshape(-1, 3)
        x, y, w, h = a["bbox"]
        rot = 0.0
        if self.aumentar:
            if random.random() < 0.5:  # volteo: la derecha pasa a ser la izquierda
                img = img[:, ::-1]
                kps[:, 0] = img.shape[1] - 1 - kps[:, 0]
                kps = kps[VOLTEO]
                x = img.shape[1] - 1 - x - w
            # Caja desplazada y con otro tamano, como las que dara el detector.
            s = random.uniform(0.8, 1.25)
            cx = x + w / 2 + random.uniform(-0.08, 0.08) * w
            cy = y + h / 2 + random.uniform(-0.08, 0.08) * h
            w, h = w * s, h * s
            x, y = cx - w / 2, cy - h / 2
            if random.random() < 0.5:
                rot = random.uniform(-30, 30)
        centro, escala = box_to_center_and_scale([x, y, w, h], image_width=ANCHO, image_height=ALTO)
        M = get_warp_matrix(rot, centro * 2.0, np.array([ANCHO - 1.0, ALTO - 1.0]), escala * 200.0)
        recorte = cv2.warpAffine(np.ascontiguousarray(img), M, (ANCHO, ALTO), flags=cv2.INTER_LINEAR)
        recorte = recorte.astype(np.float32)
        if self.aumentar:
            # Suave: la nieve ya esta cerca del blanco y se quemaria entera.
            recorte = np.clip(recorte * random.uniform(0.8, 1.15) + random.uniform(-20, 10), 0, 255)
            if random.random() < 0.3:  # compresion y desenfoque de las retransmisiones
                recorte = cv2.GaussianBlur(recorte, (0, 0), random.uniform(0.5, 1.5))

        # Puntos al espacio del mapa de calor (misma convencion que transform_preds).
        en_entrada = kps[:, :2] @ M[:, :2].T + M[:, 2]
        u = en_entrada[:, 0] * (HM_ANCHO - 1) / (ANCHO - 1)
        v = en_entrada[:, 1] * (HM_ALTO - 1) / (ALTO - 1)
        mapas = np.zeros((len(PUNTOS), HM_ALTO, HM_ANCHO), np.float32)
        peso = np.zeros(len(PUNTOS), np.float32)
        for j in range(len(PUNTOS)):
            if kps[j, 2] == 0 or not (0 <= u[j] <= HM_ANCHO - 1 and 0 <= v[j] <= HM_ALTO - 1):
                continue
            mapas[j] = np.exp(-((self.xx - u[j]) ** 2 + (self.yy - v[j]) ** 2) / (2 * SIGMA ** 2))
            peso[j] = 1.0 if kps[j, 2] == 2 else PESO_OCULTO

        recorte = (recorte / 255.0 - self.media) / self.desv
        return torch.from_numpy(recorte.transpose(2, 0, 1)), torch.from_numpy(mapas), torch.from_numpy(peso)


def iniciar_cabeza(modelo):
    """Copia a la capa final nueva los filtros de la cabeza COCO segun DESDE_COCO."""
    coco = VitPoseForPoseEstimation.from_pretrained(BASE).head.conv
    with torch.no_grad():
        for j, origen in enumerate(DESDE_COCO):
            modelo.head.conv.weight[j] = coco.weight[origen].mean(0)
            modelo.head.conv.bias[j] = coco.bias[origen].mean(0)


def ver_aumentos(ds, salida, n):
    """Mosaico de recortes aumentados con los puntos sacados de sus mapas de calor."""
    celdas = []
    for i in random.sample(range(len(ds)), n):
        x, mapas, peso = ds[i]
        img = ((x.numpy().transpose(1, 2, 0) * ds.desv + ds.media) * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        kps = []
        for m, p in zip(mapas.numpy(), peso.numpy()):
            fy, fx = np.unravel_index(m.argmax(), m.shape)
            kps.append((fx * (ANCHO - 1) / (HM_ANCHO - 1), fy * (ALTO - 1) / (HM_ALTO - 1), 2 if p == 1 else int(p > 0)))
        celdas.append(cv2.resize(dibujar(img, kps, grosor=1), (ANCHO * 2, ALTO * 2)))
    filas = [np.hstack(celdas[i:i + 6]) for i in range(0, len(celdas) - len(celdas) % 6, 6)]
    salida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida / "aumentos.jpg"), np.vstack(filas))
    print(salida / "aumentos.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", type=Path)
    ap.add_argument("salida", type=Path)
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="la cabeza nueva usa 10 veces mas")
    ap.add_argument("--cada", type=int, default=5, help="epocas entre evaluaciones")
    ap.add_argument("--trabajadores", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--max-iters", type=int, default=0, help="para pruebas rapidas")
    ap.add_argument("--limite-val", type=int, default=0, help="para pruebas rapidas")
    ap.add_argument("--ver", type=int, default=0, help="solo guarda N recortes aumentados y sale")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    if disp == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    proc = AutoProcessor.from_pretrained(BASE)
    ds = Ski2DPose(args.raiz, "train", proc.image_mean, proc.image_std, aumentar=True)
    if args.ver:
        return ver_aumentos(ds, args.salida, args.ver)

    modelo = VitPoseForPoseEstimation.from_pretrained(
        BASE, num_labels=len(PUNTOS), id2label=dict(enumerate(PUNTOS)),
        label2id={n: i for i, n in enumerate(PUNTOS)}, ignore_mismatched_sizes=True)
    iniciar_cabeza(modelo)
    modelo.to(disp)
    cargador = DataLoader(ds, batch_size=args.lote, shuffle=True, num_workers=args.trabajadores,
                          pin_memory=disp == "cuda", drop_last=True, persistent_workers=args.trabajadores > 0)
    cabeza = [p for n, p in modelo.named_parameters() if n.startswith("head.")]
    resto = [p for n, p in modelo.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": resto, "lr": args.lr}, {"params": cabeza, "lr": args.lr * 10}],
                            weight_decay=0.05)
    total = args.max_iters or args.epocas * len(cargador)
    calentamiento = max(1, min(300, total // 10))
    plan = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / calentamiento if it < calentamiento
                                             else 0.5 * (1 + math.cos(math.pi * (it - calentamiento) / max(1, total - calentamiento))))
    escalador = torch.amp.GradScaler(enabled=disp == "cuda")

    args.salida.mkdir(parents=True, exist_ok=True)
    historial = open(args.salida / "historial.csv", "w", newline="")
    log = csv.writer(historial)
    log.writerow(["epoca", "iteracion", "perdida", "pck05_todos", "pck05_esquis", "pck10_todos", "minutos"])
    mejor, it, t0 = -1.0, 0, time.time()
    print(f"{disp}: {len(ds)} imagenes, {len(cargador)} lotes por epoca, {total} iteraciones")
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        suma, n, t_epoca = 0.0, 0, time.time()
        for x, mapas, peso in cargador:
            x, mapas, peso = x.to(disp, non_blocking=True), mapas.to(disp, non_blocking=True), peso.to(disp)
            with torch.autocast(disp, dtype=torch.float16, enabled=disp == "cuda"):
                pred = modelo(pixel_values=x).heatmaps
            perdida = (((pred.float() - mapas) ** 2).mean((2, 3)) * peso).mean()
            opt.zero_grad(set_to_none=True)
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            plan.step()
            suma, n, it = suma + perdida.item(), n + 1, it + 1
            if it >= total:
                break
        fin = it >= total or epoca == args.epocas
        print(f"epoca {epoca}: {(time.time() - t_epoca) / n:.2f} s por iteracion de {args.lote} imagenes")
        if epoca % args.cada == 0 or fin:
            errores, _ = evaluar(modelo, proc, args.raiz, dispositivo=disp, lote=16, limite=args.limite_val or None)
            todos = [d for v in errores.values() for d in v]
            esquis = [d for k, v in errores.items() if k.startswith(("ski_", "toes_", "heel_")) for d in v]
            p05 = pck(todos, 0.05)
            log.writerow([epoca, it, f"{suma / n:.6f}", f"{p05:.4f}", f"{pck(esquis, 0.05):.4f}",
                          f"{pck(todos, 0.10):.4f}", f"{(time.time() - t0) / 60:.1f}"])
            historial.flush()
            print(f"epoca {epoca}  perdida {suma / n:.6f}  ({(time.time() - t0) / 60:.1f} min)\n{resumen(errores)}")
            if p05 > mejor:
                mejor = p05
                modelo.save_pretrained(args.salida / "mejor")
                proc.save_pretrained(args.salida / "mejor")
                (args.salida / "mejor" / "resultado.txt").write_text(
                    f"epoca {epoca}\n{resumen(errores)}\n\n{por_punto(errores)}\n")
        else:
            print(f"epoca {epoca}  perdida {suma / n:.6f}  ({(time.time() - t0) / 60:.1f} min)")
        if fin:
            break
    historial.close()
    print(f"Mejor PCK@0.05 (24 puntos): {mejor:.1%}  ->  {args.salida / 'mejor'}")


if __name__ == "__main__":
    main()
