"""Detecta al esquiador (RT-DETR: persona + esquis) y su pose de 24 puntos (ViTPose ajustado)
en un tramo de una manga, por tiempo de carrera. Guarda los puntos en JSON y los fotogramas
dibujados. Uso:
    python src/pose_video.py carreras/solden2026_gs_hombres.json "ODERMATT Marco/manga1" 11.2 14.8 \
        runs/vitpose24/mejor out/pose/odermatt_m1 [--hilos 3]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, VitPoseForPoseEstimation

from dibujo import dibujar
from to_coco import PUNTOS

DETECTOR = "PekingU/rtdetr_v2_r18vd"
MARGEN = 0.05  # el mismo margen que las cajas de entrenamiento (to_coco)


def toca(persona, esquis):
    x0, y0, x1, y1 = persona
    return (esquis[0] < x1 + 0.3 * (x1 - x0) and esquis[2] > x0 - 0.3 * (x1 - x0)
            and esquis[1] < y1 + 0.3 * (y1 - y0) and esquis[3] > y0)


def caja_esquiador(det, id_persona, id_esquis, ancho, alto, anterior=None):
    """Persona con mas puntuacion (o la mas cercana a la anterior) unida a los esquis que la tocan.
    Si alguna persona lleva esquis pegados, solo se mira entre esas: asi no se cuela el publico."""
    personas = [(s, b) for s, l, b in det if l == id_persona]
    esquis = [b for s, l, b in det if l == id_esquis]
    con_esquis = [(s, b) for s, b in personas if any(toca(b, e) for e in esquis)]
    personas = con_esquis or personas
    if not personas:
        return None
    if anterior is not None:
        cx, cy = anterior[0] + anterior[2] / 2, anterior[1] + anterior[3] / 2
        s, b = min(personas, key=lambda p: np.hypot((p[1][0] + p[1][2]) / 2 - cx, (p[1][1] + p[1][3]) / 2 - cy) - 200 * p[0])
    else:
        s, b = max(personas, key=lambda p: p[0] * (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))
    x0, y0, x1, y1 = b
    for e in esquis:
        if toca(b, e):
            x0, y0, x1, y1 = min(x0, e[0]), min(y0, e[1]), max(x1, e[2]), max(y1, e[3])
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0.0, x0 - MARGEN * w), max(0.0, y0 - MARGEN * h)
    x1, y1 = min(float(ancho), x1 + MARGEN * w), min(float(alto), y1 + MARGEN * h)
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carrera")
    ap.add_argument("manga")
    ap.add_argument("desde", type=float)
    ap.add_argument("hasta", type=float)
    ap.add_argument("modelo")
    ap.add_argument("salida", type=Path)
    ap.add_argument("--hilos", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.hilos)

    carrera = json.loads(Path(args.carrera).read_text())
    video = carrera["video"]
    desfase = video["mangas"][args.manga]["desfase_s"]
    proc_det = AutoImageProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForObjectDetection.from_pretrained(DETECTOR).eval()
    etiquetas = {v: int(k) for k, v in detector.config.id2label.items()}
    id_persona, id_esquis = etiquetas["person"], etiquetas["skis"]
    proc_pose = AutoProcessor.from_pretrained(args.modelo)
    pose = VitPoseForPoseEstimation.from_pretrained(args.modelo).eval()

    args.salida.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video["archivo"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    primero, ultimo = int(np.ceil((args.desde + desfase) * fps)), int(np.floor((args.hasta + desfase) * fps))
    resultados, anterior, n = [], None, 0
    while n <= ultimo:
        ok, img = cap.read()  # sin saltos: el numero de fotograma tiene que ser exacto
        if not ok:
            break
        if n >= primero:
            foto = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                ent = proc_det(images=foto, return_tensors="pt")
                sal = detector(**ent)
            r = proc_det.post_process_object_detection(sal, threshold=0.4, target_sizes=[(img.shape[0], img.shape[1])])[0]
            det = [(float(s), int(l), [float(v) for v in b]) for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]
            caja = caja_esquiador(det, id_persona, id_esquis, img.shape[1], img.shape[0], anterior)
            fila = {"fotograma": n, "crono_s": round(n / fps - desfase, 3), "caja": caja}
            if caja is not None:
                anterior = caja
                with torch.no_grad():
                    ent = proc_pose(foto, boxes=[[caja]], return_tensors="pt")
                    sal = pose(**ent)
                p = proc_pose.post_process_pose_estimation(sal, boxes=[[caja]])[0][0]
                kps = p["keypoints"].numpy()
                scores = p["scores"].numpy()
                fila["puntos"] = {nombre: [round(float(kps[i][0]), 1), round(float(kps[i][1]), 1), round(float(scores[i]), 3)]
                                  for i, nombre in enumerate(PUNTOS)}
                dibujar(img, [(x, y, 2) for x, y in kps], grosor=3)
                x, y, w, h = caja
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
            cv2.imwrite(str(args.salida / f"{n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resultados.append(fila)
        n += 1
    (args.salida / "pose.json").write_text(json.dumps({"manga": args.manga, "desfase_s": desfase, "fps": fps,
                                                       "modelo": args.modelo, "fotogramas": resultados}))
    con = sum(1 for f in resultados if f["caja"] is not None)
    print(f"{args.manga}: {len(resultados)} fotogramas, esquiador encontrado en {con}")


if __name__ == "__main__":
    main()
