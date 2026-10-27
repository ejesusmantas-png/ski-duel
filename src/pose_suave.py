"""Repite la pose de un tramo con las cajas del detector suavizadas en el tiempo (mediana de 5
fotogramas), para que no salten cuando el detector une la puerta al esquiador o coge solo una
parte de el. El tramo tiene que ser un solo plano, sin cortes. Guarda tambien los fotogramas
limpios en PNG para montar el video. Uso:
    python src/pose_suave.py out/pose/braathen_m1_completa 1247 1300 runs/vitpose24/mejor out/pose/braathen_puerta8
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

from to_coco import PUNTOS

VENTANA = 5


def suavizar_cajas(cajas):
    """cajas: [x, y, w, h] o None por fotograma. Mediana de las esquinas en la ventana."""
    esquinas = [None if c is None else np.array([c[0], c[1], c[0] + c[2], c[1] + c[3]]) for c in cajas]
    salida = []
    for i in range(len(cajas)):
        vecinos = [e for e in esquinas[max(0, i - VENTANA // 2):i + VENTANA // 2 + 1] if e is not None]
        if not vecinos:
            salida.append(None)
            continue
        x0, y0, x1, y1 = np.median(vecinos, axis=0)
        salida.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", type=Path, help="carpeta con el pose.json de pose_video.py")
    ap.add_argument("desde", type=int, help="primer fotograma")
    ap.add_argument("hasta", type=int, help="ultimo fotograma")
    ap.add_argument("modelo")
    ap.add_argument("salida", type=Path)
    args = ap.parse_args()

    d = json.loads((args.pose / "pose.json").read_text())
    filas = [f for f in d["fotogramas"] if args.desde <= f["fotograma"] <= args.hasta]
    cajas = suavizar_cajas([f["caja"] for f in filas])
    proc = AutoProcessor.from_pretrained(args.modelo)
    modelo = VitPoseForPoseEstimation.from_pretrained(args.modelo).eval()
    carrera = json.loads(Path("carreras/solden2026_gs_hombres.json").read_text())
    cap = cv2.VideoCapture(carrera["video"]["archivo"])
    (args.salida / "fotogramas").mkdir(parents=True, exist_ok=True)

    por_numero = {f["fotograma"]: (f, c) for f, c in zip(filas, cajas)}
    resultados, n = [], 0
    while n <= args.hasta:
        ok, img = cap.read()  # sin saltos: el numero de fotograma tiene que ser exacto
        if not ok:
            break
        if n in por_numero:
            f, caja = por_numero[n]
            cv2.imwrite(str(args.salida / "fotogramas" / f"{n:06d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            fila = {"fotograma": n, "crono_s": f["crono_s"], "caja": caja, "caja_detector": f["caja"]}
            if caja is not None:
                foto = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    sal = modelo(**proc(foto, boxes=[[caja]], return_tensors="pt"))
                p = proc.post_process_pose_estimation(sal, boxes=[[caja]])[0][0]
                kps, scores = p["keypoints"].numpy(), p["scores"].numpy()
                fila["puntos"] = {nombre: [round(float(kps[i][0]), 1), round(float(kps[i][1]), 1), round(float(scores[i]), 3)]
                                  for i, nombre in enumerate(PUNTOS)}
            resultados.append(fila)
        n += 1
    (args.salida / "pose.json").write_text(json.dumps({**{k: v for k, v in d.items() if k != "fotogramas"},
                                                       "cajas": "suavizadas", "fotogramas": resultados}))
    print(f"{d['manga']}: {len(resultados)} fotogramas -> {args.salida}")


if __name__ == "__main__":
    main()
