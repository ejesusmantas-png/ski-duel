"""Hojas de contactos de un video: un fotograma cada N segundos, con su minuto encima. Uso:
    python src/hoja_contactos.py video.mp4 out/hojas/nombre [--cada 6] [--desde 0] [--hasta 0]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("salida", type=Path, help="prefijo de las imagenes de salida")
ap.add_argument("--cada", type=float, default=6.0)
ap.add_argument("--desde", type=float, default=0.0)
ap.add_argument("--hasta", type=float, default=0.0)
ap.add_argument("--columnas", type=int, default=6)
ap.add_argument("--filas", type=int, default=6)
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
duracion = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
hasta = args.hasta or duracion
W, H = 320, 180
celdas = []
t = args.desde
while t < hasta:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if not ok:
        break
    img = cv2.resize(img, (W, H))
    m, s = divmod(t, 60)
    cv2.rectangle(img, (0, 0), (92, 24), (0, 0, 0), -1)
    cv2.putText(img, f"{int(m)}:{s:05.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    celdas.append(img)
    t += args.cada

por_hoja = args.columnas * args.filas
args.salida.parent.mkdir(parents=True, exist_ok=True)
for n in range(0, len(celdas), por_hoja):
    grupo = celdas[n:n + por_hoja]
    grupo += [np.zeros_like(celdas[0])] * (por_hoja - len(grupo))
    filas = [np.hstack(grupo[i:i + args.columnas]) for i in range(0, por_hoja, args.columnas)]
    ruta = f"{args.salida}_{n // por_hoja + 1}.jpg"
    cv2.imwrite(ruta, np.vstack(filas), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(ruta)
print(f"{fps:.2f} fps, {duracion:.1f} s")
