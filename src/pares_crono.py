"""Hoja con los dos esquiadores en el mismo instante de carrera, cada N segundos de crono. Uso:
    python src/pares_crono.py carreras/solden2026_gs_hombres.json "A/manga1" "B/manga1" out/hojas/pares.jpg [--cada 2]
"""
import argparse
import json

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("carrera")
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("salida")
ap.add_argument("--cada", type=float, default=2.0)
ap.add_argument("--hasta", type=float, default=53.0)
ap.add_argument("--por-fila", type=int, default=3)
args = ap.parse_args()

carrera = json.load(open(args.carrera))
video = carrera["video"]
cap = cv2.VideoCapture(video["archivo"])
fps = cap.get(cv2.CAP_PROP_FPS)
cronos = np.arange(0, args.hasta + 1e-9, args.cada)
quiero = {}
for k in (args.a, args.b):
    d = video["mangas"][k]["desfase_s"]
    for c in cronos:
        quiero[int(round((c + d) * fps))] = (k, c)

W, H = 320, 180
fotos, n = {}, 0
while n <= max(quiero):
    ok, img = cap.read()
    if not ok:
        break
    if n in quiero:
        k, c = quiero[n]
        mini = cv2.resize(img, (W, H))
        cv2.rectangle(mini, (0, 0), (W, 20), (0, 0, 0), -1)
        cv2.putText(mini, f"{k.split()[0]}  {c:05.2f} s", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        fotos[(k, round(c, 3))] = mini
    n += 1

pares = [np.hstack([fotos[(args.a, round(c, 3))], fotos[(args.b, round(c, 3))]]) for c in cronos]
separador = np.full((H, 6, 3), 255, np.uint8)
pares += [np.zeros_like(pares[0])] * (-len(pares) % args.por_fila)
filas = [np.hstack(sum(([p, separador] for p in pares[i:i + args.por_fila]), [])[:-1])
         for i in range(0, len(pares), args.por_fila)]
cv2.imwrite(args.salida, np.vstack(filas), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(args.salida)
