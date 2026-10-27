"""Mosaico con las etiquetas COCO dibujadas, para revisar la conversion a ojo. Uso:
    python src/ver_etiquetas.py data/ski2dpose out/etiquetas.jpg
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from dibujo import dibujar

RAIZ, SALIDA = Path(sys.argv[1]), Path(sys.argv[2])
POR_SPLIT, LADO = 3, 480

random.seed(0)
celdas = []
for split in ("train", "val"):
    coco = json.loads((RAIZ / f"coco_{split}.json").read_text())
    anns = {a["image_id"]: a for a in coco["annotations"]}
    for im in random.sample(coco["images"], POR_SPLIT):
        a = anns[im["id"]]
        img = cv2.imread(str(RAIZ / im["file_name"]))
        k = a["keypoints"]
        dibujar(img, [k[i:i + 3] for i in range(0, len(k), 3)])
        x, y, w, h = a["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        # Recorte cuadrado alrededor de la caja, para que el esquiador se vea grande.
        cx, cy, r = x + w / 2, y + h / 2, max(w, h) * 0.65
        x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
        recorte = img[y0:int(cy + r), x0:int(cx + r)]
        recorte = cv2.resize(recorte, (LADO, LADO))
        cv2.putText(recorte, f"{split} {im['video']}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        celdas.append(recorte)

filas = [np.hstack(celdas[i:i + POR_SPLIT]) for i in range(0, len(celdas), POR_SPLIT)]
SALIDA.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(SALIDA), np.vstack(filas))
print(SALIDA)
