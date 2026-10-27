"""Punto de partida: ViTPose preentrenado en COCO (17 puntos, sin esquis) sobre validacion.

Solo puede evaluarse en los 12 puntos que COCO y Ski2DPose comparten. Uso:
    python src/baseline_vitpose.py data/ski2dpose out/baseline
"""
import sys
import time
from pathlib import Path

import cv2
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from dibujo import dibujar
from evaluar import evaluar, por_punto, resumen
from to_coco import PUNTOS

MODELO = "usyd-community/vitpose-base-simple"
# indice en Ski2DPose -> indice en COCO
COMUNES = {2: 6, 3: 8, 4: 10, 6: 5, 7: 7, 8: 9, 10: 12, 11: 14, 12: 16, 13: 11, 14: 13, 15: 15}
MUESTRAS = 6

RAIZ, SALIDA = Path(sys.argv[1]), Path(sys.argv[2])
SALIDA.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(10)

proc = AutoProcessor.from_pretrained(MODELO)
modelo = VitPoseForPoseEstimation.from_pretrained(MODELO)
t0 = time.time()
errores, predicciones = evaluar(modelo, proc, RAIZ, mapa=COMUNES)
seg = (time.time() - t0) / len(predicciones)

for n, (im, pred) in enumerate(predicciones):
    if n % (len(predicciones) // MUESTRAS) == 0:
        img = cv2.imread(str(RAIZ / im["file_name"]))
        kps = [(0, 0, 0)] * len(PUNTOS)
        for i, c in COMUNES.items():
            kps[i] = (*pred[c], 2)
        cv2.imwrite(str(SALIDA / f"{im['video']}_{n:03d}.jpg"), dibujar(img, kps))

texto = "\n".join([
    f"ViTPose-base COCO sin ajustar, {len(predicciones)} imagenes de validacion, {seg:.2f} s/imagen en CPU",
    "Solo los 12 puntos comunes con COCO; esquis y bastones: sin prediccion posible", "",
    resumen(errores), "", por_punto(errores)])
(SALIDA / "resultado.txt").write_text(texto + "\n")
print(texto)
