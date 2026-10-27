"""Evaluacion por el mismo camino que la inferencia real: procesador de transformers
(recorte de la caja + decodificacion de los mapas de calor) y PCK por punto."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from metricas import errores_normalizados, pck, tamano
from to_coco import PUNTOS

GRUPOS = {
    "cuerpo": [n for n in PUNTOS if not n.startswith(("pole_", "ski_", "toes_", "heel_"))],
    # Los 12 que tambien tiene COCO: para comparar con el modelo sin ajustar.
    "comunes": [n for n in PUNTOS if n.startswith(("shoulder_", "elbow_", "hand_", "hip_", "knee_", "ankle_"))],
    "bastones": [n for n in PUNTOS if n.startswith("pole_")],
    "esquis": [n for n in PUNTOS if n.startswith(("ski_", "toes_", "heel_"))],
}


def evaluar(modelo, proc, raiz, split="val", dispositivo="cpu", lote=8, limite=None, mapa=None):
    """Devuelve ({punto: [error normalizado, ...]}, [(imagen, prediccion), ...]).

    mapa: {indice en Ski2DPose: indice en la salida del modelo}; por defecto, el mismo.
    Se usa la caja anotada (con el margen de to_coco) como si viniera del detector.
    """
    raiz = Path(raiz)
    coco = json.loads((raiz / f"coco_{split}.json").read_text())
    anns = {a["image_id"]: a for a in coco["annotations"]}
    imagenes = coco["images"][:limite] if limite else coco["images"]
    mapa = mapa or {i: i for i in range(len(PUNTOS))}
    idx = list(mapa)
    errores, predicciones = defaultdict(list), []
    modelo.eval()
    for i in range(0, len(imagenes), lote):
        bloque = imagenes[i:i + lote]
        fotos = [Image.open(raiz / im["file_name"]).convert("RGB") for im in bloque]
        cajas = [[anns[im["id"]]["bbox"]] for im in bloque]
        entrada = proc(fotos, boxes=cajas, return_tensors="pt").to(dispositivo)
        with torch.no_grad():
            salida = modelo(**entrada)
        for im, r in zip(bloque, proc.post_process_pose_estimation(salida, boxes=cajas)):
            pred = r[0]["keypoints"].cpu().numpy()
            k = np.array(anns[im["id"]]["keypoints"], float).reshape(-1, 3)
            for j, d in zip(idx, errores_normalizados(pred[[mapa[j] for j in idx]], k[idx], tamano(k))):
                errores[PUNTOS[j]].append(d)
            predicciones.append((im, pred))
    return errores, predicciones


def resumen(errores):
    """PCK y error mediano por grupo de puntos (solo los grupos con datos)."""
    lineas = []
    for grupo, nombres in {**GRUPOS, "todos": PUNTOS}.items():
        v = [d for n in nombres for d in errores.get(n, [])]
        if v:
            lineas.append(f"{grupo:<9} PCK@0.05 = {pck(v, 0.05):6.1%}  PCK@0.10 = {pck(v, 0.10):6.1%}  "
                          f"error mediano = {np.median(v):.1%} del tamano")
    return "\n".join(lineas)


def por_punto(errores):
    return "\n".join(f"  {n:<18} PCK@0.05 = {pck(v, 0.05):6.1%}  PCK@0.10 = {pck(v, 0.10):6.1%}"
                     for n, v in errores.items())
