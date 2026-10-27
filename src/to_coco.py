"""Convierte Ski2DPose a formato COCO de keypoints (train.json y val.json).

El reparto es por video, para que ninguna imagen de validacion se parezca a
otra de entrenamiento (fotogramas casi iguales del mismo descenso). Uso:
    python3 src/to_coco.py data/ski2dpose
"""
import json
import struct
import sys
from pathlib import Path

# Universiada de Granada (slalom) y un gigante de Almaty: dos tipos de
# carrera distintos y el gigante es el mismo tipo que el duelo de Solden.
VIDEOS_VAL = {"qxfgw1Kd98A", "uLW74013Wp0"}

PUNTOS = [
    "head", "neck",
    "shoulder_right", "elbow_right", "hand_right", "pole_basket_right",
    "shoulder_left", "elbow_left", "hand_left", "pole_basket_left",
    "hip_right", "knee_right", "ankle_right",
    "hip_left", "knee_left", "ankle_left",
    "ski_tip_right", "toes_right", "heel_right", "ski_tail_right",
    "ski_tip_left", "toes_left", "heel_left", "ski_tail_left",
]
# Pares que se intercambian al voltear la imagen en horizontal.
VOLTEO = [0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 13, 14, 15, 10, 11, 12, 20, 21, 22, 23, 16, 17, 18, 19]
_I = {n: i for i, n in enumerate(PUNTOS)}
ESQUELETO = [(_I[a], _I[b]) for a, b in [
    ("head", "neck"),
    ("neck", "shoulder_right"), ("shoulder_right", "elbow_right"), ("elbow_right", "hand_right"),
    ("hand_right", "pole_basket_right"),
    ("neck", "shoulder_left"), ("shoulder_left", "elbow_left"), ("elbow_left", "hand_left"),
    ("hand_left", "pole_basket_left"),
    ("shoulder_right", "hip_right"), ("shoulder_left", "hip_left"), ("hip_right", "hip_left"),
    ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
    ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
    ("ski_tip_right", "toes_right"), ("toes_right", "heel_right"), ("heel_right", "ski_tail_right"),
    ("ski_tip_left", "toes_left"), ("toes_left", "heel_left"), ("heel_left", "ski_tail_left"),
]]
MARGEN = 0.05  # caja = extremos de los puntos + 5 % por cada lado


def tam_webp(ruta):
    """Ancho y alto de un WebP leyendo solo la cabecera (VP8, VP8L o VP8X)."""
    with open(ruta, "rb") as f:
        c = f.read(30)
    tipo = c[12:16]
    if tipo == b"VP8 ":
        w, h = struct.unpack("<HH", c[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if tipo == b"VP8L":
        b = int.from_bytes(c[21:25], "little")
        return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
    if tipo == b"VP8X":
        return int.from_bytes(c[24:27], "little") + 1, int.from_bytes(c[27:30], "little") + 1
    raise ValueError(f"WebP desconocido: {ruta}")


def main():
    RAIZ = Path(sys.argv[1])
    IMAGENES = RAIZ / "images" / "Images_webp"
    etiquetas = json.loads((RAIZ / "labels" / "ski2dpose_labels.json").read_text())
    categoria = {"id": 1, "name": "skier", "supercategory": "person",
                 "keypoints": PUNTOS, "skeleton": [[a + 1, b + 1] for a, b in ESQUELETO],
                 "flip_index": VOLTEO}
    salida = {s: {"images": [], "annotations": [], "categories": [categoria]} for s in ("train", "val")}
    for video, tramos in etiquetas.items():
        s = "val" if video in VIDEOS_VAL else "train"
        for tramo, imagenes in tramos.items():
            for id_img, datos in imagenes.items():
                ruta = IMAGENES / video / tramo / f"{id_img}.webp"
                w, h = tam_webp(ruta)
                n = len(salida[s]["images"]) + 1
                kps, xs, ys = [], [], []
                for x, y, visible in datos["annotation"]:
                    px, py = x * w, y * h
                    kps += [round(px, 2), round(py, 2), 2 if visible else 1]
                    xs.append(px)
                    ys.append(py)
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                mx, my = (x1 - x0) * MARGEN, (y1 - y0) * MARGEN
                x0, y0 = max(0.0, x0 - mx), max(0.0, y0 - my)
                x1, y1 = min(float(w), x1 + mx), min(float(h), y1 + my)
                salida[s]["images"].append({"id": n, "file_name": str(ruta.relative_to(RAIZ)),
                                            "width": w, "height": h, "video": video})
                salida[s]["annotations"].append({
                    "id": n, "image_id": n, "category_id": 1, "iscrowd": 0,
                    "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                    "area": round((x1 - x0) * (y1 - y0), 2),
                    "keypoints": kps, "num_keypoints": len(PUNTOS)})
    for s, coco in salida.items():
        (RAIZ / f"coco_{s}.json").write_text(json.dumps(coco))
        ocultos = sum(a["keypoints"][2::3].count(1) for a in coco["annotations"])
        total = len(coco["annotations"]) * len(PUNTOS)
        print(f"{s}: {len(coco['images'])} imagenes, {ocultos / total:.1%} de puntos ocultos")


if __name__ == "__main__":
    main()
