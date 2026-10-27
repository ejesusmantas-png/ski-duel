"""Dibujo del esqueleto de 24 puntos (cuerpo, bastones y esquis) sobre una imagen."""
import cv2

from to_coco import ESQUELETO, PUNTOS

# BGR. Derecha e izquierda en colores distintos para ver si se cruzan.
DERECHA, IZQUIERDA, CENTRO = (60, 200, 255), (255, 160, 60), (230, 230, 230)
ESQUI_DER, ESQUI_IZQ = (0, 90, 255), (255, 60, 0)


def color_de(nombre):
    if nombre.startswith(("ski_", "toes_", "heel_")):
        return ESQUI_DER if nombre.endswith("right") else ESQUI_IZQ
    if nombre.endswith("right"):
        return DERECHA
    if nombre.endswith("left"):
        return IZQUIERDA
    return CENTRO


def dibujar(img, kps, grosor=3, color=None):
    """kps: lista de (x, y, v) en pixeles; v = 0 no hay punto, 1 oculto, 2 visible."""
    for a, b in ESQUELETO:
        (xa, ya, va), (xb, yb, vb) = kps[a], kps[b]
        if va and vb:
            c = color or color_de(PUNTOS[b])
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), c, grosor, cv2.LINE_AA)
    for i, (x, y, v) in enumerate(kps):
        if v:
            c = color or color_de(PUNTOS[i])
            # Los ocultos, solo con borde: su posicion es estimada.
            cv2.circle(img, (int(x), int(y)), grosor + 2, c, -1 if v == 2 else 1, cv2.LINE_AA)
    return img
