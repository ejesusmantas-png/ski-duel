"""Registro de fondo entre fotogramas de una misma camara (gira y hace zoom desde un punto fijo):
una homografia por fotograma hacia un fotograma de referencia, calculada con SIFT sobre el
fondo (sin el esquiador ni los graficos de la tele)."""
import cv2
import numpy as np

ESCALA = 0.5  # se registra a media resolucion (960x540)
# Zonas con graficos de la retransmision, en 1920x1080: barra derecha, crono, rotulos, logo.
GRAFICOS = [(1660, 0, 1920, 1080), (0, 930, 1920, 1080), (1600, 0, 1920, 90)]

_sift = cv2.SIFT_create(nfeatures=3000)
_emparejador = cv2.BFMatcher(cv2.NORM_L2)


def mascara(forma, cajas):
    """1 donde hay fondo util; 0 en los graficos y en las cajas de los esquiadores (con margen)."""
    m = np.ones(forma[:2], np.uint8)
    for x0, y0, x1, y1 in GRAFICOS:
        m[int(y0 * ESCALA):int(y1 * ESCALA), int(x0 * ESCALA):int(x1 * ESCALA)] = 0
    for c in cajas:
        if c is None:
            continue
        x, y, w, h = c
        x0, y0 = int(max(0, (x - 0.15 * w) * ESCALA)), int(max(0, (y - 0.15 * h) * ESCALA))
        m[y0:int((y + 1.15 * h) * ESCALA), x0:int((x + 1.15 * w) * ESCALA)] = 0
    return m


def rasgos(img, cajas=()):
    gris = cv2.cvtColor(cv2.resize(img, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return _sift.detectAndCompute(gris, mascara(gris.shape, cajas))


def homografia(a, b, minimo=25):
    """Homografia (en pixeles de 1920x1080) que lleva el fondo de a al de b, o (None, inliers)."""
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    pares = [m for m, n in _emparejador.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(pares) < minimo:
        return None, len(pares)
    pa = np.float32([ka[m.queryIdx].pt for m in pares]) / ESCALA
    pb = np.float32([kb[m.trainIdx].pt for m in pares]) / ESCALA
    H, ok = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    inl = int(ok.sum()) if ok is not None else 0
    return (H if inl >= minimo else None), inl


ESCALA_ECC = 0.25  # alineacion por intensidad a 480x270
_S = np.diag([ESCALA_ECC, ESCALA_ECC, 1.0])


def gris_ecc(img):
    g = cv2.cvtColor(cv2.resize(img, None, fx=ESCALA_ECC, fy=ESCALA_ECC, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32) / 255.0


def mascara_ecc(forma, cajas):
    m = np.full(forma[:2], 255, np.uint8)
    for x0, y0, x1, y1 in GRAFICOS:
        m[int(y0 * ESCALA_ECC):int(y1 * ESCALA_ECC), int(x0 * ESCALA_ECC):int(x1 * ESCALA_ECC)] = 0
    for c in cajas:
        if c is None:
            continue
        x, y, w, h = c
        x0, y0 = int(max(0, (x - 0.2 * w) * ESCALA_ECC)), int(max(0, (y - 0.2 * h) * ESCALA_ECC))
        m[y0:int((y + 1.2 * h) * ESCALA_ECC), x0:int((x + 1.2 * w) * ESCALA_ECC)] = 0
    return m


def desplazamiento(origen, destino):
    """Traslacion global (pixeles de 1920x1080) de origen a destino por correlacion de fase."""
    ventana = cv2.createHanningWindow(origen.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(origen, destino, ventana)
    H = np.eye(3)
    H[0, 2], H[1, 2] = dx / ESCALA_ECC, dy / ESCALA_ECC
    return H, resp


def ecc_robusto(origen, destino, mascara_destino):
    """ECC desde la identidad y desde la traslacion por correlacion de fase; se queda con el mejor."""
    mejor = ecc(origen, destino, mascara_destino)
    T, _ = desplazamiento(origen, destino)
    otro = ecc(origen, destino, mascara_destino, T)
    return otro if otro[1] > mejor[1] else mejor


def ecc(origen, destino, mascara_destino, H0=None, iteraciones=80):
    """Homografia (pixeles de 1920x1080) que lleva el fondo de origen al de destino, por intensidad.
    origen/destino: salidas de gris_ecc. H0: estimacion inicial en pixeles de 1920x1080."""
    W = np.eye(3, dtype=np.float32) if H0 is None else (_S @ np.linalg.inv(H0) @ np.linalg.inv(_S)).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iteraciones, 1e-6)
    try:
        # W lleva coordenadas de destino (plantilla) a coordenadas de origen.
        cc, W = cv2.findTransformECC(destino, origen, W, cv2.MOTION_HOMOGRAPHY, criterio, mascara_destino, 5)
    except cv2.error:
        return None, 0.0
    H = np.linalg.inv(_S) @ np.linalg.inv(W.astype(np.float64)) @ _S
    return H, float(cc)


def llevar(H, puntos):
    p = np.asarray(puntos, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)
