"""Por que puerta va cada esquiador, leyendo la barra de seguimiento de la retransmision.

La barra vertical de la derecha marca con un triangulo el numero de la ultima
puerta pasada; cada puerta tiene su altura fija en la barra. Este script saca
la altura del triangulo en cada fotograma, la agrupa en niveles y guarda un
recorte de cada nivel para leer el numero (ver --mosaico). Ojo: los datos de
seguimiento llegan a ~5 Hz, asi que el momento del cambio tiene ~0,2 s de
margen; sirve para identificar la puerta, no para cronometrarla. Uso:
    python src/puertas.py carreras/solden2026_gs_hombres.json out/puertas
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COLUMNA = (slice(280, 880), slice(1712, 1772))  # zona de la barra en 1920x1080
X_LINEA = slice(26, 33)                          # la linea de la barra, dentro de COLUMNA
ESTABLE = 3                                      # fotogramas seguidos para aceptar un nivel


def triangulo(img):
    """Altura (y absoluta) del triangulo de la barra, o None si la barra no esta."""
    z = img[COLUMNA].astype(np.int16)
    linea = z[:, X_LINEA]
    clara = (linea.min(axis=2) > 150) | (linea.max(axis=2) - linea.min(axis=2) > 90)
    if clara.any(axis=1).mean() < 0.55:  # la barra ocupa casi toda la columna
        return None
    navy = (z[..., 0] > z[..., 2] + 40) & (z[..., 0] > z[..., 1] + 25) & (z[..., 0] < 170)
    num, _, st, cen = cv2.connectedComponentsWithStats(navy.astype(np.uint8))
    cands = []
    for i in range(1, num):
        x, y, w, h, area = st[i]
        relleno = area / float(w * h)
        if 26 <= w <= 52 and 18 <= h <= 46 and 0.35 <= relleno <= 0.68:  # triangulo, no circulo
            cands.append(cen[i][1] + COLUMNA[0].start)
    return float(cands[0]) if len(cands) == 1 else None


def main():
    carrera = json.loads(Path(sys.argv[1]).read_text())
    salida = Path(sys.argv[2])
    salida.mkdir(parents=True, exist_ok=True)
    video = carrera["video"]
    mangas = {k: m["desfase_s"] for k, m in video["mangas"].items() if "desfase_s" in m}
    cap = cv2.VideoCapture(video["archivo"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    fin = max(d + 56 for d in mangas.values())
    alturas = {k: [] for k in mangas}
    recortes = {}
    n = 0
    while n / fps <= fin:
        ok, img = cap.read()
        if not ok:
            break
        for k, d in mangas.items():
            crono = n / fps - d
            if -1 <= crono <= 56:
                y = triangulo(img)
                alturas[k].append((n, crono, y))
                if y is not None:
                    clave = (k, int(round(y / 4)))  # un recorte por nivel aproximado
                    if clave not in recortes:
                        yi = int(y)
                        recortes[clave] = img[yi - 24:yi + 24, 1712:1772].copy()
        n += 1

    # Niveles estables: cambios de altura que se mantienen ESTABLE fotogramas.
    resultado = {}
    for k, serie in alturas.items():
        pasos, nivel, cuenta, candidato = [], None, 0, None
        for n, crono, y in serie:
            if y is None:
                continue
            if nivel is not None and abs(y - nivel) <= 3:
                cuenta = 0
                continue
            if candidato is not None and abs(y - candidato[1]) <= 3:
                cuenta += 1
            else:
                candidato, cuenta = (n, y, crono), 1
            if cuenta >= ESTABLE:
                nivel = candidato[1]
                pasos.append({"fotograma": candidato[0], "crono_s": round(candidato[2], 3), "y": round(candidato[1], 1)})
                candidato, cuenta = None, 0
        resultado[k] = pasos
        print(f"{k}: {len(pasos)} niveles: " + ", ".join(f"{p['crono_s']:.2f}s@{p['y']:.0f}" for p in pasos))
    (salida / "niveles.json").write_text(json.dumps(resultado, indent=1))

    # Mosaico para leer el numero de cada nivel: un recorte ampliado por altura.
    vistos = {}
    for (k, clave), rec in sorted(recortes.items(), key=lambda x: x[0][1]):
        vistos.setdefault(clave, (k, rec))
    celdas = []
    for clave, (k, rec) in sorted(vistos.items()):
        if rec.shape[0] != 48:
            continue
        g = cv2.resize(rec, (120, 96), interpolation=cv2.INTER_CUBIC)
        cv2.putText(g, str(clave * 4), (2, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        celdas.append(g)
    celdas += [np.zeros_like(celdas[0])] * (-len(celdas) % 10)
    mosaico = np.vstack([np.hstack(celdas[i:i + 10]) for i in range(0, len(celdas), 10)])
    cv2.imwrite(str(salida / "niveles.png"), mosaico)
    print(salida / "niveles.png")


if __name__ == "__main__":
    main()
