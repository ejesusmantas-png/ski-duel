"""Medidas de postura por fotograma a partir de la pose de 24 puntos, suavizadas en el tiempo.

Por fotograma: inclinacion del cuerpo (botas -> hombros, respecto a la vertical de la imagen),
altura de la cadera (en largos de pierna), rodilla mas flexionada y mas extendida (asi da
igual si el modelo confunde izquierda y derecha), angulo de los esquis respecto a la
horizontal, apertura entre esquis (paralelismo) y separacion de botas.
Los angulos dependen de la camara: solo se comparan en el mismo plano. Uso:
    python src/postura.py out/pose/braathen_m1_completa out/pose/odermatt_m1_completa out/postura \
        --alinear 12.97 13.13 --nombres "Pinheiro Braathen" "Odermatt"
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLAVE = ["shoulder_right", "shoulder_left", "hip_right", "hip_left", "knee_right", "knee_left",
         "ankle_right", "ankle_left", "toes_right", "heel_right", "toes_left", "heel_left",
         "ski_tip_right", "ski_tail_right", "ski_tip_left", "ski_tail_left"]
MIN_CONFIANZA = 0.35
VENTANA = 5  # mediana movil, en fotogramas
METRICAS = ["inclinacion", "altura_cadera", "rodilla_flex", "rodilla_ext", "angulo_esquis",
            "apertura_esquis", "separacion_botas"]


def angulo(a, b, c):
    u, w = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))))


def medir(puntos):
    p = {k: np.array(v[:2], float) for k, v in puntos.items()}
    conf = float(np.mean([puntos[k][2] for k in CLAVE]))
    botas_d = (p["toes_right"] + p["heel_right"]) / 2
    botas_i = (p["toes_left"] + p["heel_left"]) / 2
    botas = (botas_d + botas_i) / 2
    hombros = (p["shoulder_right"] + p["shoulder_left"]) / 2
    cadera = (p["hip_right"] + p["hip_left"]) / 2
    pierna = np.mean([np.linalg.norm(p[f"hip_{l}"] - p[f"knee_{l}"]) + np.linalg.norm(p[f"knee_{l}"] - p[f"ankle_{l}"])
                      for l in ("right", "left")])
    v = hombros - botas
    rod = [angulo(p[f"hip_{l}"], p[f"knee_{l}"], p[f"ankle_{l}"]) for l in ("right", "left")]
    esq = [p[f"ski_tip_{l}"] - p[f"ski_tail_{l}"] for l in ("right", "left")]
    ang_esq = [float(np.degrees(np.arctan2(-e[1], abs(e[0]) + 1e-9))) for e in esq]  # + = punta hacia arriba
    apertura = angulo(esq[0], np.zeros(2), esq[1])
    return conf, {
        "inclinacion": float(np.degrees(np.arctan2(v[0], -v[1]))),  # con signo: + hacia la derecha de la imagen
        "altura_cadera": float((botas[1] - cadera[1]) / pierna),
        "rodilla_flex": min(rod), "rodilla_ext": max(rod),
        "angulo_esquis": float(np.mean(ang_esq)),
        "apertura_esquis": apertura,
        "separacion_botas": float(np.linalg.norm(botas_d - botas_i) / pierna),
    }


def suavizar(valores, validos, cortes):
    """Mediana movil dentro de cada tramo sin cortes de plano, solo con fotogramas validos."""
    out = np.full(len(valores), np.nan)
    ini = 0
    for fin in list(cortes) + [len(valores)]:
        for i in range(ini, fin):
            if not validos[i]:
                continue
            a, b = max(ini, i - VENTANA // 2), min(fin, i + VENTANA // 2 + 1)
            vent = [valores[j] for j in range(a, b) if validos[j]]
            if len(vent) >= 3:
                out[i] = np.median(vent)
        ini = fin
    return out


def cortes_por_caja(filas):
    """Cambios de plano aproximados: la caja del esquiador salta o cambia mucho de tamano."""
    cortes = []
    for i in range(1, len(filas)):
        a, b = filas[i - 1]["caja"], filas[i]["caja"]
        if a is None or b is None:
            continue
        ta, tb = max(a[2], a[3]), max(b[2], b[3])
        salto = np.hypot(b[0] + b[2] / 2 - a[0] - a[2] / 2, b[1] + b[3] / 2 - a[1] - a[3] / 2)
        if salto > 0.6 * max(ta, tb) or max(ta, tb) / max(1, min(ta, tb)) > 1.6:
            cortes.append(i)
    return cortes


def procesar(ruta):
    d = json.loads((Path(ruta) / "pose.json").read_text())
    filas = d["fotogramas"]
    medidas, validos = [], []
    for f in filas:
        if f.get("puntos"):
            conf, m = medir(f["puntos"])
            medidas.append(m)
            validos.append(conf >= MIN_CONFIANZA)
            f["confianza"] = conf
        else:
            medidas.append({k: np.nan for k in METRICAS})
            validos.append(False)
            f["confianza"] = 0.0
    cortes = cortes_por_caja(filas)
    suaves = {k: suavizar([m[k] for m in medidas], validos, cortes) for k in METRICAS}
    return d, filas, medidas, validos, suaves, cortes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("salida", type=Path)
    ap.add_argument("--alinear", type=float, nargs=2, help="crono del mismo suceso en A y en B (p. ej. toque del palo)")
    ap.add_argument("--nombres", nargs=2, default=["A", "B"])
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    res = {}
    for nombre, ruta in zip(args.nombres, (args.a, args.b)):
        d, filas, medidas, validos, suaves, cortes = procesar(ruta)
        res[nombre] = (filas, validos, suaves)
        with open(args.salida / f"{nombre.replace(' ', '_').lower()}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fotograma", "crono_s", "confianza", "valido"] + METRICAS + [f"{k}_suave" for k in METRICAS])
            for i, f in enumerate(filas):
                w.writerow([f["fotograma"], f["crono_s"], round(f["confianza"], 3), int(validos[i])]
                           + [round(medidas[i][k], 2) for k in METRICAS] + [round(suaves[k][i], 2) for k in METRICAS])
        print(f"{nombre}: {len(filas)} fotogramas, {sum(validos)} validos ({sum(validos) / len(filas):.0%}), "
              f"{len(cortes)} cambios de plano aproximados")

    if args.alinear:
        lineas = [f"Alineado por el suceso: {args.nombres[0]} {args.alinear[0]} s, {args.nombres[1]} {args.alinear[1]} s",
                  "t respecto al suceso | " + " | ".join(f"{n}: incl. / cadera / rod. flex / rod. ext / ang. esquis" for n in args.nombres)]
        for dt in np.round(np.arange(-0.4, 0.401, 0.04), 2):
            celdas = []
            for nombre, t0 in zip(args.nombres, args.alinear):
                filas, validos, suaves = res[nombre]
                i = int(np.argmin([abs(f["crono_s"] - (t0 + dt)) for f in filas]))
                celdas.append(" / ".join("  -  " if np.isnan(suaves[k][i]) else f"{suaves[k][i]:5.1f}" if k != "altura_cadera" else f"{suaves[k][i]:4.2f}"
                                         for k in ("inclinacion", "altura_cadera", "rodilla_flex", "rodilla_ext", "angulo_esquis")))
            lineas.append(f"{dt:+.2f} s | " + " | ".join(celdas))
        texto = "\n".join(lineas)
        (args.salida / "alineado.txt").write_text(texto + "\n")
        print(texto)


if __name__ == "__main__":
    main()
