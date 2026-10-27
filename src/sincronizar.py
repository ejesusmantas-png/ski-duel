"""Tiempo de carrera de cada fotograma leyendo el crono de la retransmision.

Lee las cifras de la caja del crono en cada fotograma (plantillas aprendidas del
propio video) y calcula el desfase (crono = segundo del video - desfase) que
cumplen todas las lecturas: con decimas que cambian cada 2,5 fotogramas, queda
fijado con unos 0,01 s de margen. Los tiempos congelados con centesimas
(intermedios y meta) se comparan con los oficiales. Uso:
    python src/sincronizar.py carreras/solden2026_gs_hombres.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CAJA = (slice(951, 1023), slice(1663, 1830))   # interior de la caja del crono en 1920x1080 (sin nieve)
GLIFO = (16, 24)                                # tamano normalizado de cada cifra
MARGEN_FASE = 0.25  # para las plantillas, solo fotogramas lejos de un cambio de decima


def leer(video, tramos):
    """Una pasada sin saltos. Por tramo: fotogramas, mascara blanca y si la caja es azul."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fin = max(b for _, b in tramos.values())
    datos = {k: [] for k in tramos}
    n = 0
    while n / fps <= fin:
        ok, img = cap.read()
        if not ok:
            break
        for k, (a, b) in tramos.items():
            if a <= n / fps <= b:
                caja = img[CAJA].astype(np.int16)
                azul = (caja[..., 0] > caja[..., 2] + 40) & (caja[..., 0] > caja[..., 1] + 25)
                datos[k].append((n, caja.min(axis=2) > 170, float(azul.mean())))
        n += 1
    return datos, fps


def glifos(mascara):
    """Cifras y punto decimal, de izquierda a derecha: [(tipo, imagen normalizada)]."""
    num, _, stats, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), connectivity=8)
    comps = [stats[i] for i in range(1, num) if stats[i][cv2.CC_STAT_AREA] >= 8]
    if not comps:
        return []
    alto = max(c[cv2.CC_STAT_HEIGHT] for c in comps)
    salida = []
    for x, y, w, h, area in sorted(comps, key=lambda c: c[0]):
        if h >= 0.6 * alto:
            g = cv2.resize(mascara[y:y + h, x:x + w].astype(np.float32), GLIFO, interpolation=cv2.INTER_AREA)
            salida.append(("d", g))
        elif h <= 0.35 * alto and w <= 0.5 * alto:
            salida.append((".", None))
    return salida


def formato(gs):
    """'d.d' en marcha (decimas), 'd.dd' congelado (centesimas) o None."""
    tipos = "".join(t for t, _ in gs)
    if tipos.count(".") != 1:
        return None
    entera, dec = tipos.split(".")
    if not entera or set(entera + dec) != {"d"}:
        return None
    return {1: "marcha", 2: "congelado"}.get(len(dec))


def plantillas(lecturas, fps, desfase):
    """Media de cada cifra 0-9 en fotogramas cuyo valor se conoce sin ambiguedad."""
    acum = {str(d): [] for d in range(10)}
    for n, gs in lecturas:
        if formato(gs) != "marcha":
            continue
        c = n / fps - desfase
        fase = (c * 10) % 1
        if not (MARGEN_FASE < fase < 1 - MARGEN_FASE) or c < 0:
            continue
        texto = f"{c:.2f}"[:-1]  # decimas por truncamiento, como el crono
        cifras = [g for t, g in gs if t == "d"]
        if len(cifras) != len(texto.replace(".", "")):
            continue
        for ch, g in zip(texto.replace(".", ""), cifras):
            acum[ch].append(g)
    falta = [d for d, v in acum.items() if not v]
    if falta:
        raise RuntimeError(f"Sin ejemplos para las cifras {falta}: revisa el desfase aproximado")
    return {d: np.mean(v, axis=0) for d, v in acum.items()}


def leer_valor(gs, plant):
    texto = ""
    for t, g in gs:
        if t == ".":
            texto += "."
        else:
            texto += min(plant, key=lambda d: float(((plant[d] - g) ** 2).sum()))
    return float(texto)


def desfase_por_lecturas(marcha, fps):
    """Desfase que cumplen mas lecturas: valor <= t - desfase < valor + 0,1."""
    t = np.array([n / fps for n, _ in marcha])
    v = np.array([x for _, x in marcha])
    candidatos = np.arange((t - v - 0.1).min(), (t - v).max() + 0.001, 0.001)
    cumplen = np.array([((v <= t - d + 1e-9) & (t - d < v + 0.1 - 1e-9)).mean() for d in candidatos])
    mejores = candidatos[cumplen >= cumplen.max() - 1e-9]
    return float((mejores.min() + mejores.max()) / 2), (float(mejores.min()), float(mejores.max())), float(cumplen.max())


def main():
    ruta = Path(sys.argv[1])
    carrera = json.loads(ruta.read_text())
    video = carrera["video"]
    mangas = {k: v for k, v in video["mangas"].items() if "desfase_s" in v}
    tramos = {k: (v["desfase_s"] - 3, v["desfase_s"] + 60) for k, v in mangas.items()}
    datos, fps = leer(video["archivo"], tramos)
    lecturas = {k: [(n, glifos(m)) for n, m, azul in d if azul > 0.3] for k, d in datos.items()}

    # Las plantillas salen de la primera manga con desfase de referencia (fijado a mano).
    ref = video["desfase_referencia"]
    plant = plantillas(lecturas[ref["manga"]], fps, ref["desfase_s"])

    for k, lec in lecturas.items():
        marcha, congelados = [], []
        for n, gs in lec:
            f = formato(gs)
            if f == "marcha":
                valor = leer_valor(gs, plant)
                if valor > 0:  # antes de la salida el crono marca 0.0 parado
                    marcha.append((n, valor))
            elif f == "congelado":
                congelados.append((n, leer_valor(gs, plant)))
        desfase, intervalo, cumplen = desfase_por_lecturas(marcha, fps)

        # Cada congelado: primer fotograma en que aparece y su valor.
        eventos = []
        for n, v in congelados:
            if not eventos or v != eventos[-1]["valor_s"] or n - eventos[-1]["ultimo"] > 2:
                eventos.append({"valor_s": v, "fotograma": n, "ultimo": n})
            else:
                eventos[-1]["ultimo"] = n
        nombre, manga = k.split("/")
        oficial = carrera["esquiadores"][nombre][manga]
        comprobacion = []
        for punto in ("int1", "int2", "meta"):
            ev = next((e for e in eventos if abs(e["valor_s"] - oficial[punto]) < 0.005), None)
            comprobacion.append({
                "punto": punto, "oficial_s": oficial[punto],
                "leido_s": ev["valor_s"] if ev else None,
                # retraso del grafico: aparicion del congelado - momento real de paso
                "retraso_grafico_s": round(ev["fotograma"] / fps - (desfase + oficial[punto]), 3) if ev else None})
        retrasos = [c["retraso_grafico_s"] for c in comprobacion if c["retraso_grafico_s"] is not None]
        mangas[k].update({"desfase_s": round(desfase, 4), "sincronizacion": {
            "desfase_intervalo_s": [round(x, 4) for x in intervalo],
            "lecturas_en_marcha": len(marcha), "lecturas_que_cumplen": round(cumplen, 4),
            # Las que no cumplen van una decima por detras justo en el cambio: el grafico
            # tarda ~1 fotograma en actualizarse, igual para todos los esquiadores.
            "retraso_medio_grafico_s": round(float(np.mean(retrasos)), 3) if retrasos else None,
            "intermedios": comprobacion}})
        print(f"{k}: desfase {desfase:.3f} s (intervalo {intervalo[0]:.3f}-{intervalo[1]:.3f}), "
              f"{len(marcha)} lecturas, cumplen {cumplen:.1%}")
        for c in comprobacion:
            print(f"   {c['punto']:<5} oficial {c['oficial_s']:6.2f}  leido {c['leido_s']}  "
                  f"retraso del grafico {c['retraso_grafico_s']} s")
    video["nota_desfase"] = "crono en pantalla = segundo del video - desfase; calculado por src/sincronizar.py"
    ruta.write_text(json.dumps(carrera, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
