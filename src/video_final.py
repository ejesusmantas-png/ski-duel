"""Video para LinkedIn (1080x1350, 4:5) del duelo en la puerta 8:
  1. mismo tiempo de carrera: Pinheiro Braathen toca el palo y a Odermatt aun le faltan 0,14 s;
  2. mismo sitio: cada manga alineada por su toque del palo, a camara lenta, con la inclinacion;
  3. grafica de la inclinacion; 4. cierre con los datos.
Antes hay que pasar pose_suave.py y postura.py por el tramo de la puerta 8. Uso:
    python src/video_final.py out/video/ski_duel.mp4 [--fijos 10 60 200]
"""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ANCHO, ALTO, FPS = 1080, 1350, 25
CABECERA = 226                  # franja de titulos; debajo, dos paneles de 560 y una raya de 4
PANEL_ALTO = 560
Y_PANEL = (CABECERA, CABECERA + PANEL_ALTO + 4)
RECORTE_Y = (42, 1038)          # del 1920x1080 original, con la proporcion del panel
FUENTES = Path("data/fuentes")
CREDITOS = "Video: FIS · ORF · Eurosport   ·   Pose: ViTPose fine-tuned on Ski2DPose (EPFL)"

# Paleta validada (skill dataviz, modo oscuro): superficie, tintas y las dos series.
SUPERFICIE, PLANO = (26, 26, 25), (13, 13, 13)
TINTA, TINTA_2, TENUE = (240, 239, 236), (195, 194, 183), (137, 135, 129)
REJILLA, EJE = (44, 44, 42), (56, 56, 53)

CARRERA = json.loads(Path("carreras/solden2026_gs_hombres.json").read_text())
ESQUIADORES = {
    "B": {"nombre": "Pinheiro Braathen", "pose": "out/pose/braathen_puerta8", "manga": "PINHEIRO BRAATHEN Lucas/manga1",
          "postura": "out/postura_puerta8/pinheiro_braathen.csv", "toque": 12.97, "color": (217, 89, 38)},
    "O": {"nombre": "Odermatt", "pose": "out/pose/odermatt_puerta8", "manga": "ODERMATT Marco/manga1",
          "postura": "out/postura_puerta8/odermatt.csv", "toque": 13.13, "color": (57, 135, 229)},
}
DESFASE = {k: CARRERA["video"]["mangas"][e["manga"]]["desfase_s"] for k, e in ESQUIADORES.items()}
MISMO_CRONO = round((DESFASE["O"] - DESFASE["B"]) * FPS)                                      # 2598
MISMO_SITIO = MISMO_CRONO + round((ESQUIADORES["O"]["toque"] - ESQUIADORES["B"]["toque"]) * FPS)  # 2602

ESCENA1 = range(1248, 1281)      # fotogramas de B: crono 11,71 -> 12,99 (toca el palo entre 1279 y 1280)
ESCENA2 = range(1267, 1296)      # fotogramas de B: -0,50 -> +0,62 s respecto a su toque
LENTO1, LENTO2 = 2, 4            # cada fotograma se repite: 0,5x y 0,25x
PAUSA1, FUNDIDO = 50, 10

CUERPO = [("head", "neck"), ("neck", "shoulder_right"), ("neck", "shoulder_left"),
          ("shoulder_right", "elbow_right"), ("elbow_right", "hand_right"),
          ("shoulder_left", "elbow_left"), ("elbow_left", "hand_left"),
          ("shoulder_right", "hip_right"), ("shoulder_left", "hip_left"), ("hip_right", "hip_left"),
          ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
          ("hip_left", "knee_left"), ("knee_left", "ankle_left")]
ESQUIS = [("ski_tip_right", "toes_right", "heel_right", "ski_tail_right"),
          ("ski_tip_left", "toes_left", "heel_left", "ski_tail_left")]
BASTONES = [("hand_right", "pole_basket_right"), ("hand_left", "pole_basket_left")]
MIN_PUNTO, MIN_CESTA = 0.3, 0.5


# ---------- datos ----------

def cargar(clave):
    e = ESQUIADORES[clave]
    d = json.loads((Path(e["pose"]) / "pose.json").read_text())
    filas = {f["fotograma"]: f for f in d["fotogramas"]}
    with open(e["postura"]) as fh:
        incl = {int(r["fotograma"]): float(r["inclinacion_suave"]) for r in csv.DictReader(fh)}
    # el hueco suelto que deja el filtro de confianza, por interpolacion
    nums = sorted(incl)
    validos = [n for n in nums if not np.isnan(incl[n])]
    for n in nums:
        if np.isnan(incl[n]):
            incl[n] = float(np.interp(n, validos, [incl[v] for v in validos]))
    return filas, incl


@lru_cache(maxsize=256)
def fotograma(clave, n):
    img = cv2.imread(str(Path(ESQUIADORES[clave]["pose"]) / "fotogramas" / f"{n:06d}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def puntos_a(fila, escala, dx, dy):
    """Puntos del fotograma original llevados al lienzo: x' = x * escala + dx."""
    return {k: (v[0] * escala + dx, v[1] * escala + dy, v[2]) for k, v in fila["puntos"].items()}


def medio(p, *nombres):
    return np.mean([p[n][:2] for n in nombres], axis=0)


# ---------- dibujo sobre el video ----------

def _pt(p):
    return int(round(p[0] * 16)), int(round(p[1] * 16))  # coma fija para cv2 (shift=4)


def esqueleto(lienzo, p, color, grosor):
    lineas = [(p[a], p[b], grosor) for a, b in CUERPO if p[a][2] >= MIN_PUNTO and p[b][2] >= MIN_PUNTO]
    for lado in ("right", "left"):
        bota = (*medio(p, f"toes_{lado}", f"heel_{lado}"), 1.0)
        if p[f"ankle_{lado}"][2] >= MIN_PUNTO:
            lineas.append((p[f"ankle_{lado}"], bota, grosor))
    for esqui in ESQUIS:
        lineas += [(p[a], p[b], grosor + 2) for a, b in zip(esqui, esqui[1:]) if p[a][2] >= MIN_PUNTO and p[b][2] >= MIN_PUNTO]
    lineas += [(p[a], p[b], max(2, grosor - 1)) for a, b in BASTONES if p[a][2] >= MIN_PUNTO and p[b][2] >= MIN_CESTA]
    juntas = [p[k] for k in ("neck", "shoulder_right", "shoulder_left", "elbow_right", "elbow_left", "hand_right",
                             "hand_left", "hip_right", "hip_left", "knee_right", "knee_left", "ankle_right", "ankle_left")
              if p[k][2] >= MIN_PUNTO]
    # sombra suave debajo, para que se lea sobre la nieve
    sombra = lienzo.copy()
    for a, b, g in lineas:
        cv2.line(sombra, _pt(a), _pt(b), (0, 0, 0), g + 5, cv2.LINE_AA, 4)
    lienzo[:] = cv2.addWeighted(sombra, 0.35, lienzo, 0.65, 0)
    for a, b, g in lineas:
        cv2.line(lienzo, _pt(a), _pt(b), color, g, cv2.LINE_AA, 4)
    r = grosor + 1
    for j in juntas:
        cv2.circle(lienzo, _pt(j), r * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
        cv2.circle(lienzo, _pt(j), r * 16, color, 2, cv2.LINE_AA, 4)
    if p["head"][2] >= MIN_PUNTO and p["neck"][2] >= MIN_PUNTO:
        radio = 0.55 * np.hypot(p["head"][0] - p["neck"][0], p["head"][1] - p["neck"][1])
        cv2.circle(lienzo, _pt(p["head"]), int(radio * 16), color, grosor, cv2.LINE_AA, 4)


def inclinacion(lienzo, p, color):
    """Linea botas -> hombros, la vertical y el arco entre las dos: lo que mide la grafica."""
    botas = medio(p, "toes_right", "heel_right", "toes_left", "heel_left")
    hombros = medio(p, "shoulder_right", "shoulder_left")
    largo = float(np.linalg.norm(hombros - botas))
    arriba = botas + np.array([0, -largo * 0.95])
    sombra = lienzo.copy()
    for a, b, g in ((botas, hombros, 9), (botas, arriba, 6)):
        cv2.line(sombra, _pt(a), _pt(b), (0, 0, 0), g, cv2.LINE_AA, 4)
    lienzo[:] = cv2.addWeighted(sombra, 0.35, lienzo, 0.65, 0)
    cv2.line(lienzo, _pt(botas), _pt(arriba), (255, 255, 255), 2, cv2.LINE_AA, 4)
    cv2.line(lienzo, _pt(botas), _pt(hombros), (255, 255, 255), 4, cv2.LINE_AA, 4)
    fin = float(np.degrees(np.arctan2(hombros[1] - botas[1], hombros[0] - botas[0])))
    r = int(largo * 0.38)
    a0, a1 = sorted((-90.0, fin))
    cv2.ellipse(lienzo, _pt(botas), (r * 16, r * 16), 0, a0, a1, color, 5, cv2.LINE_AA, 4)
    cv2.circle(lienzo, _pt(botas), 6 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)


# ---------- texto ----------

ARCHIVOS = {"titulo": "InterDisplay-Bold.ttf", "semi": "Inter-SemiBold.ttf", "medio": "Inter-Medium.ttf",
            "normal": "Inter-Regular.ttf"}


@lru_cache(maxsize=64)
def fuente(peso, tam):
    return ImageFont.truetype(str(FUENTES / ARCHIVOS[peso]), tam)


def ancho_texto(texto, peso, tam, tabular=False):
    return fuente(peso, tam).getlength(texto, features=["tnum"] if tabular else None)


def texto(dib, xy, cadena, peso, tam, color, alfa=1.0, ancla="ls", tabular=False):
    """tabular: cifras de ancho fijo, para los numeros que cambian de un fotograma a otro."""
    dib.text(xy, cadena, font=fuente(peso, tam), fill=(*color, int(255 * alfa)), anchor=ancla,
             features=["tnum"] if tabular else None)


def chip(dib, x, y, partes, alfa=1.0, alto=58, derecha=False, centro=False):
    """partes: ('punto', color) o (texto, peso, tam, color). x, y: esquina superior izquierda
    (o derecha, o el centro). Las cifras van tabulares: los chips llevan los contadores."""
    huecos, anchos = 14, []
    for parte in partes:
        anchos.append(16 if parte[0] == "punto" else ancho_texto(parte[0], parte[1], parte[2], True))
    total = sum(anchos) + huecos * (len(partes) - 1) + 2 * 20
    if derecha:
        x -= total
    elif centro:
        x -= total / 2
    dib.rounded_rectangle((x, y, x + total, y + alto), radius=alto // 2, fill=(*SUPERFICIE, int(225 * alfa)))
    cx = x + 20
    for parte, w in zip(partes, anchos):
        if parte[0] == "punto":
            dib.ellipse((cx, y + alto / 2 - 8, cx + 16, y + alto / 2 + 8), fill=(*parte[1], int(255 * alfa)))
        else:
            texto(dib, (cx, y + alto / 2), parte[0], parte[1], parte[2], parte[3], alfa, ancla="lm", tabular=True)
        cx += w + huecos


def con_texto(lienzo, pintar):
    """Pinta el texto con PIL encima del lienzo (RGB uint8) y devuelve el resultado."""
    base = Image.fromarray(lienzo).convert("RGBA")
    capa = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pintar(ImageDraw.Draw(capa))
    return np.array(Image.alpha_composite(base, capa).convert("RGB"))


def cabecera(dib, lineas, alfa=1.0):
    """lineas: [(texto, peso, tam, color)], de arriba abajo, alineadas a la izquierda."""
    y = 78
    for cadena, peso, tam, color in lineas:
        texto(dib, (48, y), cadena, peso, tam, color, alfa)
        y += tam + 14


CABECERA_1 = [("Faster through the speed trap.", "titulo", 52, TINTA),
              ("Still 0.55 s slower. Where?", "titulo", 52, TINTA),
              ("Odermatt vs Pinheiro Braathen · Sölden giant slalom, run 1", "medio", 27, TINTA_2)]
CABECERA_1B = [("Same race time, same camera:", "titulo", 52, TINTA),
               ("Odermatt is 0.14 s behind at gate 8.", "titulo", 52, TINTA),
               ("Both clocks synced to the frame from the TV graphics", "medio", 27, TINTA_2)]
CABECERA_2 = [("Now aligned on the pole contact:", "titulo", 52, TINTA),
              ("same gate, frame by frame.", "titulo", 52, TINTA),
              ("Each run starts the clock when it hits the gate-8 pole · 0.25×", "medio", 27, TINTA_2)]
X_RELOJ = 300  # el chip del reloj, entre los dos paneles y a la izquierda: a la derecha esta el rotulo de la tele


def chips_nombres(dib, extra=None, alto=58):
    """Nombres en los bordes de fuera: arriba en el panel de arriba, abajo en el de abajo.
    extra: {clave: [partes]} que se anaden detras del nombre."""
    for clave, y in (("B", Y_PANEL[0] + 22), ("O", ALTO - 22 - alto)):
        partes = [("punto", ESQUIADORES[clave]["color"]), (ESQUIADORES[clave]["nombre"], "semi", 30, TINTA)]
        chip(dib, 24, y, partes + (extra or {}).get(clave, []), alto=alto)


# ---------- escenas ----------

def panel_completo(clave, n):
    """El fotograma entero, recortado a la proporcion del panel."""
    img = fotograma(clave, n)[RECORTE_Y[0]:RECORTE_Y[1]]
    escala = ANCHO / img.shape[1]
    return cv2.resize(img, (ANCHO, PANEL_ALTO), interpolation=cv2.INTER_AREA), escala, -RECORTE_Y[0] * escala


def preparar_seguimiento(filas_b, filas_o):
    """Recorte que sigue a cada esquiador en la escena 2, con los dos al mismo tamano."""
    def talla(filas, numeros):
        t = []
        for n in numeros:
            p = filas[n]["puntos"]
            t.append(np.mean([np.hypot(*np.subtract(p[f"hip_{l}"][:2], p[f"knee_{l}"][:2]))
                              + np.hypot(*np.subtract(p[f"knee_{l}"][:2], p[f"ankle_{l}"][:2]))
                              + np.hypot(*np.subtract(p[f"shoulder_{l}"][:2], p[f"hip_{l}"][:2])) for l in ("right", "left")]))
        return float(np.median(t))
    nb = list(ESCENA2)
    no = [n + MISMO_SITIO for n in nb]
    escala_b = 0.62
    escala_o = escala_b * talla(filas_b, nb) / talla(filas_o, no)
    seg = {}
    for clave, filas, numeros, escala in (("B", filas_b, nb, escala_b), ("O", filas_o, no, escala_o)):
        centros = np.array([[filas[n]["caja"][0] + filas[n]["caja"][2] / 2, filas[n]["caja"][1] + filas[n]["caja"][3] / 2]
                            for n in numeros])
        suaves = np.array([centros[max(0, i - 3):i + 4].mean(axis=0) for i in range(len(centros))])
        w, h = ANCHO / escala, PANEL_ALTO / escala
        for n, (cx, cy) in zip(numeros, suaves):
            x0 = float(np.clip(cx - w / 2, 0, 1920 - w))
            y0 = float(np.clip(cy - h * 0.48, 0, 1080 - h))
            seg[(clave, n)] = (x0, y0, escala)
    print(f"escala de seguimiento: B {escala_b:.3f}, O {escala_o:.3f}")
    return seg


def panel_seguido(clave, n, seg):
    x0, y0, escala = seg[(clave, n)]
    M = np.array([[escala, 0, -x0 * escala], [0, escala, -y0 * escala]])
    img = cv2.warpAffine(fotograma(clave, n), M, (ANCHO, PANEL_ALTO), flags=cv2.INTER_AREA)
    return img, escala, -x0 * escala, -y0 * escala


def lienzo_vacio():
    lienzo = np.empty((ALTO, ANCHO, 3), np.uint8)
    lienzo[:] = SUPERFICIE
    lienzo[CABECERA + PANEL_ALTO:CABECERA + PANEL_ALTO + 4] = PLANO
    return lienzo


def escena_mismo_crono(filas, n_b, congelado=0.0):
    """congelado: 0 -> 1, cuanto se ha fundido ya el texto del momento del toque."""
    lienzo = lienzo_vacio()
    for clave, n, y in (("B", n_b, Y_PANEL[0]), ("O", n_b + MISMO_CRONO, Y_PANEL[1])):
        img, escala, dy = panel_completo(clave, n)
        lienzo[y:y + PANEL_ALTO] = img
        p = puntos_a(filas[clave][n], escala, 0, dy + y)
        esqueleto(lienzo, p, ESQUIADORES[clave]["color"], 4)
    crono = filas["B"][n_b]["crono_s"]

    def pintar(dib):
        if congelado < 1:
            cabecera(dib, CABECERA_1, 1 - congelado)
        if congelado > 0:
            cabecera(dib, CABECERA_1B, congelado)
        chips_nombres(dib, {"O": [("+0.14 s to the pole", "medio", 28, TINTA_2)]} if congelado > 0 else None)
        chip(dib, X_RELOJ, CABECERA + PANEL_ALTO + 2 - 29,
             [("RACE TIME", "semi", 22, TENUE), (f"{crono:5.2f} s", "semi", 34, TINTA)], centro=True)
    return con_texto(lienzo, pintar)


def escena_mismo_sitio(filas, incl, seg, n_b, lineas=None):
    """lineas: otra cabecera (la portada usa la del principio)."""
    lienzo = lienzo_vacio()
    dt = filas["B"][n_b]["crono_s"] - ESQUIADORES["B"]["toque"]
    angulos = {}
    for clave, n, y in (("B", n_b, Y_PANEL[0]), ("O", n_b + MISMO_SITIO, Y_PANEL[1])):
        img, escala, dx, dy = panel_seguido(clave, n, seg)
        lienzo[y:y + PANEL_ALTO] = img
        sub = lienzo[y:y + PANEL_ALTO]
        p = puntos_a(filas[clave][n], escala, dx, dy)
        esqueleto(sub, p, ESQUIADORES[clave]["color"], 4)
        inclinacion(sub, p, ESQUIADORES[clave]["color"])
        angulos[clave] = incl[clave][n]

    def pintar(dib):
        cabecera(dib, lineas or CABECERA_2)
        chips_nombres(dib, {k: [("lean", "medio", 26, TENUE), (f"{angulos[k]:2.0f}°", "semi", 38, TINTA)]
                            for k in angulos}, alto=64)
        if abs(dt) < 0.03:
            momento = [("POLE CONTACT", "semi", 30, TINTA)]
        else:
            momento = [("t", "medio", 24, TENUE), (f"{dt:+.2f} s".replace("-", "−"), "semi", 34, TINTA),
                       ("before pole contact" if dt < 0 else "after pole contact", "medio", 24, TENUE)]
        chip(dib, X_RELOJ, CABECERA + PANEL_ALTO + 2 - 29, momento, centro=True)
    return con_texto(lienzo, pintar)


# ---------- grafica ----------

class Grafica:
    """Inclinacion frente al tiempo desde el toque del palo, para la escena 3 (matplotlib)."""
    Y0, ALTO_G = 300, 860

    def __init__(self, filas, incl):
        for archivo in ARCHIVOS.values():
            font_manager.fontManager.addfont(str(FUENTES / archivo))
        self.fp = {peso: font_manager.FontProperties(fname=str(FUENTES / a)) for peso, a in ARCHIVOS.items()}
        self.series = {}
        for clave, desfase in (("B", 0), ("O", MISMO_SITIO)):
            ns = [n + desfase for n in ESCENA2]
            t = np.array([filas[clave][n]["crono_s"] - ESQUIADORES[clave]["toque"] for n in ns])
            self.series[clave] = (t, np.array([incl[clave][n] for n in ns]))
        self.fig = plt.figure(figsize=(ANCHO / 100, self.ALTO_G / 100), dpi=100)
        self.fig.patch.set_facecolor(np.array(SUPERFICIE) / 255)
        ax = self.ax = self.fig.add_axes([0.12, 0.13, 0.58, 0.80])
        ax.set_facecolor(np.array(SUPERFICIE) / 255)
        ax.set_xlim(-0.52, 0.64)
        ax.set_ylim(0, 75)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_yticklabels(["0°", "20°", "40°", "60°"])
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["−0.4", "−0.2", "0", "+0.2", "+0.4", "+0.6"])
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(self.fp["medio"])
            lab.set_fontsize(24)
            lab.set_color(np.array(TENUE) / 255)
        ax.tick_params(length=0, pad=12)
        ax.grid(axis="y", color=np.array(REJILLA) / 255, linewidth=2)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(np.array(EJE) / 255)
        ax.spines["bottom"].set_linewidth(2)
        ax.set_xlabel("seconds from pole contact", fontproperties=self.fp["medio"], fontsize=24,
                      color=np.array(TENUE) / 255, labelpad=14)
        ax.axvline(0, color=np.array(TINTA_2) / 255, linewidth=2, zorder=1)
        ax.text(0.012, 73, "pole contact", fontproperties=self.fp["medio"], fontsize=22, va="top",
                color=np.array(TINTA_2) / 255)
        self.lineas, self.puntos, self.rotulos = {}, {}, {}
        for clave in ("B", "O"):
            color = np.array(ESQUIADORES[clave]["color"]) / 255
            self.lineas[clave], = ax.plot([], [], color=color, linewidth=6, solid_capstyle="round",
                                          solid_joinstyle="round", zorder=3)
            self.puntos[clave], = ax.plot([], [], "o", color=color, markersize=22, markeredgewidth=6,
                                          markeredgecolor=np.array(SUPERFICIE) / 255, zorder=4)
            # rotulo directo al final de cada linea; sale con las notas (antes las lineas van juntas)
            t, y = self.series[clave]
            self.rotulos[clave] = ax.text(t[-1] + 0.04, y[-1], ESQUIADORES[clave]["nombre"],
                                          fontproperties=self.fp["semi"], fontsize=20, va="center",
                                          color=np.array(TINTA) / 255, clip_on=False, alpha=0)
        # notas que aparecen al final. Mismo angulo (±2,5°) de -0,38 a -0,14 s; justo antes del palo
        # Odermatt ya empieza a levantarse.
        self.nota_igual = ax.text(-0.26, 69.5, "same lean", fontproperties=self.fp["semi"], fontsize=26,
                                  ha="center", va="bottom", color=np.array(TINTA) / 255, alpha=0)
        self.raya_igual, = ax.plot([-0.38, -0.14], [68, 68], color=np.array(TINTA_2) / 255, linewidth=2, alpha=0)
        y_nivel = 45
        cruces = {c: self.cruce(c, y_nivel) for c in ("B", "O")}
        self.flecha = ax.annotate("", xy=(cruces["B"], y_nivel), xytext=(cruces["O"], y_nivel),
                                  arrowprops=dict(arrowstyle="<->", color=np.array(TINTA) / 255, lw=2.5,
                                                  shrinkA=0, shrinkB=0), alpha=0)
        # a 25 fps cada fotograma son 0,04 s: el hueco se da redondeado a la decima
        self.nota_hueco = ax.text(cruces["B"] + 0.03, y_nivel, f"~{cruces['B'] - cruces['O']:.1f} s",
                                  fontproperties=self.fp["semi"], fontsize=26, ha="left", va="center",
                                  color=np.array(TINTA) / 255, alpha=0)
        self.cruces = cruces
        print(f"cruce de {y_nivel}°: B {cruces['B']:+.3f} s, O {cruces['O']:+.3f} s, hueco {cruces['B'] - cruces['O']:.3f} s")

    def cruce(self, clave, nivel):
        t, y = self.series[clave]
        for i in range(len(t) - 1):
            if t[i] > 0 and y[i] >= nivel > y[i + 1]:
                return float(t[i] + (t[i + 1] - t[i]) * (y[i] - nivel) / (y[i] - y[i + 1]))
        return float("nan")

    def pintar(self, hasta, alfa_igual=0.0, alfa_hueco=0.0):
        for clave in ("B", "O"):
            t, y = self.series[clave]
            k = t <= hasta + 1e-9
            if k.sum() >= 2 and hasta < t[-1]:
                # el tramo hasta 'hasta', interpolando el ultimo punto para que avance suave
                ti = np.append(t[k], hasta)
                yi = np.append(y[k], np.interp(hasta, t, y))
            else:
                ti, yi = t[k], y[k]
            self.lineas[clave].set_data(ti, yi)
            if len(ti):
                self.puntos[clave].set_data([ti[-1]], [yi[-1]])
            self.rotulos[clave].set_alpha(alfa_igual)
        self.nota_igual.set_alpha(alfa_igual)
        self.raya_igual.set_alpha(alfa_igual)
        self.flecha.set_alpha(alfa_hueco)
        self.flecha.arrow_patch.set_alpha(alfa_hueco)
        self.nota_hueco.set_alpha(alfa_hueco)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


CABECERA_3 = [("Into the gate, the same lean.", "titulo", 52, TINTA),
              ("Out of it, Odermatt stands up first.", "titulo", 52, TINTA),
              ("Body lean: boots-to-shoulders line vs. vertical · gate 8 · same camera", "medio", 27, TINTA_2)]


def escena_grafica(graf, hasta, alfa_igual, alfa_hueco):
    lienzo = np.empty((ALTO, ANCHO, 3), np.uint8)
    lienzo[:] = SUPERFICIE
    lienzo[Grafica.Y0:Grafica.Y0 + Grafica.ALTO_G] = graf.pintar(hasta, alfa_igual, alfa_hueco)

    def pintar(dib):
        cabecera(dib, CABECERA_3)
        x = 48
        for clave in ("B", "O"):  # leyenda
            dib.ellipse((x, Grafica.Y0 - 30, x + 18, Grafica.Y0 - 12), fill=ESQUIADORES[clave]["color"])
            texto(dib, (x + 30, Grafica.Y0 - 21), ESQUIADORES[clave]["nombre"], "semi", 26, TINTA, ancla="lm")
            x += 30 + ancho_texto(ESQUIADORES[clave]["nombre"], "semi", 26) + 40
        texto(dib, (48, ALTO - 56), "25 fps: each step is one video frame (0.04 s). Pose model output, median of 5 frames.",
              "normal", 22, TENUE)
        texto(dib, (48, ALTO - 24), CREDITOS, "normal", 22, TENUE)
    return con_texto(lienzo, pintar)


def escena_cierre(alfa=1.0):
    lienzo = np.empty((ALTO, ANCHO, 3), np.uint8)
    lienzo[:] = SUPERFICIE
    of = {k: CARRERA["esquiadores"][e["manga"].split("/")[0]]["manga1"] for k, e in ESQUIADORES.items()}
    dif = {k: of["O"][k] - of["B"][k] for k in ("int1", "int2", "meta")}

    def pintar(dib):
        texto(dib, (48, 150), "SÖLDEN GIANT SLALOM · RUN 1 · 25 OCT 2026", "semi", 26, TENUE, alfa)
        y = 260
        for clave in ("B", "O"):
            dib.ellipse((48, y - 30, 76, y - 2), fill=(*ESQUIADORES[clave]["color"], int(255 * alfa)))
            texto(dib, (96, y), ESQUIADORES[clave]["nombre"], "semi", 44, TINTA, alfa)
            tiempo = f"{of[clave]['meta']:.2f}"
            texto(dib, (ANCHO - 48, y), tiempo, "titulo", 56, TINTA, alfa, ancla="rs")
            if clave == "O":
                texto(dib, (ANCHO - 48 - ancho_texto(tiempo, "titulo", 56) - 24, y), f"+{dif['meta']:.2f}",
                      "semi", 36, TINTA_2, alfa, ancla="rs")
            y += 92
        texto(dib, (48, y + 6), f"Official gap: +{dif['int1']:.2f} at split 1 · +{dif['int2']:.2f} at split 2 · "
              f"+{dif['meta']:.2f} at the finish", "medio", 28, TINTA_2, alfa)
        texto(dib, (48, y + 50), f"Speed trap: Odermatt {of['O']['radar_kmh']:.1f} km/h, "
              f"Pinheiro Braathen {of['B']['radar_kmh']:.1f} km/h", "medio", 28, TINTA_2, alfa)
        y += 190
        for linea in ("At gate 8, Pinheiro Braathen keeps", "his lean ~0.1 s longer after the pole.",
                      "Faster, or just different?"):
            texto(dib, (48, y), linea, "titulo", 56, TINTA, alfa)
            y += 72
        y += 70
        for linea in ("How it was measured", "· Race clock read from the TV graphics in every frame (±0.01 s),",
                      "  checked against the official FIS splits",
                      "· ViTPose fine-tuned on Ski2DPose: 24 points, skis and poles included;",
                      "  82 % of ski points within 5 % of the skier's size (PCK@0.05)",
                      "· Skier detection: RT-DETR v2"):
            peso, color = ("semi", TINTA) if linea.startswith("How") else ("normal", TINTA_2)
            texto(dib, (48, y), linea, peso, 28, color, alfa)
            y += 42
        texto(dib, (48, ALTO - 24), CREDITOS, "normal", 22, TENUE, alfa)
    return con_texto(lienzo, pintar)


# ---------- montaje ----------

def fotogramas_del_video(filas, incl):
    """Genera los fotogramas del video, en orden."""
    ultimo = None
    for n in ESCENA1:
        img = escena_mismo_crono(filas, n)
        for _ in range(LENTO1):
            yield img
    for i in range(PAUSA1):
        ultimo = escena_mismo_crono(filas, ESCENA1[-1], congelado=min(1.0, i / 12))
        yield ultimo
    seg = preparar_seguimiento(filas["B"], filas["O"])
    primero = escena_mismo_sitio(filas, incl, seg, ESCENA2[0])
    for i in range(FUNDIDO):
        yield cv2.addWeighted(ultimo, 1 - (i + 1) / (FUNDIDO + 1), primero, (i + 1) / (FUNDIDO + 1), 0)
    for n in ESCENA2:
        ultimo = escena_mismo_sitio(filas, incl, seg, n)
        for _ in range(LENTO2):
            yield ultimo
    graf = Grafica(filas, incl)
    t0, t1 = graf.series["B"][0][0], graf.series["B"][0][-1]
    primero = escena_grafica(graf, t0, 0, 0)
    for i in range(FUNDIDO):
        yield cv2.addWeighted(ultimo, 1 - (i + 1) / (FUNDIDO + 1), primero, (i + 1) / (FUNDIDO + 1), 0)
    dibujo, notas, quieto = 60, 30, 70  # fotogramas: la linea avanza, aparecen las notas, se queda quieta
    for i in range(dibujo):
        yield escena_grafica(graf, t0 + (t1 - t0) * (i + 1) / dibujo, 0, 0)
    for i in range(notas):
        yield escena_grafica(graf, t1, min(1, (i + 1) / 10), max(0, min(1, (i - 14) / 10)))
    fijo = escena_grafica(graf, t1, 1, 1)
    for _ in range(quieto):
        yield fijo
    ultimo = fijo
    for i in range(FUNDIDO):
        yield cv2.addWeighted(ultimo, 1 - (i + 1) / (FUNDIDO + 1), escena_cierre(), (i + 1) / (FUNDIDO + 1), 0)
    cierre = escena_cierre()
    for _ in range(100):
        yield cierre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("salida", type=Path)
    ap.add_argument("--fijos", type=int, nargs="*", help="solo guarda estos fotogramas del video, en JPG")
    args = ap.parse_args()
    filas, incl = {}, {}
    for clave in ESQUIADORES:
        filas[clave], incl[clave] = cargar(clave)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    # portada para la miniatura de LinkedIn: el gancho del principio sobre +0,34 s, donde mas se ve
    portada = escena_mismo_sitio(filas, incl, preparar_seguimiento(filas["B"], filas["O"]), 1288, CABECERA_1)
    cv2.imwrite(str(args.salida.with_name("portada.png")), cv2.cvtColor(portada, cv2.COLOR_RGB2BGR))
    if args.fijos is not None:
        quiero = set(args.fijos)
        for i, img in enumerate(fotogramas_del_video(filas, incl)):
            if i in quiero:
                ruta = args.salida.with_name(f"{args.salida.stem}_{i:04d}.jpg")
                cv2.imwrite(str(ruta), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(ruta)
            if i >= max(quiero):
                break
        return
    escritor = imageio_ffmpeg.write_frames(str(args.salida), (ANCHO, ALTO), fps=FPS, codec="libx264", quality=None,
                                           macro_block_size=2,
                                           output_params=["-crf", "16", "-preset", "slow", "-profile:v", "high",
                                                          "-movflags", "+faststart"])
    escritor.send(None)
    total = 0
    for img in fotogramas_del_video(filas, incl):
        escritor.send(np.ascontiguousarray(img).tobytes())
        total += 1
    escritor.close()
    print(f"{args.salida}: {total} fotogramas, {total / FPS:.1f} s")


if __name__ == "__main__":
    main()
