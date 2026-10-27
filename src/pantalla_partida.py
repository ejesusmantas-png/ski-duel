"""Borrador: las dos mangas una al lado de la otra, al mismo tiempo de carrera. Uso:
    python src/pantalla_partida.py carreras/solden2026_gs_hombres.json "A/manga1" "B/manga1" out/video/partida.mp4
"""
import json
import sys

import cv2
import imageio_ffmpeg
import numpy as np

DESDE, HASTA = -0.6, 54.0      # crono de carrera
MITAD = (960, 540)
CABECERA = 132  # 540 + 132 = 672, multiplo de 16 (el codec no reescala)


def rotulo(img, texto, x, y, escala=1.0, color=(255, 255, 255), grosor=2):
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_DUPLEX, escala, (0, 0, 0), grosor + 4, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_DUPLEX, escala, color, grosor, cv2.LINE_AA)


def main():
    carrera = json.loads(open(sys.argv[1]).read())
    a, b, salida = sys.argv[2], sys.argv[3], sys.argv[4]
    video = carrera["video"]
    da, db = video["mangas"][a]["desfase_s"], video["mangas"][b]["desfase_s"]
    cap = cv2.VideoCapture(video["archivo"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    ia = int(round((DESDE + da) * fps))
    salto = int(round((db - da) * fps))  # fotogramas entre A y B al mismo crono
    print(f"desfase entre mangas: {(db - da) * fps:.3f} fotogramas")
    n_fot = int((HASTA - DESDE) * fps)
    quiero_a = set(range(ia, ia + n_fot))
    quiero_b = {n + salto for n in quiero_a}
    guardados, n = {}, 0
    ancho, alto = 2 * MITAD[0], MITAD[1] + CABECERA
    escritor = imageio_ffmpeg.write_frames(salida, (ancho, alto), fps=fps, codec="libx264",
                                           output_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"])
    escritor.send(None)
    nombre = {a: a.split("/")[0].title(), b: b.split("/")[0].title()}
    while n <= max(quiero_b):
        ok, img = cap.read()
        if not ok:
            break
        if n in quiero_a or n in quiero_b:
            guardados[n] = cv2.resize(img, MITAD, interpolation=cv2.INTER_AREA)
        k = n - salto  # cuando llega el de B, el de A ya esta guardado
        if k in quiero_a and k in guardados and n in guardados:
            lienzo = np.zeros((alto, ancho, 3), np.uint8)
            lienzo[CABECERA:, :MITAD[0]] = guardados.pop(k)
            lienzo[CABECERA:, MITAD[0]:] = guardados[n]
            crono = k / fps - da
            rotulo(lienzo, nombre[a], 30, 55, 1.2, (0, 215, 255))
            rotulo(lienzo, nombre[b], MITAD[0] + 30, 55, 1.2, (60, 60, 255))
            texto = f"{max(crono, 0):05.2f} s"
            (w, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
            rotulo(lienzo, texto, ancho // 2 - w // 2, 115, 1.3)
            cv2.line(lienzo, (MITAD[0], CABECERA), (MITAD[0], alto), (255, 255, 255), 3)
            escritor.send(cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB).tobytes())
        n += 1
    escritor.close()
    print(salida)


if __name__ == "__main__":
    main()
