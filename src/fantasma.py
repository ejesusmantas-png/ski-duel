"""El «fantasma»: en un plano con la misma camara en los dos descensos, cuando pasa B por
el mismo punto de la pista que A, a que distancia lateral, y el esqueleto de B dibujado
sobre la imagen de A en ese punto.

1. Cada manga se encadena fotograma a fotograma (alineacion por intensidad, ECC, sin el
   esquiador ni los graficos) hasta un fotograma de referencia de esa manga.
2. Las dos mangas se unen por el par de fotogramas con mejor registro SIFT, afinado con ECC.
3. Con todo en el mismo encuadre, para cada punto de la trayectoria de A se busca cuando
   la cruza B en su sentido de marcha (interpolando entre fotogramas) y a que distancia.
Escala: la longitud de los esquis (1,95 m en gigante masculino, percentil 90 en la imagen). Uso:
    python src/fantasma.py carreras/solden2026_gs_hombres.json out/pose/braathen_m1 out/pose/odermatt_m1 out/fantasma/puertas_8_9
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import registro
from dibujo import dibujar
from to_coco import PUNTOS

PIES = ["toes_right", "heel_right", "toes_left", "heel_left"]
ESQUI_M = 1.95
MIN_ECC = 0.8
COLOR_A, COLOR_B = (0, 215, 255), (60, 60, 255)  # BGR: amarillo (A) y rojo (B)


def cargar(ruta):
    d = json.loads((Path(ruta) / "pose.json").read_text())
    return d, {f["fotograma"]: f for f in d["fotogramas"] if f.get("puntos")}


def pies(f):
    return np.mean([f["puntos"][p][:2] for p in PIES], axis=0)


def leer(video, fotogramas):
    cap = cv2.VideoCapture(video)
    quiero, out, n = set(fotogramas), {}, 0
    while n <= max(quiero):
        ok, img = cap.read()
        if not ok:
            break
        if n in quiero:
            out[n] = img
        n += 1
    return out


def encadenar(D, imgs):
    """Homografia de cada fotograma de la manga hacia el primero (None si se pierde el hilo)."""
    ns = sorted(D)
    g = {n: registro.gris_ecc(imgs[n]) for n in ns}
    rasgos = {}
    def sift(n):
        if n not in rasgos:
            rasgos[n] = registro.rasgos(imgs[n], [D[n]["caja"]])
        return rasgos[n]
    al_primero = {ns[0]: np.eye(3)}
    calidad = {ns[0]: 1.0}
    for i in range(1, len(ns)):
        n = ns[i]
        hecho = False
        for atras in (1, 2):  # si un fotograma falla, se salta desde el anterior al anterior
            if i - atras < 0 or ns[i - atras] not in al_primero:
                continue
            p = ns[i - atras]
            H, cc = registro.ecc_robusto(g[n], g[p], registro.mascara_ecc(g[p].shape, [D[p]["caja"]]))
            if H is None or cc < MIN_ECC:
                H, inl = registro.homografia(sift(n), sift(p))
                cc = 0.0 if H is None else MIN_ECC
            if H is not None:
                al_primero[n] = al_primero[p] @ H
                calidad[n] = cc
                hecho = True
                break
        if not hecho:
            calidad[n] = 0.0
    return al_primero, calidad


def unir(A, B, imgs, a_primero, b_primero):
    """Homografia del primero de A al primero de B, por el par A-B mejor registrado."""
    mejor = (None, 0, None, None)
    ra = {n: registro.rasgos(imgs[n], [A[n]["caja"]]) for n in sorted(A)[::3]}
    rb = {m: registro.rasgos(imgs[m], [B[m]["caja"]]) for m in sorted(B)[::3]}
    for n, fa in ra.items():
        for m, fb in rb.items():
            if abs(B[m]["crono_s"] - A[n]["crono_s"]) > 0.6 or n not in a_primero or m not in b_primero:
                continue
            H, inl = registro.homografia(fa, fb)
            if H is not None and inl > mejor[1]:
                mejor = (H, inl, n, m)
    H, inl, n, m = mejor
    if H is None:
        raise RuntimeError("No hay ningun par A-B con fondo comun suficiente")
    gb = registro.gris_ecc(imgs[m])
    He, cc = registro.ecc(registro.gris_ecc(imgs[n]), gb, registro.mascara_ecc(gb.shape, [B[m]["caja"]]), H)
    if He is not None and cc >= MIN_ECC:
        H = He
    # primero de A -> n -> m -> primero de B
    return np.linalg.inv(b_primero[m]) @ H @ a_primero[n], (n, m, inl, cc)


def main():
    carrera = json.loads(Path(sys.argv[1]).read_text())
    (da, A), (db, B) = cargar(sys.argv[2]), cargar(sys.argv[3])
    salida = Path(sys.argv[4])
    salida.mkdir(parents=True, exist_ok=True)
    fps = da["fps"]
    imgs = leer(carrera["video"]["archivo"], list(A) + list(B))

    a_al, a_cal = encadenar(A, imgs)
    b_al, b_cal = encadenar(B, imgs)
    # al_primero lleva cada fotograma al primero de su manga: invertir para ir del fotograma al primero.
    a_ref = {n: np.linalg.inv(H) for n, H in a_al.items()}
    b_ref = {m: np.linalg.inv(H) for m, H in b_al.items()}
    union, info = unir(A, B, imgs, a_ref, b_ref)
    print(f"Union por A{info[0]}-B{info[1]}: {info[2]} coincidencias SIFT, ECC {info[3]:.3f}. "
          f"Encadenados: A {len(a_ref)}/{len(A)}, B {len(b_ref)}/{len(B)}")

    # Todo al encuadre del primer fotograma de B.
    ta = {n: registro.llevar(union @ a_ref[n], [pies(A[n])])[0] for n in a_ref}
    tb = {m: registro.llevar(b_ref[m], [pies(B[m])])[0] for m in b_ref}
    largo_b = {m: np.mean([np.linalg.norm(np.subtract(*registro.llevar(b_ref[m], [B[m]["puntos"][f"ski_tip_{l}"][:2],
                                                                                   B[m]["puntos"][f"ski_tail_{l}"][:2]])))
                           for l in ("right", "left")]) for m in b_ref}
    bm = sorted(tb)
    filas = []
    for n in sorted(ta):
        p = ta[n]
        res = None
        for m0, m1 in zip(bm[:-1], bm[1:]):
            if m1 - m0 > 2:
                continue
            seg = tb[m1] - tb[m0]
            L = np.linalg.norm(seg)
            if L < 1e-6:
                continue
            u = seg / L
            f = float((p - tb[m0]) @ u) / L
            if 0 <= f < 1:
                lat = float(u[0] * (p - tb[m0])[1] - u[1] * (p - tb[m0])[0])
                if res is None or abs(lat) < abs(res[3]):
                    res = (m0, m1, f, lat)
        if res is None:
            continue
        m0, m1, f, lat = res
        t_b = B[m0]["crono_s"] + f * (B[m1]["crono_s"] - B[m0]["crono_s"])
        cercanos = [largo_b[m] for m in bm if abs(m - m0) <= fps / 2]
        escala = np.percentile(cercanos, 90) / ESQUI_M
        mb = m0 if f < 0.5 else m1
        filas.append({"fotograma_a": n, "crono_a_s": A[n]["crono_s"], "fotograma_b": mb, "crono_b_s": round(t_b, 3),
                      "diferencia_s": round(t_b - A[n]["crono_s"], 3), "lateral_m": round(lat / escala, 2),
                      "calidad_a": round(a_cal.get(n, 0), 3), "calidad_b": round(b_cal.get(mb, 0), 3)})
        # Esqueleto de B en la imagen de A: B -> referencia de B -> referencia de A -> A.
        H_ba = np.linalg.inv(union @ a_ref[n]) @ b_ref[mb]
        img = imgs[n].copy()
        kb = registro.llevar(H_ba, [B[mb]["puntos"][q][:2] for q in PUNTOS])
        dibujar(img, [(x, y, 2) for x, y in kb], grosor=3, color=COLOR_B)
        dibujar(img, [(*A[n]["puntos"][q][:2], 2) for q in PUNTOS], grosor=3, color=COLOR_A)
        cv2.putText(img, f"{t_b - A[n]['crono_s']:+.2f} s", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imwrite(str(salida / f"{n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    (salida / "fantasma.json").write_text(json.dumps({"a": da["manga"], "b": db["manga"], "union": info[:2],
                                                      "trayectoria_a": {str(k): v.tolist() for k, v in ta.items()},
                                                      "trayectoria_b": {str(k): v.tolist() for k, v in tb.items()},
                                                      "filas": filas}, indent=1))
    print(f"{len(filas)}/{len(A)} fotogramas de A emparejados")
    for r in filas[::6]:
        print(f"  crono A {r['crono_a_s']:6.2f}  B {r['crono_b_s']:6.2f}  diferencia {r['diferencia_s']:+.3f} s  "
              f"lateral {r['lateral_m']:+.2f} m")


if __name__ == "__main__":
    main()
