"""PCK: porcentaje de puntos que caen a menos de alfa * tamano del esquiador del real.

El tamano es el lado mayor de la caja de los puntos anotados (sin margen), asi
que incluye esquis y bastones: es la escala con la que se ve al esquiador.
"""
import numpy as np


def tamano(kps_reales):
    k = np.asarray(kps_reales, dtype=float)
    return max(np.ptp(k[:, 0]), np.ptp(k[:, 1]))


def errores_normalizados(pred, real, escala):
    """pred, real: arrays (N, 2+) en pixeles. Devuelve la distancia / escala por punto."""
    pred, real = np.asarray(pred, float), np.asarray(real, float)
    return np.linalg.norm(pred[:, :2] - real[:, :2], axis=1) / escala


def pck(errores, alfa):
    return float(np.mean(np.asarray(errores) < alfa))
