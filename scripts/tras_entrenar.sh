#!/usr/bin/env bash
# Espera a que acabe el entrenamiento y saca la pose de las dos 1as mangas enteras con el modelo final.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "python -u src/entrenar.py" > /dev/null; do sleep 60; done
echo "Entrenamiento terminado: $(date '+%d/%m %H:%M')"
tail -n 12 runs/vitpose24/entrenamiento.log
for par in "PINHEIRO BRAATHEN Lucas/manga1:braathen_m1_completa" "ODERMATT Marco/manga1:odermatt_m1_completa"; do
    manga="${par%%:*}"; salida="${par##*:}"
    .venv/bin/python src/pose_video.py carreras/solden2026_gs_hombres.json "$manga" -0.5 54 \
        runs/vitpose24/mejor "out/pose/$salida" --hilos 10
done
echo "Pose terminada: $(date '+%d/%m %H:%M')"
