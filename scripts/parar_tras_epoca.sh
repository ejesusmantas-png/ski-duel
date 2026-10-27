#!/usr/bin/env bash
# Espera a que termine (y se guarde, si mejora) la evaluacion de la epoca N y para el
# entrenamiento, sin lanzar despues la pose. Uso: scripts/parar_tras_epoca.sh 20
set -u
N="$1"
cd "$(dirname "$0")/.."
LOG=runs/vitpose24/entrenamiento.log
until grep -qE "^epoca $N  perdida" "$LOG"; do
    pgrep -f '^\.venv/bin/python -u src/entrenar\.py' > /dev/null || { echo "el entrenamiento ya no esta"; exit 0; }
    sleep 10
done
sleep 60  # guardar el modelo tarda unos segundos
pkill -f '^bash scripts/tras_entrenar\.sh'          # que no arranque la pose
pkill -TERM -f '^\.venv/bin/python -u src/entrenar\.py'
echo "$(date '+%d/%m %H:%M') parado tras la epoca $N"
grep -A5 "^epoca $N  perdida" "$LOG"
head -2 runs/vitpose24/mejor/resultado.txt
