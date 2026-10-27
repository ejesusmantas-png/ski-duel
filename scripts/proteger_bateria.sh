#!/usr/bin/env bash
# El entrenamiento solo corre con cargador: sin el, se pausa (SIGSTOP, entrenamiento y
# trabajadores) y se reanuda al enchufar. Termina solo cuando el entrenamiento acaba.
# Ojo: no matar systemd-inhibit para soltar el bloqueo de suspension: lanza el comando
# con PDEATHSIG y el entrenamiento moriria con el.
set -u
PATRON="^\.venv/bin/python -u src/entrenar\.py"
pausado=0
while pgrep -f "$PATRON" > /dev/null; do
    enchufado=$(cat /sys/class/power_supply/ADP1/online 2>/dev/null || echo 1)
    bateria=$(cat /sys/class/power_supply/BAT0/capacity 2>/dev/null || echo 100)
    if [ "$enchufado" = "0" ] && [ "$pausado" = "0" ]; then
        pkill -STOP -f "$PATRON" && pausado=1
        echo "$(date '+%d/%m %H:%M') PAUSADO: sin cargador (bateria ${bateria} %)"
    elif [ "$enchufado" = "1" ] && [ "$pausado" = "1" ]; then
        pkill -CONT -f "$PATRON" && pausado=0
        echo "$(date '+%d/%m %H:%M') REANUDADO: enchufado (bateria ${bateria} %)"
    fi
    sleep 15
done
echo "$(date '+%d/%m %H:%M') entrenamiento terminado; dejo de vigilar"
