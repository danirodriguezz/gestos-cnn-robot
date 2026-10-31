#!/usr/bin/env bash
# Lanza la búsqueda de hiperparámetros (src/busqueda.py) para dejar el portátil trabajando solo.
#
# Uso:  bash scripts/busqueda.sh
#
# - Comprueba que el portátil está enchufado: con batería la 1660 Ti baja de potencia y todo va
#   varias veces más lento (FORZAR=1 bash scripts/busqueda.sh para lanzarlo igualmente).
# - Impide que el sistema se suspenda, también al cerrar la tapa, mientras dura.
# - Guarda todo lo que imprime en results/busqueda.log (se añade al final si ya existe).
# - Si se corta, al relanzarlo continúa por la primera config que no esté terminada.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${BUSQUEDA_INHIBIDA:-}" ]]; then
    enchufado=$(cat /sys/class/power_supply/{AC,ADP}*/online 2>/dev/null | head -n1 || true)
    if [[ "$enchufado" == "0" && -z "${FORZAR:-}" ]]; then
        echo "El portátil no está enchufado: enchúfalo (o usa FORZAR=1) y vuelve a lanzarlo." >&2
        exit 1
    fi
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=name,enforced.power.limit,power.draw --format=csv,noheader
    else
        echo "Aviso: no se encuentra nvidia-smi; se entrenará en CPU si no hay GPU." >&2
    fi
    if command -v systemd-inhibit >/dev/null; then
        BUSQUEDA_INHIBIDA=1 exec systemd-inhibit --what=sleep:idle:handle-lid-switch \
            --why="búsqueda de hiperparámetros" bash "$0" "$@"
    fi
    echo "Aviso: sin systemd-inhibit; desactiva tú la suspensión automática." >&2
fi

mkdir -p results
echo "=== búsqueda de hiperparámetros · inicio $(date '+%F %T') ===" | tee -a results/busqueda.log
uv run python -u -m src.busqueda 2>&1 | tee -a results/busqueda.log
echo "=== fin $(date '+%F %T') ===" | tee -a results/busqueda.log
