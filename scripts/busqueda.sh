#!/usr/bin/env bash
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
