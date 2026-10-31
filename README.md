# gestos-cnn-robot

Clasificador de gestos de la mano con una CNN entrenada desde cero en PyTorch sobre HaGRID,
para controlar un robot simulado en ROS 2. Práctica de Aprendizaje Automático 2.

Autores: Daniel Rodríguez y Carlos Falcón Brito.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

    git clone git@github.com:danirodriguezz/gestos-cnn-robot.git
    cd gestos-cnn-robot
    uv sync

## Estructura

- `src/common.py`: contrato compartido (clases, rutas, semilla, recorte). No cambiar sin avisar.
- `src/`: código (datos, modelo, entrenamiento, evaluación).
- `configs/`: un `.yaml` por experimento.
- `demo_ros2/`: demo con ROS 2 + turtlesim.
- `data/` y pesos `.pt`: no se suben a git (ver `.gitignore`).

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio. Con GPU, la ablación tarda
alrededor de una hora por semilla (~25 s por época en una GTX 1660 Ti).

### 1. Datos

    uv run python -m src.download        # HaGRID 30k a 384 px (Hugging Face) -> data/raw/
    uv run python -m src.prepare_data    # recortes 128x128 de todas las manos -> data/crops/, data/crops.csv

`data/splits.csv` (partición por persona) y `data/stats.json` (normalización) ya están en el
repositorio: **no** hace falta ejecutar `src.make_splits`, que solo se usa para regenerarlos
(`--force`). Los nombres de los recortes son deterministas, así que coinciden con `splits.csv`.

### 2. Entrenamiento

    uv run python -u -m src.run_experiments 2>&1 | tee results/ablacion.log                  # ablación, semilla 42
    uv run python -u -m src.run_experiments --seed 43 2>&1 | tee results/ablacion_s43.log    # segunda semilla
    uv run python -u -m src.run_experiments configs/resnet18.yaml 2>&1 | tee results/resnet18.log

Cada experimento escribe `best.pt`, `history.csv` y `config.yaml` en `results/<nombre>/`.
Para una prueba rápida sin GPU: `--data mock --epochs 1`. Los pesos (`*.pt`) no se suben a git.

**Búsqueda de hiperparámetros** (`configs/busqueda/`, 9 configs: `completo` con un solo cambio de lr,
lote, filtros, nº de bloques, capa FC o activación). Entrena, evalúa en validación y resume:

    bash scripts/busqueda.sh       # unas 3 h en la 1660 Ti; con el portátil enchufado

El script impide la suspensión mientras dura y guarda el log en `results/busqueda.log`. Si se corta,
al relanzarlo sigue por donde iba. Resultado: `results/tablas/busqueda.csv` y
`results/figures/busqueda_curvas.png` (rehacerlos sin entrenar: `uv run python -m src.busqueda --solo-resumen`).

**Modelo final**: `hp_batch_32`, la config con mejor exactitud de validación en la búsqueda (antes era
`completo`; el informe explica el cambio y compara los dos en test):

    mkdir -p results/final && cp results/hp_batch_32/* results/final/

### 3. Evaluación y figuras

    uv run python -m src.evaluate --ckpt results/final/best.pt --split val
    uv run python -m src.evaluate --ckpt results/final/best.pt --split test   # una sola vez
    uv run python -m src.plot_curves results/completo results/sin_dropout results/sin_bn results/sin_aug results/resnet18
    uv run python -m src.gradcam --ckpt results/final/best.pt
    uv run python -m src.analysis        # IC, McNemar, semillas, coste -> informe/generado/resultados.tex

### 4. Informe

    cd informe && latexmk -pdf -jobname=informe main.tex  # -> informe/informe.pdf

### Pruebas

    uv run python -m pytest -q