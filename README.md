# gestos-cnn-robot

Clasificador de gestos de la mano con una CNN entrenada desde cero en PyTorch sobre HaGRID. Reconoce
ocho gestos y una clase nula (`no_gesture`) pensados como órdenes para un robot. Práctica de
Aprendizaje Automático 2 (curso 2026-27).

Autores: Daniel Rodríguez Alonso y Carlos Falcón Brito.

Informe: [`informe/informe.pdf`](informe/informe.pdf) (12 páginas).

## Resultados

Partición **por persona** (70/15/15, sin personas compartidas): 11.088 recortes de entrenamiento,
2.375 de validación y 2.374 de test, con 9 clases equilibradas.

| Modelo | Parámetros | Val | Test | IC 95 % test (bootstrap por persona) |
|---|---|---|---|---|
| **GestureCNN, lote 32 (modelo final)** | 2,36 M | 99,28 % | **99,03 %** | [98,61 %; 99,42 %] |
| GestureCNN, lote 64 (`completo`, final anterior) | 2,36 M | 98,99 % | 98,99 % | [98,53 %; 99,40 %] |
| ResNet18 preentrenado (referencia) | 11,18 M | 99,41 % | 99,20 % | [98,76 %; 99,57 %] |

- **Búsqueda manual de hiperparámetros** (9 configs, solo en validación): todas quedan entre el
  98,74 % y el 99,07 %, salvo el lote de 32 (99,28 %), que pasó a ser el modelo final. En test es
  igual que `completo` (McNemar p = 1,0). Con la mitad de filtros (591 k parámetros), la red rinde
  igual.
- **Ablación** (2 semillas): solo quitar la normalización por lotes empeora de forma significativa
  (McNemar p < 0,001); el efecto de quitar el dropout o el aumento de datos no se distingue del
  ruido entre semillas.
- **ResNet18** no es significativamente mejor que GestureCNN (p = 0,56).
- Confusiones principales: `palm`/`stop` y `like`/`call`. Grad-CAM confirma que la red mira la mano.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

    git clone git@github.com:danirodriguezz/gestos-cnn-robot.git
    cd gestos-cnn-robot
    uv sync

## Estructura

    src/
      common.py          clases, rutas, semilla y recorte compartidos
      download.py        descarga HaGRID 30k a 384 px
      show_boxes.py      dibuja las cajas anotadas para comprobar su formato
      prepare_data.py    recortes 128x128 de todas las manos
      make_splits.py     equilibrado y partición por persona; estadísticas de normalización
      dataset.py         Dataset, aumento de datos y get_dataloaders
      mock_data.py       datos falsos con la misma API, para probar sin el dataset
      model.py           GestureCNN (canales, capa FC y activación configurables) y ResNet18
      train.py           entrenamiento con Adam, ReduceLROnPlateau y parada anticipada
      run_experiments.py ablación y ResNet18 (configs/*.yaml)
      busqueda.py        búsqueda de hiperparámetros (configs/busqueda/*.yaml)
      evaluate.py        métricas, informe y matriz de confusión de un checkpoint
      plot_curves.py     curvas de pérdida y exactitud
      gradcam.py         mapas Grad-CAM de aciertos y errores
      analysis.py        IC, McNemar, semillas, coste y macros LaTeX del informe
    configs/             un .yaml por experimento; configs/busqueda/ para la búsqueda
    scripts/busqueda.sh  lanza la búsqueda sin que el portátil se suspenda
    tests/               pruebas con pytest (96)
    results/             métricas, curvas y figuras de cada experimento (results/final = modelo final)
    informe/             informe en LaTeX; las cifras salen de informe/generado/resultados.tex
    data/                solo splits.csv y stats.json van en git

Los datos (`data/raw`, `data/crops`) y los pesos (`*.pt`) no se suben a git. Conforme a la
entrega, el código no lleva comentarios: cada módulo tiene solo un docstring inicial con su uso.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio. En una GTX 1660 Ti, una época de
GestureCNN tarda unos 25 s; la ablación, alrededor de una hora por semilla, y la búsqueda, unas 3 h.

### 1. Datos

    uv run python -m src.download        # HaGRID 30k a 384 px (Hugging Face) -> data/raw/
    uv run python -m src.prepare_data    # recortes 128x128 -> data/crops/, data/crops.csv

`data/splits.csv` (partición por persona) y `data/stats.json` (normalización) ya están en el
repositorio: **no** hace falta ejecutar `src.make_splits`, que solo se usa para regenerarlos
(`--force`). Los nombres de los recortes son deterministas, así que coinciden con `splits.csv`.

### 2. Ablación y ResNet18

    uv run python -u -m src.run_experiments 2>&1 | tee results/ablacion.log                  # semilla 42
    uv run python -u -m src.run_experiments --seed 43 2>&1 | tee results/ablacion_s43.log    # segunda semilla
    uv run python -u -m src.run_experiments configs/resnet18.yaml 2>&1 | tee results/resnet18.log

Cada experimento escribe `best.pt`, `history.csv` y `config.yaml` en `results/<nombre>/`. Para una
prueba rápida sin GPU ni datos: `--data mock --epochs 1`.

### 3. Búsqueda de hiperparámetros y modelo final

Nueve configs en `configs/busqueda/`, cada una igual que `completo` salvo un hiperparámetro:
tasa de aprendizaje (3e-4, 3e-3), lote (32, 128), filtros a la mitad, 4 bloques, capa FC oculta de
128 neuronas y activación (ELU, LeakyReLU).

    bash scripts/busqueda.sh                          # con el portátil enchufado
    uv run python -m src.busqueda --solo-resumen      # rehace tabla y figura sin entrenar

El script comprueba que el portátil está enchufado, impide la suspensión y guarda el log en
`results/busqueda.log`; si se corta, al relanzarlo sigue por donde iba. Solo usa validación. Escribe
`results/tablas/busqueda.csv` y `results/figures/busqueda_curvas.png`.

El modelo final es `hp_batch_32`, la config con mejor exactitud de validación (antes era `completo`;
el informe explica el cambio y compara los dos en test):

    mkdir -p results/final && cp results/hp_batch_32/* results/final/

### 4. Evaluación y figuras

    uv run python -m src.evaluate --ckpt results/final/best.pt --split val
    uv run python -m src.evaluate --ckpt results/final/best.pt --split test   # una sola vez (--force para repetir)
    uv run python -m src.plot_curves results/completo results/sin_dropout results/sin_bn results/sin_aug results/resnet18
    uv run python -m src.gradcam --ckpt results/final/best.pt
    uv run python -m src.analysis

`src.analysis` calcula los intervalos de confianza, las pruebas de McNemar (ablación frente a
`completo`; modelo final frente a `completo` y a ResNet18), las medias por semilla, el coste
(parámetros, MACs y latencia) y la tabla de la búsqueda. Lo escribe todo en
`results/tablas/analisis.json` y en `informe/generado/resultados.tex`.

### 5. Informe

    cd informe && latexmk -pdf -jobname=informe main.tex  # -> informe/informe.pdf

### Pruebas

    uv run python -m pytest -q

## Trabajo futuro

La demo en tiempo real con ROS 2 y turtlesim no forma parte de la entrega; su diseño está en el
apartado de trabajo futuro del informe. La carpeta `demo_ros2/` está reservada para ella.
