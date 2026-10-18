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

Pendiente: se completará con los comandos de preparación de datos, entrenamiento y evaluación.