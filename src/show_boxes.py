"""Dibuja las cajas de HaGRID sobre una muestra de imágenes para verificar su formato (tarea D3).

Uso: uv run python -m src.show_boxes [--n 20]

Genera:
    results/figures/cajas_muestra.png     fotos con sus cajas (verde = gesto, rojo = no_gesture)
    results/figures/recortes_muestra.png  recortes 128x128 de crop_hand sobre esas mismas manos
"""
import argparse
import json
import math
import random

import matplotlib
matplotlib.use("Agg")                           # sin ventana: solo se guarda a fichero
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.common import RESULTS, SEED, crop_hand
from src.download import GESTURES, RAW_DIR

FIG_DIR = RESULTS / "figures"


def color_of(label):
    return "red" if label == "no_gesture" else "lime"


def load_image(gesture, image_id):
    return Image.open(RAW_DIR / "images" / gesture / f"{image_id}.jpg").convert("RGB")


def sample_images(n, rng):
    """n imágenes repartidas por turnos entre los gestos; en la primera vuelta, una con dos manos de cada uno."""
    anns = {g: json.loads((RAW_DIR / "annotations" / f"{g}.json").read_text()) for g in GESTURES}
    samples, chosen = [], set()
    for i in range(n):
        g = GESTURES[i % len(GESTURES)]
        ids = [k for k in sorted(anns[g]) if k not in chosen]
        if i < len(GESTURES):
            ids = [k for k in ids if len(anns[g][k]["labels"]) > 1] or ids
        image_id = rng.choice(ids)
        chosen.add(image_id)
        samples.append((g, image_id, anns[g][image_id]))
    return samples


def plot_boxes(samples, path, cols=5):
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (g, image_id, ann) in zip(axes.flat, samples):
        img = load_image(g, image_id)
        W, H = img.size
        ax.imshow(img)
        for (x, y, w, h), label in zip(ann["bboxes"], ann["labels"]):
            # [x, y, w, h] normalizados con (x, y) = esquina superior izquierda -> píxeles
            ax.add_patch(Rectangle((x * W, y * H), w * W, h * H,
                                   fill=False, edgecolor=color_of(label), linewidth=2))
            ax.text(x * W, y * H - 4, label, color=color_of(label), fontsize=7, fontweight="bold")
        ax.set_title(f"{g} · {image_id[:8]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_crops(samples, path, cols=8):
    crops = [(crop_hand(load_image(g, image_id), box), label)
             for g, image_id, ann in samples
             for box, label in zip(ann["bboxes"], ann["labels"])]
    rows = math.ceil(len(crops) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (crop, label) in zip(axes.flat, crops):
        ax.imshow(crop)
        ax.set_title(label, fontsize=8, color="red" if label == "no_gesture" else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20, help="número de imágenes de la muestra")
    args = parser.parse_args()

    samples = sample_images(args.n, random.Random(SEED))   # misma semilla -> misma muestra para los dos
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_boxes(samples, FIG_DIR / "cajas_muestra.png")
    plot_crops(samples, FIG_DIR / "recortes_muestra.png")
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
