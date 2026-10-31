"""Curvas de entrenamiento a partir de history.csv (tarea D8).

Uso:
    uv run python -m src.plot_curves results/completo
    uv run python -m src.plot_curves results/completo results/sin_bn results/sin_dropout results/sin_aug

Genera:
    results/<exp>/curves.png                pérdida y accuracy de train y val de cada experimento
    results/figures/comparacion_curvas.png  val de todos superpuestos (solo con 2 o más experimentos)

Línea vertical continua: mejor época (menor val_loss, la que guarda best.pt).
Líneas punteadas: épocas en las que ReduceLROnPlateau bajó el lr.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import RESULTS

FIG_DIR = RESULTS / "figures"


def load_history(exp_dir):
    return pd.read_csv(Path(exp_dir) / "history.csv")


def best_epoch(history):
    return int(history.loc[history["val_loss"].idxmin(), "epoch"])


def lr_changes(history):
    changed = history["lr"].ne(history["lr"].shift()) & history.index.to_series().gt(0)
    return list(zip(history.loc[changed, "epoch"].astype(int), history.loc[changed, "lr"]))


def plot_experiment(exp_dir, path=None):
    exp_dir = Path(exp_dir)
    h = load_history(exp_dir)
    best = best_epoch(h)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ("loss", "acc"), ("Pérdida (cross-entropy)", "Accuracy")):
        ax.plot(h["epoch"], h[f"train_{metric}"], label="train")
        ax.plot(h["epoch"], h[f"val_{metric}"], label="val")
        ax.axvline(best, color="black", linewidth=1, label=f"mejor época ({best})")
        for i, (epoch, lr) in enumerate(lr_changes(h)):
            ax.axvline(epoch, color="gray", linestyle=":", label="lr reducido" if i == 0 else None)
        ax.set_xlabel("época")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    row = h[h["epoch"] == best].iloc[0]
    fig.suptitle(f"{exp_dir.name} · mejor época {best}: val_loss {row['val_loss']:.3f} · val_acc {row['val_acc']:.3f}")
    fig.tight_layout()
    path = Path(path) if path else exp_dir / "curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_comparison(exp_dirs, path=FIG_DIR / "comparacion_curvas.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for exp_dir in map(Path, exp_dirs):
        h = load_history(exp_dir)
        best = h[h["epoch"] == best_epoch(h)]
        for ax, col in zip(axes, ("val_loss", "val_acc")):
            line, = ax.plot(h["epoch"], h[col], label=exp_dir.name)
            ax.plot(best["epoch"], best[col], "o", color=line.get_color())
    for ax, title in zip(axes, ("Pérdida de val", "Accuracy de val")):
        ax.set_xlabel("época")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_dirs", nargs="+", help="carpetas results/<exp> con history.csv")
    args = parser.parse_args()
    for exp_dir in args.exp_dirs:
        print(f"Guardada {plot_experiment(exp_dir)}")
    if len(args.exp_dirs) > 1:
        print(f"Guardada {plot_comparison(args.exp_dirs)}")


if __name__ == "__main__":
    main()
