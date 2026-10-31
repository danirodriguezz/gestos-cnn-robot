"""Lanza varias configs seguidas con train.py (tarea C5).

Uso:
    uv run python -m src.run_experiments                            (las 4 configs de la ablación)
    uv run python -m src.run_experiments configs/resnet18.yaml      (cualquier lista de configs)
    uv run python -m src.run_experiments --data mock --epochs 1     (prueba rápida sin GPU)
    uv run python -m src.run_experiments --seed 43                  (repetición: results/<name>_s43/)

Cada experimento escribe en results/<name>/ igual que train.py, con la misma semilla y la misma
partición, así que son comparables (plan, sección 3.8). Al final imprime un resumen con las
épocas y la mejor val_loss / val_acc; la F1 macro de la tabla de ablación sale de evaluate.py.
"""
import argparse
from pathlib import Path

from src.common import RESULTS, ROOT
from src.train import load_config, train

CONFIGS_DIR = ROOT / "configs"
ABLATION = [CONFIGS_DIR / f"{name}.yaml" for name in ("completo", "sin_dropout", "sin_bn", "sin_aug")]


def summarize(cfg, history):
    best = min(history, key=lambda row: row["val_loss"])
    return {"name": cfg["name"], "epochs": len(history), "best_epoch": best["epoch"],
            "val_loss": best["val_loss"], "val_acc": best["val_acc"]}


def run(paths, results_dir=RESULTS, **overrides):
    cfgs = [load_config(path, **overrides) for path in paths]
    if overrides.get("seed") is not None:
        for cfg in cfgs:
            cfg["name"] = f"{cfg['name']}_s{cfg['seed']}"
    names = [cfg["name"] for cfg in cfgs]
    if len(set(names)) != len(names):
        raise ValueError(f"dos configs escriben en la misma carpeta de results/: {names}")
    return [summarize(cfg, train(cfg, results_dir)[1]) for cfg in cfgs]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="*", type=Path, default=ABLATION,
                        help="yaml de los experimentos (por defecto, los 4 de la ablación)")
    parser.add_argument("--data", choices=["real", "mock"], help="sobrescribe data en todas las configs")
    parser.add_argument("--epochs", type=int, help="sobrescribe epochs en todas las configs")
    parser.add_argument("--seed", type=int, help="otra semilla; escribe en results/<name>_s<seed>/")
    args = parser.parse_args()

    rows = run(args.configs, data=args.data, epochs=args.epochs, seed=args.seed)
    print(f"\n{'experimento':<14}{'épocas':>8}{'mejor':>8}{'val_loss':>10}{'val_acc':>9}")
    for r in rows:
        print(f"{r['name']:<14}{r['epochs']:>8}{r['best_epoch']:>8}{r['val_loss']:>10.4f}{r['val_acc']:>9.3f}")


if __name__ == "__main__":
    main()
