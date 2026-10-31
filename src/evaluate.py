"""Métricas de un checkpoint sobre val o test (tarea D8).

Uso: uv run python -m src.evaluate --ckpt results/<exp>/best.pt [--split val|test] [--force]

Escribe junto al checkpoint (results/<exp>/):
    metrics_<split>.json   accuracy, f1_macro, f1_por_clase (+ n, split, ckpt)
    report_<split>.txt     classification_report y las confusiones más frecuentes
    confusion_<split>.png  matriz de confusión normalizada por filas (diagonal = recall)

Todas las decisiones se toman con val. El test se evalúa una sola vez por experimento:
si ya existe metrics_test.json, se niega a repetirlo salvo con --force.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.common import CLASSES, SPLITS_CSV
from src.dataset import SPLITS, get_dataloaders
from src.model import load_checkpoint

LABELS = list(range(len(CLASSES)))


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.to(device).eval()
    ys, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def compute_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1.mean()),
            "f1_por_clase": {c: round(float(v), 4) for c, v in zip(CLASSES, f1)},
            "n": int(len(y_true))}


def top_confusions(y_true, y_pred, k=5):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm, axis=None)[::-1][:k]
    return [(CLASSES[i], CLASSES[j], float(cm[i, j])) for i, j in zip(*np.unravel_index(order, cm.shape))
            if cm[i, j] > 0]


def report_text(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=CLASSES,
                                   digits=3, zero_division=0)
    lines = [f"  {real:<11} -> {pred:<11} {frac:6.1%}" for real, pred, frac in top_confusions(y_true, y_pred)]
    return report + "\nConfusiones más frecuentes (real -> predicha, % de la clase real):\n" + "\n".join(lines) + "\n"


def plot_confusion(y_true, y_pred, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in LABELS:
        for j in LABELS:
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(LABELS, CLASSES, rotation=45, ha="right")
    ax.set_yticks(LABELS, CLASSES)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(ckpt, split="val", force=False, batch_size=256, num_workers=2, splits_csv=SPLITS_CSV):
    ckpt = Path(ckpt)
    out = ckpt.parent
    if split == "test" and (out / "metrics_test.json").exists() and not force:
        raise SystemExit(f"{out / 'metrics_test.json'} ya existe: el test se evalúa una sola vez "
                         "(usa --force solo si de verdad hay que repetirlo)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(ckpt)
    loader = get_dataloaders(batch_size, augment=False, num_workers=num_workers, splits_csv=splits_csv)[split]
    y_true, y_pred = predict(model, loader, device)

    metrics = {**compute_metrics(y_true, y_pred), "split": split, "ckpt": str(ckpt)}
    (out / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out / f"report_{split}.txt").write_text(report_text(y_true, y_pred))
    name = cfg.get("name", out.name)
    plot_confusion(y_true, y_pred, out / f"confusion_{split}.png",
                   title=f"{name} · {split} · acc {metrics['accuracy']:.3f} · F1 macro {metrics['f1_macro']:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="results/<exp>/best.pt")
    parser.add_argument("--split", choices=SPLITS[1:], default="val", help="val (por defecto) o test")
    parser.add_argument("--force", action="store_true", help="repite el test aunque ya esté evaluado")
    args = parser.parse_args()

    metrics = evaluate(args.ckpt, args.split, args.force)
    out = Path(args.ckpt).parent
    print((out / f"report_{args.split}.txt").read_text())
    print(f"accuracy {metrics['accuracy']:.4f} · F1 macro {metrics['f1_macro']:.4f} · n {metrics['n']}")
    print(f"Guardados metrics_{args.split}.json, report_{args.split}.txt y confusion_{args.split}.png en {out}")


if __name__ == "__main__":
    main()
