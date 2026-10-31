"""Mapas Grad-CAM sobre model.gradcam_layer (tarea D9).

Uso: uv run python -m src.gradcam --ckpt results/<exp>/best.pt [--n-errores 8]

Genera junto al checkpoint (results/<exp>/), siempre sobre val (el test se reserva para D13):
    gradcam_aciertos_val.png   un acierto por clase, elegido al azar con SEED: imagen y mapa
    gradcam_errores_val.png    los errores más seguros: imagen, mapa de la clase predicha y de la real

Grad-CAM: mapa = ReLU(Σ_k α_k · A_k), con A_k los mapas de la última conv y α_k el gradiente
medio del logit de la clase respecto a A_k. Rojo = zona que empuja hacia esa clase.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import CLASSES, SEED, SPLITS_CSV
from src.dataset import denormalize, get_dataloaders, load_stats
from src.model import load_checkpoint


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.acts = self.grads = self.handle = None

    def __enter__(self):
        self.handle = self.model.gradcam_layer.register_forward_hook(self._save_activations)
        return self

    def __exit__(self, *exc):
        self.handle.remove()

    def _save_activations(self, module, inputs, output):
        self.acts = output.detach().clone()
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.grads = grad.detach()

    def __call__(self, x, classes):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        logits.gather(1, classes.view(-1, 1)).sum().backward()
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return (cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)).cpu()


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(dim=1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def pick_examples(y_true, probs, n_errors=8, seed=SEED):
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    rng = np.random.default_rng(seed)
    hits = []
    for c in range(len(CLASSES)):
        ok = np.flatnonzero((y_true == c) & (pred == c))
        if len(ok):
            hits.append(int(rng.choice(ok)))
    wrong = np.flatnonzero(pred != y_true)
    errors = wrong[np.argsort(-conf[wrong], kind="stable")][:n_errors]
    return hits, [int(i) for i in errors]


def plot_panels(examples, path, per_row, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(examples[0])
    rows = -(-len(examples) // per_row)
    cols = per_row * n_panels
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for e, panels in enumerate(examples):
        for p, (img, cam, text) in enumerate(panels):
            ax = axes[e // per_row, (e % per_row) * n_panels + p]
            ax.imshow(img)
            if cam is not None:
                ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(text, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_figures(ckpt, n_errors=8, num_workers=2, splits_csv=SPLITS_CSV):
    ckpt = Path(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(ckpt)
    model.to(device)
    stats = load_stats(splits_csv)
    loader = get_dataloaders(256, augment=False, num_workers=num_workers, splits_csv=splits_csv)["val"]
    ds = loader.dataset
    y_true, probs = predict_proba(model, loader, device)
    pred = probs.argmax(axis=1)
    hits, errors = pick_examples(y_true, probs, n_errors)

    def maps_for(idx, classes):
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        with GradCAM(model) as cam:
            return x.cpu(), cam(x, torch.tensor(classes, device=device)).numpy()

    name, out, paths = cfg.get("name", ckpt.parent.name), ckpt.parent, []
    if hits:
        x, cams = maps_for(hits, pred[hits])
        examples = [[(denormalize(x[k], stats), None, CLASSES[y_true[i]]),
                     (denormalize(x[k], stats), cams[k], f"{CLASSES[pred[i]]} ({probs[i, pred[i]]:.2f})")]
                    for k, i in enumerate(hits)]
        paths.append(out / "gradcam_aciertos_val.png")
        plot_panels(examples, paths[-1], per_row=3, title=f"{name} · Grad-CAM de aciertos (val)")
    if errors:
        x, cams_pred = maps_for(errors, pred[errors])
        _, cams_true = maps_for(errors, y_true[errors])
        examples = [[(denormalize(x[k], stats), None, f"real: {CLASSES[y_true[i]]}"),
                     (denormalize(x[k], stats), cams_pred[k], f"predicha {CLASSES[pred[i]]} ({probs[i, pred[i]]:.2f})"),
                     (denormalize(x[k], stats), cams_true[k], f"mapa de {CLASSES[y_true[i]]}")]
                    for k, i in enumerate(errors)]
        paths.append(out / "gradcam_errores_val.png")
        plot_panels(examples, paths[-1], per_row=2, title=f"{name} · Grad-CAM de los errores más seguros (val)")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="results/<exp>/best.pt")
    parser.add_argument("--n-errores", type=int, default=8, help="nº de errores a mostrar (los más seguros)")
    args = parser.parse_args()
    for path in make_figures(args.ckpt, args.n_errores):
        print(f"Guardada {path}")


if __name__ == "__main__":
    main()
