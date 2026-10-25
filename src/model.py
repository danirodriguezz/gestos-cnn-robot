"""Modelo GestureCNN, build_model y guardado/carga de checkpoints (tarea C2).

Uso: uv run python -m src.model     (imprime la forma de salida y el nº de parámetros)

Contrato (plan, sección 4.1):
    build_model(arch="gesturecnn", use_bn=True, use_dropout=True, n_classes=9) -> nn.Module
    todo modelo devuelto tiene el atributo model.gradcam_layer (última conv)
    save_checkpoint(path, model, cfg)   # {"state_dict", "cfg", "classes"}
    load_checkpoint(path) -> (model, cfg)   # modelo ya en eval()
"""
from pathlib import Path

import torch
from torch import nn

from src.common import CLASSES, IMG_SIZE

MODEL_KEYS = ("arch", "use_bn", "use_dropout")   # claves de cfg que definen la arquitectura


def block(cin, cout, p_drop=0.0, use_bn=True):
    """conv -> BN -> ReLU -> conv -> BN -> ReLU -> MaxPool (+ Dropout2d). Divide la resolución por 2."""
    # sin BN se pone Identity para que los índices no cambien: [3] es siempre la segunda conv
    bn = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
    layers = [nn.Conv2d(cin, cout, 3, padding=1), bn(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1), bn(cout), nn.ReLU(inplace=True),
              nn.MaxPool2d(2)]
    if p_drop > 0:
        layers.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*layers)


class GestureCNN(nn.Module):
    """5 bloques (128 -> 4 px) + Global Average Pooling + Dropout + Linear."""

    def __init__(self, n_classes=len(CLASSES), use_bn=True, use_dropout=True):
        super().__init__()
        d = (lambda p: p) if use_dropout else (lambda p: 0.0)
        b = lambda cin, cout, p=0.0: block(cin, cout, d(p), use_bn)
        self.features = nn.Sequential(b(3, 32), b(32, 64), b(64, 128, 0.1),
                                      b(128, 256, 0.2), b(256, 256, 0.3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(d(0.5)), nn.Linear(256, n_classes))
        self.gradcam_layer = self.features[-1][3]      # última conv (contrato con gradcam.py)

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch="gesturecnn", use_bn=True, use_dropout=True, n_classes=len(CLASSES)):
    if arch == "gesturecnn":
        return GestureCNN(n_classes, use_bn, use_dropout)
    raise ValueError(f"arquitectura desconocida: {arch!r} (disponibles: 'gesturecnn')")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": dict(cfg), "classes": list(CLASSES)}, path)


def load_checkpoint(path):
    """Reconstruye el modelo a partir de su cfg y carga los pesos. Funciona en CPU aunque se entrenara en GPU."""
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["classes"] != CLASSES:
        raise ValueError(f"las clases del checkpoint {ckpt['classes']} no coinciden con common.CLASSES")
    cfg = ckpt["cfg"]
    model = build_model(**{k: cfg[k] for k in MODEL_KEYS if k in cfg}, n_classes=len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), cfg


def main():
    for name, kwargs in [("completo", {}), ("sin_bn", {"use_bn": False}), ("sin_dropout", {"use_dropout": False})]:
        model = build_model(**kwargs).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
        print(f"{name:<12} salida {tuple(out.shape)}  ·  {count_params(model):,} parámetros")


if __name__ == "__main__":
    main()
