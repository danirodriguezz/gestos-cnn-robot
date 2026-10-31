"""Modelos GestureCNN y ResNet18, build_model y guardado/carga de checkpoints (tareas C2 y C5).

Uso: uv run python -m src.model     (imprime la forma de salida y el nº de parámetros)

Contrato (plan, sección 4.1):
    build_model(arch="gesturecnn", use_bn=True, use_dropout=True, n_classes=9) -> nn.Module
    todo modelo devuelto tiene el atributo model.gradcam_layer (última conv)
    save_checkpoint(path, model, cfg) guarda {"state_dict", "cfg", "classes"}
    load_checkpoint(path) -> (model, cfg), con el modelo ya en eval()

Ampliación de C5 (sin romper el contrato): arch="resnet18" y el parámetro opcional
pretrained=False, que solo afecta a resnet18 (pesos de ImageNet de torchvision).

Ampliación para la búsqueda de hiperparámetros: channels, fc_hidden y activation (solo gesturecnn).
Con sus valores por defecto la red es exactamente la de C2, así que los checkpoints anteriores cargan igual.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.common import CLASSES, IMG_SIZE

ARCHS = ("gesturecnn", "resnet18")
MODEL_KEYS = ("arch", "use_bn", "use_dropout", "channels", "fc_hidden", "activation")

CHANNELS = (32, 64, 128, 256, 256)
BLOCK_DROPOUT = (0.1, 0.2, 0.3)
ACTIVATIONS = {"relu": lambda: nn.ReLU(inplace=True),
               "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
               "elu": lambda: nn.ELU(inplace=True)}


def block(cin, cout, p_drop=0.0, use_bn=True, activation="relu"):
    bn = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
    act = ACTIVATIONS[activation]
    layers = [nn.Conv2d(cin, cout, 3, padding=1), bn(cout), act(),
              nn.Conv2d(cout, cout, 3, padding=1), bn(cout), act(),
              nn.MaxPool2d(2)]
    if p_drop > 0:
        layers.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*layers)


class GestureCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES), use_bn=True, use_dropout=True,
                 channels=CHANNELS, fc_hidden=0, activation="relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activación desconocida: {activation!r} (disponibles: {', '.join(ACTIVATIONS)})")
        d = (lambda p: p) if use_dropout else (lambda p: 0.0)
        drops = [0.0] * max(len(channels) - len(BLOCK_DROPOUT), 0) + list(BLOCK_DROPOUT)[-len(channels):]
        cins = [3] + list(channels[:-1])
        self.features = nn.Sequential(*[block(cin, cout, d(p), use_bn, activation)
                                        for cin, cout, p in zip(cins, channels, drops)])
        hidden = [nn.Linear(channels[-1], fc_hidden), ACTIVATIONS[activation]()] if fc_hidden else []
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), *hidden,
                                  nn.Dropout(d(0.5)), nn.Linear(fc_hidden or channels[-1], n_classes))
        self.gradcam_layer = self.features[-1][3]

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(n_classes=len(CLASSES), pretrained=False):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.gradcam_layer = model.layer4[-1].conv2
    return model


def build_model(arch="gesturecnn", use_bn=True, use_dropout=True, n_classes=len(CLASSES), pretrained=False,
                channels=CHANNELS, fc_hidden=0, activation="relu"):
    if arch == "gesturecnn":
        return GestureCNN(n_classes, use_bn, use_dropout, tuple(channels), fc_hidden, activation)
    if arch == "resnet18":
        return build_resnet18(n_classes, pretrained)
    raise ValueError(f"arquitectura desconocida: {arch!r} (disponibles: {', '.join(ARCHS)})")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": dict(cfg), "classes": list(CLASSES)}, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["classes"] != CLASSES:
        raise ValueError(f"las clases del checkpoint {ckpt['classes']} no coinciden con common.CLASSES")
    cfg = ckpt["cfg"]
    model = build_model(**{k: cfg[k] for k in MODEL_KEYS if k in cfg}, n_classes=len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), cfg


def main():
    for name, kwargs in [("completo", {}), ("sin_bn", {"use_bn": False}), ("sin_dropout", {"use_dropout": False}),
                         ("resnet18", {"arch": "resnet18"})]:
        model = build_model(**kwargs).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
        print(f"{name:<12} salida {tuple(out.shape)}  ·  {count_params(model):,} parámetros")


if __name__ == "__main__":
    main()
