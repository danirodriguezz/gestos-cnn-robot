"""Modelo falso con la interfaz de GestureCNN (plan, sección 4.2).

Sirve para probar evaluate.py y gradcam.py en segundos y sin un checkpoint entrenado.
Nunca se usa en el flujo final.
"""
from torch import nn

from src.common import CLASSES


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.gradcam_layer = self.conv              # mismo contrato que build_model
        self.head = nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, len(CLASSES)))

    def forward(self, x):
        return self.head(self.conv(x))
