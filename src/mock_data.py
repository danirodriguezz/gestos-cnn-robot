"""Datos falsos con la misma API que dataset.get_dataloaders (tarea C3).

Sustituye a src/dataset.py de Daniel hasta que esté publicado (H1): permite escribir y probar
train.py y los configs sin los recortes reales. Cuando llegue, en train.py solo cambia:
    from src.mock_data import get_dataloaders   ->   from src.dataset import get_dataloaders

Las imágenes son ruido N(0, 1) (como si ya estuvieran normalizadas con stats.json) y las
etiquetas son aleatorias: no hay nada que aprender, solo sirven para probar el código.
Nunca se usa en el flujo final.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.common import CLASSES, IMG_SIZE, SEED

SPLITS = ("train", "val", "test")


def make_split(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=g)
    y = torch.randint(0, len(CLASSES), (n,), generator=g)
    return TensorDataset(x, y)


def get_dataloaders(batch_size=64, augment=True, num_workers=2, splits_csv=None, n=512):
    return {split: DataLoader(make_split(n, SEED + i), batch_size=batch_size, shuffle=split == "train")
            for i, split in enumerate(SPLITS)}
