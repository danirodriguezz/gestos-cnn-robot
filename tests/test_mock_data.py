"""Pruebas de src/mock_data.py (tarea C3). Uso: uv run python -m pytest tests/test_mock_data.py"""
import torch

from src.common import CLASSES, IMG_SIZE
from src.mock_data import get_dataloaders


def test_claves():
    assert set(get_dataloaders()) == {"train", "val", "test"}


def test_forma_del_contrato():
    for loader in get_dataloaders(batch_size=16, n=40).values():
        x, y = next(iter(loader))
        assert x.shape == (16, 3, IMG_SIZE, IMG_SIZE) and x.dtype == torch.float32
        assert y.shape == (16,) and y.dtype == torch.int64
        assert 0 <= y.min() and y.max() < len(CLASSES)


def test_n_y_batch_size():
    loaders = get_dataloaders(batch_size=10, n=25)
    for loader in loaders.values():
        assert len(loader.dataset) == 25
        assert [len(y) for _, y in loader] == [10, 10, 5]


def test_reproducible():
    a, b = get_dataloaders(n=32), get_dataloaders(n=32)
    for split in ("train", "val", "test"):
        assert torch.equal(a[split].dataset.tensors[0], b[split].dataset.tensors[0])
        assert torch.equal(a[split].dataset.tensors[1], b[split].dataset.tensors[1])


def test_particiones_distintas():
    loaders = get_dataloaders(n=32)
    assert not torch.equal(loaders["train"].dataset.tensors[0], loaders["val"].dataset.tensors[0])


def test_acepta_parametros_del_dataset_real():
    loaders = get_dataloaders(64, augment=False, num_workers=0, splits_csv="data/splits.csv", n=8)
    assert len(loaders["val"].dataset) == 8
