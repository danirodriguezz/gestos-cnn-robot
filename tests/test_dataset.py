"""Pruebas de src/dataset.py (tarea D7). Uso: uv run python -m pytest tests/test_dataset.py

Se saltan si faltan splits.csv o los recortes (ejecuta src.prepare_data y src.make_splits).
"""
import pandas as pd
import pytest
import torch

from src.common import CLASSES, CROPS_DIR, IMG_SIZE, SPLITS_CSV, STATS_JSON
from src.dataset import SPLITS, get_dataloaders

if not (SPLITS_CSV.exists() and CROPS_DIR.exists()):
    pytest.skip("faltan data/splits.csv o data/crops/", allow_module_level=True)

splits = pd.read_csv(SPLITS_CSV)


@pytest.fixture(scope="module")
def loaders():
    return get_dataloaders(batch_size=16, num_workers=0)


def test_claves(loaders):
    assert set(loaders) == set(SPLITS)


def test_forma_del_contrato(loaders):
    for loader in loaders.values():
        x, y = next(iter(loader))
        assert x.shape == (16, 3, IMG_SIZE, IMG_SIZE) and x.dtype == torch.float32
        assert y.shape == (16,) and y.dtype == torch.int64
        assert 0 <= y.min() and y.max() < len(CLASSES)


def test_tamanos_como_splits_csv(loaders):
    for split, loader in loaders.items():
        assert len(loader.dataset) == (splits["split"] == split).sum()
    assert len(loaders["train"]) == len(loaders["train"].dataset) // 16
    assert len(loaders["val"]) == -(-len(loaders["val"].dataset) // 16)


def test_val_y_test_nunca_se_aumentan(loaders):
    for split in ("val", "test"):
        ds = loaders[split].dataset
        assert torch.equal(ds[0][0], ds[0][0])


def test_augment_true_cambia_la_imagen(loaders):
    torch.manual_seed(0)
    ds = loaders["train"].dataset
    assert not torch.equal(ds[0][0], ds[0][0])


def test_augment_false_es_determinista():
    ds = get_dataloaders(augment=False, num_workers=0)["train"].dataset
    assert torch.equal(ds[0][0], ds[0][0])


def test_normalizacion():
    x, _ = next(iter(get_dataloaders(batch_size=256, augment=False, num_workers=0)["train"]))
    assert x.mean(dim=(0, 2, 3)).abs().max() < 0.2
    assert (x.std(dim=(0, 2, 3)) - 1).abs().max() < 0.2


def test_rutas_relativas_a_splits_csv(tmp_path):
    splits.groupby("split").head(4).to_csv(tmp_path / "splits.csv", index=False)
    (tmp_path / "stats.json").write_text(STATS_JSON.read_text())
    (tmp_path / "crops").symlink_to(CROPS_DIR)
    loaders = get_dataloaders(batch_size=4, num_workers=0, splits_csv=tmp_path / "splits.csv")
    assert [len(loader.dataset) for loader in loaders.values()] == [4, 4, 4]
    assert next(iter(loaders["test"]))[0].shape == (4, 3, IMG_SIZE, IMG_SIZE)
