"""Pruebas de src/gradcam.py (tarea D9). Uso: uv run python -m pytest tests/test_gradcam.py

Las de los mapas usan DummyNet y GestureCNN sin entrenar; la de punta a punta necesita los recortes.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.common import CLASSES, CROPS_DIR, IMG_SIZE, SPLITS_CSV, STATS_JSON, set_seed
from src.gradcam import GradCAM, make_figures, pick_examples
from src.model import build_model, save_checkpoint
from tests.dummy_model import DummyNet


def batch(n=4):
    set_seed()
    return torch.randn(n, 3, IMG_SIZE, IMG_SIZE)


def test_forma_y_rango():
    with GradCAM(DummyNet()) as cam:
        maps = cam(batch(), torch.tensor([0, 1, 2, 3]))
    assert maps.shape == (4, IMG_SIZE, IMG_SIZE)
    assert maps.min() >= 0 and maps.max() <= 1


def test_depende_de_la_clase():
    model, x = DummyNet(), batch(1)
    with GradCAM(model) as cam:
        maps = [cam(x, torch.tensor([c])) for c in range(len(CLASSES))]
    assert any(not torch.allclose(maps[0], m) for m in maps[1:])


def test_quita_los_hooks():
    model = DummyNet()
    with GradCAM(model) as cam:
        cam(batch(1), torch.tensor([0]))
        assert len(model.gradcam_layer._forward_hooks) == 1
    assert len(model.gradcam_layer._forward_hooks) == 0


@pytest.mark.parametrize("use_bn", [True, False])
def test_gesturecnn(use_bn):
    model = build_model(use_bn=use_bn)
    with GradCAM(model) as cam:
        maps = cam(batch(2), torch.tensor([0, 5]))
        assert cam.acts.shape[-2:] == (8, 8)
        assert cam.acts.min() < 0
    assert maps.shape == (2, IMG_SIZE, IMG_SIZE) and maps.max() <= 1


def test_pick_examples():
    y_true = np.array([0, 0, 1, 1, 2])
    probs = np.full((5, len(CLASSES)), 0.01)
    probs[0, 0] = 0.9
    probs[1, 1] = 0.6
    probs[2, 1] = 0.8
    probs[3, 0] = 0.95
    probs[4, 2] = 0.7
    hits, errors = pick_examples(y_true, probs)
    assert sorted(hits) == [0, 2, 4] and errors == [3, 1]


@pytest.mark.skipif(not (SPLITS_CSV.exists() and CROPS_DIR.exists()), reason="faltan splits.csv o los recortes")
def test_punta_a_punta(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pd.read_csv(SPLITS_CSV).groupby(["split", "label"]).head(3).to_csv(data / "splits.csv", index=False)
    (data / "stats.json").write_text(STATS_JSON.read_text())
    (data / "crops").symlink_to(CROPS_DIR)
    ckpt = tmp_path / "exp" / "best.pt"
    save_checkpoint(ckpt, build_model(), {"name": "exp", "arch": "gesturecnn", "use_bn": True, "use_dropout": True})

    paths = make_figures(ckpt, n_errors=4, num_workers=0, splits_csv=data / "splits.csv")
    assert ckpt.parent / "gradcam_errores_val.png" in paths
    assert all(p.stat().st_size > 0 for p in paths)
