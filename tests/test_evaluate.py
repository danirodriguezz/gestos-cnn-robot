"""Pruebas de src/evaluate.py (tarea D8). Uso: uv run python -m pytest tests/test_evaluate.py

Las de métricas usan DummyNet y datos falsos; la de punta a punta necesita los recortes y se salta sin ellos.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.common import CLASSES, CROPS_DIR, IMG_SIZE, SPLITS_CSV, STATS_JSON
from src.evaluate import compute_metrics, evaluate, plot_confusion, predict, top_confusions
from src.model import build_model, save_checkpoint
from tests.dummy_model import DummyNet

ALL = np.arange(len(CLASSES))


def test_prediccion_perfecta():
    m = compute_metrics(ALL, ALL)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0 and m["n"] == len(CLASSES)
    assert list(m["f1_por_clase"]) == CLASSES


def test_metricas_conocidas():
    # 2 stop (uno se predice like) y 2 like bien: acc 3/4; F1 stop = 2/3, F1 like = 0,8
    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert m["accuracy"] == 0.75
    assert m["f1_por_clase"]["stop"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1_por_clase"]["like"] == pytest.approx(0.8, abs=1e-4)


def test_top_confusions():
    y_true = np.concatenate([ALL, [0, 0, 0]])
    y_pred = np.concatenate([ALL, [6, 6, 0]])      # 2 de los 4 stop se predicen palm
    assert top_confusions(y_true, y_pred)[0] == ("stop", "palm", 0.5)


def test_predict_con_dummynet():
    x = torch.randn(20, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, len(CLASSES), (20,))
    model = DummyNet().train()
    y_true, y_pred = predict(model, DataLoader(TensorDataset(x, y), batch_size=8))
    assert not model.training                        # predict deja el modelo en eval()
    assert np.array_equal(y_true, y.numpy()) and y_pred.shape == (20,)
    assert y_pred.min() >= 0 and y_pred.max() < len(CLASSES)


def test_plot_confusion(tmp_path):
    rng = np.random.default_rng(0)
    plot_confusion(rng.integers(0, 9, 100), rng.integers(0, 9, 100), tmp_path / "c.png", "prueba")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_test_se_evalua_una_sola_vez(tmp_path):
    (tmp_path / "metrics_test.json").write_text("{}")
    with pytest.raises(SystemExit):                  # se niega antes de cargar nada
        evaluate(tmp_path / "best.pt", split="test")


@pytest.mark.skipif(not (SPLITS_CSV.exists() and CROPS_DIR.exists()), reason="faltan splits.csv o los recortes")
def test_punta_a_punta(tmp_path):
    """GestureCNN sin entrenar guardado con save_checkpoint, evaluado sobre un subconjunto real."""
    data = tmp_path / "data"
    data.mkdir()
    pd.read_csv(SPLITS_CSV).groupby(["split", "label"]).head(3).to_csv(data / "splits.csv", index=False)
    (data / "stats.json").write_text(STATS_JSON.read_text())
    (data / "crops").symlink_to(CROPS_DIR)
    ckpt = tmp_path / "exp" / "best.pt"
    save_checkpoint(ckpt, build_model(), {"name": "exp", "arch": "gesturecnn", "use_bn": True, "use_dropout": True})

    for split in ("val", "test"):
        m = evaluate(ckpt, split, num_workers=0, splits_csv=data / "splits.csv")
        assert m["n"] == 3 * len(CLASSES)
        for f in (f"metrics_{split}.json", f"report_{split}.txt", f"confusion_{split}.png"):
            assert (ckpt.parent / f).exists()
    saved = json.loads((ckpt.parent / "metrics_val.json").read_text())
    assert {"accuracy", "f1_macro", "f1_por_clase"} <= set(saved)
    with pytest.raises(SystemExit):
        evaluate(ckpt, "test", num_workers=0, splits_csv=data / "splits.csv")
    evaluate(ckpt, "test", force=True, num_workers=0, splits_csv=data / "splits.csv")
