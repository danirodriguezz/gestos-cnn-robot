"""Pruebas de src/train.py (tarea C4). Uso: uv run python -m pytest tests/test_train.py"""
import csv

import pytest
import torch
from torch import nn

from src.model import load_checkpoint
from src.train import HISTORY_COLS, EarlyStopping, load_config, train


def mock_cfg(**kw):
    return load_config(**{"data": "mock", "mock_n": 32, "batch_size": 16, "num_workers": 0, **kw})


def test_early_stopping():
    model = nn.Linear(2, 2)
    stopper = EarlyStopping(patience=2)
    assert not stopper.step(1.0, model) and stopper.improved
    best = {k: v.clone() for k, v in stopper.state.items()}
    with torch.no_grad():
        model.weight.add_(1.0)              # los pesos cambian pero no mejora
    assert not stopper.step(1.5, model) and not stopper.improved
    assert stopper.step(1.2, model)         # segunda época sin mejorar -> parar
    assert all(torch.equal(best[k], stopper.state[k]) for k in best)


def test_config_rechaza_claves_desconocidas(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("lr: 0.01\nuse_droput: false\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_config_yaml_y_overrides(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("name: sin_bn\nuse_bn: false\nepochs: 30\n")
    cfg = load_config(path, epochs=1, data=None)
    assert cfg["name"] == "sin_bn" and cfg["use_bn"] is False and cfg["epochs"] == 1 and cfg["data"] == "real"


def test_ficheros_de_salida(tmp_path):
    cfg = mock_cfg(name="prueba", epochs=2)
    train(cfg, results_dir=tmp_path)
    out = tmp_path / "prueba"

    with open(out / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == HISTORY_COLS and len(rows) == 2
    assert load_config(out / "config.yaml") == cfg

    model, ckpt_cfg = load_checkpoint(out / "best.pt")
    assert ckpt_cfg == cfg and not model.training


def test_sobreajusta_64_imagenes(tmp_path):
    """Criterio de «hecho» de C4: si el bucle está bien, memoriza 64 imágenes (pérdida ≈ 0)."""
    cfg = mock_cfg(name="overfit", mock_n=64, use_dropout=False, scheduler=False,
                   epochs=15, patience=15)  # val es ruido: la parada anticipada cortaría enseguida
    _, history = train(cfg, results_dir=tmp_path)
    assert history[-1]["train_loss"] < 0.05 and history[-1]["train_acc"] == 1.0
