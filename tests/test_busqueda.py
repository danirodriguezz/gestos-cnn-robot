"""Pruebas de configs/busqueda/ y src/busqueda.py. Uso: uv run python -m pytest tests/test_busqueda.py"""
import json

import pytest
import torch

from src.busqueda import REFERENCE, SEARCH, change, run, summarize
from src.common import CLASSES, IMG_SIZE
from src.model import MODEL_KEYS, build_model, load_checkpoint
from src.train import load_config

MOCK = {"data": "mock", "epochs": 1, "mock_n": 16, "batch_size": 8, "num_workers": 0}


def fake_evaluate(ckpt):
    ckpt.parent.joinpath("metrics_val.json").write_text(json.dumps({"accuracy": 0.9, "f1_macro": 0.9, "n": 10}))


def test_al_menos_5_configs_con_nombre_del_fichero():
    assert len(SEARCH) >= 5
    assert all(load_config(p)["name"] == p.stem for p in SEARCH)


@pytest.mark.parametrize("path", SEARCH, ids=lambda p: p.stem)
def test_cada_config_cambia_un_solo_hiperparametro(path):
    ref, cfg = load_config(REFERENCE), load_config(path)
    assert len([k for k in cfg if k != "name" and cfg[k] != ref[k]]) == 1


@pytest.mark.parametrize("path", SEARCH, ids=lambda p: p.stem)
def test_cada_arquitectura_se_construye(path):
    cfg = load_config(path)
    model = build_model(**{k: cfg[k] for k in MODEL_KEYS}).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE)).shape == (2, len(CLASSES))


def test_change():
    ref = load_config(REFERENCE)
    assert change(load_config(SEARCH[0]), ref) != "(referencia)"
    assert change(ref, ref) == "(referencia)"


def test_corre_con_el_mock_y_se_salta_lo_hecho(tmp_path):
    paths = SEARCH[:2]
    assert len(run(paths, tmp_path, fake_evaluate, **MOCK)) == 2
    for p in paths:
        model, cfg = load_checkpoint(tmp_path / p.stem / "best.pt")
        assert cfg["name"] == p.stem
    assert run(paths, tmp_path, fake_evaluate, **MOCK) == []


def test_resumen(tmp_path):
    paths = SEARCH[:2]
    run([REFERENCE, *paths], tmp_path, fake_evaluate, **MOCK)
    rows = summarize(paths, tmp_path, tmp_path / "busqueda.csv", tmp_path / "curvas.png")
    assert [r["experimento"] for r in rows] == ["completo", *(p.stem for p in paths)]
    assert rows[0]["cambio"] == "(referencia)" and rows[0]["errores"] == 1
    assert (tmp_path / "busqueda.csv").exists() and (tmp_path / "curvas.png").exists()
