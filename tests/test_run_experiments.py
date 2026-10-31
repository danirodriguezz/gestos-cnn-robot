"""Pruebas de configs/ y src/run_experiments.py (tarea C5).
Uso: uv run python -m pytest tests/test_run_experiments.py"""
import pytest

from src.model import load_checkpoint
from src.run_experiments import ABLATION, CONFIGS_DIR, run
from src.train import load_config

CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))
MOCK = {"data": "mock", "epochs": 1, "mock_n": 16, "batch_size": 8, "num_workers": 0, "pretrained": False}


def test_estan_las_5_configs():
    assert [p.stem for p in CONFIGS] == ["completo", "resnet18", "sin_aug", "sin_bn", "sin_dropout"]
    assert all(p.exists() for p in ABLATION)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_name_coincide_con_el_fichero(path):
    assert load_config(path)["name"] == path.stem


def test_ablacion_cambia_un_solo_componente():
    full = load_config(CONFIGS_DIR / "completo.yaml")
    for path, key in [("sin_dropout", "use_dropout"), ("sin_bn", "use_bn"), ("sin_aug", "augment")]:
        cfg = load_config(CONFIGS_DIR / f"{path}.yaml")
        diff = {k for k in cfg if cfg[k] != full[k]} - {"name"}
        assert diff == {key} and cfg[key] is False


def test_las_5_configs_corren_1_epoca_con_el_mock(tmp_path):
    rows = run(CONFIGS, results_dir=tmp_path, **MOCK)
    assert [r["name"] for r in rows] == [p.stem for p in CONFIGS]
    for r in rows:
        assert r["epochs"] == 1
        model, cfg = load_checkpoint(tmp_path / r["name"] / "best.pt")
        assert cfg["name"] == r["name"] and hasattr(model, "gradcam_layer")


def test_valida_todas_las_configs_antes_de_entrenar(tmp_path):
    bad = tmp_path / "mal.yaml"
    bad.write_text("name: mal\nuse_droput: false\n")
    with pytest.raises(ValueError):
        run([CONFIGS_DIR / "completo.yaml", bad], results_dir=tmp_path, **MOCK)
    assert not (tmp_path / "completo").exists()


def test_rechaza_nombres_repetidos(tmp_path):
    with pytest.raises(ValueError):
        run([CONFIGS_DIR / "completo.yaml"] * 2, results_dir=tmp_path, **MOCK)


def test_otra_semilla_escribe_en_otra_carpeta(tmp_path):
    rows = run([CONFIGS_DIR / "completo.yaml"], results_dir=tmp_path, seed=43, **MOCK)
    assert rows[0]["name"] == "completo_s43"
    _, cfg = load_checkpoint(tmp_path / "completo_s43" / "best.pt")
    assert cfg["seed"] == 43 and not (tmp_path / "completo").exists()
