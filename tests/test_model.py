"""Pruebas de src/model.py (tarea C2). Uso: uv run python -m pytest tests/test_model.py"""
import pytest
import torch
from torch import nn

from src.common import CLASSES, IMG_SIZE, set_seed
from src.model import build_model, count_params, load_checkpoint, save_checkpoint


def batch(n=2):
    return torch.randn(n, 3, IMG_SIZE, IMG_SIZE)


@pytest.mark.parametrize("use_bn, use_dropout", [(True, True), (False, True), (True, False)])
def test_forma_salida(use_bn, use_dropout):
    model = build_model(use_bn=use_bn, use_dropout=use_dropout).eval()
    with torch.no_grad():
        assert model(batch()).shape == (2, len(CLASSES))


def test_numero_parametros():
    assert 2.2e6 < count_params(build_model()) < 2.6e6


def test_sin_bn_no_tiene_batchnorm():
    assert not any(isinstance(m, nn.BatchNorm2d) for m in build_model(use_bn=False).modules())


def test_sin_dropout_no_tiene_dropout():
    assert not any(isinstance(m, (nn.Dropout, nn.Dropout2d)) and m.p > 0
                   for m in build_model(use_dropout=False).modules())


@pytest.mark.parametrize("use_bn", [True, False])
def test_gradcam_layer_es_la_ultima_conv(use_bn):
    model = build_model(use_bn=use_bn)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert model.gradcam_layer is convs[-1] and model.gradcam_layer.out_channels == 256


def test_resnet18():
    model = build_model(arch="resnet18").eval()
    with torch.no_grad():
        assert model(batch()).shape == (2, len(CLASSES))
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert model.gradcam_layer is convs[-1] and model.gradcam_layer.out_channels == 512
    assert 11e6 < count_params(model) < 11.5e6


def test_guardar_y_cargar_resnet18(tmp_path):
    set_seed()
    model = build_model(arch="resnet18").eval()
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, {"arch": "resnet18", "pretrained": True})
    loaded, _ = load_checkpoint(path)
    x = batch()
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_arch_desconocida():
    with pytest.raises(ValueError):
        build_model(arch="vgg")


def test_guardar_y_cargar(tmp_path):
    set_seed()
    cfg = {"arch": "gesturecnn", "use_bn": False, "use_dropout": True, "lr": 1e-3}
    model = build_model(use_bn=False).eval()
    path = tmp_path / "exp" / "best.pt"
    save_checkpoint(path, model, cfg)

    loaded, loaded_cfg = load_checkpoint(path)
    assert loaded_cfg == cfg
    assert not loaded.training
    x = batch()
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_por_defecto_es_la_red_original():
    a = build_model(channels=(32, 64, 128, 256, 256), fc_hidden=0, activation="relu")
    assert count_params(a) == count_params(build_model()) == 2_357_673


def test_opciones_de_la_busqueda():
    model = build_model(channels=(32, 64, 128, 256), fc_hidden=128, activation="elu")
    assert len(model.features) == 4
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert model.gradcam_layer is convs[-1]
    assert any(isinstance(m, nn.ELU) for m in model.modules()) and not any(isinstance(m, nn.ReLU) for m in model.modules())
    with pytest.raises(ValueError):
        build_model(activation="tanhh")
