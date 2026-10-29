"""Pruebas de src/analysis.py con datos sintéticos (no necesitan checkpoints ni el dataset).
Uso: uv run python -m pytest tests/test_analysis.py"""
import numpy as np
import torch
from torch import nn

from src.analysis import bootstrap_by_group, count_macs, macro_name, mcnemar, num, pct
from tests.dummy_model import DummyNet


def test_bootstrap_contiene_el_valor_observado():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 600)
    pred = np.where(rng.random(600) < 0.9, y, (y + 1) % 9)        # ~90 % de acierto
    groups = np.repeat(np.arange(200), 3)                          # 200 personas con 3 fotos
    ic = bootstrap_by_group(y, pred, groups, n_boot=300)
    acc = (y == pred).mean()
    assert ic["acc_ic"][0] < acc < ic["acc_ic"][1] and ic["n_grupos"] == 200


def test_bootstrap_sin_errores_da_intervalo_degenerado():
    y = np.arange(90) % 9
    ic = bootstrap_by_group(y, y, np.arange(90) // 3, n_boot=50)
    assert ic["acc_ic"] == [1.0, 1.0] and ic["f1_ic"] == [1.0, 1.0]


def test_mcnemar():
    y = np.zeros(100, dtype=int)
    a = y.copy()
    b = y.copy()
    b[:10] = 1                                                     # B falla 10 que A acierta
    t = mcnemar(y, a, b)
    assert (t["solo_a"], t["solo_b"]) == (10, 0) and t["p"] < 0.01
    assert mcnemar(y, a, a)["p"] == 1.0                            # modelos idénticos: sin discordancias


def test_count_macs():
    # una conv 3->8 de 3x3 sobre 128x128 + una lineal 8->9
    assert count_macs(DummyNet()) == 128 * 128 * 8 * 3 * 3 * 3 + 8 * 9
    assert count_macs(nn.Sequential(nn.Flatten(), nn.Linear(3 * 128 * 128, 2))) == 3 * 128 * 128 * 2


def test_formato_latex():
    assert num(0.98765, 3) == "0{,}988"
    assert pct(0.9899) == r"98{,}99\,\%"
    assert macro_name("acc", "test", "sin_bn") == r"\AccTestSinBn"
    assert macro_name("lat", "cuda", "resnet18") == r"\LatCudaResnet"
    assert macro_name("acc", "val", "completo_s43") == r"\AccValCompletoS"
