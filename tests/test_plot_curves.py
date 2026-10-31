"""Pruebas de src/plot_curves.py (tarea D8) con un history.csv sintético.

Uso: uv run python -m pytest tests/test_plot_curves.py
"""
import pandas as pd
import pytest

from src.plot_curves import best_epoch, load_history, lr_changes, plot_comparison, plot_experiment
from src.train import HISTORY_COLS


def fake_history(exp_dir, best=5, n=10):
    exp_dir.mkdir(parents=True)
    rows = [[e, 2 / e, 0.1 * e / n + 0.5, 1 + abs(e - best) / n, 0.8 - abs(e - best) / 50,
             1e-3 if e < 8 else 5e-4] for e in range(1, n + 1)]
    pd.DataFrame(rows, columns=HISTORY_COLS).to_csv(exp_dir / "history.csv", index=False)
    return exp_dir


def test_best_epoch(tmp_path):
    assert best_epoch(load_history(fake_history(tmp_path / "a", best=4))) == 4


def test_lr_changes(tmp_path):
    assert lr_changes(load_history(fake_history(tmp_path / "a"))) == [(8, pytest.approx(5e-4))]


def test_plot_experiment(tmp_path):
    path = plot_experiment(fake_history(tmp_path / "completo"))
    assert path == tmp_path / "completo" / "curves.png" and path.stat().st_size > 0


def test_plot_comparison(tmp_path):
    dirs = [fake_history(tmp_path / name, best=b) for name, b in (("completo", 5), ("sin_bn", 7))]
    path = plot_comparison(dirs, tmp_path / "comparacion.png")
    assert path.stat().st_size > 0
