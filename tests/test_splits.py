"""Pruebas de data/splits.csv y data/stats.json (tarea D6). Uso: uv run python -m pytest tests/test_splits.py

Se saltan si aún no se ha ejecutado src.make_splits; la de rutas, si no están los recortes.
"""
import json

import pytest
import pandas as pd

from src.common import CLASSES, CROPS_DIR, DATA_DIR, SPLITS_CSV, STATS_JSON
from src.make_splits import SPLITS

if not SPLITS_CSV.exists():
    pytest.skip("falta data/splits.csv: ejecuta uv run python -m src.make_splits", allow_module_level=True)

df = pd.read_csv(SPLITS_CSV, dtype={"user_id": str})


def test_columnas_del_contrato():
    assert list(df.columns) == ["path", "label", "user_id", "split"]
    assert set(df["split"]) == set(SPLITS)
    assert set(df["label"]) == set(CLASSES)
    assert not df["path"].duplicated().any()


def test_ninguna_persona_en_dos_particiones():
    assert df.groupby("user_id")["split"].nunique().max() == 1


def test_proporciones_70_15_15():
    frac = df["split"].value_counts(normalize=True)
    for split, target in zip(SPLITS, (0.70, 0.15, 0.15)):
        assert abs(frac[split] - target) < 0.02, (split, frac[split])


def test_clases_estratificadas():
    """Cada clase pesa en cada partición lo mismo que en el total (±1,5 puntos)."""
    total = df["label"].value_counts(normalize=True)
    for split in SPLITS:
        part = df.loc[df["split"] == split, "label"].value_counts(normalize=True)
        assert (part.reindex(CLASSES) - total.reindex(CLASSES)).abs().max() < 0.015, split


def test_no_gesture_equilibrado():
    counts = df["label"].value_counts()
    gestures = counts.drop("no_gesture")
    assert abs(counts["no_gesture"] - gestures.mean()) <= 1


@pytest.mark.skipif(not CROPS_DIR.exists(), reason="faltan los recortes: ejecuta src.prepare_data")
def test_rutas_existen():
    assert all((DATA_DIR / p).exists() for p in df["path"])


def test_stats():
    stats = json.loads(STATS_JSON.read_text())
    assert set(stats) == {"mean", "std"}
    assert all(0 < m < 1 for m in stats["mean"]) and all(0 < s < 0.5 for s in stats["std"])
