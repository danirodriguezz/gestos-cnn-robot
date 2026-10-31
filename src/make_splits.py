"""Partición 70/15/15 por persona y estadísticas de normalización de train (tarea D6).

Uso: uv run python -m src.make_splits [--force]

Lee data/crops.csv (de prepare_data.py) y genera:
    data/splits.csv                  path,label,user_id,split   (contrato de common.py)
    data/stats.json                  {"mean": [r, g, b], "std": [r, g, b]} solo con train, en escala [0, 1]
    results/tablas/particion.csv     recortes por clase y partición

1. Equilibrado: no_gesture se submuestrea (con SEED) al tamaño medio de un gesto.
2. Partición: StratifiedGroupKFold en 20 trozos de ~5 % (14 train, 3 val, 3 test). Cada persona
   cae entera en un trozo y cada trozo tiene la misma mezcla de clases.

splits.csv queda fijado en el repositorio: no se regenera salvo con --force.
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.common import CLASSES, DATA_DIR, SEED, SPLITS_CSV, STATS_JSON
from src.prepare_data import CROPS_CSV, TABLES_DIR

SPLITS = ("train", "val", "test")
FOLDS = {"train": 14, "val": 3, "test": 3}


def balance(df, seed=SEED):
    gestures = df[df["label"] != "no_gesture"]
    n = round(gestures["label"].value_counts().mean())
    kept = df[df["label"] == "no_gesture"].sample(n, random_state=seed)
    return pd.concat([gestures, kept]).sort_index()


def assign_splits(df, seed=SEED):
    sgkf = StratifiedGroupKFold(n_splits=sum(FOLDS.values()), shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["user_id"])):
        fold[idx] = i
    split_of_fold = [s for s in SPLITS for _ in range(FOLDS[s])]
    return df.assign(split=[split_of_fold[f] for f in fold])


def channel_stats(paths):
    s, s2, n = np.zeros(3), np.zeros(3), 0
    for p in paths:
        a = np.asarray(Image.open(DATA_DIR / p).convert("RGB"), dtype=np.float64) / 255
        s += a.sum(axis=(0, 1))
        s2 += (a ** 2).sum(axis=(0, 1))
        n += a.shape[0] * a.shape[1]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()}


def split_table(df):
    table = pd.crosstab(df["label"], df["split"]).reindex(index=CLASSES, columns=SPLITS)
    table.loc["total"] = table.sum()
    return table


def print_summary(df, table, stats):
    print(f"\n{table.to_string()}")
    frac = df["split"].value_counts(normalize=True)
    people = df.groupby("split")["user_id"].nunique()
    print("\n" + "  ·  ".join(f"{s}: {frac[s]:.1%} de los recortes, {people[s]} personas" for s in SPLITS))
    print(f"Personas en más de una partición: {(df.groupby('user_id')['split'].nunique() > 1).sum()}")
    print(f"stats.json (train): mean {stats['mean']} · std {stats['std']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="regenera splits.csv aunque ya exista")
    args = parser.parse_args()
    if SPLITS_CSV.exists() and not args.force:
        raise SystemExit(f"{SPLITS_CSV} ya existe y está fijado; usa --force solo si de verdad hay que rehacerlo")

    crops = pd.read_csv(CROPS_CSV, dtype={"user_id": str})
    df = assign_splits(balance(crops))
    print(f"{len(crops)} recortes -> {len(df)} tras equilibrar no_gesture")

    df[["path", "label", "user_id", "split"]].to_csv(SPLITS_CSV, index=False)
    stats = channel_stats(df.loc[df["split"] == "train", "path"])
    STATS_JSON.write_text(json.dumps(stats, indent=2))

    table = split_table(df)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "particion.csv")
    print_summary(df, table, stats)
    print(f"\nGuardados {SPLITS_CSV}, {STATS_JSON} y {TABLES_DIR / 'particion.csv'}")


if __name__ == "__main__":
    main()
