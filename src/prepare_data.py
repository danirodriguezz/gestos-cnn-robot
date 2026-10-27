"""Recorta todas las manos a 128x128 y genera las tablas de recuento por clase y persona (tarea D4).

Uso: uv run python -m src.prepare_data

Genera:
    data/crops/<clase>/<image_id>_<k>.jpg   un recorte por mano (k = índice de la mano en la imagen)
    data/crops.csv                          índice de recortes: path,label,user_id,image_id,hand_idx,box_px
    results/tablas/recuento_por_clase.csv   manos, imágenes y personas de cada clase
    results/tablas/recuento_por_persona.csv manos de cada persona en cada clase

Aquí no se descarta ninguna mano: equilibrar no_gesture y filtrar manos pequeñas (box_px) se
decide en la partición (D6), sin tener que volver a recortar.
"""
import json

import pandas as pd
from PIL import Image

from src.common import CLASSES, CROPS_DIR, DATA_DIR, RESULTS, crop_hand
from src.download import GESTURES, RAW_DIR

CROPS_CSV = DATA_DIR / "crops.csv"     # path relativo a DATA_DIR, con "/" para que valga en Kaggle
TABLES_DIR = RESULTS / "tablas"
JPEG_QUALITY = 95                      # alta: el recorte ya pierde detalle al ampliarse


def save_crop(crop, path):
    """Guarda en un temporal y renombra: si el script se corta, nunca queda un JPEG a medias."""
    tmp = path.with_suffix(".tmp")
    crop.save(tmp, format="JPEG", quality=JPEG_QUALITY)
    tmp.replace(path)


def crop_image(gesture, image_id, ann):
    """Recorta las manos de una imagen (las que falten) y devuelve una fila del índice por mano."""
    img = Image.open(RAW_DIR / "images" / gesture / f"{image_id}.jpg")   # perezoso: aún no decodifica
    W, H = img.size
    rows, rgb = [], None
    for k, (box, label) in enumerate(zip(ann["bboxes"], ann["labels"])):
        # la carpeta sale de la etiqueta, no de la carpeta de origen (una foto de fist es no_gesture)
        path = CROPS_DIR / label / f"{image_id}_{k}.jpg"
        if not path.exists():
            if rgb is None:
                rgb = img.convert("RGB")
            save_crop(crop_hand(rgb, box), path)
        rows.append({"path": path.relative_to(DATA_DIR).as_posix(), "label": label,
                     "user_id": ann["user_id"], "image_id": image_id, "hand_idx": k,
                     "box_px": round(max(box[2] * W, box[3] * H), 1)})   # lado de la caja sin margen
    return rows


def build_crops():
    for c in CLASSES:
        (CROPS_DIR / c).mkdir(parents=True, exist_ok=True)
    rows = []
    for g in GESTURES:
        ann = json.loads((RAW_DIR / "annotations" / f"{g}.json").read_text())
        for image_id in sorted(ann):
            rows.extend(crop_image(g, image_id, ann[image_id]))
        print(f"{g}: {len(ann)} imágenes recortadas")
    df = pd.DataFrame(rows)
    df.to_csv(CROPS_CSV, index=False)    # se escribe al final: si existe, el recorte terminó
    return df


def count_tables(df):
    by_class = (df.groupby("label")
                  .agg(manos=("path", "size"), imagenes=("image_id", "nunique"), personas=("user_id", "nunique"))
                  .reindex(CLASSES))
    by_person = pd.crosstab(df["user_id"], df["label"]).reindex(columns=CLASSES, fill_value=0)
    by_person["total"] = by_person.sum(axis=1)
    by_person = by_person.sort_values("total", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    by_class.to_csv(TABLES_DIR / "recuento_por_clase.csv")
    by_person.to_csv(TABLES_DIR / "recuento_por_persona.csv")
    return by_class, by_person


def print_summary(df, by_class, by_person):
    print(f"\n{by_class.to_string()}")
    print(f"\nTotal: {len(df)} recortes de {df['image_id'].nunique()} imágenes y {len(by_person)} personas")

    bins = pd.cut(by_person["total"], [0, 1, 5, 10, 20, by_person["total"].max()],
                  labels=["1", "2-5", "6-10", "11-20", ">20"])
    print("\nManos por persona:  " + "  ·  ".join(f"{k}: {v}" for k, v in bins.value_counts(sort=False).items()))
    top = by_person["total"].iloc[0]
    print(f"La persona con más manos tiene {top} ({top / len(df):.2%} del total)")

    q = df["box_px"].quantile([0.05, 0.5, 0.95])
    print(f"Lado de la caja (px): p5 {q[0.05]:.0f} · mediana {q[0.5]:.0f} · p95 {q[0.95]:.0f}"
          f" · menores de 24 px: {(df['box_px'] < 24).sum()}")


def main():
    df = build_crops()
    by_class, by_person = count_tables(df)
    print_summary(df, by_class, by_person)
    print(f"\nRecortes en {CROPS_DIR}, índice en {CROPS_CSV}, tablas en {TABLES_DIR}")


if __name__ == "__main__":
    main()
