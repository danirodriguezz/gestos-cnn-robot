"""Descarga el subconjunto de HaGRID y extrae solo las clases del proyecto.

Uso: uv run python -m src.download

Resultado:
    data/raw/images/<gesto>/<id>.jpg
    data/raw/annotations/<gesto>.json   (solo las anotaciones de las imágenes extraídas)
"""
import json
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.common import CLASSES, DATA_DIR

REPO_ID  = "cj-mills/hagrid-sample-30k-384p"   # HaGRID v1 a 384 px, con bboxes y user_id
ZIP_NAME = "hagrid-sample-30k-384p.zip"
PREFIX   = "hagrid-sample-30k-384p"            # carpeta raíz dentro del zip
RAW_DIR  = DATA_DIR / "raw"

# no_gesture no tiene carpeta propia: sale de la segunda mano de las imágenes de los gestos
GESTURES = [c for c in CLASSES if c != "no_gesture"]


def extract_gesture(zf, gesture, names):
    """Extrae las imágenes de un gesto y guarda sus anotaciones filtradas. Devuelve las anotaciones."""
    img_dir = RAW_DIR / "images" / gesture
    ann_path = RAW_DIR / "annotations" / f"{gesture}.json"
    if ann_path.exists():                       # el JSON se escribe al final: si existe, el gesto está completo
        print(f"{gesture}: ya extraído, se omite")
        return json.loads(ann_path.read_text())

    # el JSON trae el dataset original completo (~28.000 entradas); el zip solo ~1.750 imágenes
    all_ann = json.loads(zf.read(f"{PREFIX}/ann_train_val/{gesture}.json"))
    folder = f"{PREFIX}/hagrid_30k/train_val_{gesture}/"
    members = [n for n in names if n.startswith(folder) and n.endswith(".jpg")]

    img_dir.mkdir(parents=True, exist_ok=True)
    ann, skipped = {}, 0
    for member in members:
        image_id = Path(member).stem
        if image_id not in all_ann:             # imagen sin anotación: sin caja no se puede recortar
            skipped += 1
            continue
        (img_dir / f"{image_id}.jpg").write_bytes(zf.read(member))
        ann[image_id] = all_ann[image_id]

    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(ann))
    print(f"{gesture}: {len(ann)} imágenes extraídas ({skipped} sin anotación descartadas)")
    return ann


def print_summary(annotations):
    hands = Counter(label for ann in annotations.values() for a in ann.values() for label in a["labels"])
    users = {a["user_id"] for ann in annotations.values() for a in ann.values()}
    print(f"\n{'clase':<12}{'imágenes':>9}{'manos':>8}")
    for g in GESTURES:
        print(f"{g:<12}{len(annotations[g]):>9}{hands[g]:>8}")
    print(f"{'no_gesture':<12}{'-':>9}{hands['no_gesture']:>8}")
    print(f"\nTotal: {sum(len(a) for a in annotations.values())} imágenes de {len(users)} personas")


def main():
    # se guarda en la caché de Hugging Face (~/.cache/huggingface): la segunda vez no se descarga
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        annotations = {g: extract_gesture(zf, g, names) for g in GESTURES}
    print_summary(annotations)


if __name__ == "__main__":
    main()
