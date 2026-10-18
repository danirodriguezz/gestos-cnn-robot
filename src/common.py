"""Contrato compartido del proyecto. No romper sin avisar al otro (ver plan, sección 4.1)."""
import os
import random
from pathlib import Path

import numpy as np
import torch

CLASSES = ["stop", "like", "dislike", "one", "peace", "fist", "palm", "call", "no_gesture"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE, MARGIN, SEED = 128, 0.25, 42

ROOT       = Path(__file__).resolve().parent.parent   # .../gestos-cnn-robot
DATA_DIR   = ROOT / "data"
CROPS_DIR  = DATA_DIR / "crops"        # data/crops/<clase>/<id>.jpg, RGB 128x128
SPLITS_CSV = DATA_DIR / "splits.csv"   # columnas: path,label,user_id,split (train|val|test)
STATS_JSON = DATA_DIR / "stats.json"   # {"mean": [r, g, b], "std": [r, g, b]} solo con train
RESULTS    = ROOT / "results"          # results/<experimento>/{best.pt, history.csv, config.yaml}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def crop_hand(img, box, margin=MARGIN, size=IMG_SIZE):
    """Recorte cuadrado de la mano con margen, redimensionado a size x size.

    img: PIL.Image en RGB (en la demo, convertir antes el fotograma BGR de OpenCV).
    box: [x, y, w, h] normalizados en [0, 1], esquina superior izquierda (confirmar en D3).
    Las zonas fuera de la imagen se rellenan con negro para que la mano nunca se deforme.
    """
    W, H = img.size
    x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    l, t = cx - side / 2, cy - side / 2
    return img.crop((round(l), round(t), round(l + side), round(t + side))).resize((size, size))