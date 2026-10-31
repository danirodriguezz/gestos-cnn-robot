"""Contrato compartido del proyecto. No romper sin avisar al otro (ver plan, sección 4.1)."""
import random
from pathlib import Path

import numpy as np
import torch

CLASSES = ["stop", "like", "dislike", "one", "peace", "fist", "palm", "call", "no_gesture"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE, MARGIN, SEED = 128, 0.25, 42

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
CROPS_DIR  = DATA_DIR / "crops"
SPLITS_CSV = DATA_DIR / "splits.csv"
STATS_JSON = DATA_DIR / "stats.json"
RESULTS    = ROOT / "results"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def crop_hand(img, box, margin=MARGIN, size=IMG_SIZE):
    W, H = img.size
    x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    l, t = cx - side / 2, cy - side / 2
    return img.crop((round(l), round(t), round(l + side), round(t + side))).resize((size, size))