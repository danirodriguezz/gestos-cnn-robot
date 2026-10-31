"""Dataset, transformaciones y get_dataloaders (tarea D7).

Uso: uv run python -m src.dataset    (resumen de los loaders + rejilla de augmentation)

Contrato (plan, sección 4.1):
    get_dataloaders(batch_size=64, augment=True, num_workers=2, splits_csv=SPLITS_CSV)
        -> {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    cada lote: x float32 (B, 3, 128, 128) normalizado con stats.json; y int64 (B,) en 0..8

Las rutas de splits.csv y stats.json se buscan en la carpeta de splits_csv: basta con pasar
otra ruta (p. ej. en Kaggle) para usar una copia de data/ en otro sitio.
Solo train se aumenta, y solo si augment=True; val y test nunca.

Genera: results/figures/augmentation_muestra.png   (original + 4 versiones aumentadas por fila)
"""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.common import CLASS_TO_IDX, IMG_SIZE, RESULTS, SEED, SPLITS_CSV, set_seed

SPLITS = ("train", "val", "test")
FIG_DIR = RESULTS / "figures"


def load_stats(splits_csv=SPLITS_CSV):
    return json.loads((Path(splits_csv).parent / "stats.json").read_text())


def build_transform(stats, augment):
    to_tensor = [T.ToTensor(), T.Normalize(stats["mean"], stats["std"])]
    if not augment:
        return T.Compose(to_tensor)
    return T.Compose([
        T.RandomRotation(15),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        *to_tensor,
    ])


class GestureDataset(Dataset):
    def __init__(self, df, root, transform):
        self.paths = [Path(root) / p for p in df["path"]]
        self.labels = [CLASS_TO_IDX[label] for label in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def get_dataloaders(batch_size=64, augment=True, num_workers=2, splits_csv=SPLITS_CSV):
    splits_csv = Path(splits_csv)
    df = pd.read_csv(splits_csv, dtype={"user_id": str})
    stats = load_stats(splits_csv)
    eval_tf = build_transform(stats, augment=False)
    train_tf = build_transform(stats, augment)

    loaders = {}
    for split in SPLITS:
        is_train = split == "train"
        ds = GestureDataset(df[df["split"] == split], splits_csv.parent, train_tf if is_train else eval_tf)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0)
    return loaders


def denormalize(x, stats):
    mean = torch.tensor(stats["mean"]).view(3, 1, 1)
    std = torch.tensor(stats["std"]).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_augmentation_grid(path, classes=("like", "dislike", "peace", "palm"), n_aug=4,
                           splits_csv=SPLITS_CSV, seed=SEED):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(seed)
    splits_csv = Path(splits_csv)
    df = pd.read_csv(splits_csv, dtype={"user_id": str})
    train = df[df["split"] == "train"]
    rows = pd.concat([train[train["label"] == c].sample(1, random_state=seed) for c in classes])
    stats = load_stats(splits_csv)
    plain = GestureDataset(rows, splits_csv.parent, build_transform(stats, augment=False))
    aug = GestureDataset(rows, splits_csv.parent, build_transform(stats, augment=True))

    fig, axes = plt.subplots(len(rows), n_aug + 1, figsize=((n_aug + 1) * 1.8, len(rows) * 1.9), squeeze=False)
    for r, label in enumerate(rows["label"]):
        images = [plain[r][0]] + [aug[r][0] for _ in range(n_aug)]
        for c, (ax, x) in enumerate(zip(axes[r], images)):
            ax.imshow(denormalize(x, stats))
            ax.set_title(f"{label} · original" if c == 0 else f"aumentada {c}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    set_seed()
    loaders = get_dataloaders(num_workers=0)
    for split, loader in loaders.items():
        x, y = next(iter(loader))
        print(f"{split:<5} {len(loader.dataset):>6} recortes · {len(loader):>4} lotes · "
              f"x {tuple(x.shape)} {x.dtype} · y {tuple(y.shape)} {y.dtype}")
    x, _ = next(iter(loaders["val"]))
    print(f"lote de val normalizado: media {x.mean(dim=(0, 2, 3)).numpy().round(2)} · "
          f"std {x.std(dim=(0, 2, 3)).numpy().round(2)}")

    path = FIG_DIR / "augmentation_muestra.png"
    plot_augmentation_grid(path)
    print(f"Rejilla guardada en {path}")


if __name__ == "__main__":
    main()
