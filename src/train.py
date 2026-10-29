"""Bucle de entrenamiento con parada anticipada (tarea C4).

Uso:
    uv run python -m src.train --config configs/completo.yaml
    uv run python -m src.train --data mock --epochs 3 --name prueba    (sin yaml: valores por defecto)

Escribe en results/<name>/:
    config.yaml   configuración completa usada (DEFAULTS + yaml + argumentos)
    history.csv   epoch,train_loss,train_acc,val_loss,val_acc,lr   (una fila por época)
    best.pt       checkpoint con la menor val_loss (save_checkpoint de model.py)

El test no se toca aquí: se evalúa una sola vez con evaluate.py.
"""
import argparse
import copy
import csv
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from src.common import RESULTS, SEED, set_seed
from src.model import MODEL_KEYS, build_model, save_checkpoint

DEFAULTS = {
    "name": "debug",            # carpeta dentro de results/
    "arch": "gesturecnn",       # arquitectura (build_model): gesturecnn | resnet18
    "pretrained": False,        # solo resnet18: parte de los pesos de ImageNet (necesita internet)
    "use_bn": True,
    "use_dropout": True,
    "augment": True,            # se pasa a get_dataloaders
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 50,               # máximo; la parada anticipada suele cortar antes
    "patience": 6,              # épocas sin mejorar val_loss antes de parar
    "scheduler": True,          # ReduceLROnPlateau sobre val_loss
    "num_workers": 2,
    "seed": SEED,
    "data": "real",             # "real" (src.dataset de Daniel) o "mock" (src.mock_data)
    "mock_n": 512,              # ejemplos por partición si data == "mock"
}
HISTORY_COLS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]


def load_config(path=None, **overrides):
    """DEFAULTS, sobrescritos por el yaml y después por los overrides que no sean None."""
    cfg = dict(DEFAULTS)
    if path is not None:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:                             # una errata en el yaml no debe ignorarse en silencio
        raise ValueError(f"claves desconocidas en la configuración: {sorted(unknown)}")
    return cfg


def get_loaders(cfg):
    if cfg["data"] == "mock":
        from src.mock_data import get_dataloaders
        return get_dataloaders(cfg["batch_size"], cfg["augment"], cfg["num_workers"], n=cfg["mock_n"])
    if cfg["data"] == "real":
        try:
            from src.dataset import get_dataloaders
        except ModuleNotFoundError as e:
            if e.name != "src.dataset":
                raise
            raise RuntimeError("src/dataset.py aún no existe: usa data: mock (o --data mock)") from e
        return get_dataloaders(cfg["batch_size"], augment=cfg["augment"], num_workers=cfg["num_workers"])
    raise ValueError(f"data debe ser 'real' o 'mock', no {cfg['data']!r}")


class EarlyStopping:
    """Para si val_loss no mejora en `patience` épocas y guarda una copia de los mejores pesos."""

    def __init__(self, patience=6):
        self.patience, self.best, self.count, self.state = patience, float("inf"), 0, None

    def step(self, val_loss, model):
        if val_loss < self.best:
            self.best, self.count = val_loss, 0
            self.state = copy.deepcopy(model.state_dict())   # mejores pesos
        else:
            self.count += 1
        return self.count >= self.patience                   # True -> parar

    @property
    def improved(self):
        return self.count == 0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                           # activa dropout y BN con estadísticas del lote
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()                            # sin gradientes: menos memoria y más rápido
def evaluate(model, loader, criterion, device):
    model.eval()                            # desactiva dropout; BN usa sus medias acumuladas
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train(cfg, results_dir=RESULTS):
    """Entrena según cfg. Devuelve (modelo con los mejores pesos, lista de filas de history)."""
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_loaders(cfg)
    model = build_model(**{k: cfg[k] for k in MODEL_KEYS}, pretrained=cfg["pretrained"]).to(device)

    out = Path(results_dir) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
                 if cfg["scheduler"] else None)
    stopper = EarlyStopping(cfg["patience"])

    print(f"[{cfg['name']}] {device} · {sum(p.numel() for p in model.parameters()):,} parámetros")
    history = []
    with open(out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        writer.writeheader()
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
            lr = optimizer.param_groups[0]["lr"]          # lr usado en esta época

            row = dict(zip(HISTORY_COLS, [epoch, train_loss, train_acc, val_loss, val_acc, lr]))
            history.append(row)
            writer.writerow(row)
            f.flush()                                     # si se corta la sesión, lo hecho queda escrito

            stop = stopper.step(val_loss, model)
            if stopper.improved:                          # se guarda en cada mejora, no solo al final
                save_checkpoint(out / "best.pt", model, cfg)
            print(f"época {epoch:3d} · train {train_loss:.4f} / {train_acc:.3f} · "
                  f"val {val_loss:.4f} / {val_acc:.3f} · lr {lr:.1e} · {time.perf_counter() - t0:.1f}s"
                  + (" *" if stopper.improved else ""))
            if scheduler is not None:
                scheduler.step(val_loss)
            if stop:
                print(f"Parada anticipada: {cfg['patience']} épocas sin mejorar val_loss")
                break

    model.load_state_dict(stopper.state)
    print(f"Mejor val_loss {stopper.best:.4f} · resultados en {out}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="yaml del experimento (configs/...)")
    parser.add_argument("--data", choices=["real", "mock"], help="sobrescribe data del yaml")
    parser.add_argument("--epochs", type=int, help="sobrescribe epochs del yaml")
    parser.add_argument("--name", help="sobrescribe name del yaml")
    args = parser.parse_args()
    train(load_config(args.config, data=args.data, epochs=args.epochs, name=args.name))


if __name__ == "__main__":
    main()
