"""Búsqueda manual de hiperparámetros: entrena configs/busqueda/*.yaml y las compara en validación.

Uso:
    bash scripts/busqueda.sh                                  (lo normal: evita la suspensión y guarda el log)
    uv run python -u -m src.busqueda                          (lo mismo, sin las comprobaciones del script)
    uv run python -m src.busqueda --solo-resumen              (rehace tabla y figuras sin entrenar)

Cada config es completo.yaml con un solo hiperparámetro cambiado (lr, lote, filtros, bloques, capa FC o
activación), con la misma semilla y la misma partición, así que la diferencia con completo se debe a ese
cambio. Se puede interrumpir: al relanzarla se salta las configs que ya tienen metrics_val.json.

Solo usa validación; el test no se toca. Escribe:
    results/<hp_...>/                         lo mismo que train.py + metrics_val.json, report_val.txt,
                                              confusion_val.png y curves.png
    results/tablas/busqueda.csv               una fila por config (y completo como referencia)
    results/figures/busqueda_curvas.png       val_loss y val_acc de todas superpuestas
"""
import argparse
import csv
import json
from pathlib import Path

from src.common import RESULTS
from src.model import MODEL_KEYS, build_model, count_params
from src.plot_curves import FIG_DIR, best_epoch, load_history, plot_comparison, plot_experiment
from src.run_experiments import CONFIGS_DIR
from src.train import load_config, train

SEARCH = sorted((CONFIGS_DIR / "busqueda").glob("*.yaml"))
REFERENCE = CONFIGS_DIR / "completo.yaml"         # ya entrenado en la ablación: results/completo/
TABLE_COLS = ["experimento", "cambio", "parametros", "epocas", "mejor_epoca",
              "val_loss", "val_acc", "f1_macro", "errores"]


def change(cfg, ref):
    """Texto con lo que cambia cfg respecto a la referencia, p. ej. 'lr=0.0003'."""
    keys = [k for k in cfg if k != "name" and cfg[k] != ref[k]]
    return ", ".join(f"{k}={cfg[k]}" for k in keys) or "(referencia)"


def is_done(exp_dir):
    return (Path(exp_dir) / "metrics_val.json").exists()


def evaluate_val(ckpt):
    from src.evaluate import evaluate                 # aquí para que las pruebas con mock no carguen los datos
    return evaluate(ckpt, "val")


def run(paths=SEARCH, results_dir=RESULTS, evaluate_fn=evaluate_val, **overrides):
    """Entrena y evalúa en val las configs que falten. Devuelve los nombres entrenados en esta llamada."""
    cfgs = [load_config(path, **overrides) for path in paths]   # una errata se ve antes de empezar
    pending = [cfg for cfg in cfgs if not is_done(Path(results_dir) / cfg["name"])]
    for cfg in cfgs:
        if cfg not in pending:
            print(f"[{cfg['name']}] ya hecho, se salta")
    for i, cfg in enumerate(pending, 1):
        print(f"\n=== {i}/{len(pending)} · {cfg['name']} ===")
        train(cfg, results_dir)
        evaluate_fn(Path(results_dir) / cfg["name"] / "best.pt")
    return [cfg["name"] for cfg in pending]


def summary_row(exp_dir, ref_cfg):
    exp_dir = Path(exp_dir)
    cfg = load_config(exp_dir / "config.yaml")
    metrics = json.loads((exp_dir / "metrics_val.json").read_text())
    history = load_history(exp_dir)
    best = history[history["epoch"] == best_epoch(history)].iloc[0]
    model = build_model(**{k: cfg[k] for k in MODEL_KEYS})
    return {"experimento": exp_dir.name, "cambio": change(cfg, ref_cfg), "parametros": count_params(model),
            "epocas": len(history), "mejor_epoca": int(best["epoch"]), "val_loss": round(best["val_loss"], 4),
            "val_acc": round(metrics["accuracy"], 4), "f1_macro": round(metrics["f1_macro"], 4),
            "errores": round(metrics["n"] * (1 - metrics["accuracy"]))}


GROUPS = {"Tasa de aprendizaje": ["hp_lr_3e-4", "hp_lr_3e-3"],
          "Tamaño de lote": ["hp_batch_32", "hp_batch_128"],
          "Capacidad": ["hp_filtros_mitad", "hp_4_bloques", "hp_fc_128"],
          "Activación": ["hp_elu", "hp_leaky_relu"]}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]           # orden fijo dentro de cada panel; completo en gris
STYLES = ["-", "--", ":"]


def plot_groups(ref_dir, dirs, path):
    """Un panel por tipo de hiperparámetro: completo (gris) frente a sus variantes. Fila de arriba:
    pérdida de val en escala log; abajo: accuracy de val ampliada a la zona alta (>= 0,95)."""
    import matplotlib.pyplot as plt

    names = {Path(d).name: Path(d) for d in dirs}
    groups = {t: [names[n] for n in ns if n in names] for t, ns in GROUPS.items()}
    others = [d for n, d in names.items() if not any(n in ns for ns in GROUPS.values())]
    if others:
        groups["Otros"] = others
    groups = {t: ds for t, ds in groups.items() if ds}
    fig, axes = plt.subplots(2, len(groups), figsize=(3.3 * len(groups), 5.6), sharex=True, sharey="row",
                             squeeze=False)
    for col, (title, ds) in enumerate(groups.items()):
        series = [(Path(ref_dir), "#5f5e5a", "-", "completo")] + [
            (d, COLORS[i % 3], STYLES[i % 3], d.name.removeprefix("hp_")) for i, d in enumerate(ds)]
        for d, color, style, label in series:
            h = load_history(d)
            best = h[h["epoch"] == best_epoch(h)]
            for row, metric in enumerate(("val_loss", "val_acc")):
                ax = axes[row, col]
                ax.plot(h["epoch"], h[metric], style, color=color, lw=1.6, label=label)
                ax.plot(best["epoch"], best[metric], "o", color=color, ms=6, mec="white", mew=1.2)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].legend(fontsize=7.5, frameon=False)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Pérdida de val (log)")
    axes[1, 0].set_ylim(0.95, 1.0)
    axes[1, 0].set_ylabel("Accuracy de val")
    for ax in axes.flat:
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("época")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summarize(paths=SEARCH, results_dir=RESULTS, table=RESULTS / "tablas" / "busqueda.csv",
              figure=FIG_DIR / "busqueda_curvas.png"):
    """Curvas de cada config, figura por grupos y tabla (completo + las configs ya terminadas)."""
    ref_cfg = load_config(REFERENCE)
    ref_run = Path(results_dir) / ref_cfg["name"] / "config.yaml"
    if ref_run.exists():                              # la config con la que se entrenó de verdad
        ref_cfg = load_config(ref_run)
    names = [ref_cfg["name"]] + [load_config(p)["name"] for p in paths]
    dirs = [Path(results_dir) / n for n in names if is_done(Path(results_dir) / n)]
    for d in dirs[1:]:
        plot_experiment(d)
    if len(dirs) > 1 and dirs[0].name == ref_cfg["name"]:
        plot_groups(dirs[0], dirs[1:], figure)
    elif len(dirs) > 1:
        plot_comparison(dirs, figure)
    rows = [summary_row(d, ref_cfg) for d in dirs]
    table = Path(table)
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solo-resumen", action="store_true", help="no entrena: solo tabla y figuras")
    args = parser.parse_args()

    if not args.solo_resumen:
        run()
    rows = summarize()
    print(f"\n{'experimento':<18}{'cambio':<32}{'parám.':>11}{'épocas':>8}{'val_acc':>9}{'F1':>8}{'errores':>9}")
    for r in rows:
        print(f"{r['experimento']:<18}{r['cambio']:<32}{r['parametros']:>11,}{r['epocas']:>8}"
              f"{r['val_acc']:>9.4f}{r['f1_macro']:>8.4f}{r['errores']:>9}")
    print("\nTabla en results/tablas/busqueda.csv · curvas en results/figures/busqueda_curvas.png")


if __name__ == "__main__":
    main()
