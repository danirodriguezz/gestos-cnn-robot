"""Análisis estadístico y de coste para el informe (tareas D13 y D14).

Uso: uv run python -m src.analysis        (después de entrenar y de pasar evaluate.py)

Se ejecuta cuando el modelo final ya está elegido con validación: nada de lo que calcula sobre
test se usa para tomar decisiones. Los experimentos que no tengan best.pt se saltan.

Escribe:
    results/<exp>/preds_<split>.csv     path, user_id, y_true, y_pred por imagen (caché)
    results/tablas/analisis.json        todo lo calculado
    informe/generado/resultados.tex     macros y tablas LaTeX del informe (no editar a mano)

Qué calcula:
    - IC del 95 % por bootstrap por persona: se remuestrean user_id, no imágenes, porque las
      fotos de una misma persona no son independientes (un IC por imagen saldría demasiado estrecho)
    - prueba de McNemar exacta entre completo y cada variante, con las mismas imágenes de test, y del
      modelo final frente a completo y a ResNet18
    - tabla de la búsqueda de hiperparámetros (results/tablas/busqueda.csv, de src/busqueda.py)
    - media de las dos semillas (42 y 43) de cada variante de la ablación
    - parámetros, MACs por imagen, segundos por época (de los logs) y latencia con lote de 1
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.common import CLASSES, IMG_SIZE, RESULTS, ROOT, SEED, SPLITS_CSV
from src.dataset import get_dataloaders
from src.evaluate import LABELS, compute_metrics, predict
from src.model import count_params, load_checkpoint

ABLATION = ["completo", "sin_dropout", "sin_bn", "sin_aug"]
SEEDS = {42: "", 43: "_s43"}                     # sufijo de la carpeta de cada semilla
REFERENCE = "resnet18"
FINAL = "final"                                  # copia de results/hp_batch_32, elegido en la búsqueda
BASE = "completo"                                # referencia de la ablación y de la búsqueda (semilla 42)
SEARCH_CSV = RESULTS / "tablas" / "busqueda.csv"
SEARCH_ROWS = [  # (experimento, hiperparámetro, valor probado); en el mismo orden que la tabla del informe
    ("completo", r"Referencia (\emph{completo})", "—"),
    ("hp_lr_3e-4", "Tasa de aprendizaje", r"$3\cdot10^{-4}$"), ("hp_lr_3e-3", "", r"$3\cdot10^{-3}$"),
    ("hp_batch_32", "Tamaño de lote", "32"), ("hp_batch_128", "", "128"),
    ("hp_filtros_mitad", "Filtros", "16--128"), ("hp_4_bloques", "Nº de bloques", "4"),
    ("hp_fc_128", "Capa oculta", "128 neuronas"),
    ("hp_elu", "Activación", "ELU"), ("hp_leaky_relu", "", "LeakyReLU")]
COMPONENTS = {"completo": ("sí", "sí", "sí"), "sin_dropout": ("no", "sí", "sí"),
              "sin_bn": ("sí", "no", "sí"), "sin_aug": ("sí", "sí", "no"), "resnet18": ("—", "sí", "sí")}
LABEL = {"completo": "Completo", "sin_dropout": "Sin dropout", "sin_bn": "Sin BN",
         "sin_aug": "Sin augmentation", "resnet18": "ResNet18 preentrenado"}
COMMANDS = {"stop": "Parar", "like": "Avanzar", "dislike": "Retroceder", "one": "Girar a la izquierda",
            "peace": "Girar a la derecha", "fist": "Cerrar pinza", "palm": "Abrir pinza",
            "call": "Volver al inicio", "no_gesture": "Ninguna (quieto)"}
TEX_OUT = ROOT / "informe" / "generado" / "resultados.tex"


# --------------------------------------------------------------------------- estadística

def bootstrap_by_group(y_true, y_pred, groups, n_boot=2000, seed=SEED, alpha=0.05):
    """IC percentil de accuracy y F1 macro remuestreando grupos (personas) con reemplazo."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(len(idx_by_group), size=len(idx_by_group))
        idx = np.concatenate([idx_by_group[i] for i in pick])
        accs[b] = (y_true[idx] == y_pred[idx]).mean()
        f1s[b] = f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"acc_ic": [float(v) for v in np.percentile(accs, q)],
            "f1_ic": [float(v) for v in np.percentile(f1s, q)],
            "n_grupos": len(idx_by_group), "n_boot": n_boot}


def mcnemar(y_true, pred_a, pred_b):
    """McNemar exacto: b = solo acierta A, c = solo acierta B; p bilateral con Binomial(b + c, 1/2)."""
    ok_a, ok_b = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"solo_a": b, "solo_b": c, "p": p}


# --------------------------------------------------------------------------- coste

def count_macs(model, size=IMG_SIZE):
    """Multiplicaciones-acumulaciones por imagen de las capas Conv2d y Linear (el resto es despreciable)."""
    macs = []

    def hook(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs.append(output[0].numel() * k)
        else:
            macs.append(module.in_features * module.out_features)

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, size, size))
    for h in handles:
        h.remove()
    return int(sum(macs))


def macs_first_blocks(model, k=2):
    """Fracción de MACs y de parámetros de los k primeros bloques de GestureCNN (los de mayor resolución)."""
    total_macs, total_params = count_macs(model), count_params(model)
    first = torch.nn.Sequential(*model.features[:k])
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    macs = []
    hooks = [m.register_forward_hook(lambda mod, i, o: macs.append(
                o[0].numel() * mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels))
             for m in first.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        first.eval()(x)
    for h in hooks:
        h.remove()
    return {"bloques": k, "frac_macs": sum(macs) / total_macs, "frac_params": count_params(first) / total_params}


@torch.no_grad()
def latency_ms(model, device, n=200, warmup=20, size=IMG_SIZE):
    """Mediana y p95 (ms) de una pasada con lote de 1, como en la demo (un recorte por fotograma)."""
    model = model.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    times = []
    for i in range(warmup + n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()          # la GPU es asíncrona: sin esto se mide solo el lanzamiento
        if i >= warmup:
            times.append(1000 * (time.perf_counter() - t0))
    return {"mediana": float(np.median(times)), "p95": float(np.percentile(times, 95))}


def epoch_seconds(log_dir=RESULTS):
    """Segundos por época de cada experimento, leídos de los logs de run_experiments (results/*.log)."""
    secs, name = {}, None
    for log in sorted(Path(log_dir).glob("*.log")):
        for line in log.read_text().splitlines():
            if m := re.match(r"\[(\S+)\] ", line):
                name = m.group(1)
                secs[name] = []
            elif (m := re.search(r"· ([\d.]+)s\b", line)) and name:
                secs[name].append(float(m.group(1)))
    return {k: float(np.median(v)) for k, v in secs.items() if v}


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


# --------------------------------------------------------------------------- predicciones

def predictions(exp, split, loaders, rows, device):
    """y_true, y_pred del experimento en split; se guardan en preds_<split>.csv para no repetirlas."""
    cache = RESULTS / exp / f"preds_{split}.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype={"user_id": str})
    else:
        model, _ = load_checkpoint(RESULTS / exp / "best.pt")
        y_true, y_pred = predict(model, loaders[split], device)
        df = rows[split][["path", "user_id"]].assign(y_true=y_true, y_pred=y_pred)
        # el loader no baraja val/test: el orden es el de splits.csv, y aquí se comprueba
        assert (df["y_true"].to_numpy() == rows[split]["label"].map(CLASSES.index).to_numpy()).all()
        df.to_csv(cache, index=False)
    return df


def history_info(exp):
    h = pd.read_csv(RESULTS / exp / "history.csv")
    above = h[h["val_acc"] >= 0.95]["epoch"]
    return {"epocas": int(len(h)), "mejor_epoca": int(h.loc[h["val_loss"].idxmin(), "epoch"]),
            "lr_reducciones": [int(e) for e in h[h["lr"].diff() < 0]["epoch"]],
            "val_acc_epoca1": float(h["val_acc"].iloc[0]),
            "primera_epoca_95": int(above.iloc[0]) if len(above) else None,
            "brecha_acc_final": float(h["train_acc"].iloc[-1] - h["val_acc"].iloc[-1])}


# --------------------------------------------------------------------------- LaTeX

def num(x, d=2):
    """Coma decimal; {,} para que LaTeX no meta espacio tras la coma en modo matemático."""
    return f"{x:.{d}f}".replace(".", "{,}")


def pct(x, d=2):
    return num(100 * x, d) + r"\,\%"


def pval(p):
    """Valor p para las celdas de una tabla."""
    return r"$<\!0{,}001$" if p < 0.001 else num(p, 3)


def pexpr(p):
    """Valor p con su relación, para el texto: $p = 0{,}541$ o $p < 0{,}001$."""
    return r"$p < 0{,}001$" if p < 0.001 else f"$p = {num(p, 3)}$"


def macro_name(*parts):
    """('acc', 'test', 'sin_bn') -> \\AccTestSinBn (los nombres de macro solo admiten letras)."""
    words = re.split(r"[_\W\d]+", "_".join(parts).replace("resnet18", "resnet"))
    return "\\" + "".join(w.capitalize() for w in words if w)


def build_tex(res):
    out = ["% Generado por src/analysis.py a partir de results/: no editar a mano.", ""]
    macro = lambda name, value: out.append(f"\\newcommand{{{name}}}{{{value}}}")

    # --- datos
    d = res["datos"]
    for split in ("train", "val", "test"):
        macro(macro_name("n", split), f"{d['imagenes'][split]:,}".replace(",", "."))
        macro(macro_name("personas", split), f"{d['personas'][split]:,}".replace(",", "."))
    macro(r"\NTotal", f"{sum(d['imagenes'].values()):,}".replace(",", "."))
    macro(r"\PersonasTotal", f"{sum(d['personas'].values()):,}".replace(",", "."))
    macro(r"\PersonasCompartidas", str(d["personas_compartidas"]))
    miles = lambda n: f"{n:,}".replace(",", ".")
    c = res["recortes"]
    macro(r"\NImagenesOrig", miles(c["imagenes"]))
    macro(r"\NManos", miles(c["manos"]))
    macro(r"\NPersonasOrig", miles(c["personas"]))
    macro(r"\NoGestureOrig", miles(c["no_gesture"]))
    macro(r"\NoGestureKept", miles(d["por_clase"]["no_gesture"]["train"] + d["por_clase"]["no_gesture"]["val"]
                                    + d["por_clase"]["no_gesture"]["test"]))
    macro(r"\LadoCajaMediana", num(c["lado_caja_mediana"], 0) + r"\,px")
    macro(r"\LadoCajaPdiez", num(c["lado_caja_p10"], 0) + r"\,px")

    # --- métricas por experimento
    for exp, m in res["experimentos"].items():
        for split in ("val", "test"):
            if split in m:
                macro(macro_name("acc", split, exp), pct(m[split]["accuracy"]))
                macro(macro_name("fone", split, exp), pct(m[split]["f1_macro"]))
                macro(macro_name("errores", split, exp), str(m[split]["errores"]))
        if "epocas" in m:
            macro(macro_name("epocas", exp), str(m["epocas"]))
            macro(macro_name("mejorepoca", exp), str(m["mejor_epoca"]))
            red = [str(e) for e in m["lr_reducciones"]]
            macro(macro_name("reduccioneslr", exp), " y ".join([", ".join(red[:-1]), red[-1]]) if len(red) > 1
                  else (red[0] if red else "ninguna"))
            macro(macro_name("nreduccioneslr", exp), str(len(red)))
            macro(macro_name("accepocauno", exp), pct(m["val_acc_epoca1"], 1))
            macro(macro_name("epocanoventaycinco", exp), str(m["primera_epoca_95"]) if m["primera_epoca_95"] else "—")
            macro(macro_name("brecha", exp), num(100 * m["brecha_acc_final"], 1))

    # --- modelo final: IC y confusiones
    f = res["final"]
    macro(r"\AccTestFinalIC", f"[{pct(f['bootstrap']['acc_ic'][0])};\\ {pct(f['bootstrap']['acc_ic'][1])}]")
    macro(r"\FoneTestFinalIC", f"[{pct(f['bootstrap']['f1_ic'][0])};\\ {pct(f['bootstrap']['f1_ic'][1])}]")
    macro(r"\NBoot", f"{f['bootstrap']['n_boot']:,}".replace(",", "."))
    if BASE in res and "bootstrap" in res[BASE]:
        r = res[BASE]["bootstrap"]
        macro(r"\AccTestCompletoIC", f"[{pct(r['acc_ic'][0])};\\ {pct(r['acc_ic'][1])}]")
    if "resnet18" in res and "bootstrap" in res["resnet18"]:
        r = res["resnet18"]["bootstrap"]
        macro(r"\AccTestResnetIC", f"[{pct(r['acc_ic'][0])};\\ {pct(r['acc_ic'][1])}]")
        macro(r"\FoneTestResnetIC", f"[{pct(r['f1_ic'][0])};\\ {pct(r['f1_ic'][1])}]")
    conf = ", ".join(f"\\textit{{{a}}}$\\to$\\textit{{{b}}} ({n})" for a, b, n in f["confusiones_test"][:4])
    macro(r"\ConfusionesTestFinal", conf.replace("_", r"\_"))
    worst = min(f["por_clase_test"], key=lambda c: f["por_clase_test"][c]["f1"])
    macro(r"\PeorClaseTest", r"\textit{" + worst.replace("_", r"\_") + "}")
    macro(r"\PeorFoneTest", pct(f["por_clase_test"][worst]["f1"], 1))
    sg = f["seguridad_test"]
    macro(r"\FalsasActivaciones", str(sg["falsas_activaciones"]))
    macro(r"\NNoGestureTest", str(sg["n_no_gesture"]))
    macro(r"\TasaFalsasActivaciones", pct(sg["falsas_activaciones"] / sg["n_no_gesture"], 1))
    macro(r"\Omisiones", str(sg["omisiones"]))
    macro(r"\ErroresEnPares", str(sg["errores_en_pares"]))

    # --- semillas y McNemar
    for v, s in res["semillas"].items():
        for k in ("acc_val", "acc_test", "f1_val", "f1_test"):
            macro(macro_name(k.replace("f1", "fone"), "media", v), pct(s[k]["media"]))
    for v, t in res["mcnemar_test"].items():
        macro(macro_name("pmcnemar", v), pexpr(t["p"]))
        macro(macro_name("solocompleto", v), str(t["solo_a"]))
        macro(macro_name("solovariante", v), str(t["solo_b"]))

    # --- modelo final frente a completo y a ResNet18 (solo_a = solo acierta el final)
    for v, t in res.get("mcnemar_final", {}).items():
        macro(macro_name("pmcnemar", "final", v), pexpr(t["p"]))
        macro(macro_name("solofinal", v), str(t["solo_a"]))
        macro(macro_name("solovariante", v), str(t["solo_b"]))

    # --- búsqueda de hiperparámetros
    if "busqueda" in res:
        b = res["busqueda"]
        macro(r"\NConfigsBusqueda", str(len(b) - 1))
        macro(r"\ParamsFiltrosMitad", num(b["hp_filtros_mitad"]["parametros"] / 1e3, 0) + r"\,k")
        macro(r"\ParamsCuatroBloques", num(b["hp_4_bloques"]["parametros"] / 1e6, 2) + r"\,M")
        macro(r"\EpocasLrAlta", str(b["hp_lr_3e-3"]["epocas"]))
        macro(r"\AccValLrAlta", pct(b["hp_lr_3e-3"]["val_acc"]))
        macro(r"\AccValFiltrosMitad", pct(b["hp_filtros_mitad"]["val_acc"]))
        accs = [r["val_acc"] for k, r in b.items() if k != "hp_batch_32"]
        macro(r"\AccValBusquedaMin", pct(min(accs)))
        macro(r"\AccValBusquedaMax", pct(max(accs)))

    # --- entorno
    e = res["entorno"]
    macro(r"\VersionTorch", e["torch"].split("+")[0])
    macro(r"\GPU", (e["gpu"] or "—").replace("NVIDIA GeForce ", ""))
    macro(r"\CPU", (e["cpu"] or "—").replace("(R)", "").replace("(TM)", "").replace(" CPU", ""))
    macro(r"\CPUHilos", str(e["cpu_hilos"]))

    # --- coste
    for arch, c in res["coste"].items():
        macro(macro_name("params", arch), num(c["parametros"] / 1e6, 2) + r"\,M")
        macro(macro_name("macs", arch), num(c["macs"] / 1e6, 0) + r"\,M")
        for dev in ("cpu", "cuda"):
            if dev in c["latencia"]:
                macro(macro_name("lat", dev, arch), num(c["latencia"][dev]["mediana"], 2) + r"\,ms")
        if c.get("s_epoca"):
            macro(macro_name("sepoca", arch), num(c["s_epoca"], 1) + r"\,s")
        if c.get("s_epoca_lote_final"):
            macro(r"\SepocaLoteFinal", num(c["s_epoca_lote_final"], 1) + r"\,s")
    if "primeros_bloques" in res["coste"].get("final", {}):
        pb = res["coste"]["final"]["primeros_bloques"]
        macro(r"\FracMacsPrimerosBloques", pct(pb["frac_macs"], 0))
        macro(r"\FracParamsPrimerosBloques", pct(pb["frac_params"], 0))
    if {"final", "resnet18"} <= set(res["coste"]):
        g, r = res["coste"]["final"], res["coste"]["resnet18"]
        macro(r"\RatioParams", num(r["parametros"] / g["parametros"], 1))
        macro(r"\RatioMacs", num(g["macs"] / r["macs"], 1))           # GestureCNN / ResNet18
        if g.get("s_epoca") and r.get("s_epoca"):
            macro(r"\RatioSEpoca", num(g["s_epoca"] / r["s_epoca"], 1))
    out.append("")

    # --- tablas
    out += table_datos(res["datos"]) + table_ablacion(res) + table_semillas(res)
    out += table_por_clase(f["por_clase_test"]) + table_coste(res["coste"])
    if "busqueda" in res:
        out += table_busqueda(res["busqueda"])
    return "\n".join(out) + "\n"


def table_datos(d):
    lines = [r"\newcommand{\TablaDatos}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Clase & Orden & Train & Val & Test & Total \\", r"\midrule"]
    for c in CLASSES:
        n = d["por_clase"][c]
        lines.append(f"\\textit{{{c.replace('_', chr(92) + '_')}}} & {COMMANDS[c]} & "
                     f"{n['train']} & {n['val']} & {n['test']} & {sum(n.values())} \\\\")
    i, p = d["imagenes"], d["personas"]
    lines += [r"\midrule", f"Total imágenes & & {i['train']} & {i['val']} & {i['test']} & {sum(i.values())} \\\\",
              f"Personas & & {p['train']} & {p['val']} & {p['test']} & {sum(p.values())} \\\\",
              r"\bottomrule", r"\end{tabular}}", ""]
    return lines


def table_ablacion(res):
    """Media de las dos semillas; la ResNet18 tiene una sola ejecución."""
    lines = [r"\newcommand{\TablaAblacion}{%", r"\begin{tabular}{lccc cc cc c}", r"\toprule",
             r"& & & & \multicolumn{2}{c}{Validación} & \multicolumn{2}{c}{Test} & \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Variante & Dropout & BN & Augm. & Acc. & F1 macro & Acc. & F1 macro & Épocas \\", r"\midrule"]
    for v in ABLATION:
        s = res["semillas"][v]
        ep = " / ".join(str(e) for e in s["epocas"])
        lines.append(f"{LABEL[v]} & {' & '.join(COMPONENTS[v])} & {pct(s['acc_val']['media'])} & "
                     f"{pct(s['f1_val']['media'])} & {pct(s['acc_test']['media'])} & "
                     f"{pct(s['f1_test']['media'])} & {ep} \\\\")
    if REFERENCE in res["experimentos"]:
        m = res["experimentos"][REFERENCE]
        lines += [r"\midrule", f"{LABEL[REFERENCE]}$^\\dagger$ & {' & '.join(COMPONENTS[REFERENCE])} & "
                  f"{pct(m['val']['accuracy'])} & {pct(m['val']['f1_macro'])} & {pct(m['test']['accuracy'])} & "
                  f"{pct(m['test']['f1_macro'])} & {m['epocas']} \\\\"]
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return lines


def table_semillas(res):
    """Detalle por semilla: errores en test y McNemar frente a completo con la misma semilla."""
    lines = [r"\newcommand{\TablaSemillas}{%", r"\begin{tabular}{l rr rr rrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Acc. val} & \multicolumn{2}{c}{Errores test} & "
             r"\multicolumn{3}{c}{McNemar vs.\ completo (semilla 42)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Variante & s42 & s43 & s42 & s43 & solo compl. & solo var. & $p$ \\", r"\midrule"]
    for v in ABLATION:
        s, t = res["semillas"][v], res["mcnemar_test"].get(v)
        mc = f"{t['solo_a']} & {t['solo_b']} & {pval(t['p'])}" if t else "— & — & —"
        acc = [pct(a) for a in s["acc_val"]["valores"]] + ["—"]
        err = [str(e) for e in s["errores_test"]] + ["—"]
        lines.append(f"{LABEL[v]} & {acc[0]} & {acc[1]} & {err[0]} & {err[1]} & {mc} \\\\")
    if REFERENCE in res["mcnemar_test"]:
        t, m = res["mcnemar_test"][REFERENCE], res["experimentos"][REFERENCE]
        lines += [r"\midrule", f"{LABEL[REFERENCE]} & {pct(m['val']['accuracy'])} & — & {m['test']['errores']} & — & "
                  f"{t['solo_a']} & {t['solo_b']} & {pval(t['p'])} \\\\"]
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return lines


def table_por_clase(pc):
    lines = [r"\newcommand{\TablaPorClase}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Clase & Orden & Precisión & Recall & F1 & $n$ \\", r"\midrule"]
    for c in CLASSES:
        m = pc[c]
        lines.append(f"\\textit{{{c.replace('_', chr(92) + '_')}}} & {COMMANDS[c]} & {pct(m['precision'], 1)} & "
                     f"{pct(m['recall'], 1)} & {pct(m['f1'], 1)} & {m['n']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return lines


def table_coste(coste):
    lines = [r"\newcommand{\TablaCoste}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"& & & & \multicolumn{2}{c}{CPU (ms)} & \multicolumn{2}{c}{GPU (ms)} \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Modelo & Parámetros & MACs & s/época & mediana & p95 & mediana & p95 \\", r"\midrule"]
    for arch, c in coste.items():
        lat = c["latencia"]
        cells = [num(lat[d][k], 2) if d in lat else "—" for d in ("cpu", "cuda") for k in ("mediana", "p95")]
        sep = num(c["s_epoca"], 1) if c.get("s_epoca") else "—"
        lines.append(f"{c['nombre']} & {num(c['parametros'] / 1e6, 2)}\\,M & {num(c['macs'] / 1e6, 0)}\\,M & "
                     f"{sep} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return lines


def table_busqueda(b):
    """Una fila por config; la elegida (mayor exactitud de val) en negrita. Solo validación."""
    best = max(b, key=lambda k: b[k]["val_acc"])
    lines = [r"\newcommand{\TablaBusqueda}{%", r"\begin{tabular}{llrcrrrr}", r"\toprule",
             r"Hiperparámetro & Valor & Parám. & Épocas (mejor) & Pérdida & Acc. & F1 macro & Errores \\",
             r"\midrule"]
    for i, (exp, name, value) in enumerate(SEARCH_ROWS):
        if exp not in b:
            continue
        r = b[exp]
        if i in (1, 3, 5, 8):
            lines.append(r"\midrule")
        cells = [num(r["parametros"] / 1e6, 2) + r"\,M", f"{r['epocas']} ({r['mejor_epoca']})",
                 num(r["val_loss"], 4), pct(r["val_acc"]), pct(r["f1_macro"]), str(r["errores"])]
        if exp == best:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(f"{name} & {value} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return lines


# --------------------------------------------------------------------------- principal

def crops_summary(crops_csv=ROOT / "data" / "crops.csv"):
    """Antes de equilibrar: imágenes, manos y personas extraídas, y tamaño de las cajas (lado mayor, px)."""
    c = pd.read_csv(crops_csv, dtype={"user_id": str})
    return {"imagenes": int(c["image_id"].nunique()), "manos": int(len(c)), "personas": int(c["user_id"].nunique()),
            "lado_caja_mediana": float(c["box_px"].median()), "lado_caja_p10": float(c["box_px"].quantile(0.1)),
            "no_gesture": int((c["label"] == "no_gesture").sum())}


def data_summary(df):
    d = {"imagenes": df["split"].value_counts().to_dict(),
         "personas": df.groupby("split")["user_id"].nunique().to_dict(),
         "por_clase": {c: df[df["label"] == c]["split"].value_counts().reindex(["train", "val", "test"],
                                                                              fill_value=0).to_dict()
                       for c in CLASSES}}
    users = {s: set(df[df["split"] == s]["user_id"]) for s in ("train", "val", "test")}
    d["personas_compartidas"] = len((users["train"] & users["val"]) | (users["train"] & users["test"])
                                    | (users["val"] & users["test"]))
    d = {k: ({kk: int(vv) for kk, vv in v.items()} if k in ("imagenes", "personas") else v) for k, v in d.items()}
    return d


def per_class(y_true, y_pred):
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(n[i])}
            for i, c in enumerate(CLASSES)}


PAIRS = [("stop", "palm"), ("like", "call")]       # pares visualmente próximos (sección de datos)


def safety_errors(y_true, y_pred):
    """Errores que importan en el robot: no_gesture -> orden (mueve el robot sin querer) y al revés."""
    ng = CLASSES.index("no_gesture")
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wrong = y_true != y_pred
    in_pairs = np.zeros_like(wrong)
    for a, b in PAIRS:
        ia, ib = CLASSES.index(a), CLASSES.index(b)
        in_pairs |= ((y_true == ia) & (y_pred == ib)) | ((y_true == ib) & (y_pred == ia))
    return {"falsas_activaciones": int(((y_true == ng) & (y_pred != ng)).sum()), "n_no_gesture": int((y_true == ng).sum()),
            "omisiones": int(((y_true != ng) & (y_pred == ng)).sum()), "errores": int(wrong.sum()),
            "errores_en_pares": int(in_pairs.sum())}


def top_confusions_counts(y_true, y_pred, k=6):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm, axis=None)[::-1][:k]
    return [(CLASSES[i], CLASSES[j], int(cm[i, j])) for i, j in zip(*np.unravel_index(order, cm.shape)) if cm[i, j]]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(SPLITS_CSV, dtype={"user_id": str})
    rows = {s: df[df["split"] == s].reset_index(drop=True) for s in ("val", "test")}
    loaders = get_dataloaders(256, augment=False, num_workers=2)

    exps = [FINAL] + [v + suf for v in ABLATION for suf in SEEDS.values()] + [REFERENCE]
    exps = [e for e in exps if (RESULTS / e / "best.pt").exists()]
    res = {"datos": data_summary(df), "recortes": crops_summary(), "experimentos": {}, "semillas": {}, "mcnemar_test": {}}
    preds = {}
    for exp in exps:
        info = history_info(exp)
        for split in ("val", "test"):
            p = predictions(exp, split, loaders, rows, device)
            preds[exp, split] = p
            m = compute_metrics(p["y_true"].to_numpy(), p["y_pred"].to_numpy())
            info[split] = {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                           "errores": int((p["y_true"] != p["y_pred"]).sum())}
        res["experimentos"][exp] = info
        print(f"{exp:16} val {info['val']['accuracy']:.4f}  test {info['test']['accuracy']:.4f}  épocas {info['epocas']}")

    # modelo final: IC por persona, F1 por clase y confusiones (test)
    pt = preds[FINAL, "test"]
    res["final"] = {"bootstrap": bootstrap_by_group(pt["y_true"], pt["y_pred"], pt["user_id"]),
                    "por_clase_test": per_class(pt["y_true"], pt["y_pred"]),
                    "confusiones_test": top_confusions_counts(pt["y_true"], pt["y_pred"]),
                    "seguridad_test": safety_errors(pt["y_true"], pt["y_pred"])}
    for exp in (BASE, REFERENCE):
        if exp in exps:
            pr = preds[exp, "test"]
            res[exp] = {"bootstrap": bootstrap_by_group(pr["y_true"], pr["y_pred"], pr["user_id"])}

    # McNemar en test: completo (semilla 42) frente a cada variante de la ablación con la misma semilla
    pb = preds[BASE, "test"]
    for v in ABLATION[1:]:
        if v in exps:
            res["mcnemar_test"][v] = mcnemar(pb["y_true"].to_numpy(), pb["y_pred"], preds[v, "test"]["y_pred"])
    # y el modelo final frente a completo (el final anterior) y a la ResNet18
    res["mcnemar_final"] = {v: mcnemar(pt["y_true"].to_numpy(), pt["y_pred"], preds[v, "test"]["y_pred"])
                            for v in (BASE, REFERENCE) if v in exps}

    if SEARCH_CSV.exists():
        b = pd.read_csv(SEARCH_CSV)
        res["busqueda"] = {r["experimento"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                           for r in b.to_dict("records")}

    # media de las semillas disponibles de cada variante (normalmente 42 y 43)
    for v in ABLATION:
        runs = [v + suf for suf in SEEDS.values() if v + suf in exps]
        if not runs:
            continue
        e = [res["experimentos"][r] for r in runs]
        res["semillas"][v] = {f"{k}_{s}": {"valores": [x[s][kk] for x in e], "media": float(np.mean([x[s][kk] for x in e]))}
                              for k, kk in (("acc", "accuracy"), ("f1", "f1_macro")) for s in ("val", "test")}
        res["semillas"][v]["epocas"] = [x["epocas"] for x in e]
        res["semillas"][v]["errores_test"] = [x["test"]["errores"] for x in e]

    # coste: GestureCNN (final) y ResNet18. El s/época de GestureCNN es el de completo (misma red, lotes
    # de 64 como la ResNet18); el del lote del modelo final se guarda aparte
    secs = epoch_seconds()
    final_name = yaml.safe_load((RESULTS / FINAL / "config.yaml").read_text())["name"]
    res["coste"] = {}
    for exp, nombre, log_name in ((FINAL, "GestureCNN", "completo"), (REFERENCE, "ResNet18", "resnet18")):
        if exp not in exps:
            continue
        model, _ = load_checkpoint(RESULTS / exp / "best.pt")
        lat = {"cpu": latency_ms(model, torch.device("cpu"))}
        if torch.cuda.is_available():
            lat["cuda"] = latency_ms(model, torch.device("cuda"))
        res["coste"][exp] = {"nombre": nombre, "parametros": count_params(model), "macs": count_macs(model.cpu()),
                             "s_epoca": secs.get(log_name), "latencia": lat}
        if exp == FINAL:
            res["coste"][exp]["primeros_bloques"] = macs_first_blocks(model.cpu())
            res["coste"][exp]["s_epoca_lote_final"] = secs.get(final_name)
    res["entorno"] = {"torch": torch.__version__, "cpu_hilos": torch.get_num_threads(), "cpu": cpu_name(),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    out = RESULTS / "tablas" / "analisis.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text(build_tex(res))
    print(f"Guardados {out} y {TEX_OUT}")


if __name__ == "__main__":
    main()
