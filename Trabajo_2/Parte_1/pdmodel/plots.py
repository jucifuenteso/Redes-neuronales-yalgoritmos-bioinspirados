"""Figuras (todas regenerables desde el código)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdmodel.metrics import calibration_table


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def roc_ks_calibration(y, probs, path_prefix):
    """probs: dict nombre -> probabilidades. Guarda ROC, KS y calibración."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probs.items():
        order = np.argsort(-p)
        ys = y[order]
        tpr = np.cumsum(ys) / ys.sum()
        fpr = np.cumsum(1 - ys) / (1 - ys).sum()
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC (prueba)")
    ax.legend()
    _save(fig, f"{path_prefix}_roc.png")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probs.items():
        order = np.argsort(p)
        ys = y[order]
        cb, cg = np.cumsum(ys) / ys.sum(), np.cumsum(1 - ys) / (1 - ys).sum()
        ax.plot(np.arange(1, len(p) + 1) / len(p), np.abs(cb - cg), label=f"{name} (KS={np.max(np.abs(cb - cg)):.3f})")
    ax.set(xlabel="Fracción de la población (de menor a mayor PD)", ylabel="|F_malos - F_buenos|", title="Estadístico KS (prueba)")
    ax.legend()
    _save(fig, f"{path_prefix}_ks.png")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probs.items():
        t = calibration_table(y, p, 10)
        ax.plot(t["pred_media"], t["tasa_obs"], "o-", label=name)
    m = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, m], [0, m], "k--", lw=0.8, label="Calibración perfecta")
    ax.set(xlabel="PD predicha (media por bin)", ylabel="Tasa observada de malos", title="Curva de calibración (prueba)")
    ax.legend()
    _save(fig, f"{path_prefix}_calibration.png")


def loss_curves(history, path):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(h["epoch"], h["train_loss"], label="Entrenamiento (con dropout y L2)")
    ax.plot(h["epoch"], h["val_logloss"], label="Validación (log-loss)")
    ax.set(xlabel="Época", ylabel="Pérdida", title="Curvas de aprendizaje")
    ax.legend()
    _save(fig, path)


def score_distribution(scores, y, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(np.nanpercentile(scores, 0.5), np.nanpercentile(scores, 99.5), 40)
    ax.hist(scores[y == 0], bins=bins, alpha=0.6, density=True, label="Buenos (0)")
    ax.hist(scores[y == 1], bins=bins, alpha=0.6, density=True, label="Malos (1)")
    ax.set(xlabel="Puntaje", ylabel="Densidad", title="Distribución del puntaje (prueba)")
    ax.legend()
    _save(fig, path)


def importance_bar(imp_by_model, path, top=15):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    first = next(iter(imp_by_model.values()))
    order = first.head(top)["variable"].tolist()[::-1]
    h = 0.8 / len(imp_by_model)
    for i, (name, imp) in enumerate(imp_by_model.items()):
        s = imp.set_index("variable").reindex(order)
        ax.barh(np.arange(len(order)) + i * h, s["caida_auc_media"], h, xerr=s["caida_auc_std"], label=name)
    ax.set_yticks(np.arange(len(order)) + h * (len(imp_by_model) - 1) / 2)
    ax.set_yticklabels(order)
    ax.set(xlabel="Caída del AUC al permutar (prueba)", title="Importancia por permutación")
    ax.legend()
    _save(fig, path)


def points_curves(table, variables, path, ncols=3):
    n = len(variables)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, v in zip(axes.ravel(), variables):
        t = table[table.variable == v]
        if (t.tipo == "num").all():
            t = t[t.value != "__missing__"].copy()
            t["x"] = t.value.astype(float)
            t = t.sort_values("x")
            ax.plot(t["x"], t["points"], "o-")
        else:
            t = t.sort_values("points")
            ax.barh(t["value"].astype(str), t["points"])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(v, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Puntos de la scorecard por variable (0 = perfil promedio)")
    _save(fig, path)


def bad_rate_panel(df, specs, path, ncols=3):
    """specs: lista de (titulo, serie_agrupadora). Tasa de malos por grupo."""
    n = len(specs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), squeeze=False)
    for ax, (title, g) in zip(axes.ravel(), specs):
        t = df.groupby(g, observed=True)["y"].agg(["mean", "size"])
        ax.bar(range(len(t)), t["mean"])
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([str(i) for i in t.index], rotation=60, ha="right", fontsize=7)
        ax.axhline(df["y"].mean(), color="r", ls="--", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Tasa de malos")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Tasa de incumplimiento por grupo (línea roja = promedio)")
    _save(fig, path)
