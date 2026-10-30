"""Métricas de discriminación y calibración. La clase positiva (1) es el cliente MALO."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc_score(y, p):
    """AUC por rangos (equivale al estadístico U de Mann-Whitney normalizado)."""
    y, p = np.asarray(y), np.asarray(p)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini_score(y, p):
    return 2.0 * auc_score(y, p) - 1.0


def ks_score(y, p):
    """KS = max_t | F_malos(t) - F_buenos(t) |, evaluado en los cortes distintos de p."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    ys, ps = y[order], p[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]   # un valor por empate
    return float(np.max(np.abs(cum_bad[last] - cum_good[last])))


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p, eps=1e-12):
    y, p = np.asarray(y), np.clip(np.asarray(p), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y, p, n_bins=10):
    """Curva de calibración con bins por cuantiles de p (mismo número de casos por bin)."""
    y, p = np.asarray(y), np.asarray(p)
    bins = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    t = df.groupby("bin", observed=True).agg(pred_media=("p", "mean"), tasa_obs=("y", "mean"),
                                             n=("y", "size")).reset_index(drop=True)
    return t


def expected_calibration_error(y, p, n_bins=10):
    t = calibration_table(y, p, n_bins)
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["tasa_obs"] - t["pred_media"])))


def all_metrics(y, p):
    return {
        "auc": auc_score(y, p), "gini": gini_score(y, p), "ks": ks_score(y, p),
        "brier": brier_score(y, p), "logloss": log_loss(y, p), "ece": expected_calibration_error(y, p),
    }
