"""Importancia por permutación (Breiman, 2001; Fisher, Rudin & Dominici, 2019) sobre variables de solicitud."""
import numpy as np
import pandas as pd

from pdmodel.metrics import auc_score


def permutation_importance(predict_proba, df, y, cols, n_repeats=5, seed=0):
    """Caída del AUC al permutar cada variable (más caída = más importante)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    base = auc_score(y, predict_proba(df))
    rows = []
    for c in cols:
        drops = []
        for _ in range(n_repeats):
            tmp = df.copy()
            tmp[c] = tmp[c].to_numpy()[rng.permutation(len(tmp))]
            drops.append(base - auc_score(y, predict_proba(tmp)))
        rows.append({"variable": c, "caida_auc_media": float(np.mean(drops)), "caida_auc_std": float(np.std(drops))})
    out = pd.DataFrame(rows).sort_values("caida_auc_media", ascending=False).reset_index(drop=True)
    out.attrs["auc_base"] = base
    return out
