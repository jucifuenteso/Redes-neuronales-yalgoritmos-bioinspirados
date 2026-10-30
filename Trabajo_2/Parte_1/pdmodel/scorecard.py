"""Escala de puntaje (base, odds, PDO) y scorecard aditiva con puntos por variable continua.

Derivación (ver docs/guia_conceptual.md):
    odds_buenos = (1 - PD) / PD = exp(-logit(PD))
    Puntaje = Offset + Factor * ln(odds_buenos),   Factor = PDO / ln 2,   Offset = Base - Factor * ln(odds_base)
    => Puntaje = Offset - Factor * logit(PD)
Puntos por variable (efecto parcial en el logit, estilo dependencia parcial de Friedman, 2001):
    puntos_j(v) = -Factor * ( E_x[ logit f(x_j = v, x_{-j}) ] - E_x[ logit f(x) ] )
    Puntaje_aditivo(x) = Puntos_base + sum_j puntos_j(x_j),  Puntos_base = Offset - Factor * E_x[logit f(x)]
Para regresión logística el modelo ya es aditivo en el logit y la aproximación es (casi) exacta;
para la red neuronal es una aproximación aditiva cuya fidelidad se mide explícitamente.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdmodel.mlp import sigmoid


@dataclass
class ScoreScale:
    base_score: float = 600.0
    base_odds: float = 50.0     # odds buenos:malos en el puntaje base
    pdo: float = 20.0           # puntos para duplicar las odds

    @property
    def factor(self):
        return self.pdo / np.log(2.0)

    @property
    def offset(self):
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_logit(self, logit_pd):
        return self.offset - self.factor * np.asarray(logit_pd)

    def score(self, pd_):
        p = np.clip(np.asarray(pd_, dtype=float), 1e-9, 1 - 1e-9)
        return self.score_from_logit(np.log(p / (1 - p)))

    def pd_from_score(self, score):
        return sigmoid((self.offset - np.asarray(score, dtype=float)) / self.factor)


class Scorecard:
    def __init__(self, scale, base_points, table, num_cols, cat_cols):
        self.scale, self.base_points, self.table = scale, base_points, table
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self._num = {}
        self._cat = {}
        for c in self.num_cols:
            t = table[table.variable == c]
            val = t[t.value != "__missing__"]
            order = np.argsort(val.value.astype(float).to_numpy())
            self._num[c] = (val.value.astype(float).to_numpy()[order], val.points.to_numpy()[order],
                            t.loc[t.value == "__missing__", "points"].tolist())
        for c in self.cat_cols:
            t = table[table.variable == c]
            self._cat[c] = dict(zip(t.value.astype(str), t.points))

    def contributions(self, df):
        """DataFrame (n x variables) con los puntos de cada variable."""
        out = {}
        for c in self.num_cols:
            grid, pts, miss = self._num[c]
            x = df[c].astype(float).to_numpy()
            v = np.interp(x, grid, pts)
            v = np.where(np.isnan(x), miss[0] if miss else 0.0, v)
            out[c] = v
        for c in self.cat_cols:
            m = self._cat[c]
            out[c] = df[c].fillna("__missing__").astype(str).map(m).fillna(0.0).to_numpy()
        return pd.DataFrame(out, index=df.index)

    def additive_score(self, df):
        return self.base_points + self.contributions(df).sum(axis=1).to_numpy()


def build_scorecard(predict_logit, app_df, num_cols, cat_cols, scale, n_background=5000, n_grid=12, seed=0):
    bg = app_df.sample(min(n_background, len(app_df)), random_state=seed).reset_index(drop=True)
    base_logit = float(np.mean(predict_logit(bg)))
    rows = []
    for c in num_cols:
        qs = np.linspace(0.02, 0.98, n_grid)
        grid = np.unique(np.nanquantile(bg[c].astype(float), qs))
        for v in grid:
            tmp = bg.copy()
            tmp[c] = v
            eff = float(np.mean(predict_logit(tmp))) - base_logit
            rows.append((c, repr(float(v)), "num", eff, -scale.factor * eff))
        if bg[c].isna().any():
            tmp = bg.copy()
            tmp[c] = np.nan
            eff = float(np.mean(predict_logit(tmp))) - base_logit
            rows.append((c, "__missing__", "num", eff, -scale.factor * eff))
    for c in cat_cols:
        for v in bg[c].fillna("__missing__").astype(str).value_counts().index:
            tmp = bg.copy()
            tmp[c] = np.nan if v == "__missing__" else v
            eff = float(np.mean(predict_logit(tmp))) - base_logit
            rows.append((c, v, "cat", eff, -scale.factor * eff))
    table = pd.DataFrame(rows, columns=["variable", "value", "tipo", "efecto_logit", "points"])
    base_points = float(scale.score_from_logit(base_logit))
    return Scorecard(scale, base_points, table, num_cols, cat_cols)
