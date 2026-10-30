"""Preprocesamiento ajustado SOLO con entrenamiento (sin fuga de información hacia val/test).

Pasos (en este orden):
 1. Derivadas: log1p(annual_inc), log1p(revol_bal), loan_to_income = loan_amnt / annual_inc.
 2. Winsorización a los cuantiles [q_lo, q_hi] de train.
 3. Indicadores de dato faltante (solo para columnas con faltantes en train) e imputación por mediana.
 4. Estandarización (media 0, desviación 1) con estadísticas de train.
 5. One-hot de categóricas (categorías con frecuencia >= min_freq; resto -> '__other__').
"""
import numpy as np
import pandas as pd

LOG_COLS = {"annual_inc", "revol_bal"}


class Preprocessor:
    def __init__(self, num_cols, cat_cols, clip_q=(0.01, 0.99), min_freq=0.005):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.clip_q, self.min_freq = tuple(clip_q), min_freq

    def _derive(self, df):
        cols, names = [], []
        for c in self.num_cols:
            v = df[c].astype(float).to_numpy()
            if c in LOG_COLS:
                v = np.log1p(np.clip(v, 0, None))
                names.append("log_" + c)
            else:
                names.append(c)
            cols.append(v)
        if "loan_amnt" in self.num_cols and "annual_inc" in self.num_cols:
            inc = np.clip(df["annual_inc"].astype(float).to_numpy(), 1.0, None)
            cols.append(df["loan_amnt"].astype(float).to_numpy() / inc)
            names.append("loan_to_income")
        return np.column_stack(cols), names

    def fit(self, df):
        arr, self.base_names_ = self._derive(df)
        q = np.nanquantile(arr, list(self.clip_q), axis=0)
        self.lo_, self.hi_ = q[0], q[1]
        arr = np.clip(arr, self.lo_, self.hi_)
        miss = np.isnan(arr)
        self.miss_idx_ = np.where(miss.any(axis=0))[0]
        self.median_ = np.nanmedian(arr, axis=0)
        arr = np.where(miss, self.median_, arr)
        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        self.categories_ = {}
        for c in self.cat_cols:
            s = df[c].fillna("__missing__").astype(str)
            freq = s.value_counts(normalize=True)
            self.categories_[c] = sorted(freq[freq >= self.min_freq].index.tolist()) + ["__other__"]
        self.feature_names_ = (
            list(self.base_names_)
            + [f"{self.base_names_[i]}__missing" for i in self.miss_idx_]
            + [f"{c}={k}" for c in self.cat_cols for k in self.categories_[c]]
        )
        return self

    def transform(self, df):
        arr, _ = self._derive(df)
        arr = np.clip(arr, self.lo_, self.hi_)
        miss = np.isnan(arr)
        flags = miss[:, self.miss_idx_].astype(np.float32)
        arr = (np.where(miss, self.median_, arr) - self.mean_) / self.std_
        blocks = [arr.astype(np.float32), flags]
        for c in self.cat_cols:
            s = df[c].fillna("__missing__").astype(str).to_numpy()
            cats = self.categories_[c]
            known = set(cats[:-1])
            s = np.array([v if v in known else "__other__" for v in s])
            blocks.append((s[:, None] == np.array(cats)[None, :]).astype(np.float32))
        return np.hstack(blocks)

    @property
    def n_features(self):
        return len(self.feature_names_)
