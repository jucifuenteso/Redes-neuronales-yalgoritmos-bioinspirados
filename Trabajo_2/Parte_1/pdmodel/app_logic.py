"""Lógica de la aplicación (separada de la interfaz para poder probarla sin Streamlit)."""
import json

import joblib
import numpy as np
import pandas as pd

from pdmodel.scorecard import ScoreScale


class ScoringService:
    def __init__(self, models_dir):
        from pathlib import Path
        d = Path(models_dir)
        self.pipe = joblib.load(d / "mlp_final.joblib")
        self.card = joblib.load(d / "scorecard_mlp.joblib")
        self.ref = np.load(d / "score_reference.npy")
        self.meta = json.load(open(d / "app_meta.json", encoding="utf-8"))
        s = self.meta["scale"]
        self.scale = ScoreScale(s["base_score"], s["base_odds"], s["pdo"])
        self.cols = self.meta["num_cols"] + self.meta["cat_cols"]

    def _frame(self, applicant):
        row = {c: (np.nan if applicant.get(c) is None else applicant.get(c)) for c in self.cols}
        return pd.DataFrame([row])

    def score(self, applicant):
        p = float(self.pipe.predict_proba(self._frame(applicant))[0])
        s = float(self.scale.score(p))
        pct = float(np.searchsorted(self.ref, s) / len(self.ref) * 100)   # % de la población con puntaje menor
        return {"pd": p, "score": s, "percentile": pct}

    def contributions(self, applicant):
        df = self._frame(applicant)
        c = self.card.contributions(df).iloc[0].sort_values()
        return pd.DataFrame({"variable": c.index, "puntos": c.values})

    def what_if(self, applicant, changes):
        base = self.score(applicant)
        new = self.score({**applicant, **changes})
        return base, new, new["score"] - base["score"]

    def sweep(self, applicant, var, values):
        out = []
        for v in values:
            out.append(self.score({**applicant, var: v})["score"])
        return pd.DataFrame({var: values, "puntaje": out})
