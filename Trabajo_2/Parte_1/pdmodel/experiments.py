"""Experimentos: baseline (regresión logística), búsqueda de la red, corridas con N semillas, scorecard."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pdmodel import plots
from pdmodel.config import path_of
from pdmodel.data import app_columns, load_processed
from pdmodel.features import Preprocessor
from pdmodel.interpret import permutation_importance
from pdmodel.metrics import all_metrics, auc_score, log_loss
from pdmodel.mlp import MLPClassifier
from pdmodel.pipeline import PDPipeline, SkLogit
from pdmodel.scorecard import ScoreScale, build_scorecard

HIGHER_IS_BETTER = {"auc": True, "gini": True, "ks": True, "brier": False, "logloss": False}


# ---------------------------------------------------------------- datos compartidos
class Data:
    """Mismos datos, mismo preprocesamiento y mismas particiones para TODOS los modelos."""

    def __init__(self, cfg):
        self.cfg = cfg
        df = load_processed(cfg)
        self.num_cols, self.cat_cols = app_columns(cfg["data"]["use_lc_pricing"])
        self.cols = self.num_cols + self.cat_cols
        self.parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
        for s, d in self.parts.items():
            if len(d) == 0 or d["y"].nunique() < 2:
                raise ValueError(f"La partición '{s}' está vacía o tiene una sola clase. Revise data.split en la config.")
        mdir = path_of(cfg, "models", create=True)
        self.pre = Preprocessor(self.num_cols, self.cat_cols, cfg["data"]["clip_quantiles"],
                                cfg["data"]["min_category_freq"]).fit(self.parts["train"])
        joblib.dump(self.pre, mdir / "preprocessor.joblib")
        self.X = {s: self.pre.transform(d) for s, d in self.parts.items()}
        self.y = {s: d["y"].to_numpy() for s, d in self.parts.items()}


def _score_metric(metric, y, p):
    m = all_metrics(y, p)
    return m[metric], m


def _write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


# ---------------------------------------------------------------- baseline
def run_baseline(cfg, data):
    bc = cfg["baseline"]
    metric = bc["select_metric"]
    rows, best, t0 = [], None, time.time()
    for C in bc["C_grid"]:
        t1 = time.time()
        sk = LogisticRegression(C=C, max_iter=2000).fit(data.X["train"], data.y["train"])
        pv = sk.predict_proba(data.X["val"])[:, 1]
        val, _ = _score_metric(metric, data.y["val"], pv)
        rows.append({"C": C, f"val_{metric}": val, "segundos": time.time() - t1})
        key = val if HIGHER_IS_BETTER[metric] else -val
        if best is None or key > best[0]:
            best = (key, C, sk)
    total = time.time() - t0
    tdir = path_of(cfg, "tables", create=True)
    pd.DataFrame(rows).to_csv(tdir / "baseline_search.csv", index=False)
    pipe = PDPipeline(data.pre, SkLogit(best[2]), "Regresión logística")
    joblib.dump(pipe, path_of(cfg, "models", create=True) / "lr_final.joblib")
    res = {"C_elegido": best[1], "n_configuraciones": len(rows), "segundos_total": total, "n_params": pipe.n_params,
           "val": all_metrics(data.y["val"], best[2].predict_proba(data.X["val"])[:, 1]),
           "test": all_metrics(data.y["test"], best[2].predict_proba(data.X["test"])[:, 1])}
    _write_json(res, tdir / "baseline_result.json")
    return res


# ---------------------------------------------------------------- búsqueda de la red
def sample_config(space, rng):
    n_layers = int(rng.choice(space["n_layers"]))
    units = int(rng.choice(space["units"]))
    opt = str(rng.choice(space["optimizer"]))
    lo, hi = space["lr_adam"] if opt == "adam" else space["lr_sgd"]
    return {
        "hidden": [units] * n_layers,
        "activation": str(rng.choice(space["activation"])),
        "dropout": float(rng.choice(space["dropout"])),
        "l2": float(np.exp(rng.uniform(np.log(space["l2"][0]), np.log(space["l2"][1])))),
        "optimizer": opt,
        "lr": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "batch_size": int(rng.choice(space["batch_size"])),
    }


REFERENCE_CONFIG = {"hidden": [32, 16], "activation": "relu", "dropout": 0.1, "l2": 1e-5,
                    "optimizer": "adam", "lr": 1e-3, "batch_size": 1024}


def run_search(cfg, data):
    sc = cfg["mlp_search"]
    metric = sc["select_metric"]
    rng = np.random.RandomState(cfg["seed"])
    configs = [REFERENCE_CONFIG] + [sample_config(sc["space"], rng) for _ in range(sc["n_trials"] - 1)]
    rows, best, t0 = [], None, time.time()
    for i, c in enumerate(configs):
        t1 = time.time()
        m = MLPClassifier(**c, seed=cfg["seed"]).fit(data.X["train"], data.y["train"], data.X["val"],
                                                    data.y["val"], sc["max_epochs"], sc["patience"])
        val, allm = _score_metric(metric, data.y["val"], m.predict_proba(data.X["val"]))
        rows.append({"trial": i, **{k: (str(v) if k == "hidden" else v) for k, v in c.items()},
                     "n_params": m.n_params, "mejor_epoca": m.best_epoch_, f"val_{metric}": val,
                     "val_auc": allm["auc"], "val_logloss": allm["logloss"], "segundos": time.time() - t1})
        print(f"  prueba {i + 1:3d}/{len(configs)}  val_{metric}={val:.4f}  {c['hidden']} {c['activation']} "
              f"{c['optimizer']} lr={c['lr']:.4g}  ({time.time() - t1:.1f}s)")
        key = val if HIGHER_IS_BETTER[metric] else -val
        if best is None or key > best[0]:
            best = (key, c, val)
    total = time.time() - t0
    tdir = path_of(cfg, "tables", create=True)
    pd.DataFrame(rows).to_csv(tdir / "mlp_search_trials.csv", index=False)
    mdir = path_of(cfg, "models", create=True)
    _write_json(best[1], mdir / "best_mlp_config.json")
    res = {"n_configuraciones": len(configs), "segundos_total": total, "mejor_config": best[1],
           f"mejor_val_{metric}": best[2]}
    _write_json(res, tdir / "mlp_search_result.json")
    return res


# ---------------------------------------------------------------- 30 semillas y comparación
def run_final(cfg, data):
    fc = cfg["final"]
    mdir = path_of(cfg, "models", create=True)
    tdir = path_of(cfg, "tables", create=True)
    fdir = path_of(cfg, "figures", create=True)
    mcfg = fc["config"] or json.load(open(mdir / "best_mlp_config.json", encoding="utf-8"))
    rows, models, t0 = [], [], time.time()
    for k in range(fc["n_seeds"]):
        seed = cfg["seed"] + 1000 + k
        m = MLPClassifier(**mcfg, seed=seed).fit(data.X["train"], data.y["train"], data.X["val"], data.y["val"],
                                                fc["max_epochs"], fc["patience"])
        mv = all_metrics(data.y["val"], m.predict_proba(data.X["val"]))
        mt = all_metrics(data.y["test"], m.predict_proba(data.X["test"]))
        rows.append({"seed": seed, "mejor_epoca": m.best_epoch_, "segundos": m.fit_seconds_,
                     **{f"val_{a}": b for a, b in mv.items()}, **{f"test_{a}": b for a, b in mt.items()}})
        models.append(m)
        print(f"  semilla {k + 1:2d}/{fc['n_seeds']} ({seed})  test_auc={mt['auc']:.4f}  test_brier={mt['brier']:.4f}")
    total = time.time() - t0
    seeds_df = pd.DataFrame(rows)
    seeds_df.to_csv(tdir / "mlp_seeds_metrics.csv", index=False)

    summary = []
    for part in ["val", "test"]:
        for met in ["auc", "gini", "ks", "brier", "logloss"]:
            s = seeds_df[f"{part}_{met}"]
            best_v, worst_v = (s.max(), s.min()) if HIGHER_IS_BETTER[met] else (s.min(), s.max())
            summary.append({"particion": part, "metrica": met, "media": s.mean(), "desv_est": s.std(ddof=1) if len(s) > 1 else 0.0,
                            "mejor": best_v, "peor": worst_v})
    summary = pd.DataFrame(summary)
    summary.to_csv(tdir / "mlp_seeds_summary.csv", index=False)

    # representante: la semilla con mejor log-loss en VALIDACIÓN (nunca se elige con prueba)
    rep = int(seeds_df["val_logloss"].idxmin())
    pipe = PDPipeline(data.pre, models[rep], "Red neuronal (MLP)")
    joblib.dump(pipe, mdir / "mlp_final.joblib")
    plots.loss_curves(models[rep].history_, fdir / "mlp_learning_curves.png")

    lr_pipe = joblib.load(mdir / "lr_final.joblib")
    base = json.load(open(tdir / "baseline_result.json", encoding="utf-8"))
    srch = json.load(open(tdir / "mlp_search_result.json", encoding="utf-8")) if (tdir / "mlp_search_result.json").exists() else {}
    comp = []
    t = summary[summary.particion == "test"].set_index("metrica")
    for met in ["auc", "gini", "ks", "brier"]:
        comp.append({"metrica_test": met, "LR": base["test"][met], "MLP_media": t.loc[met, "media"],
                     "MLP_desv_est": t.loc[met, "desv_est"], "MLP_mejor": t.loc[met, "mejor"], "MLP_peor": t.loc[met, "peor"]})
    comp = pd.DataFrame(comp)
    comp.to_csv(tdir / "comparison_lr_vs_mlp.csv", index=False)
    budget = {"LR": {"n_configuraciones": base["n_configuraciones"], "segundos_busqueda": base["segundos_total"], "n_params": base["n_params"]},
              "MLP": {"n_configuraciones": srch.get("n_configuraciones"), "segundos_busqueda": srch.get("segundos_total"),
                      "segundos_corridas_finales": total, "n_semillas": fc["n_seeds"], "n_params": pipe.n_params, "config": mcfg}}
    _write_json(budget, tdir / "compute_budget.json")

    yt = data.y["test"]
    p_lr, p_mlp = lr_pipe.model.predict_proba(data.X["test"]), pipe.model.predict_proba(data.X["test"])
    plots.roc_ks_calibration(yt, {"Regresión logística": p_lr, "Red neuronal": p_mlp}, str(fdir / "test"))
    pd.DataFrame({"y": yt, "p_lr": p_lr, "p_mlp": p_mlp}).to_csv(tdir / "test_predictions.csv.gz", index=False, compression="gzip")
    return {"resumen": summary, "comparacion": comp, "presupuesto": budget, "semilla_representante": int(seeds_df.loc[rep, "seed"])}


# ---------------------------------------------------------------- scorecard + interpretabilidad
def run_scorecard(cfg, data):
    sc = cfg["scorecard"]
    scale = ScoreScale(sc["base_score"], sc["base_odds"], sc["pdo"])
    mdir = path_of(cfg, "models", create=True)
    tdir = path_of(cfg, "tables", create=True)
    fdir = path_of(cfg, "figures", create=True)
    mlp, lr = joblib.load(mdir / "mlp_final.joblib"), joblib.load(mdir / "lr_final.joblib")
    train, test = data.parts["train"], data.parts["test"]
    out = {}
    cards = {}
    for name, pipe in [("mlp", mlp), ("lr", lr)]:
        card = build_scorecard(pipe.predict_logit, train[data.cols], data.num_cols, data.cat_cols, scale,
                               sc["n_background"], sc["n_grid"], cfg["seed"])
        card.table.to_csv(tdir / f"scorecard_{name}_points.csv", index=False)
        exact = scale.score(pipe.predict_proba(test[data.cols]))
        add = card.additive_score(test[data.cols])
        out[name] = {"puntos_base": card.base_points,
                     "corr_aditivo_vs_exacto": float(np.corrcoef(add, exact)[0, 1]),
                     "mae_puntos": float(np.mean(np.abs(add - exact))),
                     "puntaje_exacto_min_max": [float(exact.min()), float(exact.max())]}
        cards[name] = card
    out["escala"] = {"base_score": scale.base_score, "base_odds": scale.base_odds, "pdo": scale.pdo,
                     "factor": scale.factor, "offset": scale.offset}
    _write_json(out, tdir / "scorecard_summary.json")
    joblib.dump(cards["mlp"], mdir / "scorecard_mlp.joblib")

    yt = data.y["test"]
    exact_test = scale.score(mlp.predict_proba(test[data.cols]))
    plots.score_distribution(exact_test, yt, fdir / "score_distribution.png")
    plots.points_curves(cards["mlp"].table, [c for c in data.cols][:12], fdir / "scorecard_points_mlp.png")

    # población de referencia para el percentil de la app (todos los créditos etiquetados)
    allapp = pd.concat(data.parts.values())
    ref = np.sort(scale.score(mlp.predict_proba(allapp[data.cols])))
    np.save(mdir / "score_reference.npy", ref)
    meta = {"num_cols": data.num_cols, "cat_cols": data.cat_cols, "scale": out["escala"],
            "defaults": {c: float(np.nanmedian(train[c])) for c in data.num_cols}
                        | {c: str(train[c].mode().iloc[0]) for c in data.cat_cols},
            "ranges": {c: [float(np.nanquantile(train[c], 0.01)), float(np.nanquantile(train[c], 0.99))] for c in data.num_cols},
            "categories": {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in data.cat_cols}}
    _write_json(meta, mdir / "app_meta.json")

    # importancia por permutación (en prueba, submuestreada)
    ic = cfg["interpret"]
    samp = test.sample(min(ic["n_sample"], len(test)), random_state=cfg["seed"])
    imps = {}
    for name, pipe in [("Red neuronal", mlp), ("Regresión logística", lr)]:
        imp = permutation_importance(pipe.predict_proba, samp[data.cols], samp["y"].to_numpy(), data.cols,
                                     ic["n_repeats"], cfg["seed"])
        imp.to_csv(tdir / f"permutation_importance_{'mlp' if name.startswith('Red') else 'lr'}.csv", index=False)
        imps[name] = imp
    plots.importance_bar(imps, fdir / "permutation_importance.png")
    return out, imps
