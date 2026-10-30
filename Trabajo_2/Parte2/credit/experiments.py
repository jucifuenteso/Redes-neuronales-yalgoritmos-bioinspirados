"""Experimentos de la Parte 2: LGD (regresión), ablación de la función de pérdida y pérdida esperada."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

from . import config
from . import metrics
from . import nn as neural_network
from .features import make_bundle


def _dump(obj, path):
    def convert_numpy_value(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value))
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert_numpy_value), "utf-8")


def sample_hyperparameters(search_space, base_hyperparameters, random_generator, epochs):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters["hidden"] = ([int(random_generator.choice(search_space["units"]))]
                                 * int(random_generator.choice(search_space["n_layers"])))
    for name in ("activation", "dropout", "weight_decay", "optimizer", "lr", "batch_size"):
        sampled_value = random_generator.choice(search_space[name])
        hyperparameters[name] = sampled_value.item() if hasattr(sampled_value, "item") else sampled_value
    hyperparameters["epochs"] = epochs
    return hyperparameters


def hyperparameters_from_trial_row(trial_row, base_hyperparameters, epochs):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters["hidden"] = json.loads(str(trial_row["hidden"]))
    for name in ("activation", "optimizer"):
        hyperparameters[name] = str(trial_row[name])
    for name in ("dropout", "weight_decay", "lr"):
        hyperparameters[name] = float(trial_row[name])
    hyperparameters["batch_size"] = int(trial_row["batch_size"])
    hyperparameters["epochs"] = int(epochs)
    return hyperparameters


def random_search(cfg, features_train, target_train, features_val, target_val, base_hyperparameters, loss_name):
    """Búsqueda aleatoria (Bergstra & Bengio, 2012): n pruebas con la misma semilla de entrenamiento.

    Se prefiere a la cuadrícula porque, con muchas dimensiones y pocas relevantes, explora más
    valores distintos de cada hiperparámetro con el mismo presupuesto. Criterio: RMSE de validación.
    """
    search_config = cfg["lgd"]["search"]
    random_generator = np.random.default_rng(cfg["seed"])
    trial_rows, search_start_time = [], time.time()
    for trial_index in range(int(search_config["trials"])):
        hyperparameters = sample_hyperparameters(search_config["space"], base_hyperparameters,
                                                 random_generator, search_config["epochs"])
        model, training_info = neural_network.fit_mlp(features_train, target_train, features_val, target_val,
                                         hyperparameters, seed=cfg["seed"], loss=loss_name)
        validation_rmse = metrics.regression_metrics(target_val, neural_network.predict(model, features_val))["rmse"]
        trial_rows.append({
            "trial": trial_index,
            **{name: str(value) if name == "hidden" else value for name, value in hyperparameters.items()},
            "val_score": -validation_rmse, "val_rmse": validation_rmse,
            "epochs_run": training_info["epochs"], "seconds": training_info["seconds"],
            "n_params": training_info["n_params"]})
        print(f"[LGD búsqueda {trial_index + 1}/{search_config['trials']}] val_rmse={validation_rmse:.4f} "
              f"hidden={hyperparameters['hidden']} act={hyperparameters['activation']} "
              f"opt={hyperparameters['optimizer']} lr={hyperparameters['lr']}")
    return pd.DataFrame(trial_rows), time.time() - search_start_time


# ----------------------------------------------------------------------------- Descriptivo de la LGD
def run_lgd_describe(cfg):
    """LGD observada por año de originación, plazo y propósito: insumo de la discusión de limitaciones."""
    output_dir = config.path(cfg, "outputs")
    bundle = make_bundle(cfg)
    loans_with_observed_lgd = bundle.loans[bundle.loans["lgd"].notna()].copy()
    loans_with_observed_lgd["issue_year"] = loans_with_observed_lgd["issue_d"].dt.year
    loans_with_observed_lgd["term_months"] = (
        loans_with_observed_lgd["term"].astype(str).str.extract(r"(\d+)")[0].astype(float))
    segment_tables = []
    for segment_variable in ("issue_year", "term_months", "purpose", "home_ownership", "verification_status"):
        segment_table = loans_with_observed_lgd.groupby(segment_variable)["lgd"].agg(
            n="size", lgd_mean="mean", lgd_median="median",
            share_lgd_ge_099=lambda lgd_values: float((lgd_values >= 0.99).mean()),
            share_lgd_le_001=lambda lgd_values: float((lgd_values <= 0.01).mean())).reset_index()
        segment_table.insert(0, "variable", segment_variable)
        segment_table = segment_table.rename(columns={segment_variable: "level"})
        segment_table["level"] = segment_table["level"].astype(str)
        segment_tables.append(segment_table)
    pd.concat(segment_tables, ignore_index=True).to_csv(output_dir / "lgd_descriptive.csv", index=False)

    # por qué la LGD es una variable en [0,1] con masa en los extremos
    observed_lgd = loans_with_observed_lgd["lgd"]
    descriptive_summary = {
        "n_lgd_observada": int(len(loans_with_observed_lgd)), "lgd_media": float(observed_lgd.mean()),
        "lgd_mediana": float(observed_lgd.median()), "lgd_std": float(observed_lgd.std()),
        "share_lgd_ge_099": float((observed_lgd >= 0.99).mean()),
        "share_lgd_le_001": float((observed_lgd <= 0.01).mean()),
        "share_default_con_lgd_observada": float(bundle.loans.loc[bundle.loans["default"] == 1, "lgd"].notna().mean())}
    _dump(descriptive_summary, output_dir / "lgd_descriptive_summary.json")
    print(json.dumps(descriptive_summary, indent=2, ensure_ascii=False))
    return descriptive_summary


# ----------------------------------------------------------------------------- Distribución
def distribution_report(observed_values, predictions_by_model):
    """Compara la distribución de las predicciones con la observada (momentos, cuantiles, masa en extremos,
    estadístico KS de dos muestras y distancia de Wasserstein-1)."""
    quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    observed_values = np.asarray(observed_values, float)

    def describe_distribution(values, reference_values=None):
        report = {"mean": float(values.mean()), "std": float(values.std()),
                  **{f"q{int(level * 100):02d}": float(np.quantile(values, level)) for level in quantile_levels},
                  "share_ge_099": float((values >= 0.99).mean()), "share_le_001": float((values <= 0.01).mean())}
        if reference_values is not None:
            report["ks_vs_obs"] = float(stats.ks_2samp(values, reference_values).statistic)
            report["wasserstein_vs_obs"] = float(stats.wasserstein_distance(values, reference_values))
        return report

    full_report = {"observada": describe_distribution(observed_values)}
    for model_name, model_predictions in predictions_by_model.items():
        full_report[model_name] = describe_distribution(np.asarray(model_predictions, float), observed_values)
    return full_report


# ----------------------------------------------------------------------------- LGD
def run_lgd(cfg, skip_search=False):
    output_dir, models_dir = config.path(cfg, "outputs"), config.path(cfg, "models")
    run_start_time = time.time()
    bundle = make_bundle(cfg)
    features_train, target_train, _ = bundle.lgd_part("train")
    features_val, target_val, positions_val = bundle.lgd_part("val")
    features_test, target_test, positions_test = bundle.lgd_part("test")
    if min(len(target_train), len(target_val), len(target_test)) == 0:
        raise RuntimeError("Alguna partición no tiene créditos con LGD observada; revise data.train_end / data.val_end.")
    print(f"LGD train={len(target_train)} val={len(target_val)} test={len(target_test)}  "
          f"LGD media(train)={target_train.mean():.3f}  variables={features_train.shape[1]}")
    rmse_of = lambda observed, predicted: metrics.regression_metrics(observed, predicted)["rmse"]

    # --- referencia: regresión lineal con L2 (alpha elegido por RMSE de validación)
    linear_start_time = time.time()
    best_linear_model, best_validation_rmse = None, np.inf
    for ridge_alpha in cfg["lgd"]["ridge_alpha_grid"]:
        candidate_model = Ridge(alpha=ridge_alpha).fit(features_train, target_train)
        validation_rmse = rmse_of(target_val, np.clip(candidate_model.predict(features_val), 0, 1))
        if validation_rmse < best_validation_rmse:
            best_linear_model, best_validation_rmse, best_ridge_alpha = candidate_model, validation_rmse, ridge_alpha
    linear_seconds = time.time() - linear_start_time
    joblib.dump(best_linear_model, models_dir / "lgd_linear.joblib")
    linear_predictions = {
        "val": np.clip(best_linear_model.predict(features_val), 0, 1),
        "test": np.clip(best_linear_model.predict(features_test), 0, 1)}

    # --- búsqueda de hiperparámetros del MLP
    base_hyperparameters = dict(cfg["lgd"]["mlp"])
    loss_name = base_hyperparameters.pop("loss")
    saved_hyperparameters_path = models_dir / "lgd_best_hp.json"
    if int(cfg["lgd"]["search"]["trials"]) > 0 and not skip_search:
        search_trials, search_seconds = random_search(cfg, features_train, target_train, features_val,
                                                      target_val, base_hyperparameters, loss_name)
        search_trials.to_csv(output_dir / "lgd_search_trials.csv", index=False)
        hyperparameters = hyperparameters_from_trial_row(
            search_trials.sort_values("val_score", ascending=False).iloc[0], base_hyperparameters,
            cfg["lgd"]["mlp"]["epochs"])
    elif skip_search and saved_hyperparameters_path.exists():
        hyperparameters = json.loads(saved_hyperparameters_path.read_text("utf-8"))
        search_trials, search_seconds = None, 0.0
    else:
        hyperparameters, search_trials, search_seconds = base_hyperparameters, None, 0.0
    saved_hyperparameters_path.write_text(json.dumps(hyperparameters, indent=2), "utf-8")
    print("Mejor configuración MLP-LGD:", hyperparameters, "| pérdida:", loss_name)

    # --- configuración final con n semillas
    metrics_per_seed, trained_models, final_training_start_time = [], [], time.time()
    for seed_offset in range(int(cfg["n_seeds"])):
        seed = int(cfg["seeds_final_start"]) + seed_offset
        model, training_info = neural_network.fit_mlp(features_train, target_train, features_val, target_val,
                                         hyperparameters, seed=seed, loss=loss_name)
        metrics_per_seed.append({
            "seed": seed,
            **{f"val_{name}": value
               for name, value in metrics.regression_metrics(target_val, neural_network.predict(model, features_val)).items()},
            **{f"test_{name}": value
               for name, value in metrics.regression_metrics(target_test, neural_network.predict(model, features_test)).items()},
            "epochs": training_info["epochs"], "seconds": training_info["seconds"],
            "n_params": training_info["n_params"]})
        trained_models.append(model)
        print(f"[LGD semilla {seed}] val_rmse={metrics_per_seed[-1]['val_rmse']:.4f} "
              f"test_rmse={metrics_per_seed[-1]['test_rmse']:.4f}")
    final_training_seconds = time.time() - final_training_start_time
    seeds_table = pd.DataFrame(metrics_per_seed)
    seeds_table.to_csv(output_dir / "lgd_mlp_seeds.csv", index=False)

    # modelo representativo: menor RMSE de VALIDACIÓN (nunca se elige con prueba)
    selected_seed_position = int(seeds_table["val_rmse"].idxmin())
    selected_model = trained_models[selected_seed_position]
    import torch
    torch.save(selected_model.state_dict(), models_dir / "lgd_mlp.pt")
    neural_network.export_numpy(selected_model, hyperparameters["activation"], models_dir / "lgd_mlp.npz")
    mlp_predictions = {"val": neural_network.predict(selected_model, features_val), "test": neural_network.predict(selected_model, features_test)}

    pd.concat([
        pd.DataFrame({"part": "val", "row": positions_val, "lgd_obs": target_val,
                      "lgd_linear": linear_predictions["val"], "lgd_mlp": mlp_predictions["val"]}),
        pd.DataFrame({"part": "test", "row": positions_test, "lgd_obs": target_test,
                      "lgd_linear": linear_predictions["test"], "lgd_mlp": mlp_predictions["test"]}),
    ]).to_csv(output_dir / "lgd_predictions.csv", index=False)

    train_mean_lgd = float(target_train.mean())
    summary = {
        "n_train": len(target_train), "n_val": len(target_val), "n_test": len(target_test),
        "n_features": features_train.shape[1], "loss": loss_name, "split": cfg["data"]["split"],
        "include_lc_pricing": cfg["data"]["include_lc_pricing"],
        "baseline_constante_media": {
            "mean_train": train_mean_lgd,
            "test": metrics.regression_metrics(target_test, np.full_like(target_test, train_mean_lgd))},
        "linear": {"alpha": best_ridge_alpha, "configs_trained": len(cfg["lgd"]["ridge_alpha_grid"]),
                   "seconds": linear_seconds, "n_params": int(best_linear_model.coef_.size + 1),
                   "metrics": {"val": metrics.regression_metrics(target_val, linear_predictions["val"]),
                               "test": metrics.regression_metrics(target_test, linear_predictions["test"])}},
        "mlp": {"hp": hyperparameters,
                "configs_trained_search": 0 if search_trials is None else len(search_trials),
                "seconds_search": search_seconds, "seconds_final_all_seeds": final_training_seconds,
                "n_seeds": len(metrics_per_seed), "n_params": int(seeds_table["n_params"].iloc[0]),
                "seed_summary": metrics.summarize(metrics_per_seed),
                "selected_seed": int(seeds_table["seed"].iloc[selected_seed_position]),
                "selected_metrics": {"val": metrics.regression_metrics(target_val, mlp_predictions["val"]),
                                     "test": metrics.regression_metrics(target_test, mlp_predictions["test"])}},
        "distribution_test": distribution_report(target_test, {"lineal": linear_predictions["test"],
                                                               "mlp": mlp_predictions["test"]}),
        "total_seconds": time.time() - run_start_time,
    }
    _dump(summary, output_dir / "lgd_summary.json")
    print(f"Listo LGD. MLP test RMSE media={summary['mlp']['seed_summary']['test_rmse']['mean']:.4f} "
          f"vs lineal={summary['linear']['metrics']['test']['rmse']:.4f} "
          f"vs constante={summary['baseline_constante_media']['test']['rmse']:.4f}")
    return summary


def run_lgd_loss_ablation(cfg):
    """Misma arquitectura e hiperparámetros, mismas semillas: BCE (cuasi-Bernoulli) vs MSE (gaussiana).

    Evidencia para la elección de la función de pérdida: ambas se evalúan con el MISMO RMSE/MAE de prueba.
    """
    output_dir, models_dir = config.path(cfg, "outputs"), config.path(cfg, "models")
    bundle = make_bundle(cfg)
    features_train, target_train, _ = bundle.lgd_part("train")
    features_val, target_val, _ = bundle.lgd_part("val")
    features_test, target_test, _ = bundle.lgd_part("test")
    saved_hyperparameters_path = models_dir / "lgd_best_hp.json"
    base_hyperparameters = dict(cfg["lgd"]["mlp"])
    base_hyperparameters.pop("loss")
    hyperparameters = (json.loads(saved_hyperparameters_path.read_text("utf-8"))
                       if saved_hyperparameters_path.exists() else base_hyperparameters)
    result_rows, ablation_start_time = [], time.time()
    for loss_name in ("bce", "mse"):
        for seed_offset in range(int(cfg["n_seeds"])):
            seed = int(cfg["seeds_final_start"]) + seed_offset
            model, training_info = neural_network.fit_mlp(features_train, target_train, features_val, target_val,
                                             hyperparameters, seed=seed, loss=loss_name)
            test_metrics = metrics.regression_metrics(target_test, neural_network.predict(model, features_test))
            result_rows.append({"loss": loss_name, "seed": seed, "test_rmse": test_metrics["rmse"],
                                "test_mae": test_metrics["mae"], "epochs": training_info["epochs"],
                                "seconds": training_info["seconds"]})
        mean_test_rmse = np.mean([row["test_rmse"] for row in result_rows if row["loss"] == loss_name])
        print(f"[ablación {loss_name}] test_rmse media={mean_test_rmse:.4f}")
    results_table = pd.DataFrame(result_rows)
    results_table.to_csv(output_dir / "lgd_loss_ablation.csv", index=False)
    bce_results = results_table[results_table["loss"] == "bce"].sort_values("seed")
    mse_results = results_table[results_table["loss"] == "mse"].sort_values("seed")
    summary = {"hp": hyperparameters, "n_seeds": int(len(bce_results)), "seconds": time.time() - ablation_start_time}
    for metric_name in ("test_rmse", "test_mae"):
        paired_differences = bce_results[metric_name].to_numpy() - mse_results[metric_name].to_numpy()
        metric_summary = {
            "bce_mean": float(bce_results[metric_name].mean()),
            "bce_std": float(bce_results[metric_name].std(ddof=1)) if len(bce_results) > 1 else 0.0,
            "mse_mean": float(mse_results[metric_name].mean()),
            "mse_std": float(mse_results[metric_name].std(ddof=1)) if len(mse_results) > 1 else 0.0,
            "mean_diff_bce_minus_mse": float(paired_differences.mean()),
            "bce_wins": int((paired_differences < 0).sum())}
        if len(paired_differences) > 1 and np.any(paired_differences != 0):
            metric_summary["paired_t_pvalue"] = float(
                stats.ttest_rel(bce_results[metric_name], mse_results[metric_name]).pvalue)
            try:
                metric_summary["wilcoxon_pvalue"] = float(stats.wilcoxon(paired_differences).pvalue)
            except ValueError:
                pass
        summary[metric_name] = metric_summary
    _dump(summary, output_dir / "lgd_loss_ablation_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ----------------------------------------------------------------------------- Pérdida esperada
def _fit_pd_reference(cfg, bundle, models_dir):
    """PD mínima (logística + MLP de configuración fija, una semilla) para poder calcular E[L] aquí."""
    features_train, target_train = bundle.features["train"], bundle.targets["train"]
    features_val, target_val = bundle.features["val"], bundle.targets["val"]
    best_logistic_model, best_validation_auc = None, -1.0
    for regularization_strength in cfg["pd_ref"]["logreg"]["C_grid"]:
        candidate_model = LogisticRegression(C=regularization_strength, max_iter=2000).fit(
            features_train, target_train)
        validation_auc = metrics.auc(target_val, candidate_model.predict_proba(features_val)[:, 1])
        if validation_auc > best_validation_auc:
            best_logistic_model, best_validation_auc = candidate_model, validation_auc
    joblib.dump(best_logistic_model, models_dir / "pd_ref_logreg.joblib")
    hyperparameters = dict(cfg["pd_ref"]["mlp"])
    mlp_model, _ = neural_network.fit_mlp(features_train, target_train, features_val, target_val, hyperparameters,
                             seed=int(cfg["seeds_final_start"]), loss="bce")
    neural_network.export_numpy(mlp_model, hyperparameters["activation"], models_dir / "pd_ref_mlp.npz")
    validation_info = {"logreg_val_auc": best_validation_auc,
                       "mlp_val_auc": metrics.auc(target_val, neural_network.predict(mlp_model, features_val))}
    return best_logistic_model, mlp_model, validation_info


def run_expected_loss(cfg):
    """E[L] = PD · LGD · EAD sobre la partición de prueba, contra la pérdida realizada.

    EAD = funded_amnt (monto desembolsado): al originar no se conoce el saldo al momento del incumplimiento.
    Coherente con la LGD, que se define como fracción de funded_amnt.
    """
    output_dir, models_dir = config.path(cfg, "outputs"), config.path(cfg, "models")
    for required_file in ("lgd_linear.joblib", "lgd_mlp.npz"):
        if not (models_dir / required_file).exists():
            raise FileNotFoundError(f"Falta {models_dir / required_file}. Ejecute primero: python run.py lgd")
    bundle = make_bundle(cfg)
    test_positions, test_features = bundle.row_positions["test"], bundle.features["test"]
    test_loans = bundle.loans.iloc[test_positions]
    pd_logistic_model, pd_mlp_model, pd_validation_info = _fit_pd_reference(cfg, bundle, models_dir)
    pd_logistic = pd_logistic_model.predict_proba(test_features)[:, 1]
    pd_mlp = neural_network.predict(pd_mlp_model, test_features)
    lgd_linear = np.clip(joblib.load(models_dir / "lgd_linear.joblib").predict(test_features), 0, 1)
    lgd_mlp = neural_network.NumpyMLP(models_dir / "lgd_mlp.npz").predict(test_features)
    constant_lgd_from_train = float(bundle.loans.loc[bundle.row_positions["train"], "lgd"].mean())
    exposure_at_default = test_loans["funded_amnt"].to_numpy(float)

    # Pérdida realizada solo donde el desenlace es definitivo: pagó (0) o fue castigado (LGD observada).
    # "Late (31-120 days)" cuenta como malo en la PD pero su recuperación aún no ocurrió => no observable.
    loss_is_observable = ((test_loans["default"] == 0) | test_loans["loss_final"]).to_numpy()
    realized_loss = np.where(test_loans["default"] == 1, test_loans["lgd"].fillna(0), 0.0) * exposure_at_default
    expected_loss_table = pd.DataFrame({
        "row": test_positions, "ead": exposure_at_default, "default": test_loans["default"].to_numpy(),
        "observable": loss_is_observable, "pd_mlp": pd_mlp, "lgd_mlp": lgd_mlp,
        "pd_logreg": pd_logistic, "lgd_linear": lgd_linear,
        "el_mlp": pd_mlp * lgd_mlp * exposure_at_default,
        "el_ref": pd_logistic * lgd_linear * exposure_at_default,
        "el_pd_lgdconst": pd_logistic * constant_lgd_from_train * exposure_at_default,
        "realized_loss": realized_loss})
    expected_loss_table.to_csv(output_dir / "expected_loss_test.csv.gz", index=False, compression="gzip")
    observable_loans = expected_loss_table[expected_loss_table["observable"]]
    mean_absolute_error_per_loan = lambda column: float(
        np.mean(np.abs(observable_loans[column] - observable_loans["realized_loss"])))
    total_realized_loss = float(observable_loans["realized_loss"].sum())
    summary = {
        "n_test_total": int(len(expected_loss_table)), "n_observable": int(len(observable_loans)),
        "ead_total": float(observable_loans["ead"].sum()), "realized_loss_total": total_realized_loss,
        "lgd_const_train": constant_lgd_from_train, "pd_models_val": pd_validation_info,
        "modelos": {
            model_label: {
                "el_total": float(observable_loans[column].sum()),
                "ratio_vs_realized": (float(observable_loans[column].sum() / total_realized_loss)
                                      if total_realized_loss else None),
                "mae_loan": mean_absolute_error_per_loan(column)}
            for model_label, column in (("mlp_pd+mlp_lgd", "el_mlp"), ("logistica+lineal", "el_ref"),
                                        ("logistica+lgd_constante", "el_pd_lgdconst"))},
        "nota": "Pérdida realizada = default * LGD observada * EAD, solo créditos con desenlace definitivo. "
                "La PD usada es una referencia mínima (ver README), no la de la Parte 1. "
                "Sesgo por censura: en los créditos recientes los 'Current' (que casi siempre terminan pagando) y los "
                "'Late (31-120 days)' quedan fuera, así que la pérdida realizada observable está sobreestimada "
                "frente a la que tendrá la cartera completa; la razón E[L]/realizada es una cota, no un acierto.",
    }
    _dump(summary, output_dir / "expected_loss_summary.json")
    observable_loans = observable_loans.assign(
        decile=pd.qcut(observable_loans["pd_mlp"].rank(method="first"), 10, labels=False) + 1)
    observable_loans.groupby("decile").agg(
        n=("ead", "size"), pd_pred=("pd_mlp", "mean"), default_obs=("default", "mean"),
        el_mlp=("el_mlp", "sum"), el_ref=("el_ref", "sum"),
        realized=("realized_loss", "sum")).to_csv(output_dir / "expected_loss_deciles.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
