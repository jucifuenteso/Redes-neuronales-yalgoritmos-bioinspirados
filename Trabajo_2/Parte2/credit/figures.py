"""Regenera todas las figuras de la Parte 2 desde los archivos de outputs/ (sin reentrenar)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
COLOR_OBSERVED, COLOR_LINEAR, COLOR_MLP = "#4C72B0", "#DD8452", "#55A868"


def _save(figure, figures_dir, file_name):
    figure.tight_layout()
    figure.savefig(figures_dir / file_name)
    plt.close(figure)
    print("figura:", figures_dir / file_name)


def _plot_descriptive_lgd(descriptive_table, figures_dir):
    figure, (axis_by_year, axis_by_purpose) = plt.subplots(1, 2, figsize=(11, 4))
    by_year = descriptive_table[descriptive_table["variable"] == "issue_year"]
    axis_by_year.bar(by_year["level"].astype(str), by_year["lgd_mean"], color=COLOR_OBSERVED)
    axis_by_year.set_xlabel("año de originación")
    axis_by_year.set_ylabel("LGD media observada")
    for bar_position, (mean_lgd, n_loans) in enumerate(zip(by_year["lgd_mean"], by_year["n"])):
        axis_by_year.text(bar_position, mean_lgd, f"n={n_loans}", ha="center", va="bottom", fontsize=7)
    by_purpose = descriptive_table[descriptive_table["variable"] == "purpose"].sort_values("lgd_mean")
    axis_by_purpose.barh(by_purpose["level"], by_purpose["lgd_mean"], color=COLOR_OBSERVED)
    axis_by_purpose.set_xlabel("LGD media observada")
    _save(figure, figures_dir, "fig_lgd_descriptive.png")


def _plot_lgd_predictions(test_predictions, figures_dir):
    # distribución observada vs predicha, un panel por fuente
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    panels = (("lgd_obs", "observada", COLOR_OBSERVED), ("lgd_linear", "regresión lineal", COLOR_LINEAR),
              ("lgd_mlp", "MLP", COLOR_MLP))
    for axis, (column, label, color) in zip(axes, panels):
        axis.hist(test_predictions[column], bins=40, range=(0, 1), color=color, density=True)
        axis.set_title(f"LGD {label} (prueba)")
        axis.set_xlabel("LGD")
    axes[0].set_ylabel("densidad")
    _save(figure, figures_dir, "fig_lgd_distributions.png")

    # distribuciones superpuestas
    figure, axis = plt.subplots(figsize=(6, 4))
    histogram_bins = np.linspace(0, 1, 41)
    axis.hist(test_predictions["lgd_obs"], bins=histogram_bins, density=True, alpha=0.5, color=COLOR_OBSERVED,
              label="observada")
    axis.hist(test_predictions["lgd_linear"], bins=histogram_bins, histtype="step", density=True, lw=1.8,
              color=COLOR_LINEAR, label="lineal")
    axis.hist(test_predictions["lgd_mlp"], bins=histogram_bins, histtype="step", density=True, lw=1.8,
              color=COLOR_MLP, label="MLP")
    axis.set_xlabel("LGD")
    axis.set_ylabel("densidad")
    axis.legend()
    _save(figure, figures_dir, "fig_lgd_overlay.png")

    # calibración: media observada por decil de la predicción
    figure, axis = plt.subplots(figsize=(5, 5))
    for column, label, color in (("lgd_linear", "lineal", COLOR_LINEAR), ("lgd_mlp", "MLP", COLOR_MLP)):
        prediction_decile = pd.qcut(test_predictions[column].rank(method="first"), 10, labels=False)
        mean_by_decile = test_predictions.groupby(prediction_decile).agg(
            predicted=(column, "mean"), observed=("lgd_obs", "mean"))
        axis.plot(mean_by_decile["predicted"], mean_by_decile["observed"], "o-", label=label, color=color)
    minimum_observed, maximum_observed = test_predictions["lgd_obs"].min(), test_predictions["lgd_obs"].max()
    axis.plot([minimum_observed, maximum_observed], [minimum_observed, maximum_observed], "k--", lw=0.8)
    axis.set_xlabel("LGD predicha (media por decil)")
    axis.set_ylabel("LGD observada (media por decil)")
    axis.legend()
    _save(figure, figures_dir, "fig_lgd_calibration.png")


def _plot_seed_variability(metrics_per_seed, lgd_summary, figures_dir):
    figure, axes = plt.subplots(1, 2, figsize=(7, 4))
    for axis, metric_column in zip(axes, ("test_rmse", "test_mae")):
        axis.boxplot(metrics_per_seed[metric_column])
        axis.scatter(np.ones(len(metrics_per_seed)), metrics_per_seed[metric_column], alpha=0.5)
        linear_reference_value = lgd_summary["linear"]["metrics"]["test"][metric_column.split("_")[1]]
        axis.axhline(linear_reference_value, color="r", ls="--", label="lineal")
        axis.set_xticks([1])
        axis.set_xticklabels([f"MLP ({len(metrics_per_seed)} semillas)"])
        axis.set_ylabel(metric_column)
        axis.legend()
    _save(figure, figures_dir, "fig_lgd_seeds.png")


def _plot_search_trials(search_trials, figures_dir):
    figure, axis = plt.subplots(figsize=(5, 4))
    scatter = axis.scatter(search_trials["n_params"], search_trials["val_rmse"],
                           c=np.log10(search_trials["lr"]), cmap="viridis")
    figure.colorbar(scatter, ax=axis, label="log10(tasa de aprendizaje)")
    axis.set_xscale("log")
    axis.set_xlabel("parámetros entrenables")
    axis.set_ylabel("RMSE de validación")
    axis.set_title(f"Búsqueda aleatoria ({len(search_trials)} pruebas)")
    _save(figure, figures_dir, "fig_lgd_search.png")


def _plot_loss_ablation(ablation_results, figures_dir):
    figure, axis = plt.subplots(figsize=(4.5, 4))
    rmse_by_loss = [ablation_results.loc[ablation_results["loss"] == loss_name, "test_rmse"]
                    for loss_name in ("bce", "mse")]
    axis.boxplot(rmse_by_loss, tick_labels=["BCE", "MSE"])
    axis.set_ylabel("RMSE de prueba")
    axis.set_title("Función de pérdida (misma arquitectura y semillas)")
    _save(figure, figures_dir, "fig_lgd_loss_ablation.png")


def _plot_expected_loss_deciles(decile_table, figures_dir):
    figure, axis = plt.subplots(figsize=(6.5, 4))
    bar_width, decile_positions = 0.28, np.arange(len(decile_table))
    axis.bar(decile_positions - bar_width, decile_table["realized"], bar_width, label="pérdida realizada")
    axis.bar(decile_positions, decile_table["el_mlp"], bar_width, label="E[L] (MLP + MLP)")
    axis.bar(decile_positions + bar_width, decile_table["el_ref"], bar_width, label="E[L] (logística + lineal)")
    axis.set_xticks(decile_positions)
    axis.set_xticklabels(decile_table["decile"])
    axis.set_xlabel("decil de PD (MLP)")
    axis.set_ylabel("USD (suma en el decil)")
    axis.legend()
    _save(figure, figures_dir, "fig_expected_loss_deciles.png")


def make_all(cfg):
    output_dir, figures_dir = config.path(cfg, "outputs"), config.path(cfg, "figures")

    if (output_dir / "lgd_descriptive.csv").exists():
        _plot_descriptive_lgd(pd.read_csv(output_dir / "lgd_descriptive.csv"), figures_dir)

    if (output_dir / "lgd_predictions.csv").exists():
        all_predictions = pd.read_csv(output_dir / "lgd_predictions.csv")
        _plot_lgd_predictions(all_predictions[all_predictions["part"] == "test"], figures_dir)

    if (output_dir / "lgd_mlp_seeds.csv").exists() and (output_dir / "lgd_summary.json").exists():
        _plot_seed_variability(pd.read_csv(output_dir / "lgd_mlp_seeds.csv"),
                               json.loads((output_dir / "lgd_summary.json").read_text("utf-8")), figures_dir)

    if (output_dir / "lgd_search_trials.csv").exists():
        _plot_search_trials(pd.read_csv(output_dir / "lgd_search_trials.csv"), figures_dir)

    if (output_dir / "lgd_loss_ablation.csv").exists():
        _plot_loss_ablation(pd.read_csv(output_dir / "lgd_loss_ablation.csv"), figures_dir)

    if (output_dir / "expected_loss_deciles.csv").exists():
        _plot_expected_loss_deciles(pd.read_csv(output_dir / "expected_loss_deciles.csv"), figures_dir)
