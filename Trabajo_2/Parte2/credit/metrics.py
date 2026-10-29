"""Métricas de evaluación (definiciones en la sección matemática del reporte)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc(observed_labels, predicted_probabilities):
    """AUC por la relación con el estadístico U de Mann-Whitney (con rangos promedio para empates)."""
    observed_labels = np.asarray(observed_labels)
    predicted_probabilities = np.asarray(predicted_probabilities)
    n_positives = observed_labels.sum()
    n_negatives = len(observed_labels) - n_positives
    if n_positives == 0 or n_negatives == 0:
        return float("nan")
    ranks = rankdata(predicted_probabilities)
    return float((ranks[observed_labels == 1].sum() - n_positives * (n_positives + 1) / 2)
                 / (n_positives * n_negatives))


def gini(observed_labels, predicted_probabilities):
    return 2 * auc(observed_labels, predicted_probabilities) - 1


def ks(observed_labels, predicted_probabilities):
    """KS = max_t |F_malos(t) - F_buenos(t)| sobre los puntajes de probabilidad."""
    observed_labels = np.asarray(observed_labels)
    predicted_probabilities = np.asarray(predicted_probabilities)
    labels_sorted_by_score = observed_labels[np.argsort(predicted_probabilities)]
    cumulative_share_of_bad = np.cumsum(labels_sorted_by_score) / max(labels_sorted_by_score.sum(), 1)
    cumulative_share_of_good = (np.cumsum(1 - labels_sorted_by_score)
                                / max((1 - labels_sorted_by_score).sum(), 1))
    return float(np.max(np.abs(cumulative_share_of_bad - cumulative_share_of_good)))


def brier(observed_labels, predicted_probabilities):
    return float(np.mean((np.asarray(predicted_probabilities) - np.asarray(observed_labels)) ** 2))


def calibration_curve(observed_labels, predicted_probabilities, n_bins=10):
    """Bins por cuantiles de la predicción: (prob. media predicha, frecuencia observada, n)."""
    observed_labels = np.asarray(observed_labels)
    predicted_probabilities = np.asarray(predicted_probabilities)
    bin_edges = np.unique(np.quantile(predicted_probabilities, np.linspace(0, 1, n_bins + 1)))
    bin_of_each_row = np.clip(np.searchsorted(bin_edges, predicted_probabilities, side="right") - 1,
                              0, len(bin_edges) - 2)
    mean_predicted, observed_frequency, rows_per_bin = [], [], []
    for bin_index in range(len(bin_edges) - 1):
        in_bin = bin_of_each_row == bin_index
        if in_bin.any():
            mean_predicted.append(predicted_probabilities[in_bin].mean())
            observed_frequency.append(observed_labels[in_bin].mean())
            rows_per_bin.append(int(in_bin.sum()))
    return np.array(mean_predicted), np.array(observed_frequency), np.array(rows_per_bin)


def classification_metrics(observed_labels, predicted_probabilities):
    return {"auc": auc(observed_labels, predicted_probabilities),
            "gini": gini(observed_labels, predicted_probabilities),
            "ks": ks(observed_labels, predicted_probabilities),
            "brier": brier(observed_labels, predicted_probabilities)}


def regression_metrics(observed_values, predicted_values):
    observed_values = np.asarray(observed_values)
    predicted_values = np.asarray(predicted_values)
    errors = observed_values - predicted_values
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2)))}


def summarize(metrics_per_seed):
    """metrics_per_seed: lista de dicts de métricas (una por semilla) -> media, desv., mejor, peor."""
    metrics_table = pd.DataFrame(metrics_per_seed)
    summary = {}
    higher_is_better_suffixes = ("auc", "gini", "ks")
    for column in metrics_table.columns:
        if column == "seed" or not np.issubdtype(metrics_table[column].dtype, np.number):
            continue
        if column.endswith(higher_is_better_suffixes):
            best_value, worst_value = metrics_table[column].max(), metrics_table[column].min()
        else:
            best_value, worst_value = metrics_table[column].min(), metrics_table[column].max()
        summary[column] = {
            "mean": metrics_table[column].mean(),
            "std": metrics_table[column].std(ddof=1) if len(metrics_table) > 1 else 0.0,
            "best": best_value, "worst": worst_value}
    return summary
