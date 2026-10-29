import numpy as np
import pandas as pd
import pytest

from credit import metrics
from credit.data import build_lgd
from credit.experiments import distribution_report


def test_lgd_formula():
    loans = pd.DataFrame({"funded_amnt": [1000., 1000., 1000., 1000.],
                       "total_rec_prncp": [0., 400., 1000., 0.],
                       "recoveries": [0., 100., 50., 50.],
                       "collection_recovery_fee": [0., 10., 0., 80.]})
    # LGD = 1 - (principal recuperado + max(recoveries - fee, 0)) / funded_amnt, recortada a [0, 1]
    assert np.allclose(build_lgd(loans), [1.0, 1 - 490 / 1000, 0.0, 1.0])


def test_lgd_in_unit_interval_with_odd_values():
    loans = pd.DataFrame({"funded_amnt": [1000., 1000.], "total_rec_prncp": [1200., np.nan],
                       "recoveries": [np.nan, 5000.], "collection_recovery_fee": [0., 0.]})
    lgd_values = build_lgd(loans)
    assert ((lgd_values >= 0) & (lgd_values <= 1)).all()


def test_regression_metrics():
    observed = np.array([0., 0.5, 1.0])
    result = metrics.regression_metrics(observed, np.array([0.1, 0.5, 0.7]))
    assert result["mae"] == pytest.approx((0.1 + 0 + 0.3) / 3)
    assert result["rmse"] == pytest.approx(np.sqrt((0.01 + 0 + 0.09) / 3))
    assert metrics.regression_metrics(observed, observed) == {"mae": 0.0, "rmse": 0.0}


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    random_generator = np.random.default_rng(0)
    labels = random_generator.integers(0, 2, 500)
    scores = random_generator.random(500) + 0.3 * labels
    assert metrics.auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_numpy_mlp_matches_torch(tmp_path):
    pytest.importorskip("torch")
    from credit import nn as neural_network
    for activation_name in ("relu", "tanh", "elu", "silu"):
        neural_network.set_seed(0)
        torch_model = neural_network.build_mlp(7, [5, 4], activation_name, 0.3).eval()
        neural_network.export_numpy(torch_model, activation_name, tmp_path / "exported_model.npz")
        features = np.random.default_rng(1).normal(size=(20, 7)).astype(np.float32)
        numpy_model = neural_network.NumpyMLP(tmp_path / "exported_model.npz")
        assert np.allclose(neural_network.predict(torch_model, features), numpy_model.predict(features), atol=1e-5)


def test_bce_with_fractional_targets_minimised_at_target():
    """La BCE con objetivo y in [0,1] se minimiza en p = y: es una cuasi-verosimilitud válida (Papke & Wooldridge, 1996)."""
    fractional_target = 0.3
    candidate_probabilities = np.linspace(0.01, 0.99, 981)
    bce_loss = -(fractional_target * np.log(candidate_probabilities)
                 + (1 - fractional_target) * np.log(1 - candidate_probabilities))
    assert candidate_probabilities[np.argmin(bce_loss)] == pytest.approx(fractional_target, abs=1e-3)


def test_distribution_report_identical_distributions():
    observed_values = np.random.default_rng(0).random(1000)
    report = distribution_report(observed_values, {"copia": observed_values.copy()})
    assert report["copia"]["ks_vs_obs"] == 0.0
    assert report["copia"]["wasserstein_vs_obs"] == pytest.approx(0.0)
    assert report["observada"]["mean"] == pytest.approx(observed_values.mean())
