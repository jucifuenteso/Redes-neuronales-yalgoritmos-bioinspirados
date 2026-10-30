"""Pruebas unitarias (unittest, sin dependencias extra):  python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from pdmodel.data import build_target
from pdmodel.features import Preprocessor
from pdmodel.metrics import auc_score, brier_score, gini_score, ks_score
from pdmodel.mlp import MLPClassifier, bce_with_logits
from pdmodel.scorecard import ScoreScale


class TestTarget(unittest.TestCase):
    def test_coding_matches_assignment(self):
        s = pd.Series(["Fully Paid", "Does not meet the credit policy. Status:Fully Paid", "Charged Off", "Default",
                       "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off", "Current",
                       "Issued", "In Grace Period", "Late (16-30 days)"])
        y = build_target(s)
        self.assertEqual(y.iloc[:6].tolist(), [0, 0, 1, 1, 1, 1])
        self.assertTrue(y.iloc[6:].isna().all())


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.y = (rng.rand(5000) < 0.2).astype(int)
        self.p = np.clip(0.2 + 0.3 * (self.y - 0.2) + rng.normal(0, 0.15, 5000), 0.001, 0.999)
        self.p = np.round(self.p, 2)  # fuerza empates

    def test_auc_gini_brier_vs_sklearn(self):
        self.assertAlmostEqual(auc_score(self.y, self.p), roc_auc_score(self.y, self.p), places=10)
        self.assertAlmostEqual(gini_score(self.y, self.p), 2 * roc_auc_score(self.y, self.p) - 1, places=10)
        self.assertAlmostEqual(brier_score(self.y, self.p), brier_score_loss(self.y, self.p), places=10)

    def test_ks_vs_roc(self):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.y, self.p)
        self.assertAlmostEqual(ks_score(self.y, self.p), float(np.max(tpr - fpr)), places=10)


class TestMLPGradients(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 5)
        y = (rng.rand(40) < 0.4).astype(np.float64)
        for act in ["relu", "tanh", "leaky_relu"]:
            m = MLPClassifier(hidden=(6, 4), activation=act, dropout=0.0, l2=0.01, dtype=np.float64, seed=3)
            m._init_params(5, 0.4, np.random.RandomState(3))
            _, grads = m._loss_grads(X, y, np.random.RandomState(0))
            params = m._params()
            eps = 1e-6
            for p, g in zip(params, grads):
                flat_idx = rng.choice(p.size, size=min(5, p.size), replace=False)
                for fi in flat_idx:
                    idx = np.unravel_index(fi, p.shape)
                    old = p[idx]
                    p[idx] = old + eps
                    lp = m._loss_grads(X, y, np.random.RandomState(0))[0]
                    p[idx] = old - eps
                    lm = m._loss_grads(X, y, np.random.RandomState(0))[0]
                    p[idx] = old
                    num = (lp - lm) / (2 * eps)
                    self.assertAlmostEqual(g[idx], num, places=5, msg=f"{act}")

    def test_learns_a_simple_signal(self):
        rng = np.random.RandomState(0)
        X = rng.randn(4000, 3)
        y = (rng.rand(4000) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1])))).astype(float)
        m = MLPClassifier(hidden=(8,), lr=0.01, seed=0).fit(X[:3000], y[:3000], X[3000:], y[3000:], 30, 5)
        self.assertGreater(auc_score(y[3000:], m.predict_proba(X[3000:])), 0.8)

    def test_bce_stable(self):
        self.assertTrue(np.isfinite(bce_with_logits(np.array([800.0, -800.0]), np.array([0.0, 1.0]))))


class TestScale(unittest.TestCase):
    def test_base_and_pdo(self):
        sc = ScoreScale(600, 50, 20)
        odds = lambda pd_: (1 - pd_) / pd_
        p_base = 1 / (1 + 50)                                   # odds buenos:malos = 50:1
        self.assertAlmostEqual(float(sc.score(p_base)), 600, places=6)
        p_double = 1 / (1 + 100)                                # odds duplicadas
        self.assertAlmostEqual(float(sc.score(p_double)) - 600, 20, places=6)
        self.assertAlmostEqual(float(sc.pd_from_score(sc.score(0.13))), 0.13, places=9)


class TestPreprocessorNoLeak(unittest.TestCase):
    def test_stats_come_from_train_only(self):
        tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "c": ["x", "x", "y", "y"]})
        te = pd.DataFrame({"a": [1000.0, np.nan], "c": ["z", "x"]})
        pre = Preprocessor(["a"], ["c"], (0.0, 1.0), 0.0).fit(tr)
        Xt = pre.transform(te)
        self.assertEqual(Xt.shape[0], 2)
        self.assertAlmostEqual(float(pre.mean_[0]), 2.5)           # media de train, no de test
        self.assertLessEqual(Xt[0, 0], (4.0 - 2.5) / pre.std_[0] + 1e-6)  # winsorizado al máximo de train


if __name__ == "__main__":
    unittest.main()
