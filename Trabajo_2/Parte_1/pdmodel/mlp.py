"""Perceptrón multicapa para clasificación binaria, implementado en NumPy desde cero.

Se implementa a mano (forward, retropropagación, Adam / SGD con momentum, dropout, L2 y parada
temprana) para que cada ecuación del reporte tenga su línea de código correspondiente.
La corrección de los gradientes se verifica en tests/test_mlp.py con diferencias finitas.

Modelo:  a^0 = x;  z^l = a^{l-1} W^l + b^l;  h^l = g(z^l);  a^l = dropout(h^l)
         logit = a^L w + b;   p = sigmoid(logit) = P(y = 1 | x)   (1 = cliente malo)
Pérdida: J = BCE_media + (lambda/2) * sum_l ||W^l||_F^2   (los sesgos no se penalizan)
"""
import copy
import time

import numpy as np


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # estable para |z| grande


def bce_with_logits(z, y):
    """-mean[y log p + (1-y) log(1-p)] calculado de forma estable a partir del logit z."""
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


# nombre -> (g(z), g'(z) en función de z y de a = g(z))
_ACT = {
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(z.dtype)),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.01 * z),
                   lambda z, a: np.where(z > 0, 1.0, 0.01).astype(z.dtype)),
    "tanh": (np.tanh, lambda z, a: 1.0 - a * a),
}


class MLPClassifier:
    def __init__(self, hidden=(64, 32), activation="relu", dropout=0.0, l2=0.0, optimizer="adam",
                 lr=1e-3, batch_size=1024, momentum=0.9, dtype=np.float32, seed=0):
        if activation not in _ACT:
            raise ValueError(f"activation debe estar en {list(_ACT)}")
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer debe ser 'adam' o 'sgd'")
        self.hidden, self.activation, self.dropout, self.l2 = tuple(hidden), activation, dropout, l2
        self.optimizer, self.lr, self.batch_size, self.momentum = optimizer, lr, batch_size, momentum
        self.dtype, self.seed = dtype, seed
        self.W, self.b = None, None

    # ---------- parámetros ----------
    def _init_params(self, n_in, base_rate, rng):
        sizes = [n_in, *self.hidden, 1]
        self.W, self.b = [], []
        for i, (fi, fo) in enumerate(zip(sizes[:-1], sizes[1:])):
            hidden_layer = i < len(sizes) - 2
            if hidden_layer and self.activation in ("relu", "leaky_relu"):
                std = np.sqrt(2.0 / fi)              # He
            else:
                std = np.sqrt(2.0 / (fi + fo))       # Glorot
            self.W.append((rng.randn(fi, fo) * std).astype(self.dtype))
            self.b.append(np.zeros(fo, dtype=self.dtype))
        r = float(np.clip(base_rate, 1e-4, 1 - 1e-4))
        self.b[-1][:] = np.log(r / (1 - r))          # arranca prediciendo la tasa base

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def _params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    # ---------- forward / backward ----------
    def _forward(self, X, train, rng=None):
        g = _ACT[self.activation][0]
        a, cache = X, []
        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            h = g(z)
            mask = None
            if train and self.dropout > 0:
                mask = (rng.random_sample(h.shape) >= self.dropout).astype(h.dtype) / (1.0 - self.dropout)
                out = h * mask
            else:
                out = h
            cache.append((a, z, h, mask))
            a = out
        logit = (a @ self.W[-1] + self.b[-1]).ravel()
        return logit, cache, a

    def _loss_grads(self, X, y, rng):
        """Pérdida total (BCE + L2) y gradientes por retropropagación."""
        dg = _ACT[self.activation][1]
        n, L = len(y), len(self.W)
        z, cache, a_last = self._forward(X, True, rng)
        loss = bce_with_logits(z, y)
        if self.l2 > 0:
            loss += 0.5 * self.l2 * sum(float((w.astype(np.float64) ** 2).sum()) for w in self.W)
        gW, gb = [None] * L, [None] * L
        d = ((sigmoid(z) - y) / n).reshape(-1, 1).astype(self.dtype)   # dJ/dlogit = (p - y)/n
        gW[-1] = a_last.T @ d + self.l2 * self.W[-1]
        gb[-1] = d.sum(axis=0)
        da = d @ self.W[-1].T
        for l in range(L - 2, -1, -1):
            a_in, zl, h, mask = cache[l]
            if mask is not None:
                da = da * mask                       # retropropagar a través del dropout
            dz = da * dg(zl, h)
            gW[l] = a_in.T @ dz + self.l2 * self.W[l]
            gb[l] = dz.sum(axis=0)
            if l > 0:
                da = dz @ self.W[l].T
        grads = [g for pair in zip(gW, gb) for g in pair]
        return loss, grads

    # ---------- optimizadores ----------
    def _init_opt(self):
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _step(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        params = self._params()
        if self.optimizer == "adam":
            self._t += 1
            c1, c2 = 1 - b1 ** self._t, 1 - b2 ** self._t
            for p, g, m, v in zip(params, grads, self._m, self._v):
                m *= b1
                m += (1 - b1) * g
                v *= b2
                v += (1 - b2) * g * g
                p -= self.lr * (m / c1) / (np.sqrt(v / c2) + eps)
        else:  # SGD con momentum
            for p, g, v in zip(params, grads, self._m):
                v *= self.momentum
                v += g
                p -= self.lr * v

    # ---------- API pública ----------
    def logit(self, X, batch=100_000):
        X = np.asarray(X, dtype=self.dtype)
        out = [self._forward(X[i:i + batch], False)[0] for i in range(0, len(X), batch)]
        return np.concatenate(out).astype(np.float64)

    def predict_proba(self, X):
        return sigmoid(self.logit(X))

    def fit(self, Xtr, ytr, Xva, yva, max_epochs=50, patience=5, verbose=False):
        t0 = time.time()
        rng = np.random.RandomState(self.seed)
        Xtr, Xva = np.ascontiguousarray(Xtr, dtype=self.dtype), np.ascontiguousarray(Xva, dtype=self.dtype)
        ytr, yva = np.asarray(ytr, dtype=self.dtype), np.asarray(yva, dtype=self.dtype)
        self._init_params(Xtr.shape[1], float(ytr.mean()), rng)
        self._init_opt()
        n = len(ytr)
        best, best_state, bad, self.history_ = np.inf, None, 0, []
        for epoch in range(1, max_epochs + 1):
            perm = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                loss, grads = self._loss_grads(Xtr[idx], ytr[idx], rng)
                self._step(grads)
                tot += loss * len(idx)
            val = bce_with_logits(self.logit(Xva), yva.astype(np.float64))
            self.history_.append({"epoch": epoch, "train_loss": tot / n, "val_logloss": val})
            if verbose:
                print(f"  época {epoch:3d}  train={tot / n:.4f}  val_logloss={val:.4f}")
            if not np.isfinite(val):
                break
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = (copy.deepcopy(self.W), copy.deepcopy(self.b))
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:                    # parada temprana: restaurar mejores pesos
            self.W, self.b = best_state
        self.best_epoch_ = int(np.argmin([h["val_logloss"] for h in self.history_]) + 1)
        self.fit_seconds_ = time.time() - t0
        return self
