"""Logit fraccional (Papke y Wooldridge, 1996): modelo de referencia de baja complejidad para la LGD.

LGD_predicha = sigmoide(X w + b), estimado minimizando la misma cuasi-verosimilitud Bernoulli que la red:
    J(w, b) = -(1/n) sum[ y log p + (1-y) log(1-p) ] + (lambda/2) ||w||^2       (el sesgo no se penaliza)
    dJ/dw = X^T (p - y) / n + lambda w ;   dJ/db = sum(p - y) / n
Es una red SIN capas ocultas, pero el problema es convexo, así que se resuelve hasta convergencia con
L-BFGS (lote completo) en lugar de un número fijo de épocas de Adam: entrenarlo con épocas y lotes lo
dejaba sin converger y sesgaba la comparación a favor de la red (ver docs/caceria_alucinacion.md).
"""
import numpy as np
from pdmodel.mlp import bce_with_logits, sigmoid
from scipy.optimize import minimize


class LogitFraccional:
    def __init__(self, l2=0.0, max_iteraciones=2000, tolerancia=1e-10):
        self.l2, self.max_iteraciones, self.tolerancia = l2, max_iteraciones, tolerancia

    def _perdida_y_gradiente(self, parametros, variables, objetivo):
        pesos, sesgo = parametros[:-1], parametros[-1]
        logit = variables @ pesos + sesgo
        error = (sigmoid(logit) - objetivo) / len(objetivo)
        perdida = bce_with_logits(logit, objetivo) + 0.5 * self.l2 * float(pesos @ pesos)
        gradiente = np.append(variables.T @ error + self.l2 * pesos, error.sum())
        return perdida, gradiente

    def fit(self, variables, objetivo):
        variables = np.asarray(variables, dtype=np.float64)
        objetivo = np.asarray(objetivo, dtype=np.float64)
        media_objetivo = float(np.clip(objetivo.mean(), 1e-4, 1 - 1e-4))
        parametros_iniciales = np.append(np.zeros(variables.shape[1]), np.log(media_objetivo / (1 - media_objetivo)))
        resultado = minimize(self._perdida_y_gradiente, parametros_iniciales, args=(variables, objetivo), jac=True,
                             method="L-BFGS-B", options={"maxiter": self.max_iteraciones, "gtol": self.tolerancia})
        self.coef_, self.intercept_ = resultado.x[:-1], float(resultado.x[-1])
        self.convergio_, self.iteraciones_ = bool(resultado.success), int(resultado.nit)
        return self

    def predict_proba(self, variables):
        return sigmoid(np.asarray(variables, dtype=np.float64) @ self.coef_ + self.intercept_)

    @property
    def n_params(self):
        return int(self.coef_.size + 1)
