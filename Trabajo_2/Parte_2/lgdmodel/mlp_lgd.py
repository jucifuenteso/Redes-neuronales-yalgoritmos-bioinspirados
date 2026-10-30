"""Perceptrón multicapa para la LGD: el MLP en NumPy de la Parte 1 con la pérdida generalizada.

La LGD es continua y vive en [0, 1], así que la salida es la misma de la Parte 1: un logit z y
LGD_predicha = sigmoide(z), que garantiza predicciones dentro del soporte del objetivo.
Lo único que cambia es la pérdida y, con ella, el error que se retropropaga desde el logit:

  perdida='bce'  J = -(1/n) sum[ y log p + (1-y) log(1-p) ]   =>  dJ/dz = (p - y) / n
                 Cuasi-verosimilitud Bernoulli: admite y fraccional (Papke y Wooldridge, 1996).
                 Es la misma ecuación y el mismo código que la PD de la Parte 1.
  perdida='mse'  J = (1/n) sum (p - y)^2                        =>  dJ/dz = 2 (p - y) p (1 - p) / n
                 Verosimilitud gaussiana con varianza constante sobre p = sigmoide(z).

A ambas se suma (lambda/2) sum_l ||W^l||^2 (L2, sin penalizar sesgos). Forward, dropout, Adam/SGD e
inicialización se heredan sin cambios de pdmodel.mlp.MLPClassifier.
"""
import copy
import time

import numpy as np
from pdmodel.mlp import _ACT as ACTIVACIONES_PARTE1
from pdmodel.mlp import MLPClassifier, bce_with_logits, sigmoid

PERDIDAS_DISPONIBLES = ("bce", "mse")


def error_cuadratico_medio_desde_logit(logit, objetivo):
    return float(np.mean((sigmoid(logit) - objetivo) ** 2))


class MLPLGD(MLPClassifier):
    def __init__(self, perdida="bce", **hiperparametros):
        if perdida not in PERDIDAS_DISPONIBLES:
            raise ValueError(f"perdida debe estar en {PERDIDAS_DISPONIBLES}")
        super().__init__(**hiperparametros)
        self.perdida = perdida

    # ---------- pérdida y su derivada respecto al logit ----------
    def calcular_perdida(self, logit, objetivo):
        if self.perdida == "bce":
            return bce_with_logits(logit, objetivo)
        return error_cuadratico_medio_desde_logit(logit, objetivo)

    def gradiente_respecto_al_logit(self, logit, objetivo):
        prediccion = sigmoid(logit)
        n_filas = len(objetivo)
        if self.perdida == "bce":
            return (prediccion - objetivo) / n_filas
        return 2.0 * (prediccion - objetivo) * prediccion * (1.0 - prediccion) / n_filas

    # ---------- retropropagación (misma estructura que MLPClassifier._loss_grads) ----------
    def _loss_grads(self, X, y, rng):
        derivada_activacion = ACTIVACIONES_PARTE1[self.activation][1]
        n_capas = len(self.W)
        logit, cache_por_capa, activacion_ultima_oculta = self._forward(X, True, rng)
        perdida_total = self.calcular_perdida(logit, y)
        if self.l2 > 0:
            perdida_total += 0.5 * self.l2 * sum(float((pesos.astype(np.float64) ** 2).sum()) for pesos in self.W)
        gradientes_pesos, gradientes_sesgos = [None] * n_capas, [None] * n_capas
        delta = self.gradiente_respecto_al_logit(logit, y).reshape(-1, 1).astype(self.dtype)
        gradientes_pesos[-1] = activacion_ultima_oculta.T @ delta + self.l2 * self.W[-1]
        gradientes_sesgos[-1] = delta.sum(axis=0)
        gradiente_activacion = delta @ self.W[-1].T
        for capa in range(n_capas - 2, -1, -1):
            entrada_capa, preactivacion, activacion, mascara_dropout = cache_por_capa[capa]
            if mascara_dropout is not None:
                gradiente_activacion = gradiente_activacion * mascara_dropout
            delta_capa = gradiente_activacion * derivada_activacion(preactivacion, activacion)
            gradientes_pesos[capa] = entrada_capa.T @ delta_capa + self.l2 * self.W[capa]
            gradientes_sesgos[capa] = delta_capa.sum(axis=0)
            if capa > 0:
                gradiente_activacion = delta_capa @ self.W[capa].T
        gradientes = [gradiente for par in zip(gradientes_pesos, gradientes_sesgos) for gradiente in par]
        return perdida_total, gradientes

    # ---------- entrenamiento con parada temprana sobre la MISMA pérdida en validación ----------
    def fit(self, Xtr, ytr, Xva, yva, max_epochs=50, patience=5, verbose=False):
        """Igual que MLPClassifier.fit, pero la pérdida de validación es la elegida (bce o mse)."""
        inicio = time.time()
        generador_aleatorio = np.random.RandomState(self.seed)
        variables_entrenamiento = np.ascontiguousarray(Xtr, dtype=self.dtype)
        variables_validacion = np.ascontiguousarray(Xva, dtype=self.dtype)
        objetivo_entrenamiento = np.asarray(ytr, dtype=self.dtype)
        objetivo_validacion = np.asarray(yva, dtype=np.float64)
        self._init_params(variables_entrenamiento.shape[1], float(objetivo_entrenamiento.mean()), generador_aleatorio)
        self._init_opt()
        n_filas = len(objetivo_entrenamiento)
        mejor_perdida_validacion, mejores_pesos, epocas_sin_mejora = np.inf, None, 0
        self.history_ = []
        for epoca in range(1, max_epochs + 1):
            permutacion = generador_aleatorio.permutation(n_filas)
            perdida_acumulada = 0.0
            for inicio_lote in range(0, n_filas, self.batch_size):
                filas_lote = permutacion[inicio_lote:inicio_lote + self.batch_size]
                perdida_lote, gradientes = self._loss_grads(variables_entrenamiento[filas_lote],
                                                            objetivo_entrenamiento[filas_lote], generador_aleatorio)
                self._step(gradientes)
                perdida_acumulada += perdida_lote * len(filas_lote)
            perdida_validacion = self.calcular_perdida(self.logit(variables_validacion), objetivo_validacion)
            self.history_.append({"epoca": epoca, "perdida_entrenamiento": perdida_acumulada / n_filas,
                                  "perdida_validacion": perdida_validacion})
            if verbose:
                print(f"  época {epoca:3d}  entrenamiento={perdida_acumulada / n_filas:.4f}  "
                      f"validación={perdida_validacion:.4f}")
            if not np.isfinite(perdida_validacion):
                break
            if perdida_validacion < mejor_perdida_validacion - 1e-5:   # mismo umbral que la Parte 1
                mejor_perdida_validacion, epocas_sin_mejora = perdida_validacion, 0
                mejores_pesos = (copy.deepcopy(self.W), copy.deepcopy(self.b))
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= patience:
                    break
        if mejores_pesos is not None:                 # parada temprana: restaurar mejores pesos
            self.W, self.b = mejores_pesos
        self.best_epoch_ = int(np.argmin([fila["perdida_validacion"] for fila in self.history_]) + 1)
        self.fit_seconds_ = time.time() - inicio
        return self
