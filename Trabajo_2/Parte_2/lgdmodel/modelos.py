"""Une el preprocesamiento con cada modelo de LGD para predecir directamente desde la tabla de solicitud."""
import numpy as np


class ModeloLGD:
    """Preprocesador (el de la Parte 1, ajustado con train) + modelo de LGD.

    Los modelos con salida sigmoide (logit fraccional y MLP) ya predicen en (0, 1). La regresión lineal
    no tiene esa garantía: sus predicciones se recortan a [0, 1] y eso se declara en el reporte.
    """

    def __init__(self, preprocesador, modelo, nombre):
        self.preprocesador, self.modelo, self.nombre = preprocesador, modelo, nombre

    def predecir_desde_matriz(self, matriz_preprocesada):
        if hasattr(self.modelo, "predict_proba"):
            return np.asarray(self.modelo.predict_proba(matriz_preprocesada), dtype=float)
        return np.clip(np.asarray(self.modelo.predict(matriz_preprocesada), dtype=float), 0.0, 1.0)

    def predecir(self, solicitudes):
        return self.predecir_desde_matriz(self.preprocesador.transform(solicitudes))

    @property
    def n_parametros(self):
        if hasattr(self.modelo, "n_params"):
            return int(self.modelo.n_params)
        return int(self.modelo.coef_.size + np.size(self.modelo.intercept_))
