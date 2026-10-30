"""Une preprocesamiento + modelo para predecir directamente desde la tabla de solicitud."""
import numpy as np

from pdmodel.mlp import sigmoid


class SkLogit:
    """Adaptador de sklearn.LogisticRegression a la interfaz logit()/n_params."""

    def __init__(self, sk_model):
        self.sk = sk_model

    def logit(self, X):
        return self.sk.decision_function(X).astype(np.float64)

    def predict_proba(self, X):
        return sigmoid(self.logit(X))

    @property
    def n_params(self):
        return int(self.sk.coef_.size + np.size(self.sk.intercept_))


class PDPipeline:
    def __init__(self, pre, model, name):
        self.pre, self.model, self.name = pre, model, name

    def predict_logit(self, df):
        return self.model.logit(self.pre.transform(df))

    def predict_proba(self, df):
        return sigmoid(self.predict_logit(df))

    @property
    def n_params(self):
        return self.model.n_params
