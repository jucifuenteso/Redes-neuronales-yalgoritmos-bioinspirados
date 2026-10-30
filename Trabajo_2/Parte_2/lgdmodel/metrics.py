"""Métricas de la regresión de la LGD y comparación de distribuciones (predicha frente a observada)."""
import numpy as np
import pandas as pd
from scipy import stats

CUANTILES_REPORTADOS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def metricas_regresion(observado, predicho):
    """MAE = (1/n) sum |y - y_hat| ;  RMSE = sqrt( (1/n) sum (y - y_hat)^2 )."""
    errores = np.asarray(observado, dtype=float) - np.asarray(predicho, dtype=float)
    return {"mae": float(np.mean(np.abs(errores))), "rmse": float(np.sqrt(np.mean(errores ** 2)))}


def describir_distribucion(valores, valores_de_referencia=None):
    valores = np.asarray(valores, dtype=float)
    descripcion = {"media": float(valores.mean()), "desv_est": float(valores.std()),
                   **{f"q{int(round(nivel * 100)):02d}": float(np.quantile(valores, nivel))
                      for nivel in CUANTILES_REPORTADOS},
                   "fraccion_mayor_igual_099": float((valores >= 0.99).mean()),
                   "fraccion_menor_igual_001": float((valores <= 0.01).mean())}
    if valores_de_referencia is not None:
        # KS de dos muestras: máxima distancia vertical entre las dos funciones de distribución empíricas.
        descripcion["ks_dos_muestras_vs_observada"] = float(stats.ks_2samp(valores, valores_de_referencia).statistic)
        # Wasserstein-1: área entre las dos funciones de distribución empíricas (en unidades de LGD).
        descripcion["wasserstein_vs_observada"] = float(stats.wasserstein_distance(valores, valores_de_referencia))
    return descripcion


def comparar_distribuciones(lgd_observada, predicciones_por_modelo):
    lgd_observada = np.asarray(lgd_observada, dtype=float)
    comparacion = {"observada": describir_distribucion(lgd_observada)}
    for nombre_modelo, predicciones in predicciones_por_modelo.items():
        comparacion[nombre_modelo] = describir_distribucion(predicciones, lgd_observada)
    return comparacion


def calibracion_por_deciles(observado, predicho, n_grupos=10):
    """Media observada frente a media predicha en grupos de igual tamaño ordenados por la predicción."""
    tabla = pd.DataFrame({"observado": np.asarray(observado, dtype=float), "predicho": np.asarray(predicho, dtype=float)})
    tabla["grupo"] = pd.qcut(tabla["predicho"].rank(method="first"), n_grupos, labels=False)
    return (tabla.groupby("grupo").agg(prediccion_media=("predicho", "mean"), observado_medio=("observado", "mean"),
                                       n=("observado", "size")).reset_index())


def resumir_semillas(tabla_por_semilla, metricas=("mae", "rmse"), particiones=("val", "test")):
    """Media, desviación estándar, mejor y peor valor de cada métrica sobre las semillas (menor es mejor)."""
    filas_resumen = []
    for particion in particiones:
        for metrica in metricas:
            valores = tabla_por_semilla[f"{particion}_{metrica}"]
            filas_resumen.append({"particion": particion, "metrica": metrica, "media": valores.mean(),
                                  "desv_est": valores.std(ddof=1) if len(valores) > 1 else 0.0,
                                  "mejor": valores.min(), "peor": valores.max()})
    return pd.DataFrame(filas_resumen)
