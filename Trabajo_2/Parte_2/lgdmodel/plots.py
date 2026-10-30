"""Figuras de la Parte 2 (todas regenerables desde reports/tables con `python -m lgdmodel figuras`)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lgdmodel.config import ruta

COLOR_OBSERVADA, COLOR_LINEAL, COLOR_LOGIT, COLOR_RED = "#4C72B0", "#DD8452", "#8172B3", "#55A868"
COLUMNAS_PREDICCION = [("pred_lineal", "Regresión lineal", COLOR_LINEAL),
                       ("pred_logit_fraccional", "Logit fraccional", COLOR_LOGIT),
                       ("pred_red", "Red neuronal", COLOR_RED)]


def _guardar(figura, ruta_archivo):
    figura.tight_layout()
    figura.savefig(ruta_archivo, dpi=140)
    plt.close(figura)
    print("  figura:", ruta_archivo)


def figura_distribucion_lgd_observada(creditos_con_lgd, ruta_archivo):
    figura, (eje_objetivo, eje_alternativa) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    intervalos = np.linspace(0, 1, 41)
    eje_objetivo.hist(creditos_con_lgd["lgd"], bins=intervalos, color=COLOR_OBSERVADA)
    eje_objetivo.set_title("LGD sobre el monto desembolsado (objetivo)")
    eje_alternativa.hist(creditos_con_lgd["lgd_sobre_saldo"].dropna(), bins=intervalos, color="#937860")
    eje_alternativa.set_title("LGD sobre el saldo al castigo (alternativa)")
    for eje in (eje_objetivo, eje_alternativa):
        eje.set_xlabel("LGD")
    eje_objetivo.set_ylabel("número de créditos")
    _guardar(figura, ruta_archivo)


def figura_lgd_por_anio(creditos_con_lgd, ruta_archivo):
    por_anio = creditos_con_lgd.groupby(creditos_con_lgd["issue_d"].dt.year)["lgd"].agg(["mean", "size"])
    figura, eje = plt.subplots(figsize=(6.5, 3.8))
    eje.bar(por_anio.index.astype(str), por_anio["mean"], color=COLOR_OBSERVADA)
    for posicion, (media, n_creditos) in enumerate(zip(por_anio["mean"], por_anio["size"])):
        eje.text(posicion, media, f"n={n_creditos}", ha="center", va="bottom", fontsize=7)
    eje.set_xlabel("año de originación")
    eje.set_ylabel("LGD media observada")
    _guardar(figura, ruta_archivo)


def figura_busqueda(pruebas, metrica, ruta_archivo):
    figura, eje = plt.subplots(figsize=(5.5, 4))
    dispersion = eje.scatter(pruebas["n_parametros"], pruebas[f"val_{metrica}"], c=np.log10(pruebas["lr"]),
                             cmap="viridis")
    figura.colorbar(dispersion, ax=eje, label="log10(tasa de aprendizaje)")
    eje.set_xscale("log")
    eje.set_xlabel("parámetros entrenables")
    eje.set_ylabel(f"{metrica.upper()} de validación")
    eje.set_title(f"Búsqueda aleatoria de la red ({len(pruebas)} configuraciones)")
    _guardar(figura, ruta_archivo)


def figura_semillas(tabla_semillas, referencias, ruta_archivo):
    figura, ejes = plt.subplots(1, 2, figsize=(8, 4))
    for eje, metrica in zip(ejes, ["mae", "rmse"]):
        valores = tabla_semillas[f"test_{metrica}"]
        eje.boxplot(valores)
        eje.scatter(np.ones(len(valores)), valores, alpha=0.5, color=COLOR_RED)
        eje.axhline(referencias["lineal"]["test"][metrica], color=COLOR_LINEAL, ls="--", label="regresión lineal")
        eje.axhline(referencias["logit_fraccional"]["test"][metrica], color=COLOR_LOGIT, ls=":",
                    label="logit fraccional")
        eje.axhline(referencias["constante"]["test"][metrica], color="grey", ls="-.", label="media constante")
        eje.set_xticks([1])
        eje.set_xticklabels([f"Red ({len(valores)} semillas)"])
        eje.set_ylabel(f"{metrica.upper()} en prueba")
        eje.legend(fontsize=7)
    _guardar(figura, ruta_archivo)


def figura_distribucion_predicciones(predicciones_prueba, ruta_archivo):
    figura, (eje_histograma, eje_acumulada) = plt.subplots(1, 2, figsize=(11, 4))
    intervalos = np.linspace(0, 1, 41)
    eje_histograma.hist(predicciones_prueba["lgd_observada"], bins=intervalos, density=True, alpha=0.45,
                        color=COLOR_OBSERVADA, label="observada")
    for columna, etiqueta, color in COLUMNAS_PREDICCION:
        eje_histograma.hist(predicciones_prueba[columna], bins=intervalos, density=True, histtype="step", lw=1.8,
                            color=color, label=etiqueta)
    eje_histograma.set_xlabel("LGD")
    eje_histograma.set_ylabel("densidad")
    eje_histograma.legend(fontsize=8)
    for columna, etiqueta, color in [("lgd_observada", "observada", COLOR_OBSERVADA)] + COLUMNAS_PREDICCION:
        valores_ordenados = np.sort(predicciones_prueba[columna].to_numpy())
        eje_acumulada.plot(valores_ordenados, np.arange(1, len(valores_ordenados) + 1) / len(valores_ordenados),
                           color=color, label=etiqueta)
    eje_acumulada.set_xlabel("LGD")
    eje_acumulada.set_ylabel("distribución acumulada empírica")
    eje_acumulada.legend(fontsize=8)
    _guardar(figura, ruta_archivo)


def figura_calibracion(calibracion, ruta_archivo):
    figura, eje = plt.subplots(figsize=(5, 5))
    for columna, etiqueta, color in COLUMNAS_PREDICCION:
        deciles = calibracion[calibracion["modelo"] == columna]
        eje.plot(deciles["prediccion_media"], deciles["observado_medio"], "o-", color=color, label=etiqueta)
    limite_inferior = min(calibracion["prediccion_media"].min(), calibracion["observado_medio"].min())
    limite_superior = max(calibracion["prediccion_media"].max(), calibracion["observado_medio"].max())
    eje.plot([limite_inferior, limite_superior], [limite_inferior, limite_superior], "k--", lw=0.8)
    eje.set_xlabel("LGD predicha (media por decil)")
    eje.set_ylabel("LGD observada (media por decil)")
    eje.legend(fontsize=8)
    _guardar(figura, ruta_archivo)


def figura_ablacion(tabla_ablacion, ruta_archivo):
    figura, ejes = plt.subplots(1, 2, figsize=(8, 4))
    for eje, metrica in zip(ejes, ["test_mae", "test_rmse"]):
        eje.boxplot([tabla_ablacion.loc[tabla_ablacion["perdida"] == perdida, metrica] for perdida in ["bce", "mse"]],
                    tick_labels=["BCE", "MSE"])
        eje.set_ylabel(metrica.replace("test_", "").upper() + " en prueba")
    figura.suptitle("Misma arquitectura y semillas: pérdida BCE frente a MSE")
    _guardar(figura, ruta_archivo)


def figura_curva_aprendizaje(historia, ruta_archivo):
    figura, eje = plt.subplots(figsize=(5.5, 4))
    eje.plot(historia["epoca"], historia["perdida_entrenamiento"], label="entrenamiento")
    eje.plot(historia["epoca"], historia["perdida_validacion"], label="validación")
    eje.axvline(historia.loc[historia["perdida_validacion"].idxmin(), "epoca"], color="grey", ls="--",
                label="pesos restaurados (parada temprana)")
    eje.set_xlabel("época")
    eje.set_ylabel("pérdida")
    eje.legend(fontsize=8)
    _guardar(figura, ruta_archivo)


def figura_perdida_esperada_deciles(deciles, ruta_archivo):
    figura, eje = plt.subplots(figsize=(7, 4))
    ancho, posiciones = 0.28, np.arange(len(deciles))
    eje.bar(posiciones - ancho, deciles["perdida_realizada"], ancho, label="pérdida realizada", color=COLOR_OBSERVADA)
    eje.bar(posiciones, deciles["perdida_esperada_red"], ancho, label="E[L] red (PD MLP × LGD MLP)", color=COLOR_RED)
    eje.bar(posiciones + ancho, deciles["perdida_esperada_referencia"], ancho,
            label="E[L] referencia (PD logística × LGD lineal)", color=COLOR_LINEAL)
    eje.set_xticks(posiciones)
    eje.set_xticklabels(deciles["decil_pd"])
    eje.set_xlabel("decil de PD (red de la Parte 1), créditos con desenlace definitivo")
    eje.set_ylabel("USD (suma en el decil)")
    eje.legend(fontsize=8)
    _guardar(figura, ruta_archivo)


def generar_todas_las_figuras(configuracion):
    from lgdmodel.data import cargar_datos_procesados
    carpeta_tablas = ruta(configuracion, "tables")
    carpeta_figuras = ruta(configuracion, "figures", crear=True)
    ruta_procesado = ruta(configuracion, "processed") / "lgd_dataset.csv.gz"
    if ruta_procesado.exists():
        prestamos = cargar_datos_procesados(configuracion)
        con_lgd = prestamos[prestamos["lgd"].notna()]
        figura_distribucion_lgd_observada(con_lgd, carpeta_figuras / "lgd_distribucion_observada.png")
        figura_lgd_por_anio(con_lgd, carpeta_figuras / "lgd_media_por_anio.png")
    if (carpeta_tablas / "busqueda_pruebas.csv").exists():
        figura_busqueda(pd.read_csv(carpeta_tablas / "busqueda_pruebas.csv"), configuracion["red"]["metrica_seleccion"],
                        carpeta_figuras / "busqueda_red_lgd.png")
    if (carpeta_tablas / "semillas_metricas.csv").exists() and (carpeta_tablas / "referencias_resultado.json").exists():
        with open(carpeta_tablas / "referencias_resultado.json", encoding="utf-8") as archivo:
            referencias = json.load(archivo)
        figura_semillas(pd.read_csv(carpeta_tablas / "semillas_metricas.csv"), referencias,
                        carpeta_figuras / "semillas_lgd.png")
    if (carpeta_tablas / "predicciones_lgd.csv.gz").exists():
        predicciones = pd.read_csv(carpeta_tablas / "predicciones_lgd.csv.gz")
        figura_distribucion_predicciones(predicciones[predicciones["particion"] == "test"],
                                         carpeta_figuras / "distribucion_predicciones_lgd.png")
    if (carpeta_tablas / "calibracion_lgd_deciles.csv").exists():
        figura_calibracion(pd.read_csv(carpeta_tablas / "calibracion_lgd_deciles.csv"),
                           carpeta_figuras / "calibracion_lgd.png")
    if (carpeta_tablas / "ablacion_perdida.csv").exists():
        figura_ablacion(pd.read_csv(carpeta_tablas / "ablacion_perdida.csv"), carpeta_figuras / "ablacion_perdida.png")
    if (carpeta_tablas / "curva_aprendizaje_representante.csv").exists():
        figura_curva_aprendizaje(pd.read_csv(carpeta_tablas / "curva_aprendizaje_representante.csv"),
                                 carpeta_figuras / "curva_aprendizaje_lgd.png")
    if (carpeta_tablas / "perdida_esperada_deciles.csv").exists():
        figura_perdida_esperada_deciles(pd.read_csv(carpeta_tablas / "perdida_esperada_deciles.csv"),
                                        carpeta_figuras / "perdida_esperada_deciles.png")
