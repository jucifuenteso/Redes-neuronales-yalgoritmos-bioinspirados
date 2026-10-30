"""Experimentos de la Parte 2: descriptivo de la LGD, modelos de referencia, búsqueda de la red,
corridas con N semillas, ablación de la función de pérdida y pérdida esperada E[L] = PD · LGD · EAD."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from pdmodel.experiments import sample_config
from pdmodel.features import Preprocessor
from scipy import stats
from sklearn.linear_model import Ridge

from lgdmodel.config import ruta, ruta_modelos_parte1
from lgdmodel.data import cargar_datos_procesados, columnas_predictoras, verificar_consistencia_con_parte1
from lgdmodel.logit_fraccional import LogitFraccional
from lgdmodel.metrics import (calibracion_por_deciles, comparar_distribuciones, describir_distribucion,
                              metricas_regresion, resumir_semillas)
from lgdmodel.mlp_lgd import MLPLGD
from lgdmodel.modelos import ModeloLGD

PARTICIONES = ["train", "val", "test"]
HIPERPARAMETROS_DE_LA_RED = {"hidden", "activation", "dropout", "l2", "optimizer", "lr", "batch_size"}


def _escribir_json(objeto, ruta_archivo):
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        json.dump(objeto, archivo, indent=2, ensure_ascii=False, default=float)


def _leer_json(ruta_archivo):
    with open(ruta_archivo, encoding="utf-8") as archivo:
        return json.load(archivo)


# ---------------------------------------------------------------- datos compartidos
class DatosLGD:
    """Mismos datos, mismo preprocesamiento y mismas particiones para TODOS los modelos de LGD.

    El preprocesador es el de la Parte 1 (misma clase, misma configuración, ajustado con los mismos
    créditos de entrenamiento), de modo que PD y LGD ven exactamente la misma representación.
    Los modelos de LGD se entrenan y evalúan solo con incumplidos cuya pérdida es definitiva.
    """

    def __init__(self, configuracion):
        self.configuracion = configuracion
        prestamos = cargar_datos_procesados(configuracion)
        self.columnas_numericas, self.columnas_categoricas = columnas_predictoras(configuracion)
        self.columnas_solicitud = self.columnas_numericas + self.columnas_categoricas
        self.creditos_por_particion = {
            particion: prestamos[prestamos["split"] == particion].reset_index(drop=True) for particion in PARTICIONES}
        self.incumplidos_con_lgd = {
            particion: creditos[creditos["lgd"].notna()].reset_index(drop=True)
            for particion, creditos in self.creditos_por_particion.items()}
        for particion, incumplidos in self.incumplidos_con_lgd.items():
            if len(incumplidos) < 2:
                raise ValueError(f"La partición '{particion}' tiene {len(incumplidos)} créditos con LGD observada. "
                                 "Revise data.split de la Parte 1 o lgd.estados_perdida_definitiva.")
        configuracion_datos = configuracion["datos"]
        self.preprocesador = Preprocessor(self.columnas_numericas, self.columnas_categoricas,
                                          configuracion_datos["clip_quantiles"],
                                          configuracion_datos["min_category_freq"]).fit(
            self.creditos_por_particion["train"])
        joblib.dump(self.preprocesador, ruta(configuracion, "models", crear=True) / "preprocesador.joblib")
        self.variables = {particion: self.preprocesador.transform(incumplidos)
                          for particion, incumplidos in self.incumplidos_con_lgd.items()}
        self.lgd = {particion: incumplidos["lgd"].to_numpy(dtype=float)
                    for particion, incumplidos in self.incumplidos_con_lgd.items()}


def _metrica_de_seleccion(configuracion, observado, predicho):
    return metricas_regresion(observado, predicho)[configuracion["red"]["metrica_seleccion"]]


# ---------------------------------------------------------------- descriptivo de la LGD
def ejecutar_descriptivo(configuracion):
    from lgdmodel import plots
    prestamos = cargar_datos_procesados(configuracion)
    carpeta_tablas = ruta(configuracion, "tables", crear=True)
    carpeta_figuras = ruta(configuracion, "figures", crear=True)
    incumplidos = prestamos[prestamos["y"] == 1].copy()

    # ¿Qué incumplidos tienen LGD observable en cada partición?
    estados_por_particion = pd.crosstab(incumplidos["loan_status"], incumplidos["split"]).reindex(
        columns=PARTICIONES, fill_value=0)
    estados_por_particion["lgd_observable"] = [
        estado in configuracion["lgd"]["estados_perdida_definitiva"] for estado in estados_por_particion.index]
    estados_por_particion.to_csv(carpeta_tablas / "lgd_estados_por_particion.csv")

    # Evidencia para decidir qué estados tienen pérdida definitiva: ¿hubo recuperaciones post-castigo?
    incumplidos["fraccion_capital_pagado"] = incumplidos["total_rec_prncp"] / incumplidos["funded_amnt"]
    incumplidos["recuperacion_sobre_desembolso"] = incumplidos["recoveries"] / incumplidos["funded_amnt"]
    recuperaciones_por_estado = incumplidos.groupby("loan_status").agg(
        n=("y", "size"),
        fraccion_con_recuperaciones=("recoveries", lambda valores: float((valores > 0).mean())),
        recuperacion_media_sobre_desembolso=("recuperacion_sobre_desembolso", "mean"),
        capital_pagado_medio_sobre_desembolso=("fraccion_capital_pagado", "mean"))
    recuperaciones_por_estado.to_csv(carpeta_tablas / "lgd_recuperaciones_por_estado.csv")

    con_lgd = prestamos[prestamos["lgd"].notna()].copy()
    distribucion = {
        "lgd_sobre_desembolso (objetivo)": describir_distribucion(con_lgd["lgd"]),
        "lgd_sobre_saldo_al_castigo (alternativa)": describir_distribucion(con_lgd["lgd_sobre_saldo"].dropna()),
        "lgd_media_por_particion": con_lgd.groupby("split")["lgd"].mean().to_dict(),
        "n_lgd_observada_por_particion": con_lgd.groupby("split")["lgd"].size().to_dict(),
    }
    _escribir_json(distribucion, carpeta_tablas / "lgd_distribucion.json")

    con_lgd["anio_originacion"] = con_lgd["issue_d"].dt.year
    tablas_por_segmento = []
    for variable_segmento in ["anio_originacion", "term_months", "purpose", "home_ownership", "verification_status"]:
        tabla_segmento = con_lgd.groupby(variable_segmento)["lgd"].agg(
            n="size", lgd_media="mean", lgd_mediana="median").reset_index()
        tabla_segmento.columns = ["grupo", "n", "lgd_media", "lgd_mediana"]
        tabla_segmento.insert(0, "variable", variable_segmento)
        tabla_segmento["grupo"] = tabla_segmento["grupo"].astype(str)
        tablas_por_segmento.append(tabla_segmento)
    pd.concat(tablas_por_segmento).to_csv(carpeta_tablas / "lgd_por_segmento.csv", index=False)

    plots.figura_distribucion_lgd_observada(con_lgd, carpeta_figuras / "lgd_distribucion_observada.png")
    plots.figura_lgd_por_anio(con_lgd, carpeta_figuras / "lgd_media_por_anio.png")
    return distribucion, recuperaciones_por_estado


# ---------------------------------------------------------------- modelos de referencia
def ejecutar_referencias(configuracion, datos):
    """Referencias de baja complejidad: media constante, regresión lineal (Ridge) y logit fraccional."""
    carpeta_modelos = ruta(configuracion, "models", crear=True)
    carpeta_tablas = ruta(configuracion, "tables", crear=True)
    filas_busqueda = []

    lgd_media_entrenamiento = float(datos.lgd["train"].mean())
    resultado = {"constante": {
        "valor": lgd_media_entrenamiento, "n_parametros": 1,
        "val": metricas_regresion(datos.lgd["val"], np.full_like(datos.lgd["val"], lgd_media_entrenamiento)),
        "test": metricas_regresion(datos.lgd["test"], np.full_like(datos.lgd["test"], lgd_media_entrenamiento))}}

    # Regresión lineal con L2: alpha elegido en validación. Predicciones recortadas a [0, 1].
    inicio_lineal, mejor_lineal = time.time(), None
    for alpha in configuracion["lineal"]["alpha_grid"]:
        inicio_prueba = time.time()
        modelo = ModeloLGD(datos.preprocesador, Ridge(alpha=alpha).fit(datos.variables["train"], datos.lgd["train"]),
                           "Regresión lineal")
        valor_validacion = _metrica_de_seleccion(configuracion, datos.lgd["val"],
                                                 modelo.predecir_desde_matriz(datos.variables["val"]))
        filas_busqueda.append({"modelo": "lineal", "hiperparametro": "alpha", "valor": alpha,
                               f"val_{configuracion['red']['metrica_seleccion']}": valor_validacion,
                               "segundos": time.time() - inicio_prueba})
        if mejor_lineal is None or valor_validacion < mejor_lineal[0]:
            mejor_lineal = (valor_validacion, alpha, modelo)
    segundos_lineal = time.time() - inicio_lineal

    # Logit fraccional: GLM con enlace logístico y pérdida BCE (= red sin capas ocultas), resuelto
    # hasta convergencia con L-BFGS para que la referencia no quede sub-entrenada.
    parametros_logit = configuracion["logit_fraccional"]
    inicio_logit, mejor_logit = time.time(), None
    for l2 in parametros_logit["l2_grid"]:
        inicio_prueba = time.time()
        glm = LogitFraccional(l2=l2, max_iteraciones=parametros_logit["max_iteraciones"]).fit(
            datos.variables["train"], datos.lgd["train"])
        if not glm.convergio_:
            print(f"  AVISO: el logit fraccional con l2={l2} no convergió en {glm.iteraciones_} iteraciones")
        modelo = ModeloLGD(datos.preprocesador, glm, "Logit fraccional")
        valor_validacion = _metrica_de_seleccion(configuracion, datos.lgd["val"],
                                                 modelo.predecir_desde_matriz(datos.variables["val"]))
        filas_busqueda.append({"modelo": "logit_fraccional", "hiperparametro": "l2", "valor": l2,
                               f"val_{configuracion['red']['metrica_seleccion']}": valor_validacion,
                               "convergio": glm.convergio_, "iteraciones": glm.iteraciones_,
                               "segundos": time.time() - inicio_prueba})
        if mejor_logit is None or valor_validacion < mejor_logit[0]:
            mejor_logit = (valor_validacion, l2, modelo)
    segundos_logit = time.time() - inicio_logit

    for nombre, (_, valor_hiperparametro, modelo), nombre_hiperparametro, n_configuraciones, segundos in [
            ("lineal", mejor_lineal, "alpha", len(configuracion["lineal"]["alpha_grid"]), segundos_lineal),
            ("logit_fraccional", mejor_logit, "l2", len(parametros_logit["l2_grid"]), segundos_logit)]:
        joblib.dump(modelo, carpeta_modelos / f"lgd_{nombre}.joblib")
        resultado[nombre] = {
            f"{nombre_hiperparametro}_elegido": valor_hiperparametro, "n_configuraciones": n_configuraciones,
            "segundos_busqueda": segundos, "n_parametros": modelo.n_parametros,
            "val": metricas_regresion(datos.lgd["val"], modelo.predecir_desde_matriz(datos.variables["val"])),
            "test": metricas_regresion(datos.lgd["test"], modelo.predecir_desde_matriz(datos.variables["test"]))}
    pd.DataFrame(filas_busqueda).to_csv(carpeta_tablas / "referencias_busqueda.csv", index=False)
    _escribir_json(resultado, carpeta_tablas / "referencias_resultado.json")
    return resultado


# ---------------------------------------------------------------- búsqueda de la red
def ejecutar_busqueda(configuracion, datos):
    parametros_busqueda = configuracion["busqueda"]
    perdida = configuracion["red"]["perdida"]
    metrica = configuracion["red"]["metrica_seleccion"]
    generador_aleatorio = np.random.RandomState(configuracion["semilla"])
    configuraciones = [dict(parametros_busqueda["configuracion_referencia"])] + [
        sample_config(parametros_busqueda["espacio"], generador_aleatorio)
        for _ in range(parametros_busqueda["n_pruebas"] - 1)]
    filas, mejor, inicio_busqueda = [], None, time.time()
    for numero_prueba, hiperparametros in enumerate(configuraciones):
        inicio_prueba = time.time()
        red = MLPLGD(perdida=perdida, **hiperparametros, seed=configuracion["semilla"]).fit(
            datos.variables["train"], datos.lgd["train"], datos.variables["val"], datos.lgd["val"],
            parametros_busqueda["max_epocas"], parametros_busqueda["paciencia"])
        metricas_validacion = metricas_regresion(datos.lgd["val"], red.predict_proba(datos.variables["val"]))
        if red.best_epoch_ == parametros_busqueda["max_epocas"]:
            print(f"  AVISO: prueba {numero_prueba}: la mejor época es la última; puede estar sub-entrenada")
        filas.append({"prueba": numero_prueba,
                      **{clave: (str(valor) if clave == "hidden" else valor) for clave, valor in hiperparametros.items()},
                      "n_parametros": red.n_params, "mejor_epoca": red.best_epoch_,
                      "val_mae": metricas_validacion["mae"], "val_rmse": metricas_validacion["rmse"],
                      "segundos": time.time() - inicio_prueba})
        print(f"  prueba {numero_prueba + 1:3d}/{len(configuraciones)}  val_{metrica}={metricas_validacion[metrica]:.4f}  "
              f"{hiperparametros['hidden']} {hiperparametros['activation']} {hiperparametros['optimizer']} "
              f"lr={hiperparametros['lr']:.4g}  ({time.time() - inicio_prueba:.1f}s)")
        if mejor is None or metricas_validacion[metrica] < mejor[0]:
            mejor = (metricas_validacion[metrica], hiperparametros)
    segundos_busqueda = time.time() - inicio_busqueda
    pd.DataFrame(filas).to_csv(ruta(configuracion, "tables", crear=True) / "busqueda_pruebas.csv", index=False)
    _escribir_json(mejor[1], ruta(configuracion, "models", crear=True) / "lgd_mejor_configuracion.json")
    resultado = {"n_configuraciones": len(configuraciones), "segundos_total": segundos_busqueda, "perdida": perdida,
                 "mejor_configuracion": mejor[1], f"mejor_val_{metrica}": mejor[0]}
    _escribir_json(resultado, ruta(configuracion, "tables") / "busqueda_resultado.json")
    return resultado


def configuracion_final_efectiva(configuracion_base, sobreescrituras):
    """Combina CLAVE POR CLAVE la mejor configuración con lo pedido en final.sobreescribir.

    Así `--set final.sobreescribir.hidden=[8]` cambia solo la arquitectura y conserva el resto, en lugar de
    ignorarse porque existe un archivo guardado (error que tenía la versión anterior).
    """
    sobreescrituras = dict(sobreescrituras or {})
    claves_desconocidas = set(sobreescrituras) - HIPERPARAMETROS_DE_LA_RED
    if claves_desconocidas:
        raise ValueError(f"final.sobreescribir tiene claves desconocidas {sorted(claves_desconocidas)}; "
                         f"válidas: {sorted(HIPERPARAMETROS_DE_LA_RED)}")
    efectiva = {**configuracion_base, **sobreescrituras}
    efectiva["hidden"] = [int(neuronas) for neuronas in efectiva["hidden"]]
    return efectiva


def cargar_configuracion_de_la_red(configuracion):
    ruta_mejor = ruta(configuracion, "models") / "lgd_mejor_configuracion.json"
    if ruta_mejor.exists():
        base, origen = _leer_json(ruta_mejor), "búsqueda (models/lgd_mejor_configuracion.json)"
    else:
        base, origen = dict(configuracion["busqueda"]["configuracion_referencia"]), \
            "configuración de referencia (no se encontró el resultado de la búsqueda)"
    efectiva = configuracion_final_efectiva(base, configuracion["final"].get("sobreescribir"))
    if configuracion["final"].get("sobreescribir"):
        origen += f" + sobreescrituras {configuracion['final']['sobreescribir']}"
    return efectiva, origen


def _entrenar_con_semillas(configuracion, datos, hiperparametros, perdida, n_semillas):
    filas, redes = [], []
    for indice_semilla in range(n_semillas):
        semilla = configuracion["semilla"] + 1000 + indice_semilla
        red = MLPLGD(perdida=perdida, **hiperparametros, seed=semilla).fit(
            datos.variables["train"], datos.lgd["train"], datos.variables["val"], datos.lgd["val"],
            configuracion["final"]["max_epocas"], configuracion["final"]["paciencia"])
        metricas_validacion = metricas_regresion(datos.lgd["val"], red.predict_proba(datos.variables["val"]))
        metricas_prueba = metricas_regresion(datos.lgd["test"], red.predict_proba(datos.variables["test"]))
        if red.best_epoch_ == configuracion["final"]["max_epocas"]:
            print(f"  AVISO: semilla {semilla}: la mejor época es la última ({red.best_epoch_}); "
                  "puede no haber convergido (suba final.max_epocas)")
        filas.append({"perdida": perdida, "semilla": semilla, "mejor_epoca": red.best_epoch_,
                      "segundos": red.fit_seconds_,
                      **{f"val_{nombre}": valor for nombre, valor in metricas_validacion.items()},
                      **{f"test_{nombre}": valor for nombre, valor in metricas_prueba.items()}})
        redes.append(red)
        print(f"  [{perdida}] semilla {indice_semilla + 1:2d}/{n_semillas} ({semilla})  "
              f"test_mae={metricas_prueba['mae']:.4f}  test_rmse={metricas_prueba['rmse']:.4f}")
    return pd.DataFrame(filas), redes


# ---------------------------------------------------------------- N semillas y comparación
def ejecutar_final(configuracion, datos):
    carpeta_modelos = ruta(configuracion, "models", crear=True)
    carpeta_tablas = ruta(configuracion, "tables", crear=True)
    hiperparametros, origen = cargar_configuracion_de_la_red(configuracion)
    perdida = configuracion["red"]["perdida"]
    metrica = configuracion["red"]["metrica_seleccion"]
    print(f"Configuración efectiva de la red (origen: {origen}), pérdida={perdida}:\n  {hiperparametros}")

    inicio = time.time()
    tabla_semillas, redes = _entrenar_con_semillas(configuracion, datos, hiperparametros, perdida,
                                                   configuracion["final"]["n_semillas"])
    segundos_finales = time.time() - inicio
    tabla_semillas.to_csv(carpeta_tablas / "semillas_metricas.csv", index=False)
    resumen_semillas = resumir_semillas(tabla_semillas)
    resumen_semillas.to_csv(carpeta_tablas / "semillas_resumen.csv", index=False)

    # representante: la semilla con mejor métrica en VALIDACIÓN (nunca se elige con prueba)
    posicion_representante = int(tabla_semillas[f"val_{metrica}"].idxmin())
    red_representante = redes[posicion_representante]
    modelo_red = ModeloLGD(datos.preprocesador, red_representante, "Red neuronal (MLP)")
    joblib.dump(modelo_red, carpeta_modelos / "lgd_mlp_final.joblib")
    _escribir_json({"hiperparametros": hiperparametros, "perdida": perdida, "origen": origen,
                    "semilla_representante": int(tabla_semillas.loc[posicion_representante, "semilla"])},
                   carpeta_modelos / "lgd_mlp_configuracion_final.json")
    pd.DataFrame(red_representante.history_).to_csv(carpeta_tablas / "curva_aprendizaje_representante.csv", index=False)

    referencias = _leer_json(carpeta_tablas / "referencias_resultado.json")
    modelo_lineal = joblib.load(carpeta_modelos / "lgd_lineal.joblib")
    modelo_logit = joblib.load(carpeta_modelos / "lgd_logit_fraccional.joblib")
    resultado_busqueda = (_leer_json(carpeta_tablas / "busqueda_resultado.json")
                          if (carpeta_tablas / "busqueda_resultado.json").exists() else {})

    resumen_prueba = resumen_semillas[resumen_semillas["particion"] == "test"].set_index("metrica")
    comparacion = []
    for nombre, clave in [("Media constante", "constante"), ("Regresión lineal", "lineal"),
                          ("Logit fraccional", "logit_fraccional")]:
        comparacion.append({"modelo": nombre, "test_mae": referencias[clave]["test"]["mae"],
                            "test_rmse": referencias[clave]["test"]["rmse"],
                            "n_parametros": referencias[clave]["n_parametros"],
                            "n_configuraciones": referencias[clave].get("n_configuraciones", 0)})
    comparacion.append({"modelo": f"Red neuronal (media de {len(tabla_semillas)} semillas)",
                        "test_mae": resumen_prueba.loc["mae", "media"], "test_rmse": resumen_prueba.loc["rmse", "media"],
                        "test_mae_desv_est": resumen_prueba.loc["mae", "desv_est"],
                        "test_rmse_desv_est": resumen_prueba.loc["rmse", "desv_est"],
                        "test_rmse_mejor": resumen_prueba.loc["rmse", "mejor"],
                        "test_rmse_peor": resumen_prueba.loc["rmse", "peor"],
                        "n_parametros": red_representante.n_params,
                        "n_configuraciones": resultado_busqueda.get("n_configuraciones", 0)})
    pd.DataFrame(comparacion).to_csv(carpeta_tablas / "comparacion_modelos.csv", index=False)

    presupuesto = {
        "lineal": {"n_configuraciones": referencias["lineal"]["n_configuraciones"],
                   "segundos_busqueda": referencias["lineal"]["segundos_busqueda"],
                   "n_parametros": referencias["lineal"]["n_parametros"]},
        "logit_fraccional": {"n_configuraciones": referencias["logit_fraccional"]["n_configuraciones"],
                             "segundos_busqueda": referencias["logit_fraccional"]["segundos_busqueda"],
                             "n_parametros": referencias["logit_fraccional"]["n_parametros"]},
        "red": {"n_configuraciones_busqueda": resultado_busqueda.get("n_configuraciones"),
                "segundos_busqueda": resultado_busqueda.get("segundos_total"),
                "segundos_corridas_finales": segundos_finales, "n_semillas": len(tabla_semillas),
                "n_parametros": red_representante.n_params, "configuracion": hiperparametros, "perdida": perdida}}
    _escribir_json(presupuesto, carpeta_tablas / "presupuesto_computo.json")

    filas_predicciones = []
    for particion in ["val", "test"]:
        filas_predicciones.append(pd.DataFrame({
            "particion": particion, "lgd_observada": datos.lgd[particion],
            "pred_constante": referencias["constante"]["valor"],
            "pred_lineal": modelo_lineal.predecir_desde_matriz(datos.variables[particion]),
            "pred_logit_fraccional": modelo_logit.predecir_desde_matriz(datos.variables[particion]),
            "pred_red": modelo_red.predecir_desde_matriz(datos.variables[particion])}))
    predicciones = pd.concat(filas_predicciones, ignore_index=True)
    predicciones.to_csv(carpeta_tablas / "predicciones_lgd.csv.gz", index=False, compression="gzip")

    prueba = predicciones[predicciones["particion"] == "test"]
    distribucion = comparar_distribuciones(prueba["lgd_observada"], {
        "lineal": prueba["pred_lineal"], "logit_fraccional": prueba["pred_logit_fraccional"],
        "red": prueba["pred_red"]})
    _escribir_json(distribucion, carpeta_tablas / "distribucion_predicciones.json")
    calibracion = pd.concat([calibracion_por_deciles(prueba["lgd_observada"], prueba[columna]).assign(modelo=columna)
                             for columna in ["pred_lineal", "pred_logit_fraccional", "pred_red"]])
    calibracion.to_csv(carpeta_tablas / "calibracion_lgd_deciles.csv", index=False)
    return {"comparacion": pd.DataFrame(comparacion), "resumen": resumen_semillas, "presupuesto": presupuesto}


# ---------------------------------------------------------------- ablación de la pérdida
def ejecutar_ablacion(configuracion, datos):
    """Misma arquitectura, mismos hiperparámetros y mismas semillas: BCE frente a MSE, medidas con el
    MISMO MAE/RMSE de prueba. Es la evidencia empírica para la elección de la función de pérdida."""
    hiperparametros, origen = cargar_configuracion_de_la_red(configuracion)
    print(f"Ablación con la configuración (origen: {origen}):\n  {hiperparametros}")
    n_semillas = configuracion["ablacion"]["n_semillas"]
    tablas = [_entrenar_con_semillas(configuracion, datos, hiperparametros, perdida, n_semillas)[0]
              for perdida in ("bce", "mse")]
    tabla_ablacion = pd.concat(tablas, ignore_index=True)
    carpeta_tablas = ruta(configuracion, "tables", crear=True)
    tabla_ablacion.to_csv(carpeta_tablas / "ablacion_perdida.csv", index=False)
    resultados_bce = tabla_ablacion[tabla_ablacion["perdida"] == "bce"].sort_values("semilla")
    resultados_mse = tabla_ablacion[tabla_ablacion["perdida"] == "mse"].sort_values("semilla")
    resumen = {"hiperparametros": hiperparametros, "n_semillas": n_semillas}
    for metrica in ["test_mae", "test_rmse"]:
        diferencias_pareadas = resultados_bce[metrica].to_numpy() - resultados_mse[metrica].to_numpy()
        resumen_metrica = {
            "bce_media": float(resultados_bce[metrica].mean()), "mse_media": float(resultados_mse[metrica].mean()),
            "bce_desv_est": float(resultados_bce[metrica].std(ddof=1)) if n_semillas > 1 else 0.0,
            "mse_desv_est": float(resultados_mse[metrica].std(ddof=1)) if n_semillas > 1 else 0.0,
            "diferencia_media_bce_menos_mse": float(diferencias_pareadas.mean()),
            "semillas_donde_bce_es_mejor": int((diferencias_pareadas < 0).sum())}
        if n_semillas > 1 and np.any(diferencias_pareadas != 0):
            resumen_metrica["valor_p_t_pareada"] = float(stats.ttest_rel(resultados_bce[metrica],
                                                                         resultados_mse[metrica]).pvalue)
            resumen_metrica["valor_p_wilcoxon"] = float(stats.wilcoxon(diferencias_pareadas).pvalue)
        resumen[metrica] = resumen_metrica
    _escribir_json(resumen, carpeta_tablas / "ablacion_perdida_resumen.json")
    return resumen


# ---------------------------------------------------------------- pérdida esperada
def _verificar_mismo_preprocesamiento(datos, carpeta_modelos_parte1):
    ruta_preprocesador_parte1 = carpeta_modelos_parte1 / "preprocessor.joblib"
    if not ruta_preprocesador_parte1.exists():
        return {"verificado": False, "motivo": f"No existe {ruta_preprocesador_parte1}"}
    preprocesador_parte1 = joblib.load(ruta_preprocesador_parte1)
    mismas_variables = list(preprocesador_parte1.feature_names_) == list(datos.preprocesador.feature_names_)
    mismas_estadisticas = mismas_variables and bool(
        np.allclose(preprocesador_parte1.mean_, datos.preprocesador.mean_)
        and np.allclose(preprocesador_parte1.std_, datos.preprocesador.std_))
    return {"verificado": True, "mismas_variables": mismas_variables, "mismas_estadisticas": mismas_estadisticas}


def ejecutar_perdida_esperada(configuracion, datos):
    """E[L]_i = PD_i · LGD_i · EAD_i en la partición de prueba, con EAD = funded_amnt (ver data.py)."""
    carpeta_modelos = ruta(configuracion, "models")
    carpeta_tablas = ruta(configuracion, "tables", crear=True)
    carpeta_modelos_parte1 = ruta_modelos_parte1(configuracion)

    todos_los_creditos = pd.concat(datos.creditos_por_particion.values(), ignore_index=True)
    consistencia = verificar_consistencia_con_parte1(todos_los_creditos, configuracion)
    if configuracion["perdida_esperada"]["exigir_consistencia_con_parte1"] and not consistencia["coincide"]:
        raise RuntimeError(
            "La partición de la Parte 2 no coincide con la que generó los modelos de PD de la Parte 1 "
            f"({json.dumps(consistencia, ensure_ascii=False)}). Combinarlos usaría créditos de prueba que la PD vio "
            "en entrenamiento. Corra la Parte 1 con la misma configuración o use "
            "--set perdida_esperada.exigir_consistencia_con_parte1=false bajo su responsabilidad.")
    for nombre_archivo in ["lr_final.joblib", "mlp_final.joblib"]:
        if not (carpeta_modelos_parte1 / nombre_archivo).exists():
            raise FileNotFoundError(f"Falta {carpeta_modelos_parte1 / nombre_archivo}: corra primero la Parte 1.")
    pd_logistica = joblib.load(carpeta_modelos_parte1 / "lr_final.joblib")
    pd_red = joblib.load(carpeta_modelos_parte1 / "mlp_final.joblib")
    lgd_lineal = joblib.load(carpeta_modelos / "lgd_lineal.joblib")
    lgd_red = joblib.load(carpeta_modelos / "lgd_mlp_final.joblib")
    lgd_constante = float(datos.lgd["train"].mean())

    prueba = datos.creditos_por_particion["test"]
    probabilidad_incumplimiento_logistica = pd_logistica.predict_proba(prueba)
    probabilidad_incumplimiento_red = pd_red.predict_proba(prueba)
    exposicion = prueba["ead"].to_numpy(dtype=float)
    combinaciones = {
        "referencia_pd_logistica_x_lgd_lineal": probabilidad_incumplimiento_logistica * lgd_lineal.predecir(prueba),
        "red_pd_mlp_x_lgd_mlp": probabilidad_incumplimiento_red * lgd_red.predecir(prueba),
        "pd_mlp_x_lgd_constante": probabilidad_incumplimiento_red * lgd_constante,
    }
    tabla_creditos = pd.DataFrame({
        "issue_d": prueba["issue_d"], "loan_status": prueba["loan_status"], "y": prueba["y"], "ead": exposicion,
        "pd_logistica": probabilidad_incumplimiento_logistica, "pd_red": probabilidad_incumplimiento_red,
        "lgd_observada": prueba["lgd"],
        **{f"perdida_esperada_{nombre}": factor * exposicion for nombre, factor in combinaciones.items()}})
    # Pérdida realizada: solo observable cuando el desenlace es definitivo (pagó, o fue castigado con LGD medida).
    tabla_creditos["desenlace_definitivo"] = (tabla_creditos["y"] == 0) | tabla_creditos["lgd_observada"].notna()
    tabla_creditos["perdida_realizada"] = tabla_creditos["lgd_observada"].fillna(0.0) * exposicion
    tabla_creditos.to_csv(carpeta_tablas / "perdida_esperada_prueba.csv.gz", index=False, compression="gzip")

    evaluados = tabla_creditos[tabla_creditos["desenlace_definitivo"]]
    perdida_realizada_total = float(evaluados["perdida_realizada"].sum())
    incumplidos_evaluados = evaluados[evaluados["y"] == 1]
    resumen = {
        "ead": "funded_amnt (exposición al originar, coherente con la LGD sobre el monto desembolsado)",
        "consistencia_con_parte1": consistencia,
        "mismo_preprocesamiento_que_parte1": _verificar_mismo_preprocesamiento(datos, carpeta_modelos_parte1),
        "n_prueba": int(len(tabla_creditos)), "n_desenlace_definitivo": int(len(evaluados)),
        "excluidos_sin_desenlace_por_estado": {
            str(estado): int(conteo) for estado, conteo in
            tabla_creditos.loc[~tabla_creditos["desenlace_definitivo"], "loan_status"].value_counts().items()},
        "ead_total_evaluado": float(evaluados["ead"].sum()),
        "perdida_realizada_total": perdida_realizada_total,
        # Descomposición para explicar cualquier diferencia entre E[L] y la pérdida realizada:
        "diagnostico": {
            "pd_media_red_en_evaluados": float(evaluados["pd_red"].mean()),
            "pd_media_logistica_en_evaluados": float(evaluados["pd_logistica"].mean()),
            "tasa_incumplimiento_observada_en_evaluados": float(evaluados["y"].mean()),
            "tasa_incumplimiento_observada_en_toda_la_prueba": float(tabla_creditos["y"].mean()),
            "lgd_media_observada_en_incumplidos_evaluados": float(incumplidos_evaluados["lgd_observada"].mean()),
            "lgd_media_predicha_red_en_incumplidos_evaluados": float(
                lgd_red.predecir(prueba[tabla_creditos["desenlace_definitivo"].to_numpy()
                                        & (tabla_creditos["y"] == 1).to_numpy()]).mean()),
            "lgd_constante_entrenamiento": lgd_constante},
        "combinaciones": {
            nombre: {"perdida_esperada_total_evaluados": float(evaluados[f"perdida_esperada_{nombre}"].sum()),
                     "razon_esperada_sobre_realizada": (float(evaluados[f"perdida_esperada_{nombre}"].sum()
                                                              / perdida_realizada_total)
                                                        if perdida_realizada_total else None),
                     "error_absoluto_medio_por_credito": float(np.mean(np.abs(
                         evaluados[f"perdida_esperada_{nombre}"] - evaluados["perdida_realizada"]))),
                     "perdida_esperada_total_toda_la_prueba": float(tabla_creditos[f"perdida_esperada_{nombre}"].sum())}
            for nombre in combinaciones},
        "nota": ("La comparación con la pérdida realizada usa solo créditos con desenlace definitivo; los excluidos "
                 "se listan arriba. Si la razón E[L]/realizada se aleja de 1, el bloque 'diagnostico' separa cuánto "
                 "se explica por la PD (PD media frente a tasa observada) y cuánto por la LGD."),
    }
    _escribir_json(resumen, carpeta_tablas / "perdida_esperada_resumen.json")

    evaluados = evaluados.assign(decil_pd=pd.qcut(evaluados["pd_red"].rank(method="first"), 10, labels=False) + 1)
    evaluados.groupby("decil_pd").agg(
        n=("ead", "size"), ead_total=("ead", "sum"), pd_media_red=("pd_red", "mean"),
        tasa_incumplimiento_observada=("y", "mean"),
        perdida_esperada_red=("perdida_esperada_red_pd_mlp_x_lgd_mlp", "sum"),
        perdida_esperada_referencia=("perdida_esperada_referencia_pd_logistica_x_lgd_lineal", "sum"),
        perdida_realizada=("perdida_realizada", "sum")).to_csv(carpeta_tablas / "perdida_esperada_deciles.csv")
    return resumen
