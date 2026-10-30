"""Del loan.csv original a la tabla de la Parte 2: los mismos créditos y particiones de la Parte 1,
más las columnas de recuperación con las que se construye la LGD.

Definiciones (LCDataDictionary.xlsx, hoja LoanStats, incluido en el conjunto de Kaggle):
  funded_amnt              "The total amount committed to that loan at that point in time."
  total_rec_prncp          "Principal received to date"
  recoveries               "post charge off gross recovery"
  collection_recovery_fee  "post charge off collection fee"

LGD (fracción de la exposición que se pierde), solo para créditos con pérdida definitiva:
  LGD = recortar_[0,1]( 1 - (total_rec_prncp + max(recoveries - collection_recovery_fee, 0)) / funded_amnt )
EAD = funded_amnt: la decisión se toma al recibir la solicitud (caso de uso de la Parte 1), cuando el saldo
al momento del incumplimiento todavía no existe. Ver docs/guia_conceptual.md para las limitaciones.
"""
import json

import numpy as np
import pandas as pd
from pdmodel.data import RAW_COLS, app_columns, assign_split, clean_applications

from lgdmodel.config import ruta, ruta_archivo_original, ruta_tablas_parte1

COLUMNAS_DE_RECUPERACION = ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]


def construir_lgd(prestamos):
    """LGD sobre el monto desembolsado (coherente con EAD = funded_amnt)."""
    recuperacion_neta_posterior_al_castigo = (
        prestamos["recoveries"].fillna(0) - prestamos["collection_recovery_fee"].fillna(0)).clip(lower=0)
    monto_recuperado = prestamos["total_rec_prncp"].fillna(0) + recuperacion_neta_posterior_al_castigo
    monto_desembolsado = prestamos["funded_amnt"].where(prestamos["funded_amnt"] > 0)
    return (1.0 - monto_recuperado / monto_desembolsado).clip(0.0, 1.0)


def construir_lgd_sobre_saldo(prestamos):
    """Definición alternativa (solo descriptiva): pérdida sobre el saldo de capital al castigo.

    Saldo al castigo = funded_amnt - total_rec_prncp. Es la definición habitual cuando la EAD es el saldo
    en el incumplimiento; aquí no se modela porque ese saldo no se conoce al originar.
    """
    saldo_al_castigo = (prestamos["funded_amnt"] - prestamos["total_rec_prncp"].fillna(0))
    recuperacion_neta_posterior_al_castigo = (
        prestamos["recoveries"].fillna(0) - prestamos["collection_recovery_fee"].fillna(0)).clip(lower=0)
    return (1.0 - recuperacion_neta_posterior_al_castigo / saldo_al_castigo.where(saldo_al_castigo > 0)).clip(0.0, 1.0)


def leer_archivo_original(ruta_archivo):
    columnas_a_leer = set(RAW_COLS) | set(COLUMNAS_DE_RECUPERACION)
    prestamos_originales = pd.read_csv(ruta_archivo, usecols=lambda columna: columna in columnas_a_leer,
                                       low_memory=False)
    columnas_faltantes = set(COLUMNAS_DE_RECUPERACION) - set(prestamos_originales.columns)
    if columnas_faltantes:
        raise ValueError(f"El archivo {ruta_archivo} no tiene las columnas {sorted(columnas_faltantes)}, "
                         "necesarias para construir la LGD.")
    return prestamos_originales


def construir_tabla_lgd(prestamos_originales, configuracion):
    """Aplica EXACTAMENTE los pasos de pdmodel.data.prepare_dataset y añade las columnas de la LGD."""
    configuracion_datos = configuracion["datos"]
    solicitudes = clean_applications(prestamos_originales, configuracion_datos["use_lc_pricing"])
    for columna in COLUMNAS_DE_RECUPERACION:
        solicitudes[columna] = pd.to_numeric(prestamos_originales[columna], errors="coerce")

    # mismo filtro, misma submuestra (misma semilla) y mismo orden que la Parte 1
    solicitudes = solicitudes[solicitudes["y"].notna() & solicitudes["issue_d"].notna()].copy()
    fraccion_de_muestra = configuracion_datos["sample_frac"]
    if fraccion_de_muestra < 1.0:
        solicitudes = solicitudes.sample(frac=fraccion_de_muestra, random_state=configuracion["semilla"])
    solicitudes = solicitudes.sort_values("issue_d").reset_index(drop=True)
    solicitudes["split"] = assign_split(solicitudes, configuracion["configuracion_parte1"])

    estados_con_perdida_definitiva = set(configuracion["lgd"]["estados_perdida_definitiva"])
    solicitudes["ead"] = solicitudes["funded_amnt"]
    solicitudes["perdida_definitiva"] = (solicitudes["y"] == 1) & solicitudes["loan_status"].isin(
        estados_con_perdida_definitiva)
    solicitudes["lgd"] = construir_lgd(solicitudes).where(solicitudes["perdida_definitiva"])
    solicitudes["lgd_sobre_saldo"] = construir_lgd_sobre_saldo(solicitudes).where(solicitudes["perdida_definitiva"])
    return solicitudes


def verificar_consistencia_con_parte1(solicitudes, configuracion):
    """Compara tamaño y tasa de malos por partición con data_summary.json de la Parte 1."""
    ruta_resumen_parte1 = ruta_tablas_parte1(configuracion) / "data_summary.json"
    if not ruta_resumen_parte1.exists():
        return {"verificado": False, "coincide": False,
                "motivo": f"No existe {ruta_resumen_parte1}: corra primero la Parte 1 con la misma configuración."}
    with open(ruta_resumen_parte1, encoding="utf-8") as archivo:
        resumen_parte1 = json.load(archivo)
    comparacion_por_particion = {}
    for particion, valores_parte1 in resumen_parte1["particiones"].items():
        filas_particion = solicitudes[solicitudes["split"] == particion]
        comparacion_por_particion[particion] = {
            "n_parte1": int(valores_parte1["n"]), "n_parte2": int(len(filas_particion)),
            "tasa_malos_parte1": float(valores_parte1["tasa_malos"]),
            "tasa_malos_parte2": float(filas_particion["y"].mean()) if len(filas_particion) else None}
    coincide = all(
        valores["n_parte1"] == valores["n_parte2"]
        and valores["tasa_malos_parte2"] is not None
        and abs(valores["tasa_malos_parte1"] - valores["tasa_malos_parte2"]) < 1e-9
        for valores in comparacion_por_particion.values())
    return {"verificado": True, "coincide": bool(coincide), "por_particion": comparacion_por_particion,
            "archivo_parte1": str(ruta_resumen_parte1)}


def preparar_datos(configuracion):
    ruta_archivo = ruta_archivo_original(configuracion)
    if not ruta_archivo.exists():
        raise FileNotFoundError(
            f"No existe {ruta_archivo}. La Parte 2 lee el MISMO loan.csv que la Parte 1 (Kaggle: "
            "ranadeep/credit-risk-dataset). Para la demostración corra: python -m lgdmodel todo --config configs/demo.yaml")
    n_columnas_en_archivo = len(pd.read_csv(ruta_archivo, nrows=0).columns)
    prestamos_originales = leer_archivo_original(ruta_archivo)
    solicitudes = construir_tabla_lgd(prestamos_originales, configuracion)

    carpeta_procesados = ruta(configuracion, "processed", crear=True)
    solicitudes.to_csv(carpeta_procesados / "lgd_dataset.csv.gz", index=False, compression="gzip")

    consistencia = verificar_consistencia_con_parte1(solicitudes, configuracion)
    incumplidos = solicitudes[solicitudes["y"] == 1]
    resumen = {
        "archivo": str(ruta_archivo),
        "fecha_descarga": configuracion["datos"]["download_date"],
        "registros_crudos": int(len(prestamos_originales)),
        "columnas_en_archivo": int(n_columnas_en_archivo),
        "columnas_usadas": len(set(RAW_COLS) | set(COLUMNAS_DE_RECUPERACION)),
        "registros_etiquetados": int(len(solicitudes)),
        "periodo_originacion": [str(solicitudes["issue_d"].min().date()), str(solicitudes["issue_d"].max().date())],
        "estados_perdida_definitiva": configuracion["lgd"]["estados_perdida_definitiva"],
        "por_particion": {
            particion: {
                "n_creditos": int((solicitudes["split"] == particion).sum()),
                "n_incumplidos": int((incumplidos["split"] == particion).sum()),
                "n_lgd_observada": int(solicitudes.loc[solicitudes["split"] == particion, "lgd"].notna().sum()),
                "lgd_media": float(solicitudes.loc[solicitudes["split"] == particion, "lgd"].mean()),
            } for particion in ["train", "val", "test"]},
        "incumplidos_por_estado": {str(estado): int(conteo) for estado, conteo in
                                   incumplidos["loan_status"].value_counts().items()},
        "consistencia_con_parte1": consistencia,
    }
    carpeta_tablas = ruta(configuracion, "tables", crear=True)
    with open(carpeta_tablas / "lgd_data_summary.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)
    return solicitudes, resumen


def cargar_datos_procesados(configuracion):
    ruta_procesado = ruta(configuracion, "processed") / "lgd_dataset.csv.gz"
    if not ruta_procesado.exists():
        raise FileNotFoundError(f"No existe {ruta_procesado}. Corra primero: python -m lgdmodel preparar --config <cfg>")
    return pd.read_csv(ruta_procesado, low_memory=False, parse_dates=["issue_d"])


def columnas_predictoras(configuracion):
    """Las mismas variables de solicitud que la Parte 1 (numéricas, categóricas)."""
    return app_columns(configuracion["datos"]["use_lc_pricing"])
