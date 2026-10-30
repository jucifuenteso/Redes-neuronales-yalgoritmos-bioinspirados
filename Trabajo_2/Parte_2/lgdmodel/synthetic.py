"""CSV SINTÉTICO para la demostración: el de la Parte 1 más las columnas de recuperación.

Las filas base salen de pdmodel.synthetic.generate con la misma semilla y el mismo número de filas que la
demostración de la Parte 1, así que las solicitudes, los estados y las particiones son los mismos.
Las columnas de recuperación usan otro generador aleatorio para no alterar esas filas.
La LGD sintética depende (de forma no lineal) de variables de solicitud solo para que la demostración tenga
señal que aprender. NADA de esto se reporta: los resultados del trabajo salen del loan.csv real.
"""
import numpy as np
import pandas as pd
from pdmodel.synthetic import generate as generar_solicitudes_parte1

ESTADOS_CASTIGADOS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]
ESTADOS_MALOS_SIN_CASTIGO = ["Default", "Late (31-120 days)"]
ESTADOS_PAGADOS = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]


def _sigmoide(valores):
    return 1.0 / (1.0 + np.exp(-valores))


def _beta_con_media(generador, media, concentracion):
    media = np.clip(media, 0.02, 0.98)
    return generador.beta(media * concentracion, (1.0 - media) * concentracion)


def generar(n_filas=60000, semilla=42):
    prestamos = generar_solicitudes_parte1(n_filas, semilla)
    generador = np.random.RandomState(semilla + 2)
    n_prestamos = len(prestamos)
    monto = prestamos["loan_amnt"].to_numpy(dtype=float)
    plazo_meses = prestamos["term"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    deuda_sobre_ingreso = prestamos["dti"].to_numpy(dtype=float)
    ingreso_anual = prestamos["annual_inc"].to_numpy(dtype=float)
    tiene_vivienda = prestamos["home_ownership"].isin(["OWN", "MORTGAGE"]).to_numpy()
    es_pequena_empresa = (prestamos["purpose"] == "small_business").to_numpy()
    estado = prestamos["loan_status"].to_numpy()

    # Fracción del capital pagada antes de incumplir: menor en plazos largos y con DTI alto.
    media_fraccion_pagada = _sigmoide(0.2 - 0.9 * (plazo_meses == 60) - 0.04 * (deuda_sobre_ingreso - 17))
    fraccion_pagada = _beta_con_media(generador, media_fraccion_pagada, 6.0)
    fraccion_pagada = np.where(np.isin(estado, ESTADOS_PAGADOS), 1.0, fraccion_pagada)

    # Tasa de recuperación del saldo tras el castigo: depende de vivienda, propósito, ingreso y una interacción.
    indice_recuperacion = (-1.4 + 0.7 * tiene_vivienda - 0.6 * es_pequena_empresa
                           + 0.5 * np.tanh(2.0 * (np.log(ingreso_anual) - 11.0))
                           - 0.6 * ((deuda_sobre_ingreso > 25) & (plazo_meses == 60)))
    tasa_recuperacion = _beta_con_media(generador, _sigmoide(indice_recuperacion), 5.0)
    sin_recuperacion = generador.rand(n_prestamos) < 0.35
    saldo_al_castigo = monto * (1.0 - fraccion_pagada)
    es_castigado = np.isin(estado, ESTADOS_CASTIGADOS)
    recuperacion_bruta = np.where(es_castigado & ~sin_recuperacion, saldo_al_castigo * tasa_recuperacion, 0.0)

    prestamos["funded_amnt"] = monto
    prestamos["total_rec_prncp"] = np.round(monto * fraccion_pagada, 2)
    prestamos["recoveries"] = np.round(recuperacion_bruta, 2)
    prestamos["collection_recovery_fee"] = np.round(recuperacion_bruta * generador.uniform(0.05, 0.2, n_prestamos), 2)
    return prestamos


def escribir(ruta_archivo, n_filas, semilla):
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    prestamos = generar(n_filas, semilla)
    prestamos.to_csv(ruta_archivo, index=False)
    return prestamos
