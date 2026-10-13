"""
Genera la matriz de costo del TSP (data/matriz_costo_47x47.csv) a partir de
las 3 tablas crudas (tiempos, peajes, distancias), y deja disponibles las
funciones para evaluar el costo de una ruta (usadas por aco_tsp.py y
ga_tsp.py en cada iteracion/generacion).

Costo(i, j) = valor_hora * tiempo_horas(i, j)
            + peaje_eur(i, j)
            + (distancia_km(i, j) / 100) * consumo_l_100km * precio_combustible_eur_l

(formula exigida en la rubrica, Parte 2, punto 2)

Correr una sola vez (o cuando cambien el vehiculo/valor de la hora):
    python costos.py
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parametros del vehiculo y valor de la hora (ajustar aqui antes de correr)
# ---------------------------------------------------------------------------
VALOR_HORA_EUR = 20.0
CONSUMO_L_100KM = 6.5
PRECIO_COMBUSTIBLE_EUR_L = 1.75

DIR_DATOS = "."
SALIDA_MATRIZ_COSTO = f"{DIR_DATOS}/matriz_costo_47x47.csv"


def cargar_datos(dir_datos=DIR_DATOS):
    coordenadas = pd.read_csv(f"{dir_datos}/coordenadas_47_ciudades.csv")
    nombres = coordenadas["ciudad"].tolist()

    tiempos = pd.read_csv(f"{dir_datos}/matriz_tiempos_47x47.csv", index_col=0)
    peajes = pd.read_csv(f"{dir_datos}/matriz_peajes_47x47.csv", index_col=0)
    distancias = pd.read_csv(f"{dir_datos}/matriz_distancias_47x47.csv", index_col=0)

    return {
        "nombres": nombres,
        "coords": coordenadas[["latitud", "longitud"]].to_numpy(dtype=float),
        "tiempos_horas": tiempos.to_numpy(dtype=float),
        "peajes_eur": peajes.to_numpy(dtype=float),
        "distancia_km": distancias.to_numpy(dtype=float),
    }


def construir_matriz_costo(datos, valor_hora_eur, consumo_l_100km, precio_combustible_eur_l):
    tiempo_total = valor_hora_eur * datos["tiempos_horas"]
    peaje_total = datos["peajes_eur"]
    combustible_total = (datos["distancia_km"] / 100.0) * consumo_l_100km * precio_combustible_eur_l

    C = tiempo_total + peaje_total + combustible_total
    np.fill_diagonal(C, 0.0)
    return C


def costo_ruta(ruta, C):
    """ruta: lista/array de indices de ciudades (permutacion de 0..n-1).
    El recorrido es cerrado: se suma tambien el regreso a la ciudad inicial.
    Usada por aco_tsp.py y ga_tsp.py en cada iteracion/generacion."""
    ruta = np.asarray(ruta)
    costo = C[ruta[:-1], ruta[1:]].sum()
    costo += C[ruta[-1], ruta[0]]
    return float(costo)


def desglosar_costo_ruta(ruta, datos, valor_hora_eur, consumo_l_100km, precio_combustible_eur_l):
    """Separa cuanto del costo total de una ruta viene de tiempo, peajes
    y combustible. Util para el reporte final, no se usa en la optimizacion."""
    ruta = np.asarray(ruta)
    i = np.append(ruta, ruta[0])
    pares_i, pares_j = i[:-1], i[1:]

    horas = datos["tiempos_horas"][pares_i, pares_j].sum()
    peajes = datos["peajes_eur"][pares_i, pares_j].sum()
    km = datos["distancia_km"][pares_i, pares_j].sum()

    return {
        "horas_totales": float(horas),
        "costo_tiempo_eur": float(valor_hora_eur * horas),
        "costo_peajes_eur": float(peajes),
        "km_totales": float(km),
        "costo_combustible_eur": float((km / 100.0) * consumo_l_100km * precio_combustible_eur_l),
    }


if __name__ == "__main__":
    datos = cargar_datos()
    C = construir_matriz_costo(datos, VALOR_HORA_EUR, CONSUMO_L_100KM, PRECIO_COMBUSTIBLE_EUR_L)

    df_costo = pd.DataFrame(C, index=datos["nombres"], columns=datos["nombres"])
    df_costo.to_csv(SALIDA_MATRIZ_COSTO)

    print(f"Parametros usados: valor_hora={VALOR_HORA_EUR} EUR/h, "
          f"consumo={CONSUMO_L_100KM} l/100km, precio={PRECIO_COMBUSTIBLE_EUR_L} EUR/l")
    print(f"Matriz de costo generada: {SALIDA_MATRIZ_COSTO}  ({df_costo.shape[0]}x{df_costo.shape[1]})")
    print(df_costo.iloc[:5, :5])