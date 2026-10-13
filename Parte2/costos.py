"""
Construccion de la matriz de costo del TSP y evaluacion del costo de una ruta.

Costo(i, j) = valor_hora * tiempo_horas(i, j)
            + peaje_eur(i, j)
            + (distancia_km(i, j) / 100) * consumo_l_100km * precio_combustible_eur_l

(formula exigida en la rubrica, Parte 2, punto 2)
"""
import numpy as np
import pandas as pd


def cargar_datos(dir_datos="data"):
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
    El recorrido es cerrado: se suma tambien el regreso a la ciudad inicial."""
    ruta = np.asarray(ruta)
    costo = C[ruta[:-1], ruta[1:]].sum()
    costo += C[ruta[-1], ruta[0]]
    return float(costo)


def desglosar_costo_ruta(ruta, datos, valor_hora_eur, consumo_l_100km, precio_combustible_eur_l):
    """Separa cuanto del costo total viene de tiempo, peajes y combustible."""
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