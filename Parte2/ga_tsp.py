"""
Algoritmo genetico para el TSP con codificacion por PERMUTACION.

Basado en:
  - prep_Algoritmos_Evolutivos.ipynb (estructura general: poblacion
    inicial, ranking, elitismo, mutacion, cruzamiento, bucle generacional).
    Ese notebook solo cubre codificacion real (funciones continuas) y
    binaria/entera (knapsack), no permutaciones, asi que se adapta aqui.
  - Notas manuscritas de ACO, seccion de codificacion de TSP para GA
    (ejemplo padre/madre/hijo con permutaciones de ciudades):
      - Cruzamiento: Order Crossover (OX) -> conserva un segmento del
        padre y completa con el orden relativo de la madre, sin repetir
        ciudades.
      - Mutacion: swap -> intercambia la posicion de dos ciudades al azar.
"""
import numpy as np

from costos import costo_ruta


def poblacion_inicial(n_poblacion, n_ciudades, rng):
    pob = np.tile(np.arange(n_ciudades), (n_poblacion, 1))
    for i in range(n_poblacion):
        rng.shuffle(pob[i])
    return pob


def evaluar_poblacion(poblacion, C):
    return np.array([costo_ruta(ind, C) for ind in poblacion])


def seleccion_torneo(poblacion, costos, k, rng):
    idx = rng.integers(0, len(poblacion), size=k)
    mejor = idx[np.argmin(costos[idx])]
    return poblacion[mejor]


def cruce_ox(padre, madre, rng):
    n = len(padre)
    i, j = sorted(rng.integers(0, n, size=2))
    hijo = -np.ones(n, dtype=int)
    hijo[i:j] = padre[i:j]

    resto = [g for g in madre if g not in hijo[i:j]]
    pos = 0
    for k in range(n):
        if hijo[k] == -1:
            hijo[k] = resto[pos]
            pos += 1
    return hijo


def mutacion_swap(individuo, rng):
    i, j = rng.integers(0, len(individuo), size=2)
    individuo = individuo.copy()
    individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


class GeneticoTSP:
    def __init__(
        self,
        tam_poblacion=150,
        n_generaciones=400,
        prob_cruce=0.9,
        prob_mutacion=0.2,
        elite_frac=0.1,
        tam_torneo=3,
        semilla=1983,
    ):
        self.tam_poblacion = tam_poblacion
        self.n_generaciones = n_generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.elite_frac = elite_frac
        self.tam_torneo = tam_torneo
        self.semilla = semilla

    def fit(self, C, verbose=False):
        rng = np.random.default_rng(self.semilla)
        n_ciudades = C.shape[0]
        n_elite = max(1, int(round(self.tam_poblacion * self.elite_frac)))

        poblacion = poblacion_inicial(self.tam_poblacion, n_ciudades, rng)
        costos = evaluar_poblacion(poblacion, C)
        n_evaluaciones = len(poblacion)

        idx_orden = np.argsort(costos)
        mejor_ruta = poblacion[idx_orden[0]].copy()
        mejor_costo = costos[idx_orden[0]]

        historial_mejor_costo = [mejor_costo]
        historial_mejor_ruta = [mejor_ruta.copy()]

        for gen in range(self.n_generaciones):
            idx_orden = np.argsort(costos)
            poblacion = poblacion[idx_orden]
            costos = costos[idx_orden]

            nueva_poblacion = [poblacion[i].copy() for i in range(n_elite)]

            while len(nueva_poblacion) < self.tam_poblacion:
                padre = seleccion_torneo(poblacion, costos, self.tam_torneo, rng)
                madre = seleccion_torneo(poblacion, costos, self.tam_torneo, rng)

                if rng.random() < self.prob_cruce:
                    hijo = cruce_ox(padre, madre, rng)
                else:
                    hijo = padre.copy()

                if rng.random() < self.prob_mutacion:
                    hijo = mutacion_swap(hijo, rng)

                nueva_poblacion.append(hijo)

            poblacion = np.array(nueva_poblacion[: self.tam_poblacion])
            costos = evaluar_poblacion(poblacion, C)
            n_evaluaciones += len(poblacion) - n_elite

            idx_mejor = np.argmin(costos)
            if costos[idx_mejor] < mejor_costo:
                mejor_costo = costos[idx_mejor]
                mejor_ruta = poblacion[idx_mejor].copy()

            historial_mejor_costo.append(mejor_costo)
            historial_mejor_ruta.append(mejor_ruta.copy())

            if verbose and (gen % max(1, self.n_generaciones // 10) == 0):
                print(f"[GA] gen {gen:4d}  mejor costo = {mejor_costo:,.2f}")

        return {
            "mejor_ruta": mejor_ruta,
            "mejor_costo": mejor_costo,
            "historial_mejor_costo": historial_mejor_costo,
            "historial_mejor_ruta": historial_mejor_ruta,
            "n_evaluaciones": n_evaluaciones,
        }