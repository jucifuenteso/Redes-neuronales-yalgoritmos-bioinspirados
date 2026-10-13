"""
Colonia de hormigas (Ant System) para el TSP.

Basado en:
  - ant_colony.ipynb (estructura general: generar ciudades, matriz de
    distancias/costos, optimizar, graficar la mejor ruta).
  - Notas manuscritas IRNA202002_Ant_Colony_Optimization.pdf (formulas):

      p_ij^k = (tau_ij^alpha * eta_ij^beta) / sum_{l no visitado} (tau_il^alpha * eta_il^beta)
      eta_ij = 1 / costo_ij          (atractivo = inverso del costo)
      delta_tau_ij^k = Q / L_k       (L_k = costo total del recorrido de la hormiga k)
      tau_ij <- (1 - rho) * tau_ij + sum_k delta_tau_ij^k

Reimplementado (no clona el repo externo de Berroa) para:
  - trabajar sobre la matriz de COSTO (no solo distancia),
  - exponer todos los hiperparametros via config.yaml,
  - recibir la matriz de costo ya construida (no toca archivos de disco),
  - guardar el historial de la mejor ruta por iteracion (para el GIF).
"""
import numpy as np

from costos import costo_ruta


class AntColonyTSP:
    def __init__(
        self,
        n_hormigas=47,
        n_iter=300,
        alpha=1.0,
        beta=2.0,
        rho=0.1,          # tasa de evaporacion
        Q=1.0,
        tau0=1.0,
        semilla=1983,
    ):
        self.n_hormigas = n_hormigas
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau0 = tau0
        self.semilla = semilla

    def fit(self, C, verbose=False):
        rng = np.random.default_rng(self.semilla)
        n = C.shape[0]

        eta = 1.0 / np.where(C == 0, np.inf, C)
        np.fill_diagonal(eta, 0.0)
        tau = np.full((n, n), self.tau0, dtype=float)

        mejor_ruta = None
        mejor_costo = np.inf
        historial_mejor_costo = []
        historial_mejor_ruta = []
        n_evaluaciones = 0

        for it in range(self.n_iter):
            rutas_it = np.zeros((self.n_hormigas, n), dtype=int)
            costos_it = np.zeros(self.n_hormigas)

            for k in range(self.n_hormigas):
                inicio = rng.integers(0, n)
                visitado = np.zeros(n, dtype=bool)
                ruta = np.empty(n, dtype=int)
                ruta[0] = inicio
                visitado[inicio] = True

                actual = inicio
                for paso in range(1, n):
                    no_visitadas = np.where(~visitado)[0]
                    pesos = (tau[actual, no_visitadas] ** self.alpha) * (
                        eta[actual, no_visitadas] ** self.beta
                    )
                    suma = pesos.sum()
                    probs = pesos / suma if suma > 0 else np.ones(len(pesos)) / len(pesos)
                    siguiente = rng.choice(no_visitadas, p=probs)

                    ruta[paso] = siguiente
                    visitado[siguiente] = True
                    actual = siguiente

                rutas_it[k] = ruta
                costos_it[k] = costo_ruta(ruta, C)
                n_evaluaciones += 1

            # Evaporacion + deposito de feromona
            tau *= (1 - self.rho)
            for k in range(self.n_hormigas):
                ruta = rutas_it[k]
                deposito = self.Q / costos_it[k]
                arcos_i = np.append(ruta, ruta[0])[:-1]
                arcos_j = np.append(ruta, ruta[0])[1:]
                tau[arcos_i, arcos_j] += deposito
                tau[arcos_j, arcos_i] += deposito

            idx_mejor_it = np.argmin(costos_it)
            if costos_it[idx_mejor_it] < mejor_costo:
                mejor_costo = costos_it[idx_mejor_it]
                mejor_ruta = rutas_it[idx_mejor_it].copy()

            historial_mejor_costo.append(mejor_costo)
            historial_mejor_ruta.append(mejor_ruta.copy())

            if verbose and (it % max(1, self.n_iter // 10) == 0):
                print(f"[ACO] iter {it:4d}  mejor costo = {mejor_costo:,.2f}")

        return {
            "mejor_ruta": mejor_ruta,
            "mejor_costo": mejor_costo,
            "historial_mejor_costo": historial_mejor_costo,
            "historial_mejor_ruta": historial_mejor_ruta,
            "n_evaluaciones": n_evaluaciones,
        }