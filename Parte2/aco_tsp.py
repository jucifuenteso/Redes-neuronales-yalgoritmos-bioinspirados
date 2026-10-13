"""
Colonia de hormigas (Ant System) para el TSP.

Basado en:
  - ant_colony.ipynb (estructura general: ciudades, matriz de costos,
    optimizar, graficar la mejor ruta).
  - Notas manuscritas IRNA202002_Ant_Colony_Optimization.pdf (formulas):

      p_ij^k = (tau_ij^alpha * eta_ij^beta) / sum_{l no visitado} (tau_il^alpha * eta_il^beta)
      eta_ij = 1 / costo_ij          (atractivo = inverso del costo)
      delta_tau_ij^k = Q / L_k       (L_k = costo total del recorrido de la hormiga k)
      tau_ij <- (1 - rho) * tau_ij + sum_k delta_tau_ij^k

Version simple: una sola funcion, sin clases, priorizando que la logica
matematica sea clara y facil de seguir.
"""
import numpy as np

from costos import costo_ruta


def aco_tsp(C, n_hormigas=47, n_iter=300, alpha=1.0, beta=2.0, rho=0.1, Q=1.0, tau0=1.0, semilla=1983):
    rng = np.random.default_rng(semilla)
    n = C.shape[0]

    eta = 1.0 / np.where(C == 0, np.inf, C)  # atractivo = inverso del costo
    np.fill_diagonal(eta, 0.0)
    tau = np.full((n, n), tau0, dtype=float)

    # Se inicializa con una ruta valida (orden 0..n-1) en vez de None,
    # asi mejor_ruta siempre es un array real.
    mejor_ruta = np.arange(n)
    mejor_costo = costo_ruta(mejor_ruta, C)

    historial_mejor_costo = []
    historial_mejor_ruta = []
    n_evaluaciones = 0

    for it in range(n_iter):
        rutas_it = np.zeros((n_hormigas, n), dtype=int)
        costos_it = np.zeros(n_hormigas)

        for k in range(n_hormigas):
            inicio = rng.integers(0, n)
            visitado = np.zeros(n, dtype=bool)
            ruta = np.empty(n, dtype=int)
            ruta[0] = inicio
            visitado[inicio] = True
            actual = inicio

            for paso in range(1, n):
                no_visitadas = np.where(~visitado)[0]
                pesos = (tau[actual, no_visitadas] ** alpha) * (eta[actual, no_visitadas] ** beta)
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
        tau *= (1 - rho)
        for k in range(n_hormigas):
            ruta = rutas_it[k]
            deposito = Q / costos_it[k]
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

    return {
        "mejor_ruta": mejor_ruta,
        "mejor_costo": mejor_costo,
        "historial_mejor_costo": historial_mejor_costo,
        "historial_mejor_ruta": historial_mejor_ruta,
        "n_evaluaciones": n_evaluaciones,
    }