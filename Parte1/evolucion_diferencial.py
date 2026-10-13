"""
Evolucion diferencial, variante clasica DE/rand/1/bin (Storn y Price, 1997).

Para cada individuo x_i de la poblacion:

    mutacion:   v_i = x_r1 + F (x_r2 - x_r3)      r1, r2, r3 distintos entre si y de i
    cruce:      u_ij = v_ij  si rand_j < CR o j = j_rand,   x_ij en otro caso
    seleccion:  x_i <- u_i   si f(u_i) <= f(x_i)

Los mutantes que se salen de D se recortan al borde. La seleccion es uno a
uno (cada hijo compite solo contra su padre), asi que el mejor de la
poblacion nunca empeora.

Cuenta de evaluaciones: una evaluacion de f por individuo por generacion
(mas las de la poblacion inicial). No usa grad f.
"""
import numpy as np


def evolucion_diferencial(problema, tam_poblacion, presupuesto, F=0.5, CR=0.9,
                          ftol=1e-10, paciencia=50, semilla=1983, guardar_trayectoria=False):
    rng = np.random.default_rng(semilla)
    lo, hi = problema.limites
    m, n = tam_poblacion, problema.n
    filas = np.arange(m)

    X = rng.uniform(lo, hi, size=(m, n))
    fX = problema.f(X)
    mejor_f = fX.min()

    historial = [(problema.n_f, problema.n_grad, mejor_f)]
    trayectoria = [X.copy()] if guardar_trayectoria else None
    referencia, sin_mejora = mejor_f, 0
    parada = "presupuesto"

    while problema.evals_equivalentes() + m <= presupuesto:
        # Tres indices distintos por fila, elegidos entre los otros m-1 individuos
        r = np.argsort(rng.random((m, m - 1)), axis=1)[:, :3]
        r += r >= filas[:, None]
        mutante = np.clip(X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]]), lo, hi)

        cruce = rng.random((m, n)) < CR
        cruce[filas, rng.integers(0, n, size=m)] = True
        U = np.where(cruce, mutante, X)

        fU = problema.f(U)
        gana = fU <= fX
        X[gana], fX[gana] = U[gana], fU[gana]
        mejor_f = fX.min()

        historial.append((problema.n_f, problema.n_grad, mejor_f))
        if guardar_trayectoria:
            trayectoria.append(X.copy())

        if mejor_f < referencia - ftol:
            referencia, sin_mejora = mejor_f, 0
        else:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            parada = "estancamiento"
            break

    i_mejor = np.argmin(fX)
    return {
        "mejor_x": X[i_mejor].copy(),
        "mejor_f": fX[i_mejor],
        "historial": np.array(historial, dtype=float),
        "trayectoria": trayectoria,
        "parada": parada,
    }
