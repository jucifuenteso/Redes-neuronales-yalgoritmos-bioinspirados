"""
Optimizacion por enjambre de particulas (PSO), topologia gbest.

Basado en las diapositivas "Particle Swarm Optimization (PSO) 2026-2S":

    v_i <- w v_i + c1 r1 (.) (b_i - x_i) + c2 r2 (.) (g - x_i)
    x_i <- x_i + v_i

  - b_i = mejor posicion de la particula i (pbest), g = mejor del enjambre (gbest).
  - r1, r2 ~ U(0,1)^p, un numero por dimension, nuevos en cada iteracion.
  - Inicializacion uniforme en D con v = 0.
  - Velocidad limitada a v_max = k (x_max - x_min) / 2.
  - Si una particula se sale de D se ubica en el borde y su velocidad se pone en cero.
  - Parada: presupuesto de evaluaciones o g sin mejorar durante `paciencia` iteraciones.

Los valores por defecto w = 0.729, c1 = c2 = 1.494 son los del coeficiente
de constriccion de Clerc y Kennedy (2002).

Cuenta de evaluaciones: una evaluacion de f por particula por iteracion
(mas las de la poblacion inicial). No usa grad f.
"""
import numpy as np


def pso(problema, tam_enjambre, presupuesto, w=0.729, c1=1.494, c2=1.494, k_vmax=0.5,
        ftol=1e-10, paciencia=50, semilla=1983, guardar_trayectoria=False):
    rng = np.random.default_rng(semilla)
    lo, hi = problema.limites
    m, n = tam_enjambre, problema.n
    v_max = k_vmax * (hi - lo) / 2.0

    X = rng.uniform(lo, hi, size=(m, n))
    V = np.zeros((m, n))
    fX = problema.f(X)

    B, fB = X.copy(), fX.copy()          # pbest
    i_g = np.argmin(fB)
    g, fg = B[i_g].copy(), fB[i_g]       # gbest

    historial = [(problema.n_f, problema.n_grad, fg)]
    trayectoria = [X.copy()] if guardar_trayectoria else None
    referencia, sin_mejora = fg, 0
    parada = "presupuesto"

    while problema.evals_equivalentes() + m <= presupuesto:
        R1 = rng.random((m, n))
        R2 = rng.random((m, n))
        V = w * V + c1 * R1 * (B - X) + c2 * R2 * (g - X)
        V = np.clip(V, -v_max, v_max)
        X = X + V

        fuera = (X < lo) | (X > hi)
        X = np.clip(X, lo, hi)
        V[fuera] = 0.0

        fX = problema.f(X)

        mejora = fX < fB
        B[mejora], fB[mejora] = X[mejora], fX[mejora]
        i_g = np.argmin(fB)
        if fB[i_g] < fg:
            g, fg = B[i_g].copy(), fB[i_g]

        historial.append((problema.n_f, problema.n_grad, fg))
        if guardar_trayectoria:
            trayectoria.append(X.copy())

        if fg < referencia - ftol:
            referencia, sin_mejora = fg, 0
        else:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            parada = "estancamiento"
            break

    return {
        "mejor_x": g,
        "mejor_f": fg,
        "historial": np.array(historial, dtype=float),
        "trayectoria": trayectoria,
        "parada": parada,
    }
