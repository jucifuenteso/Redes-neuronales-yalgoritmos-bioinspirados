"""
Algoritmo genetico con codificacion REAL para funciones continuas.

Misma estructura que el genetico de la Parte 2 (poblacion inicial,
elitismo, seleccion por torneo, cruce, mutacion, bucle generacional),
cambiando los operadores de permutaciones por operadores para vectores
reales:

  - Cruce BLX-alpha: cada gen del hijo se sortea uniforme en el intervalo
    [min - alpha d, max + alpha d], donde min, max son los valores de los
    dos padres en ese gen y d = max - min.
  - Mutacion gaussiana: cada gen se perturba, con probabilidad
    prob_mutacion, con ruido N(0, (sigma_mut * (hi - lo))^2).

Los hijos que se salen de D se recortan al borde.

Cuenta de evaluaciones: en cada generacion solo se evaluan los hijos
(tam_poblacion - n_elite); la elite conserva su valor ya calculado.
No usa grad f.
"""
import numpy as np


def genetico(problema, tam_poblacion, presupuesto, prob_cruce=0.9, prob_mutacion=0.2,
             sigma_mut=0.1, alpha_blx=0.5, elite_frac=0.1, tam_torneo=3,
             ftol=1e-10, paciencia=50, semilla=1983, guardar_trayectoria=False):
    rng = np.random.default_rng(semilla)
    lo, hi = problema.limites
    m, n = tam_poblacion, problema.n
    n_elite = max(1, int(round(m * elite_frac)))
    n_hijos = m - n_elite

    X = rng.uniform(lo, hi, size=(m, n))
    fX = problema.f(X)
    mejor_f = fX.min()

    historial = [(problema.n_f, problema.n_grad, mejor_f)]
    trayectoria = [X.copy()] if guardar_trayectoria else None
    referencia, sin_mejora = mejor_f, 0
    parada = "presupuesto"

    while problema.evals_equivalentes() + n_hijos <= presupuesto:
        # Seleccion por torneo: dos padres por hijo
        candidatos = rng.integers(0, m, size=(2 * n_hijos, tam_torneo))
        ganadores = candidatos[np.arange(2 * n_hijos), np.argmin(fX[candidatos], axis=1)]
        P1, P2 = X[ganadores[:n_hijos]], X[ganadores[n_hijos:]]

        # Cruce BLX-alpha
        minimo, maximo = np.minimum(P1, P2), np.maximum(P1, P2)
        d = maximo - minimo
        H = rng.uniform(minimo - alpha_blx * d, maximo + alpha_blx * d)
        sin_cruce = rng.random(n_hijos) >= prob_cruce
        H[sin_cruce] = P1[sin_cruce]

        # Mutacion gaussiana gen a gen
        muta = rng.random((n_hijos, n)) < prob_mutacion
        H = H + muta * rng.normal(0.0, sigma_mut * (hi - lo), size=(n_hijos, n))
        H = np.clip(H, lo, hi)

        fH = problema.f(H)

        elite = np.argsort(fX)[:n_elite]
        X = np.vstack([X[elite], H])
        fX = np.concatenate([fX[elite], fH])
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
