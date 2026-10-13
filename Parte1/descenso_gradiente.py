"""
Descenso por gradiente con dos reglas para el tamano de paso.

Basado en las notas de clase "Optimizacion numerica (descenso por
gradiente y Newton)":

  - Paso fijo (Algoritmo 3.2):
        x_{k+1} = x_k - alpha * grad f(x_k)
  - Backtracking (Algoritmo 4.6): se reduce alpha <- beta * alpha hasta
    cumplir la condicion de Armijo
        f(x_k + alpha p_k) <= f(x_k) + c1 * alpha * grad f(x_k)^T p_k

Como el problema es min f(x) sujeto a x en D (una caja), cada iterado se
proyecta sobre D con np.clip. Asi el gradiente busca en la misma region
que los metodos heuristicos. Lejos del borde la proyeccion no hace nada y
se recupera el algoritmo de las notas.

Cuenta de evaluaciones:
  - paso fijo:    1 evaluacion de grad f por iteracion, 0 de f.
                  (f se evalua solo para dibujar la curva; no se cuenta)
  - backtracking: 1 evaluacion de grad f por iteracion + 1 de f por cada
                  paso de prueba.
"""
import numpy as np


def descenso_gradiente(problema, x0, presupuesto, paso=1e-3, backtracking=False,
                       paso_max=1.0, beta=0.5, c1=1e-4, tol=1e-6,
                       guardar_trayectoria=False):
    lo, hi = problema.limites
    x = np.array(x0, dtype=float)

    # Armijo necesita f(x_k); con paso fijo f(x_k) es solo monitoreo.
    fx = problema.f(x, contar=backtracking)
    alpha = paso_max if backtracking else paso

    historial = [(problema.n_f, problema.n_grad, fx)]
    trayectoria = [x.copy()] if guardar_trayectoria else None
    parada = "presupuesto"

    while True:
        costo_iter = problema.costo_grad + (1 if backtracking else 0)
        if problema.evals_equivalentes() + costo_iter > presupuesto:
            break

        g = problema.grad(x)

        if backtracking:
            # Se arranca desde el doble del ultimo paso aceptado (no desde
            # paso_max) para no gastar evaluaciones de f repitiendo recortes.
            alpha = min(paso_max, alpha / beta)
            aceptado = False
            while problema.evals_equivalentes() + 1 <= presupuesto and alpha > 1e-14:
                x_nuevo = np.clip(x - alpha * g, lo, hi)
                f_nuevo = problema.f(x_nuevo)
                # Armijo sobre el paso proyectado: g^T (x_nuevo - x) = -alpha ||g||^2 lejos del borde
                if f_nuevo <= fx + c1 * g @ (x_nuevo - x):
                    aceptado = True
                    break
                alpha *= beta
            if not aceptado:
                break
        else:
            x_nuevo = np.clip(x - alpha * g, lo, hi)
            f_nuevo = problema.f(x_nuevo, contar=False)

        # Norma del gradiente proyectado (igual a ||grad f|| lejos del borde)
        norma_g = np.linalg.norm(x_nuevo - x) / alpha
        x, fx = x_nuevo, f_nuevo

        historial.append((problema.n_f, problema.n_grad, fx))
        if guardar_trayectoria:
            trayectoria.append(x.copy())

        if norma_g < tol:
            parada = "gradiente"
            break

    return {
        "mejor_x": x,
        "mejor_f": fx,
        "historial": np.array(historial, dtype=float),
        "trayectoria": trayectoria,
        "parada": parada,
    }
