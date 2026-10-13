"""
Funciones de prueba de la Parte 1 y contador de evaluaciones.

Se eligieron Rosenbrock y Rastrigin (ver README_P1.md, seccion 1):

    Rosenbrock:  f(x) = sum_{i=1}^{n-1} [ 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
                 minimo global f = 0 en x = (1, ..., 1)
    Rastrigin:   f(x) = 10 n + sum_{i=1}^{n} [ x_i^2 - 10 cos(2 pi x_i) ]
                 minimo global f = 0 en x = (0, ..., 0)

Las dos estan definidas para cualquier dimension n, tienen gradiente
analitico sencillo y optimo conocido.

La clase Problema envuelve la funcion y lleva la cuenta, por separado, de
las evaluaciones de f y de grad f (regla de comparacion justa).
"""
import numpy as np


def rosenbrock(X):
    """X: array (m, n). Devuelve array (m,)."""
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def grad_rosenbrock(x):
    d = x[1:] - x[:-1] ** 2
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * d - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * d
    return g


def rastrigin(X):
    """X: array (m, n). Devuelve array (m,)."""
    return 10.0 * X.shape[1] + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def grad_rastrigin(x):
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


FUNCIONES = {
    "rosenbrock": {
        "f": rosenbrock,
        "grad": grad_rosenbrock,
        "limites": (-2.048, 2.048),
        "optimo": lambda n: np.ones(n),
        "f_opt": 0.0,
    },
    "rastrigin": {
        "f": rastrigin,
        "grad": grad_rastrigin,
        "limites": (-5.12, 5.12),
        "optimo": lambda n: np.zeros(n),
        "f_opt": 0.0,
    },
}


class Problema:
    """min f(x) sujeto a x en D = [lo, hi]^n, contando evaluaciones.

    n_f    : numero de evaluaciones de f (una por punto evaluado).
    n_grad : numero de evaluaciones de grad f.

    Equivalencia usada para comparar metodos: una evaluacion de grad f
    cuesta 2n evaluaciones de f (lo que costaria aproximarlo por
    diferencias centrales en dimension n).
    """

    def __init__(self, nombre, n):
        spec = FUNCIONES[nombre]
        self.nombre = nombre
        self.n = n
        self.limites = spec["limites"]
        self.x_opt = spec["optimo"](n)
        self.f_opt = spec["f_opt"]
        self._f = spec["f"]
        self._grad = spec["grad"]
        self.costo_grad = 2 * n
        self.n_f = 0
        self.n_grad = 0

    def f(self, X, contar=True):
        """Evalua f en un punto (n,) o en una poblacion (m, n).

        contar=False se usa solo para registrar la curva de convergencia de
        un metodo que no necesita f (descenso por gradiente con paso fijo).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            if contar:
                self.n_f += 1
            return float(self._f(X[None, :])[0])
        if contar:
            self.n_f += X.shape[0]
        return self._f(X)

    def grad(self, x):
        self.n_grad += 1
        return self._grad(np.asarray(x, dtype=float))

    def evals_equivalentes(self):
        return self.n_f + self.costo_grad * self.n_grad
