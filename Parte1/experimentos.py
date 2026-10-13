"""
Protocolo experimental de la Parte 1 (regla de comparacion justa).

  - Todos los metodos reciben el mismo presupuesto de evaluaciones
    equivalentes de f:   evals_equivalentes = n_f + 2n * n_grad.
  - Cada configuracion (funcion, dimension, metodo) se corre n_corridas
    veces con semillas distintas. La corrida r usa la misma semilla en
    todos los metodos, asi que los tres heuristicos parten de la misma
    poblacion inicial y los dos descensos por gradiente del mismo x0.
  - Se reportan media, desviacion estandar, mejor y peor valor final, tasa
    de exito, y las evaluaciones de f y de grad f por separado.
"""
import numpy as np
import pandas as pd

from funciones import Problema
from descenso_gradiente import descenso_gradiente
from ga_continuo import genetico
from pso import pso
from evolucion_diferencial import evolucion_diferencial

METODOS = ["GD paso fijo", "GD backtracking", "GA", "PSO", "DE"]
HEURISTICOS = {"GA": genetico, "PSO": pso, "DE": evolucion_diferencial}


def correr(metodo, funcion, n, semilla, cfg, guardar_trayectoria=False):
    """Una corrida de un metodo. Devuelve el resultado del metodo mas las
    cuentas de evaluaciones."""
    problema = Problema(funcion, n)
    comun = {"presupuesto": cfg["presupuesto_por_dim"] * n, "guardar_trayectoria": guardar_trayectoria}

    if metodo in HEURISTICOS:
        res = HEURISTICOS[metodo](problema, cfg["pob_por_dim"] * n, semilla=semilla,
                                  **cfg["parada"], **cfg[metodo.lower()], **comun)
    else:
        x0 = np.random.default_rng(semilla).uniform(*problema.limites, size=n)
        if metodo == "GD paso fijo":
            res = descenso_gradiente(problema, x0, paso=cfg["gd_paso_fijo"]["paso"][funcion],
                                     tol=cfg["gd_paso_fijo"]["tol"], **comun)
        else:
            res = descenso_gradiente(problema, x0, backtracking=True, **cfg["gd_backtracking"], **comun)

    res["n_f"] = problema.n_f
    res["n_grad"] = problema.n_grad
    res["evals_equiv"] = problema.evals_equivalentes()
    # Evaluaciones equivalentes acumuladas en cada punto del historial
    res["evals_hist"] = res["historial"][:, 0] + problema.costo_grad * res["historial"][:, 1]
    res["error_hist"] = res["historial"][:, 2] - problema.f_opt
    return res


def evals_hasta_umbral(res, umbral):
    """Evaluaciones equivalentes gastadas la primera vez que f - f* <= umbral
    (nan si nunca se alcanza)."""
    alcanzado = np.where(res["error_hist"] <= umbral)[0]
    return res["evals_hist"][alcanzado[0]] if len(alcanzado) else np.nan


def curva_en_rejilla(res, rejilla):
    """Valor de f - f* despues de gastar cada presupuesto de la rejilla
    (nan si el metodo aun no ha evaluado nada con ese presupuesto)."""
    idx = np.searchsorted(res["evals_hist"], rejilla, side="right") - 1
    return np.where(idx >= 0, res["error_hist"][np.maximum(idx, 0)], np.nan)


def experimento(cfg, verbose=True):
    """Corre todas las configuraciones.

    Devuelve:
      corridas : DataFrame con una fila por corrida.
      curvas   : dict (funcion, n, metodo) -> array (n_corridas, len(rejilla)).
      rejillas : dict (funcion, n) -> presupuestos donde se muestrean las curvas.
    """
    umbral = cfg["umbral_exito"]
    filas, curvas, rejillas = [], {}, {}

    for funcion in cfg["funciones"]:
        for n in cfg["dimensiones"]:
            presupuesto = cfg["presupuesto_por_dim"] * n
            rejilla = np.unique(np.round(np.geomspace(1, presupuesto, 300)))
            rejillas[(funcion, n)] = rejilla

            for metodo in METODOS:
                curvas_metodo = []
                for r in range(cfg["n_corridas"]):
                    res = correr(metodo, funcion, n, cfg["semilla"] + r, cfg)
                    error = res["error_hist"][-1]
                    filas.append({
                        "funcion": funcion, "n": n, "metodo": metodo, "corrida": r,
                        "f_final": error,
                        "exito": error <= umbral,
                        "n_f": res["n_f"],
                        "n_grad": res["n_grad"],
                        "evals_equiv": res["evals_equiv"],
                        "evals_hasta_umbral": evals_hasta_umbral(res, umbral) if error <= umbral else np.nan,
                        "parada": res["parada"],
                    })
                    curvas_metodo.append(curva_en_rejilla(res, rejilla))
                curvas[(funcion, n, metodo)] = np.array(curvas_metodo)
                if verbose:
                    ultimas = pd.DataFrame(filas[-cfg["n_corridas"]:])
                    print(f"  {funcion:10s} {n}D  {metodo:16s} media = {ultimas['f_final'].mean():.2e}"
                          f"   exito = {100 * ultimas['exito'].mean():5.1f} %")

    return pd.DataFrame(filas), curvas, rejillas


def resumir(corridas):
    """Estadisticas por configuracion (funcion, n, metodo)."""
    filas = []
    for (funcion, n, metodo), g in corridas.groupby(["funcion", "n", "metodo"], sort=False):
        filas.append({
            "funcion": funcion, "n": n, "metodo": metodo,
            "media": g["f_final"].mean(),
            "desv_est": g["f_final"].std(ddof=1),
            "mejor": g["f_final"].min(),
            "peor": g["f_final"].max(),
            "tasa_exito": 100 * g["exito"].mean(),
            "n_f_media": g["n_f"].mean(),
            "n_grad_media": g["n_grad"].mean(),
            "evals_equiv_media": g["evals_equiv"].mean(),
            # Mediana, entre las corridas exitosas, de lo gastado al cruzar el umbral
            "evals_hasta_umbral_mediana": g["evals_hasta_umbral"].median() if g["exito"].any() else np.nan,
        })
    return pd.DataFrame(filas)


def probabilidad_cuenca(cfg):
    """Muchas corridas de descenso por gradiente en Rastrigin, cada una desde
    un x0 aleatorio distinto, para estimar con que probabilidad se cae en la
    cuenca del minimo global y cuanto costaria encontrarlo a punta de reinicios.
    """
    n_corridas = cfg["cuenca_rastrigin"]["n_corridas"]
    filas = []
    for n in cfg["dimensiones"]:
        for metodo in ["GD paso fijo", "GD backtracking"]:
            exitos, evals = 0, 0
            for i in range(n_corridas):
                # semillas lejos de las del experimento principal
                res = correr(metodo, "rastrigin", n, cfg["semilla"] + 100000 + i, cfg)
                exitos += res["error_hist"][-1] <= cfg["umbral_exito"]
                evals += res["evals_equiv"]
            p = exitos / n_corridas
            evals_media = evals / n_corridas
            filas.append({
                "n": n, "metodo": metodo, "n_corridas": n_corridas, "exitos": exitos,
                "prob_exito": p,
                "evals_equiv_por_corrida": evals_media,
                # numero esperado de reinicios = 1/p
                "evals_equiv_esperadas_con_reinicios": evals_media / p if p > 0 else np.nan,
            })
    return pd.DataFrame(filas)
