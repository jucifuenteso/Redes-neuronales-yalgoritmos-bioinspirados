"""
Punto de entrada unico para la Parte 1 (optimizacion numerica de
Rosenbrock y Rastrigin en 2 y 3 dimensiones).

Lee config.yaml, corre el descenso por gradiente (paso fijo y
backtracking) y los tres heuristicos (GA, PSO, DE) con el mismo
presupuesto de evaluaciones, y guarda en resultados/ las tablas, las
figuras y un GIF por funcion.

Uso:
    python main.py
"""
import os

import yaml

from funciones import Problema
from experimentos import METODOS, correr, experimento, resumir, probabilidad_cuenca
from visualizacion import graficar_convergencia, graficar_valores_finales, animar_optimizacion

DIR_SALIDA = "resultados"


def miles(valor):
    """Entero con espacio como separador de miles (12 345)."""
    return f"{valor:,.0f}".replace(",", " ")


def tabla_markdown(estadisticas):
    """Tabla de estadisticas lista para pegar en el informe."""
    lineas = [
        "| Funcion | n | Metodo | Media | Desv. est. | Mejor | Peor | Exito (%) "
        "| Eval. f | Eval. grad f | Eval. equiv. | Eval. equiv. hasta el umbral |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, f in estadisticas.iterrows():
        hasta = f["evals_hasta_umbral_mediana"]
        hasta = "-" if hasta != hasta else miles(hasta)   # nan: ninguna corrida exitosa
        lineas.append(
            f"| {f['funcion'].capitalize()} | {f['n']} | {f['metodo']} | {f['media']:.2e} | {f['desv_est']:.2e} "
            f"| {f['mejor']:.2e} | {f['peor']:.2e} | {f['tasa_exito']:.1f} | {miles(f['n_f_media'])} "
            f"| {miles(f['n_grad_media'])} | {miles(f['evals_equiv_media'])} | {hasta} |"
        )
    return "\n".join(lineas) + "\n"


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(DIR_SALIDA, exist_ok=True)

    print(f"Corriendo {cfg['n_corridas']} corridas por configuracion...")
    corridas, curvas, rejillas = experimento(cfg)
    estadisticas = resumir(corridas)
    corridas.to_csv(f"{DIR_SALIDA}/corridas.csv", index=False)
    estadisticas.to_csv(f"{DIR_SALIDA}/estadisticas.csv", index=False)
    with open(f"{DIR_SALIDA}/tabla_resultados.md", "w", encoding="utf-8") as f:
        f.write(tabla_markdown(estadisticas))

    if "rastrigin" in cfg["funciones"]:
        print("\nEstimando la probabilidad de caer en la cuenca global de Rastrigin...")
        cuenca = probabilidad_cuenca(cfg)
        cuenca.to_csv(f"{DIR_SALIDA}/cuenca_rastrigin.csv", index=False)
        print(cuenca.to_string(index=False))

    print("\nGenerando figuras...")
    graficar_convergencia(curvas, rejillas, cfg, METODOS, f"{DIR_SALIDA}/convergencia.png")
    graficar_valores_finales(corridas, cfg, METODOS, f"{DIR_SALIDA}/valores_finales.png")

    # GIF en 2D: se anima la corrida 0 (la de la semilla base), sin escogerla.
    for funcion in cfg["funciones"]:
        print(f"Generando GIF de {funcion}...")
        resultados = {m: correr(m, funcion, 2, cfg["semilla"], cfg, guardar_trayectoria=True) for m in METODOS}
        animar_optimizacion(Problema(funcion, 2), resultados, cfg["umbral_exito"],
                            f"{DIR_SALIDA}/optimizacion_{funcion}_2d.gif", **cfg["gif"])

    print(f"\nListo. Resultados en {DIR_SALIDA}/: corridas.csv, estadisticas.csv, tabla_resultados.md, "
          "cuenca_rastrigin.csv, convergencia.png, valores_finales.png y un GIF por funcion.")


if __name__ == "__main__":
    main()
