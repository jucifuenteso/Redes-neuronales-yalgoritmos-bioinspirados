"""
Punto de entrada unico para la Parte 2 (TSP de las 47 capitales peninsulares).

Lee config.yaml, construye la matriz de costo, corre ACO y GA, y guarda
las figuras y el GIF de evolucion de cada uno.

Uso:
    python main.py
"""
import yaml
import matplotlib.pyplot as plt

from costos import cargar_datos, construir_matriz_costo, desglosar_costo_ruta
from aco_tsp import aco_tsp
from ga_tsp import GeneticoTSP
from visualizacion import graficar_ruta, animar_evolucion, graficar_convergencia


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    datos = cargar_datos(cfg["dir_datos"])

    C = construir_matriz_costo(
        datos,
        valor_hora_eur=cfg["valor_hora_eur"],
        consumo_l_100km=cfg["vehiculo"]["consumo_l_100km"],
        precio_combustible_eur_l=cfg["vehiculo"]["precio_combustible_eur_l"],
    )

    print("Corriendo ACO...")
    res_aco = aco_tsp(C, semilla=cfg["semilla"], **cfg["aco"])

    print("Corriendo GA...")
    res_ga = GeneticoTSP(semilla=cfg["semilla"], **cfg["ga"]).fit(C)

    for nombre, res in [("aco", res_aco), ("ga", res_ga)]:
        desglose = desglosar_costo_ruta(
            res["mejor_ruta"], datos, cfg["valor_hora_eur"],
            cfg["vehiculo"]["consumo_l_100km"], cfg["vehiculo"]["precio_combustible_eur_l"],
        )
        print(f"\n[{nombre.upper()}] mejor costo = {res['mejor_costo']:,.2f} EUR")
        print(f"  desglose: {desglose}")

        plt.figure(figsize=(7, 8))
        graficar_ruta(datos["coords"], res["mejor_ruta"], costo=res["mejor_costo"],
                       titulo=f"Mejor ruta - {nombre.upper()}")
        plt.savefig(f"ruta_{nombre}.png", dpi=150, bbox_inches="tight")
        plt.close()

        animar_evolucion(
            datos["coords"], res["historial_mejor_ruta"], res["historial_mejor_costo"],
            out_path=f"evolucion_{nombre}.gif",
            cada_n=max(1, len(res["historial_mejor_ruta"]) // 40),
        )

    plt.figure()
    graficar_convergencia(
        [res_aco["historial_mejor_costo"], res_ga["historial_mejor_costo"]],
        ["ACO", "GA"],
        out_path="convergencia.png",
    )
    plt.close()

    print("\nFiguras guardadas: ruta_aco.png, ruta_ga.png, evolucion_aco.gif, evolucion_ga.gif, convergencia.png")


if __name__ == "__main__":
    main()