"""Interfaz de línea de comandos de la Parte 2.

Uso:  python -m lgdmodel <comando> [--config configs/default.yaml] [--set clave=valor] [--set-parte1 clave=valor]

Comandos:
  sintetico   genera el CSV sintético de la demostración (solo con configs/demo.yaml)
  preparar    loan.csv -> data/processed/lgd_dataset.csv.gz (mismos créditos y particiones que la Parte 1)
  describir   LGD observada: estados, recuperaciones por estado, distribución, segmentos
  referencias media constante, regresión lineal (Ridge) y logit fraccional
  busqueda    búsqueda aleatoria de la arquitectura e hiperparámetros de la red
  final       red final con N semillas + comparación con las referencias
  ablacion    misma red y semillas: pérdida BCE frente a MSE
  perdida     pérdida esperada E[L] = PD · LGD · EAD (PD de la Parte 1)
  figuras     regenera todas las figuras desde reports/tables
  todo        todo lo anterior en orden
"""
import argparse
import time

import numpy as np

from lgdmodel import RAIZ_PARTE2
from lgdmodel.config import (cargar_configuracion, es_configuracion_de_demostracion, ruta, ruta_archivo_original,
                             ruta_modelos_parte1)


def comando_sintetico(configuracion):
    from lgdmodel.synthetic import escribir
    if not es_configuracion_de_demostracion(configuracion):
        raise SystemExit("`sintetico` solo se usa con configs/demo.yaml: con datos reales escribiría sobre loan.csv.")
    n_filas = configuracion["configuracion_parte1"].get("synthetic", {}).get("n_rows", 60000)
    ruta_archivo = ruta_archivo_original(configuracion)
    escribir(ruta_archivo, n_filas, configuracion["semilla"])
    print(f"Datos SINTÉTICOS escritos en {ruta_archivo.resolve()} ({n_filas} filas). Solo para la demostración.")


def comando_preparar(configuracion):
    from lgdmodel.data import preparar_datos
    _, resumen = preparar_datos(configuracion)
    print(f"Etiquetados: {resumen['registros_etiquetados']:,} | periodo: {resumen['periodo_originacion']}")
    for particion, valores in resumen["por_particion"].items():
        print(f"  {particion:5s} créditos={valores['n_creditos']:>8,}  incumplidos={valores['n_incumplidos']:>7,}  "
              f"con LGD observada={valores['n_lgd_observada']:>7,}  LGD media={valores['lgd_media']:.3f}")
    consistencia = resumen["consistencia_con_parte1"]
    print("Consistencia con la Parte 1:", "COINCIDE" if consistencia["coincide"] else
          f"NO COINCIDE ({consistencia.get('motivo', consistencia.get('por_particion'))})")


def comando_describir(configuracion):
    from lgdmodel.experiments import ejecutar_descriptivo
    distribucion, recuperaciones_por_estado = ejecutar_descriptivo(configuracion)
    print(recuperaciones_por_estado.round(3).to_string())
    objetivo = distribucion["lgd_sobre_desembolso (objetivo)"]
    print(f"LGD: media={objetivo['media']:.3f} mediana={objetivo['q50']:.3f} "
          f"fracción>=0.99={objetivo['fraccion_mayor_igual_099']:.3f} fracción<=0.01={objetivo['fraccion_menor_igual_001']:.3f}")


def _datos(configuracion):
    from lgdmodel.experiments import DatosLGD
    return DatosLGD(configuracion)


def comando_referencias(configuracion, datos=None):
    from lgdmodel.experiments import ejecutar_referencias
    resultado = ejecutar_referencias(configuracion, datos or _datos(configuracion))
    for nombre in ["constante", "lineal", "logit_fraccional"]:
        print(f"  {nombre:17s} params={resultado[nombre]['n_parametros']:>5}  test MAE={resultado[nombre]['test']['mae']:.4f}"
              f"  RMSE={resultado[nombre]['test']['rmse']:.4f}")


def comando_busqueda(configuracion, datos=None):
    from lgdmodel.experiments import ejecutar_busqueda
    resultado = ejecutar_busqueda(configuracion, datos or _datos(configuracion))
    print(f"Mejor configuración ({resultado['n_configuraciones']} probadas, {resultado['segundos_total']:.0f}s):",
          resultado["mejor_configuracion"])


def comando_final(configuracion, datos=None):
    from lgdmodel.experiments import ejecutar_final
    resultado = ejecutar_final(configuracion, datos or _datos(configuracion))
    print("\nComparación en prueba:")
    print(resultado["comparacion"].round(4).to_string(index=False))


def comando_ablacion(configuracion, datos=None):
    from lgdmodel.experiments import ejecutar_ablacion
    resultado = ejecutar_ablacion(configuracion, datos or _datos(configuracion))
    for metrica in ["test_mae", "test_rmse"]:
        print(f"  {metrica}: BCE={resultado[metrica]['bce_media']:.4f}  MSE={resultado[metrica]['mse_media']:.4f}  "
              f"BCE mejor en {resultado[metrica]['semillas_donde_bce_es_mejor']}/{resultado['n_semillas']} semillas")


def comando_perdida(configuracion, datos=None):
    from lgdmodel.experiments import ejecutar_perdida_esperada
    resumen = ejecutar_perdida_esperada(configuracion, datos or _datos(configuracion))
    print(f"Créditos de prueba: {resumen['n_prueba']:,} | con desenlace definitivo: {resumen['n_desenlace_definitivo']:,} "
          f"| excluidos: {resumen['excluidos_sin_desenlace_por_estado']}")
    print(f"Pérdida realizada (evaluados): {resumen['perdida_realizada_total']:,.0f}")
    for nombre, valores in resumen["combinaciones"].items():
        print(f"  {nombre:38s} E[L]={valores['perdida_esperada_total_evaluados']:>14,.0f}  "
              f"E[L]/realizada={valores['razon_esperada_sobre_realizada']:.3f}")
    print("Diagnóstico:", {clave: round(valor, 4) for clave, valor in resumen["diagnostico"].items()})


def comando_figuras(configuracion):
    from lgdmodel.plots import generar_todas_las_figuras
    generar_todas_las_figuras(configuracion)


def _asegurar_modelos_de_pd(configuracion):
    """La pérdida esperada necesita la PD de la Parte 1. En demostración se entrena aquí si falta."""
    carpeta_modelos_parte1 = ruta_modelos_parte1(configuracion)
    if (carpeta_modelos_parte1 / "mlp_final.joblib").exists() and (carpeta_modelos_parte1 / "lr_final.joblib").exists():
        return
    if not es_configuracion_de_demostracion(configuracion):
        raise SystemExit(f"Faltan los modelos de PD en {carpeta_modelos_parte1}. Corra primero la Parte 1: "
                         "python -m pdmodel all --config configs/default.yaml (desde Parte_1).")
    from pdmodel.cli import main as ejecutar_parte1
    argumentos_parte1 = ["all", "--config", configuracion["parte1"]["ruta_config"]]
    for sobreescritura in configuracion["parte1"]["sobreescrituras_totales"]:
        argumentos_parte1 += ["--set", sobreescritura]
    print("\n===== Parte 1 (demostración, salidas en Parte_2/demo_parte1) =====")
    ejecutar_parte1(argumentos_parte1)


def comando_todo(configuracion):
    inicio = time.time()
    if es_configuracion_de_demostracion(configuracion) and not ruta_archivo_original(configuracion).exists():
        comando_sintetico(configuracion)
    _asegurar_modelos_de_pd(configuracion)
    for nombre, comando in [("preparar", comando_preparar), ("describir", comando_describir)]:
        print(f"\n===== {nombre} =====")
        comando(configuracion)
    datos = _datos(configuracion)
    for nombre, comando in [("referencias", comando_referencias), ("busqueda", comando_busqueda),
                            ("final", comando_final), ("ablacion", comando_ablacion), ("perdida", comando_perdida)]:
        print(f"\n===== {nombre} =====")
        comando(configuracion, datos)
    print("\n===== figuras =====")
    comando_figuras(configuracion)
    print(f"\nTiempo total: {time.time() - inicio:.0f} s. Resultados en {ruta(configuracion, 'tables')} "
          f"y {ruta(configuracion, 'figures')}")


COMANDOS = {"sintetico": comando_sintetico, "preparar": comando_preparar, "describir": comando_describir,
            "referencias": comando_referencias, "busqueda": comando_busqueda, "final": comando_final,
            "ablacion": comando_ablacion, "perdida": comando_perdida, "figuras": comando_figuras,
            "todo": comando_todo}


def principal(argumentos=None):
    analizador = argparse.ArgumentParser(prog="lgdmodel", description=__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("comando", choices=list(COMANDOS))
    analizador.add_argument("--config", default=str(RAIZ_PARTE2 / "configs" / "default.yaml"))
    analizador.add_argument("--set", dest="sobreescrituras", action="append", default=[], metavar="clave=valor",
                            help="sobreescribe un parámetro de la Parte 2, p. ej. --set final.n_semillas=5")
    analizador.add_argument("--set-parte1", dest="sobreescrituras_parte1", action="append", default=[],
                            metavar="clave=valor",
                            help="sobreescribe la config heredada de la Parte 1, p. ej. data.split.val_end=2013-06-30")
    argumentos_leidos = analizador.parse_args(argumentos)
    configuracion = cargar_configuracion(argumentos_leidos.config, argumentos_leidos.sobreescrituras,
                                         argumentos_leidos.sobreescrituras_parte1)
    np.random.seed(configuracion["semilla"])
    COMANDOS[argumentos_leidos.comando](configuracion)


if __name__ == "__main__":
    principal()
