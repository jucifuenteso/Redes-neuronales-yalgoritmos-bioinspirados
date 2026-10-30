"""Configuración de la Parte 2.

Se cargan DOS archivos:
  1. el YAML de la Parte 2 (modelos de LGD, búsqueda, semillas, pérdida esperada), y
  2. el YAML de la Parte 1 que indica ``parte1.config``, del que se HEREDAN la semilla, la sección
     ``data`` (partición temporal, uso de int_rate/grade, winsorización...) y las rutas del archivo
     original y de los modelos de PD.
Así la partición de la LGD es la misma que la de la PD por construcción, no por copiar fechas.

Las sobreescrituras ``--set`` se aplican DESPUÉS de leer cada YAML (también en modo demostración).
"""
from pdmodel.config import load_config as cargar_yaml_con_sobreescrituras
from pdmodel.config import path_of as ruta_de_la_parte1

from lgdmodel import RAIZ_PARTE1, RAIZ_PARTE2


def cargar_configuracion(ruta_configuracion, sobreescrituras=None, sobreescrituras_parte1=None):
    configuracion = cargar_yaml_con_sobreescrituras(ruta_configuracion, sobreescrituras)
    configuracion_de_la_parte1 = configuracion.setdefault("parte1", {})
    sobreescrituras_totales_parte1 = list(configuracion_de_la_parte1.get("sobreescrituras") or [])
    sobreescrituras_totales_parte1 += list(sobreescrituras_parte1 or [])
    configuracion_de_la_parte1["sobreescrituras_totales"] = sobreescrituras_totales_parte1
    ruta_yaml_parte1 = RAIZ_PARTE1 / configuracion_de_la_parte1["config"]
    configuracion_de_la_parte1["ruta_config"] = str(ruta_yaml_parte1)
    configuracion["configuracion_parte1"] = cargar_yaml_con_sobreescrituras(ruta_yaml_parte1,
                                                                            sobreescrituras_totales_parte1)
    configuracion["semilla"] = configuracion["configuracion_parte1"]["seed"]
    configuracion["datos"] = configuracion["configuracion_parte1"]["data"]
    return configuracion


def ruta(configuracion, clave, crear=False):
    """Ruta absoluta (relativa a la carpeta Parte_2) de una entrada de ``configuracion['paths']``."""
    ruta_absoluta = RAIZ_PARTE2 / configuracion["paths"][clave]
    if crear:
        ruta_absoluta.mkdir(parents=True, exist_ok=True)
    return ruta_absoluta


def ruta_archivo_original(configuracion):
    """El mismo loan.csv que usa la Parte 1 (``paths.raw`` de su configuración)."""
    return ruta_de_la_parte1(configuracion["configuracion_parte1"], "raw")


def ruta_modelos_parte1(configuracion):
    return ruta_de_la_parte1(configuracion["configuracion_parte1"], "models")


def ruta_tablas_parte1(configuracion):
    return ruta_de_la_parte1(configuracion["configuracion_parte1"], "tables")


def es_configuracion_de_demostracion(configuracion):
    return bool(configuracion.get("demostracion", False))
