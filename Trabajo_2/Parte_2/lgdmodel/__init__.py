"""Pérdida dado incumplimiento (LGD) y pérdida esperada con Lending Club. Trabajo 2, Parte 2.

La Parte 2 reutiliza el código de la Parte 1 (paquete ``pdmodel``) para que el objetivo de
incumplimiento, la tabla de solicitud, el preprocesamiento y la partición temporal sean idénticos.
"""
import sys
from pathlib import Path

__version__ = "1.0.0"

RAIZ_PARTE2 = Path(__file__).resolve().parents[1]
RAIZ_PARTE1 = RAIZ_PARTE2.parent / "Parte_1"

if not (RAIZ_PARTE1 / "pdmodel").is_dir():
    raise ImportError(f"No se encontró la Parte 1 en {RAIZ_PARTE1}. La Parte 2 depende de su paquete 'pdmodel'.")
if str(RAIZ_PARTE1) not in sys.path:
    sys.path.insert(0, str(RAIZ_PARTE1))
