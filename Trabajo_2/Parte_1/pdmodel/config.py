"""Carga de configuración YAML con sobreescritura desde la línea de comandos."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=None):
    """Lee el YAML y aplica overrides del estilo ``a.b.c=valor`` (el valor se interpreta como YAML)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override inválido '{item}'. Use clave=valor, p. ej. final.n_seeds=5")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def path_of(cfg, key, create=False):
    """Ruta absoluta (relativa a la raíz del repo) de una entrada de ``cfg['paths']``."""
    p = ROOT / cfg["paths"][key]
    if create and key != "raw":
        p.mkdir(parents=True, exist_ok=True)
    return p
