"""Carga de configuración YAML con sobreescritura desde la línea de comandos (--set a.b=c)."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs" / "default.yaml"


def _merge(base_config, overriding_config):
    for key, value in overriding_config.items():
        if isinstance(value, dict) and isinstance(base_config.get(key), dict):
            _merge(base_config[key], value)
        else:
            base_config[key] = value
    return base_config


def load(config_path=None, overrides=()):
    with open(DEFAULT, encoding="utf-8") as default_file:
        cfg = yaml.safe_load(default_file)
    if config_path:
        with open(config_path, encoding="utf-8") as user_config_file:
            _merge(cfg, yaml.safe_load(user_config_file) or {})
    for override in overrides:
        dotted_key, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"--set espera clave=valor, recibió '{override}'")
        config_node = cfg
        key_parts = dotted_key.strip().split(".")
        for key_part in key_parts[:-1]:
            config_node = config_node.setdefault(key_part, {})
        config_node[key_parts[-1]] = yaml.safe_load(raw_value)
    return cfg


def apply_demo(cfg):
    """Modo demostración: muestra pequeña, pocas épocas, pocas semillas/pruebas (< 2 min)."""
    cfg = copy.deepcopy(cfg)
    demo_config = cfg["demo"]
    cfg["n_seeds"] = demo_config["seeds"]
    cfg["lgd"]["search"]["trials"] = demo_config["lgd_trials"]
    cfg["lgd"]["search"]["epochs"] = demo_config["epochs"]
    cfg["lgd"]["mlp"]["epochs"] = demo_config["epochs"]
    cfg["pd_ref"]["mlp"]["epochs"] = demo_config["epochs"]
    cfg["demo_mode"] = True
    # el modo demo escribe en carpetas aparte para no pisar los resultados completos
    for path_key in ("models", "outputs", "figures"):
        cfg["paths"][path_key] = cfg["paths"][path_key] + "_demo"
    return cfg


def path(cfg, path_key):
    directory = ROOT / cfg["paths"][path_key]
    directory.mkdir(parents=True, exist_ok=True)
    return directory
