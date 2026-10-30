"""Punto de entrada único de la Parte 2 (LGD y pérdida esperada).  Ejemplos:

  python run.py prepare --download-date 2026-09-01   # loan.csv -> data/processed
  python run.py describe                              # LGD observada por segmento
  python run.py lgd                                   # lineal + búsqueda + MLP x30 semillas
  python run.py lgd-loss                              # ablación BCE vs MSE (30 semillas)
  python run.py el                                    # E[L] = PD * LGD * EAD en prueba
  python run.py figures
  python run.py all --demo                            # todo el flujo en < 2 min (muestra pequeña)
  python run.py lgd --skip-search --set lgd.mlp.hidden=[128,64] --set n_seeds=5
"""
import argparse
import sys
import time

from credit import config

COMMANDS = ["prepare", "describe", "lgd", "lgd-loss", "el", "figures", "all"]


def main(argv=None):
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("command", choices=COMMANDS)
    argument_parser.add_argument("--config", help="archivo YAML que sobreescribe configs/default.yaml")
    argument_parser.add_argument("--set", action="append", default=[], metavar="clave=valor",
                           help="sobreescribe un parámetro, p. ej. lgd.mlp.lr=0.0005 (repetible)")
    argument_parser.add_argument("--demo", action="store_true", help="modo demostración rápido (< 2 min)")
    argument_parser.add_argument("--skip-search", action="store_true",
                                 help="no buscar; usa la config guardada o la de --set")
    argument_parser.add_argument("--download-date", help="fecha de descarga del dataset (para prepare)")
    arguments = argument_parser.parse_args(argv)

    cfg = config.load(arguments.config, arguments.set)
    if arguments.demo:
        cfg = config.apply_demo(cfg)
    start_time = time.time()

    from credit import data, experiments, figures
    step_functions = {
        "prepare": lambda: print(data.prepare(cfg, arguments.download_date)),
        "describe": lambda: experiments.run_lgd_describe(cfg),
        "lgd": lambda: experiments.run_lgd(cfg, arguments.skip_search),
        "lgd-loss": lambda: experiments.run_lgd_loss_ablation(cfg),
        "el": lambda: experiments.run_expected_loss(cfg),
        "figures": lambda: figures.make_all(cfg),
    }
    all_steps = ["describe", "lgd", "lgd-loss", "el", "figures"]
    steps_to_run = all_steps if arguments.command == "all" else [arguments.command]
    for step_name in steps_to_run:
        print(f"\n===== {step_name} =====")
        step_functions[step_name]()
    print(f"\nTiempo total: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
