"""Interfaz de línea de comandos.  Uso:  python -m pdmodel <comando> --config configs/demo.yaml [--set a.b=valor]

Comandos: synth | prepare | describe | baseline | search | final | scorecard | all
"""
import argparse
import time

import numpy as np
import pandas as pd

from pdmodel import plots
from pdmodel.config import ROOT, load_config, path_of
from pdmodel.data import load_processed, prepare_dataset


def cmd_synth(cfg):
    from pdmodel.synthetic import generate
    n = cfg.get("synthetic", {}).get("n_rows", 40000)
    df = generate(n, cfg["seed"])
    out = path_of(cfg, "raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Datos SINTÉTICOS escritos en {out} ({len(df)} filas). Solo para pruebas.")


def cmd_prepare(cfg):
    _, s = prepare_dataset(cfg)
    print(f"Registros crudos: {s['registros_crudos']:,} | etiquetados: {s['registros_etiquetados']:,} | "
          f"tasa de malos: {s['tasa_malos']:.3%} | periodo: {s['periodo_originacion']}")
    for k, v in s["particiones"].items():
        print(f"  {k:5s} n={v['n']:>8,}  tasa de malos={v['tasa_malos']:.3%}")


def cmd_describe(cfg):
    df = load_processed(cfg)
    tdir, fdir = path_of(cfg, "tables", create=True), path_of(cfg, "figures", create=True)
    d = df.copy()
    d["año_originación"] = d["issue_d"].dt.year
    specs = [("Año de originación", d["año_originación"]), ("Plazo (meses)", d["term_months"]),
             ("Propósito", d["purpose"]), ("Vivienda", d["home_ownership"]),
             ("Verificación de ingresos", d["verification_status"])]
    for col, title, q in [("dti", "DTI (deciles)", 10), ("annual_inc", "Ingreso anual (deciles)", 10),
                          ("revol_util", "Utilización revolvente (deciles)", 10), ("loan_amnt", "Monto (deciles)", 10),
                          ("cr_hist_years", "Historia crediticia, años (deciles)", 10), ("emp_length", "Antigüedad laboral (años)", None)]:
        if q:
            specs.append((title, pd.qcut(d[col], q, duplicates="drop")))
        else:
            specs.append((title, d[col]))
    specs += [("# moras 2 años", d["delinq_2yrs"].clip(upper=4)), ("# consultas 6 meses", d["inq_last_6mths"].clip(upper=5)),
              ("# registros públicos", d["pub_rec"].clip(upper=3))]
    plots.bad_rate_panel(d, specs, fdir / "descriptive_bad_rate.png")
    rows = []
    for title, g in specs:
        t = d.groupby(g, observed=True)["y"].agg(tasa_malos="mean", n="size").reset_index()
        t.columns = ["grupo", "tasa_malos", "n"]
        t.insert(0, "variable", title)
        t["grupo"] = t["grupo"].astype(str)
        rows.append(t)
    pd.concat(rows).to_csv(tdir / "descriptive_bad_rate.csv", index=False)
    miss = d[[c for c in d.columns if c not in ("issue_d", "loan_status", "split")]].isna().mean().rename("pct_faltante")
    miss.to_csv(tdir / "descriptive_missing.csv")
    print("Tablas y figura descriptiva guardadas en", tdir, "y", fdir)


def _data(cfg):
    from pdmodel.experiments import Data
    return Data(cfg)


def cmd_baseline(cfg):
    from pdmodel.experiments import run_baseline
    r = run_baseline(cfg, _data(cfg))
    print(f"Regresión logística: C={r['C_elegido']} | params={r['n_params']} | "
          f"test AUC={r['test']['auc']:.4f} Gini={r['test']['gini']:.4f} KS={r['test']['ks']:.4f} Brier={r['test']['brier']:.4f}")


def cmd_search(cfg):
    from pdmodel.experiments import run_search
    r = run_search(cfg, _data(cfg))
    print(f"Mejor configuración ({r['n_configuraciones']} probadas, {r['segundos_total']:.0f}s):", r["mejor_config"])


def cmd_final(cfg):
    from pdmodel.experiments import run_final
    r = run_final(cfg, _data(cfg))
    print("\nComparación en prueba (LR vs MLP con", cfg["final"]["n_seeds"], "semillas):")
    print(r["comparacion"].round(4).to_string(index=False))


def cmd_scorecard(cfg):
    from pdmodel.experiments import run_scorecard
    out, imps = run_scorecard(cfg, _data(cfg))
    print("Escala:", {k: round(v, 3) for k, v in out["escala"].items()})
    print("Fidelidad scorecard aditiva (MLP):", {k: round(v, 3) if not isinstance(v, list) else v for k, v in out["mlp"].items()})
    print("Top-5 variables (MLP):", imps["Red neuronal"].head(5)["variable"].tolist())


COMMANDS = {"synth": cmd_synth, "prepare": cmd_prepare, "describe": cmd_describe, "baseline": cmd_baseline,
            "search": cmd_search, "final": cmd_final, "scorecard": cmd_scorecard}


def cmd_all(cfg):
    t0 = time.time()
    if cfg["paths"]["raw"].endswith("synthetic.csv") and not path_of(cfg, "raw").exists():
        cmd_synth(cfg)
    for name in ["prepare", "describe", "baseline", "search", "final", "scorecard"]:
        print(f"\n===== {name} =====")
        COMMANDS[name](cfg)
    print(f"\nTiempo total: {time.time() - t0:.0f} s. Resultados en {path_of(cfg, 'tables')} y {path_of(cfg, 'figures')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdmodel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="clave=valor",
                    help="sobreescribe un parámetro, p. ej. --set final.n_seeds=5 --set scorecard.pdo=40")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    np.random.seed(cfg["seed"])
    (cmd_all if args.command == "all" else COMMANDS[args.command])(cfg)


if __name__ == "__main__":
    main()
