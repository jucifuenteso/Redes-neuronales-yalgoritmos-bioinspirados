"""Lectura del CSV original, construcción de la variable objetivo y particiones.

Idea central (caso de uso): el modelo se usa en el MOMENTO DE LA SOLICITUD. Por eso solo se usan
variables conocidas al originar el crédito. Todo lo que ocurre después (pagos, recuperaciones,
saldo pendiente, fechas del último pago, etc.) es *fuga de información* y se excluye.
"""
import json

import numpy as np
import pandas as pd

from pdmodel.config import path_of

# Columnas del CSV original que se leen (el resto se ignora, incluidas las de fuga).
RAW_COLS = [
    "loan_status", "issue_d", "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "int_rate", "grade",
]

# Codificación EXACTA del enunciado. Cualquier otro estado -> NA (no entra al entrenamiento).
GOOD_STATUS = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
BAD_STATUS = {
    "Charged Off", "Default", "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}

APP_NUM = [
    "loan_amnt", "term_months", "emp_length", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "cr_hist_years",
]
APP_CAT = ["home_ownership", "verification_status", "purpose"]
PRICING_NUM = ["int_rate"]
PRICING_CAT = ["grade"]


def app_columns(use_lc_pricing=False):
    """(numéricas, categóricas) que entran al modelo."""
    num = APP_NUM + (PRICING_NUM if use_lc_pricing else [])
    cat = APP_CAT + (PRICING_CAT if use_lc_pricing else [])
    return num, cat


def parse_month_year(s):
    """'Dec-2011' -> Timestamp. Acepta también 'Dec-11'."""
    out = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out[bad] = pd.to_datetime(s[bad], format="%b-%y", errors="coerce")
    return out


def _to_number(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def parse_emp_length(s):
    s = s.astype(str)
    num = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    return num.where(~s.str.contains("<", regex=False), 0.0).astype(float)


def build_target(status):
    """1 = malo, 0 = bueno, NaN = excluido (Current, Issued, In Grace Period, Late 16-30...)."""
    y = pd.Series(np.nan, index=status.index, dtype=float)
    y[status.isin(GOOD_STATUS)] = 0.0
    y[status.isin(BAD_STATUS)] = 1.0
    return y


def load_raw(path):
    return pd.read_csv(path, usecols=lambda c: c in RAW_COLS, low_memory=False)


def clean_applications(raw, use_lc_pricing=False):
    """Convierte el CSV crudo en la tabla de *solicitud* (tipos limpios, solo info al originar)."""
    issue = parse_month_year(raw["issue_d"].astype(str).where(raw["issue_d"].notna()))
    earliest = parse_month_year(raw["earliest_cr_line"].astype(str).where(raw["earliest_cr_line"].notna()))
    app = pd.DataFrame(index=raw.index)
    app["loan_amnt"] = pd.to_numeric(raw["loan_amnt"], errors="coerce")
    app["term_months"] = pd.to_numeric(raw["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    app["emp_length"] = parse_emp_length(raw["emp_length"])
    app["annual_inc"] = pd.to_numeric(raw["annual_inc"], errors="coerce")
    app["dti"] = pd.to_numeric(raw["dti"], errors="coerce")
    for c in ["delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
              "revol_bal", "total_acc"]:
        app[c] = pd.to_numeric(raw[c], errors="coerce")
    app["revol_util"] = _to_number(raw["revol_util"])
    app["cr_hist_years"] = (issue - earliest).dt.days / 365.25
    for c in APP_CAT:
        app[c] = raw[c].astype(str).where(raw[c].notna(), np.nan)
    if use_lc_pricing:
        app["int_rate"] = _to_number(raw["int_rate"])
        app["grade"] = raw["grade"].astype(str).where(raw["grade"].notna(), np.nan)
    app["issue_d"] = issue
    app["loan_status"] = raw["loan_status"]
    app["y"] = build_target(raw["loan_status"])
    return app


def assign_split(df, cfg):
    """Devuelve una Serie con 'train' / 'val' / 'test'. Modo temporal = partición por fecha de originación."""
    sp = cfg["data"]["split"]
    if sp["mode"] == "temporal":
        t_end, v_end = pd.Timestamp(sp["train_end"]), pd.Timestamp(sp["val_end"])
        out = np.where(df["issue_d"] <= t_end, "train", np.where(df["issue_d"] <= v_end, "val", "test"))
        return pd.Series(out, index=df.index)
    if sp["mode"] == "random":
        from sklearn.model_selection import train_test_split
        idx = df.index.to_numpy()
        rest, test = train_test_split(idx, test_size=sp["random_test_frac"], stratify=df["y"],
                                      random_state=cfg["seed"])
        frac = sp["random_val_frac"] / (1 - sp["random_test_frac"])
        train, val = train_test_split(rest, test_size=frac, stratify=df.loc[rest, "y"], random_state=cfg["seed"])
        out = pd.Series("train", index=df.index)
        out.loc[val] = "val"
        out.loc[test] = "test"
        return out
    raise ValueError("data.split.mode debe ser 'temporal' o 'random'")


def prepare_dataset(cfg):
    """CSV original -> dataset procesado (guardado en data/processed) + resumen de datos."""
    raw_path = path_of(cfg, "raw")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No existe {raw_path}. Descargue loan.csv de Kaggle (ranadeep/credit-risk-dataset) y "
            "colóquelo allí, o corra `python -m pdmodel synth` para generar datos sintéticos de prueba.")
    n_cols_file = len(pd.read_csv(raw_path, nrows=0).columns)
    raw = load_raw(raw_path)
    n_raw = len(raw)
    use_pricing = cfg["data"]["use_lc_pricing"]
    app = clean_applications(raw, use_pricing)

    status_counts = app["loan_status"].value_counts(dropna=False).to_dict()
    n_no_date = int(app["issue_d"].isna().sum())
    period = (app["issue_d"].min(), app["issue_d"].max())
    app = app[app["y"].notna() & app["issue_d"].notna()].copy()
    frac = cfg["data"]["sample_frac"]
    if frac < 1.0:
        app = app.sample(frac=frac, random_state=cfg["seed"])
    app = app.sort_values("issue_d").reset_index(drop=True)
    app["split"] = assign_split(app, cfg)

    out_dir = path_of(cfg, "processed", create=True)
    app.to_csv(out_dir / "pd_dataset.csv.gz", index=False, compression="gzip")
    summary = {
        "archivo": str(cfg["paths"]["raw"]),
        "fecha_descarga": cfg["data"]["download_date"],
        "registros_crudos": int(n_raw),
        "columnas_en_archivo": int(n_cols_file),
        "columnas_usadas": len(RAW_COLS),
        "periodo_originacion": [str(period[0].date()), str(period[1].date())],
        "registros_sin_fecha": n_no_date,
        "registros_etiquetados": int(len(app)),
        "tasa_malos": float(app["y"].mean()),
        "conteo_loan_status": {str(k): int(v) for k, v in status_counts.items()},
        "particiones": {
            s: {"n": int((app["split"] == s).sum()), "tasa_malos": float(app.loc[app["split"] == s, "y"].mean())}
            for s in ["train", "val", "test"] if (app["split"] == s).any()
        },
        "muestra_frac": frac,
    }
    tdir = path_of(cfg, "tables", create=True)
    with open(tdir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return app, summary


def load_processed(cfg):
    p = path_of(cfg, "processed") / "pd_dataset.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}. Corra primero: python -m pdmodel prepare --config <cfg>")
    df = pd.read_csv(p, low_memory=False, parse_dates=["issue_d"])
    return df
