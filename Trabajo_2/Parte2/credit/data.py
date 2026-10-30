"""Del archivo original (loan.csv) a la tabla procesada que usan todos los modelos.

Aquí se construyen el objetivo binario (PD) y la LGD observada. La ingeniería de variables
y el preprocesamiento ajustado (solo sobre entrenamiento) viven en features.py.
"""
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Variables que se conservan del archivo original. Las de la sección "post-originación"
# solo se usan para construir la LGD observada y NUNCA entran como predictores.
ORIGINATION_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d",
    "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status", "collections_12_mths_ex_med",
    "acc_now_delinq",
]
POST_ORIGINATION_COLUMNS = ["loan_status", "total_rec_prncp", "recoveries", "collection_recovery_fee"]

# Codificación de la guía (Parte 1). None = NA, se excluye del entrenamiento.
LOAN_STATUS_TO_DEFAULT_FLAG = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Current": None,
    "Issued": None,
    "In Grace Period": None,
    "Late (16-30 days)": None,
}
# Créditos malos cuya pérdida ya es definitiva (la LGD solo es observable en estos).
# "Late (31-120 days)" es malo para la PD pero su recuperación todavía no ocurrió.
LOAN_STATUSES_WITH_FINAL_LOSS = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def build_lgd(loans):
    """LGD = 1 - (principal recuperado + recuperaciones netas) / EAD, recortada a [0, 1].

    EAD = funded_amnt (monto desembolsado). Ver README / reporte para la discusión.
    """
    net_recoveries_after_collection_fee = (
        loans["recoveries"].fillna(0) - loans["collection_recovery_fee"].fillna(0)).clip(lower=0)
    total_recovered_amount = loans["total_rec_prncp"].fillna(0) + net_recoveries_after_collection_fee
    return (1.0 - total_recovered_amount / loans["funded_amnt"]).clip(0.0, 1.0)


def prepare(cfg, download_date=None):
    raw_csv_path = config.ROOT / cfg["paths"]["raw_csv"]
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {raw_csv_path}. Descargue loan.csv de "
            "https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset y ubíquelo allí "
            "(ver README).")
    columns_to_keep = set(ORIGINATION_COLUMNS + POST_ORIGINATION_COLUMNS)
    raw_loans = pd.read_csv(raw_csv_path, usecols=lambda column: column in columns_to_keep, low_memory=False)
    n_original_rows = len(raw_loans)
    n_original_columns = pd.read_csv(raw_csv_path, nrows=0).shape[1]

    raw_loans["loan_status"] = raw_loans["loan_status"].map(_normalize_whitespace)
    uncoded_statuses = set(raw_loans["loan_status"].unique()) - set(LOAN_STATUS_TO_DEFAULT_FLAG)
    if uncoded_statuses:
        raise ValueError(f"Categorías de loan_status sin codificar: {uncoded_statuses}")
    raw_loans["issue_d"] = pd.to_datetime(raw_loans["issue_d"], format="%b-%Y")
    raw_loans["earliest_cr_line"] = pd.to_datetime(raw_loans["earliest_cr_line"], format="%b-%Y", errors="coerce")

    loan_status_counts = raw_loans["loan_status"].value_counts().to_dict()
    raw_loans["default"] = raw_loans["loan_status"].map(LOAN_STATUS_TO_DEFAULT_FLAG)
    loans = raw_loans[raw_loans["default"].notna()].copy()
    loans["default"] = loans["default"].astype(int)

    loans["loss_final"] = loans["loan_status"].isin(LOAN_STATUSES_WITH_FINAL_LOSS)
    loans["lgd"] = np.where(loans["loss_final"], build_lgd(loans), np.nan)
    loans = loans.sort_values("issue_d").reset_index(drop=True)

    processed_dir = config.ROOT / cfg["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    loans.to_csv(processed_dir / "loans_clean.csv.gz", index=False, compression="gzip")

    data_report = {
        "n_registros_originales": int(n_original_rows),
        "n_variables_originales": int(n_original_columns),
        "n_registros_con_objetivo": int(len(loans)),
        "n_variables_procesadas": int(loans.shape[1]),
        "periodo_originacion": [str(loans["issue_d"].min().date()), str(loans["issue_d"].max().date())],
        "tasa_incumplimiento": float(loans["default"].mean()),
        "n_lgd_observada": int(loans["lgd"].notna().sum()),
        "loan_status_original": loan_status_counts,
        "fecha_descarga": download_date or date.fromtimestamp(raw_csv_path.stat().st_mtime).isoformat(),
        "archivo_original": str(Path(cfg["paths"]["raw_csv"])),
    }
    (processed_dir / "data_report.json").write_text(json.dumps(data_report, indent=2, ensure_ascii=False), "utf-8")
    return data_report


def load_processed(cfg):
    processed_csv_path = config.ROOT / cfg["paths"]["processed"] / "loans_clean.csv.gz"
    if not processed_csv_path.exists():
        raise FileNotFoundError(f"{processed_csv_path} no existe. Ejecute primero: python run.py prepare")
    loans = pd.read_csv(processed_csv_path, parse_dates=["issue_d", "earliest_cr_line"], low_memory=False)
    if cfg.get("demo_mode"):
        n_demo_rows = min(cfg["data"]["demo_rows"], len(loans))
        loans = loans.sample(n_demo_rows, random_state=cfg["seed"]).sort_values("issue_d").reset_index(drop=True)
    return loans


def split_indices(loans, cfg):
    """Devuelve (train, val, test) como arreglos de posiciones.

    temporal: por fecha de originación (issue_d). random: estratificada 60/20/20.
    """
    data_config = cfg["data"]
    if data_config["split"] == "temporal":
        issue_dates = loans["issue_d"]
        train_end, validation_end = pd.Timestamp(data_config["train_end"]), pd.Timestamp(data_config["val_end"])
        train_positions = np.where(issue_dates <= train_end)[0]
        validation_positions = np.where((issue_dates > train_end) & (issue_dates <= validation_end))[0]
        test_positions = np.where(issue_dates > validation_end)[0]
        if cfg.get("demo_mode") and min(len(train_positions), len(validation_positions), len(test_positions)) < 200:
            # la muestra de demostración no alcanza en alguna ventana: usar cuantiles de fecha
            train_cutoff, validation_cutoff = issue_dates.quantile(0.6), issue_dates.quantile(0.8)
            return (np.where(issue_dates <= train_cutoff)[0],
                    np.where((issue_dates > train_cutoff) & (issue_dates <= validation_cutoff))[0],
                    np.where(issue_dates > validation_cutoff)[0])
        return train_positions, validation_positions, test_positions
    from sklearn.model_selection import train_test_split
    all_positions = np.arange(len(loans))
    train_positions, held_out_positions = train_test_split(
        all_positions, test_size=0.4, stratify=loans["default"], random_state=cfg["seed"])
    validation_positions, test_positions = train_test_split(
        held_out_positions, test_size=0.5, stratify=loans["default"].iloc[held_out_positions],
        random_state=cfg["seed"])
    return np.sort(train_positions), np.sort(validation_positions), np.sort(test_positions)
