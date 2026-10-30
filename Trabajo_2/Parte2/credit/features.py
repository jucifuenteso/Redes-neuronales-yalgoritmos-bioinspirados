"""Ingeniería de variables (solo información disponible al originar) y preprocesamiento.

El mismo preprocesador, ajustado solo con entrenamiento, lo usan la regresión logística, el MLP
de PD, los modelos de LGD y la aplicación web (regla de comparación justa: mismos datos).
"""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from . import data

BASE_NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc_log", "lti", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal_log", "revol_util", "total_acc", "credit_hist_years",
    "mths_since_last_delinq", "mths_since_last_record", "emp_length_num", "term_months",
    "collections_12_mths_ex_med", "acc_now_delinq",
]
BASE_CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "addr_state",
                             "initial_list_status"]
LENDING_CLUB_PRICING_NUMERIC_FEATURES = ["int_rate", "grade_num"]
LENDING_CLUB_PRICING_CATEGORICAL_FEATURES = []

# límites fijos (no dependen de los datos) para controlar valores extremos / códigos basura
VALUE_CLIPPING_BOUNDS = {
    "dti": (0, 60), "revol_util": (0, 150), "open_acc": (0, 60), "total_acc": (0, 120),
    "delinq_2yrs": (0, 10), "inq_last_6mths": (0, 10), "pub_rec": (0, 10),
    "lti": (0, 2), "credit_hist_years": (0, 70), "collections_12_mths_ex_med": (0, 5),
    "acc_now_delinq": (0, 5)}

NUMERIC_COLUMNS_COPIED_FROM_RAW = [
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_util", "total_acc",
    "mths_since_last_delinq", "mths_since_last_record", "collections_12_mths_ex_med", "acc_now_delinq"]


def feature_lists(include_lending_club_pricing):
    numeric_features = BASE_NUMERIC_FEATURES + (
        LENDING_CLUB_PRICING_NUMERIC_FEATURES if include_lending_club_pricing else [])
    categorical_features = BASE_CATEGORICAL_FEATURES + (
        LENDING_CLUB_PRICING_CATEGORICAL_FEATURES if include_lending_club_pricing else [])
    return numeric_features, categorical_features


def _parse_employment_length_years(employment_length):
    employment_length = employment_length.astype(str).str.strip()
    years = employment_length.str.extract(r"(\d+)")[0].astype(float)
    years[employment_length.str.startswith("<")] = 0.0
    years[employment_length.str.lower().isin(["n/a", "nan"])] = np.nan
    return years


def engineer(loans, include_lending_club_pricing=False):
    """Construye solo columnas de predictores. Funciona igual para datos de entrenamiento y la app."""
    features = pd.DataFrame(index=loans.index)
    annual_income = pd.to_numeric(loans["annual_inc"], errors="coerce").clip(lower=0)
    features["loan_amnt"] = pd.to_numeric(loans["loan_amnt"], errors="coerce")
    features["annual_inc_log"] = np.log1p(annual_income)
    features["lti"] = features["loan_amnt"] / annual_income.replace(0, np.nan)
    features["revol_bal_log"] = np.log1p(pd.to_numeric(loans["revol_bal"], errors="coerce").clip(lower=0))
    features["credit_hist_years"] = (
        (pd.to_datetime(loans["issue_d"]) - pd.to_datetime(loans["earliest_cr_line"])).dt.days / 365.25)
    features["emp_length_num"] = (_parse_employment_length_years(loans["emp_length"])
                                  if loans["emp_length"].dtype == object else loans["emp_length"].astype(float))
    term = loans["term"]
    features["term_months"] = (term.astype(str).str.extract(r"(\d+)")[0].astype(float)
                               if term.dtype == object else term.astype(float))
    for column in NUMERIC_COLUMNS_COPIED_FROM_RAW:
        features[column] = pd.to_numeric(loans[column], errors="coerce") if column in loans else np.nan
    for column, (lower_bound, upper_bound) in VALUE_CLIPPING_BOUNDS.items():
        features[column] = features[column].clip(lower_bound, upper_bound)
    features["home_ownership"] = (loans["home_ownership"].astype(str).str.upper()
                                  .replace({"NONE": "OTHER", "ANY": "OTHER"}))
    for column in ["verification_status", "purpose", "addr_state", "initial_list_status"]:
        features[column] = loans[column].astype(str) if column in loans else "NA"
    if include_lending_club_pricing:
        features["int_rate"] = pd.to_numeric(loans["int_rate"], errors="coerce")
        features["grade_num"] = loans["grade"].map({grade: position + 1 for position, grade in enumerate("ABCDEFG")})
    return features


def make_preprocessor(include_lending_club_pricing=False):
    numeric_features, categorical_features = feature_lists(include_lending_club_pricing)
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
         categorical_features),
    ])


def feature_names(preprocessor):
    return list(preprocessor.get_feature_names_out())


@dataclass
class Bundle:
    loans: pd.DataFrame            # tabla procesada completa (con issue_d, default, lgd, funded_amnt...)
    raw_features: pd.DataFrame     # predictores en bruto (ingeniería aplicada, sin escalar)
    preprocessor: ColumnTransformer
    row_positions: dict            # {"train","val","test"} -> posiciones de fila
    features: dict                 # matrices preprocesadas por partición
    targets: dict                  # objetivo binario (default) por partición
    feature_names: list
    cfg: dict

    def lgd_part(self, partition):
        """Filas con LGD observada dentro de la partición: (X, y, posiciones)."""
        partition_positions = self.row_positions[partition]
        has_observed_lgd = self.loans["lgd"].iloc[partition_positions].notna().to_numpy()
        return (self.features[partition][has_observed_lgd],
                self.loans["lgd"].iloc[partition_positions][has_observed_lgd].to_numpy(np.float32),
                partition_positions[has_observed_lgd])


def make_bundle(cfg, fit=True):
    loans = data.load_processed(cfg)
    include_lending_club_pricing = cfg["data"]["include_lc_pricing"]
    raw_features = engineer(loans, include_lending_club_pricing)
    row_positions = dict(zip(["train", "val", "test"], data.split_indices(loans, cfg)))
    models_dir = config.path(cfg, "models")
    preprocessor_path = models_dir / "preprocessor.joblib"
    if fit or not preprocessor_path.exists():
        preprocessor = make_preprocessor(include_lending_club_pricing).fit(
            raw_features.iloc[row_positions["train"]])
        joblib.dump(preprocessor, preprocessor_path)
    else:
        preprocessor = joblib.load(preprocessor_path)
    features = {partition: preprocessor.transform(raw_features.iloc[positions]).astype(np.float32)
                for partition, positions in row_positions.items()}
    targets = {partition: loans["default"].iloc[positions].to_numpy(np.float32)
               for partition, positions in row_positions.items()}
    return Bundle(loans, raw_features, preprocessor, row_positions, features, targets,
                  feature_names(preprocessor), cfg)
