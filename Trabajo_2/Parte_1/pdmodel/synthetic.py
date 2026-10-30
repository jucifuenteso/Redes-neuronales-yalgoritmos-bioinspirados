"""Genera un CSV SINTÉTICO con el mismo formato que loan.csv de Lending Club.

SOLO sirve para probar el código sin descargar los datos reales (no hay internet en todas partes)
y para el modo demostración. Los resultados con estos datos NO se reportan en el trabajo.
"""
import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt(dates):
    return np.array([f"{MONTHS[d.month - 1]}-{d.year}" for d in dates])


def generate(n=40000, seed=0):
    rng = np.random.RandomState(seed)
    months = pd.date_range("2007-06-01", "2015-12-01", freq="MS")
    w = np.exp(np.linspace(0, 4.5, len(months)))          # el volumen crece con el tiempo
    issue = months[rng.choice(len(months), n, p=w / w.sum())]
    term = rng.choice([36, 60], n, p=[0.75, 0.25])
    annual_inc = np.exp(rng.normal(np.log(65000), 0.55, n)).round(0)
    loan_amnt = (np.clip(np.exp(rng.normal(np.log(12000), 0.6, n)), 1000, 35000) // 25 * 25)
    dti = np.clip(rng.normal(17, 7, n), 0, 40).round(2)
    emp = rng.randint(0, 11, n).astype(float)
    emp[rng.rand(n) < 0.05] = np.nan
    home = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.45, 0.42, 0.11, 0.02])
    verif = rng.choice(["Not Verified", "Verified", "Source Verified"], n, p=[0.35, 0.35, 0.30])
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other", "small_business",
                          "major_purchase", "car", "medical"], n, p=[0.55, 0.19, 0.06, 0.08, 0.03, 0.04, 0.03, 0.02])
    delinq = rng.poisson(0.3, n)
    inq = rng.poisson(0.8, n)
    open_acc = np.clip(rng.normal(11, 5, n), 1, 40).round()
    pub_rec = rng.poisson(0.15, n)
    revol_bal = np.exp(rng.normal(9.2, 1.0, n)).round(0)
    revol_util = np.clip(rng.normal(55, 25, n), 0, 120).round(1)
    total_acc = open_acc + rng.poisson(14, n)
    cr_hist = np.clip(rng.gamma(5, 3.2, n), 1, 60)
    earliest = [d - pd.Timedelta(days=int(c * 365.25)) for d, c in zip(issue, cr_hist)]
    msld = np.where(delinq > 0, rng.randint(1, 120, n), np.nan)
    msld[(delinq == 0) & (rng.rand(n) < 0.25)] = rng.randint(24, 120)

    lti = loan_amnt / annual_inc
    logit = (-2.4 + 0.8 * (term == 60) + 0.03 * (dti - 17) + 0.0008 * (dti - 17) ** 2
             - 0.55 * np.log(annual_inc / 65000) + 1.2 * lti + 0.25 * (purpose == "small_business")
             + 0.30 * delinq + 0.12 * inq + 0.010 * (revol_util - 55) + 0.45 * pub_rec
             - 0.012 * cr_hist + 0.15 * (home == "RENT") - 0.10 * (home == "OWN")
             + 0.35 * ((term == 60) & (revol_util > 80)) + 0.0 * rng.randn(n))
    p_bad = 1 / (1 + np.exp(-logit))
    bad = rng.rand(n) < p_bad

    int_rate = np.round(6 + 40 * p_bad + rng.normal(0, 1, n), 2).clip(5.4, 28)
    grade = pd.cut(int_rate, [0, 8, 11, 14, 17, 20, 24, 40], labels=list("ABCDEFG")).astype(str)

    end = pd.Timestamp("2016-01-01")
    age = (end.year - issue.year) * 12 + (end.month - issue.month)
    frac_left = np.clip(1 - age / term, 0, 1)
    is_current = rng.rand(n) < frac_left * 0.95
    early = issue < pd.Timestamp("2010-06-01")
    status = np.empty(n, dtype=object)
    for i in range(n):
        if is_current[i]:
            status[i] = rng.choice(["Current", "In Grace Period", "Late (16-30 days)", "Issued"],
                                   p=[0.94, 0.03, 0.02, 0.01])
        elif bad[i]:
            status[i] = rng.choice(["Charged Off", "Default", "Late (31-120 days)",
                                    "Does not meet the credit policy. Status:Charged Off"],
                                   p=[0.82, 0.04, 0.08, 0.06] if early[i] else [0.88, 0.04, 0.08, 0.0])
        else:
            status[i] = ("Does not meet the credit policy. Status:Fully Paid"
                         if early[i] and rng.rand() < 0.1 else "Fully Paid")

    emp_txt = np.array(["n/a" if np.isnan(e) else ("< 1 year" if e == 0 else "10+ years" if e == 10
                        else f"{int(e)} year{'s' if e > 1 else ''}") for e in emp])
    return pd.DataFrame({
        "id": np.arange(n), "loan_amnt": loan_amnt, "term": [f" {t} months" for t in term],
        "int_rate": int_rate, "grade": grade, "emp_length": emp_txt, "home_ownership": home,
        "annual_inc": annual_inc, "verification_status": verif, "issue_d": _fmt(issue),
        "loan_status": status, "purpose": purpose, "dti": dti, "delinq_2yrs": delinq,
        "earliest_cr_line": _fmt(earliest), "inq_last_6mths": inq, "mths_since_last_delinq": msld,
        "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util,
        "total_acc": total_acc,
        # columnas de FUGA (existen en el archivo real; el código las ignora a propósito)
        "total_pymnt": np.where(bad, loan_amnt * rng.uniform(0.2, 0.8, n), loan_amnt * 1.1),
        "recoveries": np.where(bad, loan_amnt * rng.uniform(0, 0.1, n), 0.0),
    })
