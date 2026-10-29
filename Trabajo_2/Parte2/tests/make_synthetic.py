"""Genera un loan.csv SINTÉTICO con las columnas de Lending Club, solo para pruebas de humo.

No son datos reales: sirve para verificar que el flujo corre de extremo a extremo. La LGD sintética no
depende de los predictores, así que ningún modelo debería superar a la media constante.
Uso: python tests/make_synthetic.py <ruta_salida.csv> [n]
"""
import sys

import numpy as np
import pandas as pd


def make(n_loans=40000, seed=0):
    random_generator = np.random.default_rng(seed)
    issue_dates = pd.to_datetime("2008-01-01") + pd.to_timedelta(
        random_generator.integers(0, 8 * 365, n_loans), unit="D")
    annual_income = np.exp(random_generator.normal(11, 0.5, n_loans))
    loan_amount = random_generator.integers(1, 36, n_loans) * 1000.0
    debt_to_income = random_generator.gamma(4, 4.5, n_loans)
    revolving_utilization = np.clip(random_generator.normal(55, 25, n_loans), 0, 120)
    term_months = random_generator.choice([36, 60], n_loans, p=[0.75, 0.25])
    inquiries_last_6_months = random_generator.poisson(1, n_loans)
    delinquencies_last_2_years = random_generator.poisson(0.3, n_loans)
    is_renting = random_generator.random(n_loans) < 0.45
    default_logit = (-2.6 + 0.025 * debt_to_income + 0.01 * revolving_utilization
                     - 0.6 * np.log(annual_income / 60000) + 0.5 * (term_months == 60)
                     + 0.1 * inquiries_last_6_months + 0.15 * delinquencies_last_2_years
                     + 0.25 * is_renting + 0.00001 * (loan_amount - 12000))
    default_probability = 1 / (1 + np.exp(-default_logit))
    is_bad = random_generator.random(n_loans) < default_probability
    loan_status = np.where(
        is_bad,
        random_generator.choice(["Charged Off", "Default", "Late (31-120 days)"], n_loans, p=[0.85, 0.05, 0.10]),
        "Fully Paid")
    is_still_current = (issue_dates > "2013-06-01") & (random_generator.random(n_loans) < 0.6)
    loan_status = np.where(is_still_current & ~is_bad, "Current", loan_status)
    loan_status = np.where(random_generator.random(n_loans) < 0.02,
                           "Does not meet the credit policy. Status:Fully Paid", loan_status)
    fraction_of_principal_paid = np.where(is_bad, random_generator.beta(2, 3, n_loans), 1.0)
    recoveries = np.where(
        np.isin(loan_status, ["Charged Off", "Default"]) & (random_generator.random(n_loans) < 0.4),
        loan_amount * (1 - fraction_of_principal_paid) * random_generator.beta(1, 4, n_loans), 0.0)
    earliest_credit_line_dates = issue_dates - pd.to_timedelta(random_generator.integers(900, 9000, n_loans),
                                                               unit="D")
    return pd.DataFrame({
        "loan_amnt": loan_amount, "funded_amnt": loan_amount,
        "term": [f" {months} months" for months in term_months],
        "int_rate": 6 + 20 * default_probability + random_generator.normal(0, 1, n_loans),
        "installment": loan_amount / term_months,
        "grade": random_generator.choice(list("ABCDEFG"), n_loans), "sub_grade": "A1",
        "emp_length": random_generator.choice(["< 1 year", "1 year", "5 years", "10+ years", "n/a"], n_loans),
        "home_ownership": np.where(is_renting, "RENT",
                                   random_generator.choice(["MORTGAGE", "OWN", "OTHER"], n_loans)),
        "annual_inc": annual_income,
        "verification_status": random_generator.choice(["Verified", "Not Verified", "Source Verified"], n_loans),
        "issue_d": issue_dates.strftime("%b-%Y"), "loan_status": loan_status,
        "purpose": random_generator.choice(
            ["debt_consolidation", "credit_card", "small_business", "car", "other"], n_loans),
        "addr_state": random_generator.choice(["CA", "NY", "TX", "FL", "IL", "OH"], n_loans),
        "dti": debt_to_income, "delinq_2yrs": delinquencies_last_2_years,
        "earliest_cr_line": earliest_credit_line_dates.strftime("%b-%Y"),
        "inq_last_6mths": inquiries_last_6_months,
        "mths_since_last_delinq": np.where(delinquencies_last_2_years > 0,
                                           random_generator.integers(1, 60, n_loans), np.nan),
        "mths_since_last_record": np.nan, "open_acc": random_generator.integers(2, 30, n_loans),
        "pub_rec": random_generator.poisson(0.1, n_loans),
        "revol_bal": np.exp(random_generator.normal(9, 1, n_loans)), "revol_util": revolving_utilization,
        "total_acc": random_generator.integers(3, 60, n_loans),
        "initial_list_status": random_generator.choice(["f", "w"], n_loans),
        "collections_12_mths_ex_med": 0.0, "acc_now_delinq": 0.0,
        "total_rec_prncp": loan_amount * fraction_of_principal_paid, "recoveries": recoveries,
        "collection_recovery_fee": recoveries * 0.1,
    })


if __name__ == "__main__":
    output_path = sys.argv[1]
    make(int(sys.argv[2]) if len(sys.argv) > 2 else 40000).to_csv(output_path, index=False)
    print("escrito", output_path)
