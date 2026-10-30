"""Aplicación web (Streamlit):  streamlit run app/streamlit_app.py
Variable de entorno opcional MODELS_DIR (por defecto 'models'; para la demo use models/demo)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from pdmodel.app_logic import ScoringService

# TODO: reemplazar por los enlaces reales del equipo
LINK_REPORTE = "https://example.com/reporte-tecnico"
LINK_VIDEO = "https://example.com/video-promocional"

st.set_page_config(page_title="¿Me prestas?", page_icon="💳", layout="wide")
MODELS = ROOT / os.environ.get("MODELS_DIR", "models")


@st.cache_resource
def get_service():
    return ScoringService(MODELS)


try:
    svc = get_service()
except Exception as e:  # modelos aún no entrenados
    st.error(f"No se encontraron los modelos en {MODELS}. Entrene primero (ver README). Detalle: {e}")
    st.stop()

meta = svc.meta
LABELS = {
    "loan_amnt": "Monto solicitado (USD)", "term_months": "Plazo (meses)", "emp_length": "Años en el empleo actual",
    "annual_inc": "Ingreso anual (USD)", "dti": "Deuda/ingreso mensual, DTI (%)", "delinq_2yrs": "Moras en los últimos 2 años",
    "inq_last_6mths": "Consultas de crédito en 6 meses", "mths_since_last_delinq": "Meses desde la última mora (vacío = nunca)",
    "open_acc": "Cuentas de crédito abiertas", "pub_rec": "Registros públicos negativos", "revol_bal": "Saldo revolvente (USD)",
    "revol_util": "Utilización de crédito revolvente (%)", "total_acc": "Total de cuentas de crédito",
    "cr_hist_years": "Años de historia crediticia", "home_ownership": "Vivienda", "verification_status": "Verificación de ingresos",
    "purpose": "Propósito del préstamo", "int_rate": "Tasa de interés (%)", "grade": "Grado",
}

st.title("💳 ¿Me prestas? Simula tu puntaje de crédito")
st.caption("Modelo de red neuronal entrenado con créditos de Lending Club. Herramienta académica, no es una oferta de crédito.")
c1, c2 = st.columns(2)
c1.link_button("📄 Reporte técnico", LINK_REPORTE)
c2.link_button("🎬 Video promocional", LINK_VIDEO)

with st.sidebar:
    st.header("Tus características")
    applicant = {}
    for c in meta["num_cols"]:
        lo, hi = meta["ranges"][c]
        default = meta["defaults"][c]
        if c == "term_months":
            applicant[c] = st.selectbox(LABELS[c], [36, 60], index=0)
        elif c == "mths_since_last_delinq":
            never = st.checkbox("Nunca he tenido mora", value=True)
            applicant[c] = None if never else st.number_input(LABELS[c], 0.0, 200.0, float(default))
        else:
            step = 1.0 if hi - lo > 20 else 0.1
            applicant[c] = st.number_input(LABELS[c], min_value=0.0, max_value=float(hi * 3), value=float(round(default, 1)), step=step)
    for c in meta["cat_cols"]:
        opts = meta["categories"][c]
        applicant[c] = st.selectbox(LABELS.get(c, c), opts, index=opts.index(meta["defaults"][c]))

res = svc.score(applicant)
m1, m2, m3 = st.columns(3)
m1.metric("Puntaje", f"{res['score']:.0f}")
m2.metric("Probabilidad de incumplimiento", f"{res['pd']:.1%}")
m3.metric("Mejor que el", f"{res['percentile']:.0f}% de la población")
sc = meta["scale"]
st.caption(f"Escala: {sc['base_score']:.0f} puntos = odds {sc['base_odds']:.0f}:1 (buenos:malos); cada {sc['pdo']:.0f} puntos duplican las odds.")

left, right = st.columns(2)
with left:
    st.subheader("¿Qué suma y qué resta?")
    contrib = svc.contributions(applicant)
    st.bar_chart(contrib.set_index("variable"), horizontal=True)
    st.caption("Puntos respecto al perfil promedio (aproximación aditiva del modelo).")
with right:
    st.subheader("¿Y si cambiara algo?")
    var = st.selectbox("Característica", [c for c in meta["num_cols"] if c != "mths_since_last_delinq"],
                       format_func=lambda c: LABELS[c])
    lo, hi = meta["ranges"][var]
    new_val = st.slider("Nuevo valor", float(lo), float(hi), float(applicant[var] if applicant[var] is not None else lo))
    base, new, delta = svc.what_if(applicant, {var: new_val})
    st.metric("Puntaje con el cambio", f"{new['score']:.0f}", f"{delta:+.0f} puntos")
    sweep = svc.sweep(applicant, var, list(np.linspace(lo, hi, 25)))
    st.line_chart(sweep.set_index(var))
