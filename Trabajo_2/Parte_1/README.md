# ¿Me prestas? — Trabajo 2, Parte 1: probabilidad de incumplimiento (PD)

Clasificación binaria con regresión logística (referencia) y perceptrón multicapa sobre *Credit Risk Analysis* (Lending Club).
Incluye: construcción del objetivo, partición temporal, búsqueda de hiperparámetros, 30 semillas, métricas de
discriminación/calibración, scorecard, interpretabilidad, pruebas unitarias y una app web.

> **Estado honesto:** el flujo completo se probó de punta a punta con datos **sintéticos** (mismo formato que `loan.csv`).
> Aún no se ha corrido con el archivo real de Kaggle (no hay internet donde se construyó). Los números del reporte
> deben salir de **su** corrida con los datos reales.

## 1. Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Solo se necesita NumPy/pandas/scikit-learn (la red está escrita en NumPy; sklearn solo se usa para la regresión logística).

## 2. Datos reales
1. Descargue `loan.csv` de https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset y póngalo en `data/raw/loan.csv`.
2. **Anote la fecha de descarga** y escríbala en `configs/default.yaml` → `data.download_date`.
3. El diccionario de variables viene en el mismo conjunto: cítenlo desde allí.

## 3. Ejecutar (un solo comando por experimento)
| Qué | Comando |
|---|---|
| Demostración completa (< 2 min, datos sintéticos) | `python -m pdmodel all --config configs/demo.yaml` |
| Todo con datos reales | `python -m pdmodel all --config configs/default.yaml` |
| Pasos sueltos | `python -m pdmodel {synth,prepare,describe,baseline,search,final,scorecard} --config ...` |
| Cambiar parámetros sin editar código | `... --set final.n_seeds=5 --set scorecard.pdo=40 --set mlp_search.n_trials=50` |
| Pruebas | `python -m unittest discover -s tests -v` |
| App web (tras entrenar) | `streamlit run app/streamlit_app.py` (demo: `MODELS_DIR=models/demo streamlit run app/streamlit_app.py`) |

Con `make demo`, `make full`, `make test`, `make app` si tiene `make`. Las semillas están en `seed` (config) y las 30 corridas usan `seed+1000+k`.

## 4. Estructura
```
configs/   default.yaml (real) · demo.yaml (rápido)      pdmodel/   código (data, features, mlp, metrics, scorecard, ...)
data/      raw/ (loan.csv) · processed/ (dataset limpio)   models/    modelos entrenados (.joblib, .json, .npy)
reports/   tables/ · figures/ (todo regenerable)           docs/      guia_conceptual.md · plantilla_reporte.md
app/       streamlit_app.py                                 tests/     pruebas unitarias
```
Salidas clave en `reports/tables`: `data_summary.json` (registros, variables, periodo, fecha), `descriptive_bad_rate.csv`,
`baseline_result.json`, `mlp_search_trials.csv`, `mlp_seeds_metrics.csv` / `mlp_seeds_summary.csv` (media, desv., mejor, peor),
`comparison_lr_vs_mlp.csv`, `compute_budget.json` (presupuesto de búsqueda, tiempo, parámetros), `scorecard_*_points.csv`,
`permutation_importance_*.csv`. Figuras en `reports/figures`.

## 5. Decisiones de diseño (defiéndanlas en vivo)
- Solo variables conocidas **al solicitar** el crédito; se ignoran columnas de pagos/recuperaciones (fuga).
- `int_rate` y `grade` **excluidas por defecto** (las fija Lending Club con su propio modelo); `--set data.use_lc_pricing=true` las incluye.
- Partición temporal por `issue_d`: train ≤ 2012, val 2013, test ≥ 2014 (configurable). Preprocesamiento ajustado solo con train.
- Sin re-ponderar clases: la pérdida es la verosimilitud de Bernoulli y se busca una PD calibrada.
- Selección de hiperparámetros solo con validación; la semilla "representante" se elige en validación, nunca en prueba.
- Para subir los datos procesados al repo: `data/processed/pd_dataset.csv.gz` (se genera con `prepare`; si pesa demasiado, use `sample_frac` y documéntelo).

## 6. Pendientes del equipo
Enlaces en la app (`LINK_REPORTE`, `LINK_VIDEO`), fecha de descarga, historial de commits real, probar en un clon limpio, videos, reporte.
