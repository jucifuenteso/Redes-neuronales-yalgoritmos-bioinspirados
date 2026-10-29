# Trabajo 2 · Parte 2 — Pérdida dado incumplimiento (LGD) y pérdida esperada

Redes neuronales y algoritmos bioinspirados, UNAL Medellín 2026-02. Esta carpeta contiene **solo la Parte 2**
(regresión de la LGD y combinación `E[L] = PD · LGD · EAD`). Dataset: Lending Club,
<https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset>.

> **Estado.** El código está probado de extremo a extremo con datos **sintéticos** (`tests/make_synthetic.py`).
> **Aún no hay resultados reales:** `loan.csv` no estaba en la máquina y requiere cuenta de Kaggle. Los pasos de abajo
> los generan. Ningún número de esta carpeta proviene de datos sintéticos.

## Instalación

```bash
python -m venv .venv && .venv\Scripts\activate     # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt                    # versiones fijas (probado con Python 3.13)
python -m pytest tests -q                          # 7 pruebas, ~4 s
```

## Datos

1. Descargar `loan.csv` (y `LCDataDictionary.xlsx`, el diccionario del mismo conjunto) y ubicar `loan.csv` en `data/raw/loan.csv`.
2. `python run.py prepare --download-date AAAA-MM-DD`
   Genera `data/processed/loans_clean.csv.gz` y `data/processed/data_report.json` (n.º de registros y variables,
   periodo, fecha de descarga). **Versionar `data/processed/`** para que todo corra sin internet; `data/raw/` está en `.gitignore`.

## Ejecución (un comando por experimento)

| Comando | Qué hace |
|---|---|
| `python run.py describe` | LGD observada por año, plazo, propósito, etc. (`outputs/lgd_descriptive*.csv/json`) |
| `python run.py lgd` | regresión lineal (Ridge) + búsqueda aleatoria del MLP + MLP final × 30 semillas |
| `python run.py lgd-loss` | ablación de la pérdida: BCE vs MSE, misma arquitectura y semillas |
| `python run.py el` | `E[L] = PD·LGD·EAD` en prueba vs pérdida realizada |
| `python run.py figures` | regenera las figuras en `figures/` desde `outputs/` (sin reentrenar) |
| `python run.py all` | `describe → lgd → lgd-loss → el → figures` |

**Modo demostración (< 2 min):** `python run.py all --demo` (muestra de 30 000 filas, 3 semillas, pocas épocas); escribe
en `models_demo/`, `outputs_demo/`, `figures_demo/` sin pisar los resultados completos. Con datos sintéticos de 40 000
filas tardó ~9 s.

**Cambiar parámetros sin editar código** (valores por defecto en `configs/default.yaml`):

```bash
python run.py lgd --skip-search --set lgd.mlp.hidden=[128,64] --set lgd.mlp.lr=0.0005 --set lgd.mlp.dropout=0.3
python run.py lgd --set lgd.mlp.loss=mse            # LGD con MSE en vez de BCE
python run.py lgd --set lgd.mlp.activation=tanh --set lgd.mlp.optimizer=adamw
python run.py lgd --set data.split=random           # partición aleatoria en vez de temporal
python run.py lgd --set n_seeds=5 --set lgd.search.trials=10
python run.py lgd --config configs/mi_config.yaml
```

## Qué produce y dónde

- `outputs/lgd_summary.json`: MAE/RMSE de la referencia lineal, de la constante (media de entrenamiento) y del MLP
  (media, desviación, mejor y peor sobre 30 semillas), n.º de parámetros, configuraciones entrenadas, tiempos y
  `distribution_test` (cuantiles, masa en extremos, KS y Wasserstein contra la LGD observada).
- `outputs/lgd_mlp_seeds.csv` (una fila por semilla), `lgd_search_trials.csv`, `lgd_loss_ablation*.{csv,json}`,
  `lgd_predictions.csv`, `expected_loss_summary.json`, `expected_loss_deciles.csv`.
- `models/`: referencia lineal, MLP de LGD (`.pt` y `.npz`), preprocesador. **Versionar `models/` y `outputs/`.**
- `figures/`: `fig_lgd_descriptive`, `fig_lgd_distributions`, `fig_lgd_overlay`, `fig_lgd_calibration`, `fig_lgd_seeds`,
  `fig_lgd_search`, `fig_lgd_loss_ablation`, `fig_expected_loss_deciles`. Cada una debe citarse en el reporte.

## Decisiones de diseño (y sus límites)

- **Objetivo y fórmula de la LGD, EAD, función de pérdida, activación de salida y derivaciones:** `docs/fundamentos_lgd.md`.
- **Salida sigmoide + BCE** (cuasi-verosimilitud Bernoulli) por ser la LGD un objetivo en [0,1] con masa en los
  extremos; `lgd-loss` lo contrasta contra MSE en lugar de darlo por supuesto.
- **Comparación justa:** misma tabla procesada, mismo preprocesador (ajustado solo con entrenamiento) y mismas particiones
  para lineal y MLP. La referencia lineal recibe 5 configuraciones (α de Ridge) y el MLP 30 + las 30 semillas finales; ambos
  presupuestos y tiempos se reportan en `lgd_summary.json` y deben citarse tal cual. El MLP "representativo" que se
  guarda y usa en `el` es la semilla de menor RMSE de **validación**, nunca de prueba.
- **Sin fuga:** las columnas post-originación (`total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `loan_status`) solo
  construyen la LGD; no son predictores (`credit/data.py`, `ORIGINATION` vs `POST_ORIGINATION`). Tampoco `grade`/`int_rate`
  por defecto (`data.include_lc_pricing=false`).
- **Sesgo en `el`:** la pérdida realizada solo es observable en créditos con desenlace definitivo; en la ventana de
  prueba quedan fuera los `Current` y `Late (31-120 days)`. La razón `E[L]/realizada` es una cota, no una medida de
  acierto (ver nota en `expected_loss_summary.json`).

## Conexión con la Parte 1

`el` necesita una PD. Para que esta carpeta sea autónoma entrena internamente una **PD mínima** (regresión logística y un
MLP de configuración fija, una semilla, sin búsqueda; `pd_ref` en la config). No es la de la Parte 1: la comparación de
modelos de PD, su búsqueda y sus 30 semillas pertenecen a esa parte. Al integrar, reemplazar `_fit_pd_reference` en
`credit/experiments.py` por la carga de los modelos entrenados de la Parte 1.

## Semillas

Semilla global `seed: 42` (búsqueda, muestreos, partición aleatoria opcional). Las 30 corridas finales de cada red usan las
semillas `1000…1029` (`seeds_final_start`, `n_seeds`). La regresión lineal es determinista.

## Pendiente (no cubierto aquí)

Reporte/blog, aplicación web, videos, "cacería de la alucinación" y sección de uso de IA; Parte 1 completa.
