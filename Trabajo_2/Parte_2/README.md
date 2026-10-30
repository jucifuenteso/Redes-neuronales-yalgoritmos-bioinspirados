# ¿Me prestas? — Trabajo 2, Parte 2: pérdida dado incumplimiento (LGD) y pérdida esperada

Regresión de la LGD con tres referencias de baja complejidad (media constante, regresión lineal, logit fraccional) y un
perceptrón multicapa, y combinación con la PD de la Parte 1: $E[L] = PD \cdot LGD \cdot EAD$.

> **Estado honesto:** el flujo completo se probó de punta a punta con datos **sintéticos**. La Parte 1 ya se corrió con
> el `loan.csv` real, pero ese archivo no está en el repositorio, y la Parte 2 lo necesita porque la tabla procesada de
> la Parte 1 no incluye las columnas de recuperación. Los números del reporte deben salir de **su** corrida con
> `configs/default.yaml`.

## 1. Relación con la Parte 1
La Parte 2 **importa** el código de la Parte 1 (`../Parte_1/pdmodel`) en lugar de copiarlo:

| Hereda | Garantía |
|---|---|
| Caso de uso, variables de solicitud, definición de incumplimiento | `pdmodel.data.clean_applications` / `build_target` |
| Semilla, partición temporal, `use_lc_pricing`, winsorización | se leen del YAML de la Parte 1 (`parte1.config`) |
| Preprocesamiento | misma clase `Preprocessor`, mismos créditos de entrenamiento (verificado en `perdida`) |
| Red neuronal en NumPy | `MLPLGD` hereda de `pdmodel.mlp.MLPClassifier`; solo cambia la pérdida |
| PD para la pérdida esperada | `Parte_1/models/{lr_final,mlp_final}.joblib`, sin reentrenar |

`preparar` comprueba que créditos y particiones coinciden con `Parte_1/reports/tables/data_summary.json`, y `perdida` se
detiene si no coinciden (habría créditos de prueba vistos por la PD en entrenamiento).

## 2. Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # = las versiones fijas de la Parte 1 (un solo entorno)
```

## 3. Datos reales
Use el **mismo** archivo que la Parte 1: `Parte_1/data/raw/loan.csv` (Kaggle: ranadeep/credit-risk-dataset). La fecha de
descarga también se hereda (`data.download_date` en `Parte_1/configs/default.yaml`). Los significados de las variables
se citan desde `Parte_1/data/raw/LCDataDictionary.xlsx` (ver `docs/guia_conceptual.md` §3).

## 4. Ejecutar (desde `Parte_2/`, un solo comando por experimento)
| Qué | Comando |
|---|---|
| Demostración completa (< 2 min, datos sintéticos; incluye la Parte 1 en modo demo) | `python -m lgdmodel todo --config configs/demo.yaml` |
| Todo con datos reales (requiere la Parte 1 ya corrida) | `python -m lgdmodel todo` |
| Pasos sueltos | `python -m lgdmodel {preparar,describir,referencias,busqueda,final,ablacion,perdida,figuras} [--config ...]` |
| Pruebas | `python -m unittest discover -s tests -v` |

Con `make demo`, `make full`, `make test`, `make figuras` si tiene `make`. La demostración escribe sus salidas en
carpetas `demo/` (y las de la Parte 1 en `Parte_2/demo_parte1/`): nunca toca los resultados reales de ninguna de las dos partes.

**Cambiar parámetros sin editar código** (cada comando imprime la configuración **efectiva** antes de entrenar):
```bash
python -m lgdmodel final --set final.sobreescribir.hidden=[64,32] --set final.sobreescribir.activation=tanh
python -m lgdmodel final --set final.n_semillas=5 --set final.max_epocas=30
python -m lgdmodel final --set red.perdida=mse                  # red con MSE en vez de BCE
python -m lgdmodel busqueda --set busqueda.n_pruebas=10
python -m lgdmodel todo --set 'lgd.estados_perdida_definitiva=["Charged Off","Default"]'
python -m lgdmodel preparar --set-parte1 data.split.val_end=2013-06-30   # la pérdida esperada se detendrá: la PD usó otra partición
```
`final.sobreescribir` se combina **clave por clave** con la mejor configuración de la búsqueda; una clave mal escrita da error.

## 5. Salidas (`reports/tables`, `reports/figures`, `models`)
`lgd_data_summary.json` (registros, periodo, LGD por partición, consistencia con la Parte 1) · `lgd_estados_por_particion.csv`,
`lgd_recuperaciones_por_estado.csv`, `lgd_distribucion.json`, `lgd_por_segmento.csv` · `referencias_resultado.json`,
`referencias_busqueda.csv` · `busqueda_pruebas.csv`, `busqueda_resultado.json` · `semillas_metricas.csv`,
`semillas_resumen.csv` (media, desv., mejor, peor de MAE y RMSE) · `comparacion_modelos.csv` · `presupuesto_computo.json`
(configuraciones, segundos y parámetros de cada familia) · `predicciones_lgd.csv.gz`, `distribucion_predicciones.json`,
`calibracion_lgd_deciles.csv` · `ablacion_perdida*.{csv,json}` · `perdida_esperada_{resumen.json,deciles.csv,prueba.csv.gz}`.
Figuras: `python -m lgdmodel figuras` las regenera sin reentrenar.

## 6. Decisiones de diseño (defiéndanlas en vivo)
- **LGD** $=$ recortar$_{[0,1]}(1-(\texttt{total\_rec\_prncp}+\max(\texttt{recoveries}-\texttt{collection\_recovery\_fee},0))/\texttt{funded\_amnt})$,
  solo para incumplidos con pérdida definitiva (`Charged Off` y su variante de política). `Late (31-120 days)` y `Default`
  cuentan como malos en la PD pero no tienen LGD observable; `describir` muestra la evidencia por estado.
- **EAD** $=$ `funded_amnt`: la decisión se toma al recibir la solicitud; coherente con la LGD sobre el monto desembolsado.
- **Salida sigmoide + BCE** (cuasi-verosimilitud Bernoulli) porque la LGD vive en $[0,1]$; `ablacion` compara con MSE
  usando la misma arquitectura y las mismas semillas.
- **Tres referencias:** la media (¿aprende algo el modelo?), la regresión lineal (la que pide el enunciado) y el logit
  fraccional (la red sin capas ocultas, ajustada hasta convergencia), que aísla el aporte de las capas ocultas.
- **Selección solo con validación**; la semilla representante se elige por RMSE de validación, nunca con prueba.
  Las semillas finales son `semilla + 1000 + k`, como en la Parte 1.
- **Pérdida esperada frente a la realizada** solo en créditos con desenlace definitivo; el bloque `diagnostico`
  separa el aporte de la PD y de la LGD a cualquier diferencia.

Documentación: `docs/guia_conceptual.md` (matemática y decisiones) y `docs/caceria_alucinacion.md` (errores de la IA hallados y corregidos).

## 7. Pendientes del equipo
Correr con el `loan.csv` real y versionar `data/processed/lgd_dataset.csv.gz` (si pesa demasiado, documentar una versión
comprimida o submuestreada), `models/` y `reports/`. Verificar los pendientes de `docs/caceria_alucinacion.md`. Decidir con
quien hizo la Parte 1 el tratamiento de `Late (31-120 days)` y `Default`. Probar en un clon limpio.
