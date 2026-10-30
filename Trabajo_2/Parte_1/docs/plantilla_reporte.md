# Plantilla del reporte técnico (Parte 1) — entrada de blog

Cada figura/tabla: numerada, con título y fuente, y **citada en el texto**. Archivos generados entre paréntesis.

1. **Resumen** (qué se hizo y principal hallazgo, con números de SU corrida).
2. **Datos** — registros y variables, periodo, fecha de descarga (`data_summary.json`); diccionario citado de Kaggle; construcción del objetivo (tabla de codificación del enunciado); tasa de malos y conteo por `loan_status`.
3. **Caso de uso** (quién, qué decisión, cuándo, qué información) y consecuencias: variables excluidas por fuga, `int_rate/grade`, censura.
4. **Análisis descriptivo e hipótesis** — lista numerada H1…Hn escrita *antes* del modelo (`descriptive_bad_rate.png/csv`, `descriptive_missing.csv`). Al final: confirmadas / no confirmadas.
5. **Metodología** — particiones y por qué (`data_summary.json`, tabla de particiones); preprocesamiento; modelo de referencia; MLP; búsqueda y presupuesto (`mlp_search_trials.csv`, `compute_budget.json`); 30 semillas.
6. **Fundamentación matemática** — ver `guia_conceptual.md` §4–5 (con sus propias derivaciones).
7. **Resultados** — tabla comparativa LR vs MLP (`comparison_lr_vs_mlp.csv`: media, desv., mejor, peor, parámetros, tiempo); ROC, KS, calibración (`test_roc.png`, `test_ks.png`, `test_calibration.png`); curvas de aprendizaje; ablaciones que ustedes decidan.
8. **Scorecard** — escala, derivación, tabla de puntos (`scorecard_mlp_points.csv`), fidelidad (`scorecard_summary.json`), distribución de puntajes (`score_distribution.png`).
9. **Variables de riesgo** — `permutation_importance.png`; discusión y limitaciones; contraste con las hipótesis.
10. **Discusión** — ¿qué aportó la red frente a la regresión logística (métricas, variabilidad, costo, interpretabilidad)? **Aprendizajes sobre el problema** (lista).
11. **Uso de IA** — prompts principales e impacto.  12. **Cacería de la alucinación** (≥5 hallazgos, ≥3 categorías).
13. **Enlaces**: repositorio, app, video.  14. **Bibliografía APA 7** (todas verificadas).
