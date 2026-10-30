# Cacería de la alucinación — Parte 2

Hallazgos reales en código y textos generados por una IA (Claude) durante el desarrollo de la Parte 2. Cada uno tiene
evidencia reproducible. **Antes de llevarlos al reporte, reprodúzcanla ustedes**: lo que se declara puede preguntarse en
la sustentación. Los puntos sin evidencia todavía están al final, separados: no los reporten como hallazgos hasta
verificarlos (un hallazgo sin evidencia se penaliza más que uno faltante).

La primera versión (donde están H1–H4 y H7) quedó en el historial: `git show 639ea05` (carpeta `Trabajo_2/Parte2/`).
Para reproducir la evidencia: `git worktree add ../p2_antigua 639ea05` y ejecutar allí los comandos citados.

Prompts usados (literales):
- **P1:** "rrealiza  nada masa la parte 2 y  crea una carpeta llamada parte 2 en el trabajo 2" (con el PDF del enunciado adjunto).
- **P2:** "haz que las variables sean lo mas descriptivas posibles"
- **P3:** "que alucinaciones puedes encontrar en lo generado" (revisión de lo anterior)
- **P4:** "acabo de eliminar lo hecho de la parte 2 , ahora toma en cuenta la parte 1 para re hacer la parte 2 , tambien toma en cuenta solucionar las alucinaciones identificadas"

---

## H1 — Código: los cambios en vivo de arquitectura se ignoraban en silencio
1. **Prompt:** P1.
2. **Respuesta de la IA:** el README afirmaba que `python run.py lgd --skip-search --set lgd.mlp.hidden=[128,64] ...` cambia la arquitectura.
3. **Sospecha:** la función cargaba `models/lgd_best_hp.json` cuando existía, antes de mirar la configuración.
4. **Evidencia (experimento):** se pidió `hidden=[7]` y `activation=tanh`; el modelo entrenado usó `hidden=[16]` y `elu` (los del archivo guardado).
5. **Corrección:** `final.sobreescribir` se combina clave por clave sobre la mejor configuración (`experiments.py::configuracion_final_efectiva`), la configuración efectiva se imprime y se guarda, y una clave mal escrita produce error. Pruebas: `test_sobreescritura_parcial_de_la_red_final`, `test_set_anidado_en_sobreescribir`.
6. **Lección:** después de cada modificación por línea de comandos, imprimir y verificar la configuración **efectiva**. Era el error más peligroso: en la "modificación en vivo" la predicción del equipo no habría coincidido con nada.

## H2 — Código: el modo demostración pisaba los `--set`
1. **Prompt:** P1.
2. **Respuesta:** `python run.py lgd --demo --set ...` "cambia parámetros sin editar código".
3. **Sospecha:** `apply_demo()` se aplicaba después de leer las sobreescrituras.
4. **Evidencia:** con `--demo --set lgd.mlp.epochs=50 --set n_seeds=10` quedaron 8 épocas y 3 semillas.
5. **Corrección:** como en la Parte 1, la demostración es un YAML propio (`configs/demo.yaml`) y los `--set` se aplican después de leerlo. Prueba: `test_set_se_aplica_despues_del_yaml_de_demostracion`.
6. **Lección:** probar cada combinación de opciones que se usará en vivo, no solo la opción por separado.

## H3 — Matemáticas/código: estadístico KS mal calculado con empates
1. **Prompt:** P1 (el módulo `metrics.py` de la primera versión).
2. **Respuesta:** `ks()` ordenaba con `argsort` y tomaba el máximo sobre **filas**, no sobre umbrales distintos.
3. **Sospecha:** con puntajes repetidos, el orden dentro de un empate es arbitrario.
4. **Evidencia:** `y=[0,1]`, `p=[0.5,0.5]` daba KS = **1.0** (lo correcto es 0); con puntajes redondeados daba 0.2301 frente a 0.2171 de `scipy.stats.ks_2samp`.
5. **Corrección:** la Parte 2 ya no implementa el KS de clasificación; el KS entre distribuciones usa `scipy.stats.ks_2samp`. La Parte 1 tiene su propia `ks_score`, que **sí** evalúa un valor por empate y su prueba la contrasta con la curva ROC de scikit-learn.
6. **Lección:** probar cada métrica con empates y contra una implementación de referencia.

## H4 — Datos del mundo real: explicación falsa del sesgo de la pérdida esperada
1. **Prompt:** P1.
2. **Respuesta:** "en los créditos recientes los 'Current' quedan fuera, así que la pérdida realizada observable está sobreestimada…".
3. **Sospecha:** la tabla del enunciado ya excluye `Current` de **todo** el entrenamiento y la evaluación.
4. **Evidencia:** el conjunto procesado no contiene ningún `Current`. Con los datos reales de la Parte 1: PD media predicha en prueba 0.194 frente a una tasa de malos observada de 0.260.
5. **Corrección:** el resumen de la pérdida esperada ya no afirma un mecanismo. Muestra un `diagnostico` con datos (PD media frente a tasa observada; LGD media predicha frente a observada) y la guía plantea el mecanismo como hipótesis a contrastar.
6. **Lección:** antes de explicar un sesgo, comprobar en los datos qué filas entran realmente en cada cálculo.

## H5 — Código/conceptos: referencia sub-entrenada que favorecía a la red
1. **Prompt:** P4.
2. **Respuesta:** el logit fraccional se entrenaba como la red, con Adam, 15 épocas y lotes de 1024.
3. **Sospecha:** en la demostración el logit fraccional (un GLM) daba RMSE 0.2106, casi la media constante (0.2116), mientras la regresión lineal daba 0.1849.
4. **Evidencia:** con ~1 700 créditos y lotes de 1024 son ~2 pasos por época, ~30 pasos en total, y la mejor época era la última (15 de 15). Con 300 épocas el mismo modelo llega a RMSE 0.1849 y converge.
5. **Corrección:** el logit fraccional se resuelve con L-BFGS de lote completo hasta convergencia (`logit_fraccional.py`). Prueba: con `y` binaria coincide con `LogisticRegression(penalty=None)` de scikit-learn. Además, búsqueda y semillas avisan cuando la mejor época es la última.
6. **Lección:** en una comparación justa, verificar que **cada** modelo convergió, no solo la red; una referencia mal entrenada infla la ventaja de la red.

## H6 — Código: variable sin renombrar (`NameError`)
1. **Prompt:** P2.
2. **Respuesta:** renombrado masivo de variables con un script de reemplazos.
3. **Sospecha:** revisión con `grep` de los nombres viejos tras el reemplazo.
4. **Evidencia:** quedó `print(f"... {name} ...")` cuando la variable ya se llamaba `step_name`: todo comando habría fallado.
5. **Corrección:** se corrigió y se ejecutó el flujo completo.
6. **Lección:** después de un refactor automático, ejecutar el flujo completo y las pruebas; no basta con leer el diff.

## H7 — Afirmación sin respaldo: "la misma partición de la Parte 1"
1. **Prompt:** P1.
2. **Respuesta:** comentario en la configuración: "Partición temporal ... la misma de la Parte 1".
3. **Sospecha:** en ese momento la Parte 1 no existía; las fechas venían de otro borrador (y de hecho la Parte 1 usa `val_end: 2013-12-31`, no `2013-06-30`).
4. **Evidencia:** `Parte_1/configs/default.yaml`.
5. **Corrección:** la Parte 2 **lee** la partición del YAML de la Parte 1 y la verifica contra su `data_summary.json`; la pérdida esperada se detiene si no coincide. Prueba: `test_mismas_particiones_y_deteccion_de_diferencias`.
6. **Lección:** una afirmación de consistencia entre módulos debe estar garantizada por código y verificada por una prueba, no escrita en un comentario.

---

## Pendientes de verificar con `loan.csv` real (NO son hallazgos todavía)

- **"La LGD tiene forma de U, con masa en 0 y en 1"** (afirmado en la primera versión para justificar BCE frente a MSE). Con
  la LGD sobre `funded_amnt`, que incluye el capital pagado antes del castigo, LGD ≈ 1 exige no haber pagado capital y
  LGD ≈ 0 es casi imposible. Verificación: `python -m lgdmodel describir` → `lgd_distribucion.json`
  (`fraccion_mayor_igual_099`, `fraccion_menor_igual_001`) y `lgd_distribucion_observada.png`.
- **"`Default` tiene pérdida definitiva"** (la primera versión lo trataba así). Verificación:
  `lgd_recuperaciones_por_estado.csv`: si los créditos en `Default` casi nunca tienen `recoveries > 0`, su recuperación aún
  no ocurrió y la exclusión actual es correcta. Confirmen también la definición de cada estado en la documentación de Lending Club ⚠.
- **Referencias** (Papke y Wooldridge, 1996; Gourieroux et al., 1984; Liu y Nocedal, 1989): verifiquen volumen,
  páginas y DOI antes de citar.
