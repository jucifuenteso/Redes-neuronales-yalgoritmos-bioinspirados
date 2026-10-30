# Guía conceptual — Parte 2 (LGD y pérdida esperada)

Úsenla para **entender**, no para copiar. La IA que escribió esto ya se equivocó varias veces en este mismo trabajo
(ver `caceria_alucinacion.md`). Todo lo marcado con ⚠ verifíquenlo en una fuente primaria antes de citarlo.

## 1. Mapa: tarea del enunciado → dónde está

| Tarea (Parte 2) | Dónde |
|---|---|
| 1 Variable objetivo (LGD) | `data.py::construir_lgd`, `python -m lgdmodel describir` |
| 2 Modelo de referencia | `experiments.py::ejecutar_referencias` (media, regresión lineal, logit fraccional) |
| 3 Red neuronal + búsqueda | `mlp_lgd.py`, `experiments.py::ejecutar_busqueda`, `ejecutar_ablacion` |
| 4 Evaluación (MAE, RMSE, distribución) | `metrics.py`, `experiments.py::ejecutar_final` |
| 5–6 Pérdida esperada y EAD | `experiments.py::ejecutar_perdida_esperada` |
| Comparación justa | `DatosLGD`, `presupuesto_computo.json`, `semillas_resumen.csv`, `comparacion_modelos.csv` |

## 2. Qué se hereda de la Parte 1 (y por qué)

| Elemento | Cómo se garantiza |
|---|---|
| Caso de uso: decisión **al recibir la solicitud** | mismas variables de solicitud (`pdmodel.data.app_columns`); `int_rate`/`grade` excluidas por defecto |
| Definición de incumplimiento (tabla del enunciado) | `pdmodel.data.build_target`; la LGD se define dentro de los `y = 1` |
| Partición temporal (train ≤ 2012, val 2013, test ≥ 2014) | se lee del YAML de la Parte 1 y se verifica contra su `data_summary.json` |
| Preprocesamiento | misma clase `Preprocessor`, misma configuración y mismos créditos de entrenamiento; `perdida` comprueba que las estadísticas coinciden con las de la Parte 1 |
| PD para E[L] | los modelos guardados de la Parte 1 (`lr_final.joblib`, `mlp_final.joblib`), sin reentrenar |
| Red neuronal | `pdmodel.mlp.MLPClassifier` (NumPy): la LGD solo cambia la pérdida |

**Decisión que el equipo debe defender:** los incumplidos `Late (31-120 days)` y `Default` son malos para la PD
(enunciado), pero su cobranza no ha terminado: no tienen LGD observable. En los datos reales de la Parte 1 la
partición de prueba tiene 16 754 `Charged Off`, 9 538 `Late (31-120 days)` y 1 000 `Default`: el **38.6 % de los
incumplidos de prueba no tiene LGD observable** (cálculo sobre `Parte_1/data/processed/pd_dataset.csv.gz`).
La lista de estados con pérdida definitiva es configurable (`lgd.estados_perdida_definitiva`) y `describir` produce la
evidencia para decidirla: `lgd_recuperaciones_por_estado.csv` (fracción con `recoveries > 0` por estado).

## 3. Variable objetivo

Para el crédito $i$ incumplido con pérdida definitiva:

$$\mathrm{LGD}_i=\operatorname{recortar}_{[0,1]}\Big(1-\frac{\texttt{total\_rec\_prncp}_i+\max(\texttt{recoveries}_i-\texttt{collection\_recovery\_fee}_i,\,0)}{\texttt{funded\_amnt}_i}\Big)$$

Significados citados del diccionario del conjunto (`LCDataDictionary.xlsx`, hoja *LoanStats*, en `Parte_1/data/raw/`):

| Campo | Definición literal del diccionario | Papel en la fórmula |
|---|---|---|
| `funded_amnt` | "The total amount committed to that loan at that point in time." | denominador = EAD |
| `total_rec_prncp` | "Principal received to date" | capital recuperado por pagos antes del castigo |
| `recoveries` | "post charge off gross recovery" | lo cobrado después del castigo |
| `collection_recovery_fee` | "post charge off collection fee" | costo de cobrar: se resta (recuperación **neta**) |

**Limitaciones** (para la discusión): (a) valor nominal, sin descontar flujos ni costos internos; (b) el denominador es
el monto original, no el saldo al incumplir: un crédito que pagó mucho capital antes de incumplir tiene LGD baja aunque
haya perdido todo su saldo (por eso `describir` reporta también `lgd_sobre_saldo`); (c) censura: las recuperaciones
posteriores a la fecha de corte del archivo no se ven y los castigos recientes son incumplimientos tempranos;
(d) se excluyen los intereses cobrados (`total_rec_int`).
**No asuman la forma de la distribución** (la literatura suele describir LGD bimodales, pero con esta definición puede no
serlo): mírenla en `lgd_distribucion.json` y `lgd_distribucion_observada.png`.

## 4. Activación de salida y función de pérdida (máxima verosimilitud)

La LGD es continua y acotada en $[0,1]$. Salida: $\hat y=\sigma(z)=1/(1+e^{-z})\in(0,1)$, que respeta el soporte. Una
salida lineal (como la regresión lineal) puede predecir fuera de $[0,1]$ y hay que recortarla.

**BCE como cuasi-verosimilitud Bernoulli.** Se postula solo la media: $\mathbb E[y\mid x]=\sigma(f_\theta(x))$. Se maximiza
$\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$ aunque $y_i$ sea fraccional (no es una verosimilitud exacta: $y$ no es Bernoulli).
Como la Bernoulli pertenece a la familia exponencial lineal, el estimador es consistente para la media condicional si
esta está bien especificada (Gourieroux, Monfort y Trognon, 1984 ⚠; Papke y Wooldridge, 1996 ⚠). Pérdida:
$J=-\frac1n\sum[y_i\log p_i+(1-y_i)\log(1-p_i)]$, y $\partial J/\partial z_i=(p_i-y_i)/n$ — la misma ecuación de la Parte 1.

**MSE como verosimilitud gaussiana.** $y_i\mid x_i\sim\mathcal N(\sigma(z_i),s^2)$ ⇒ $-\log L=\frac{1}{2s^2}\sum(y_i-p_i)^2+\text{cte}$.
Con $J=\frac1n\sum(p_i-y_i)^2$: $\partial J/\partial z_i=2(p_i-y_i)\,p_i(1-p_i)/n$. El factor $p(1-p)$ se anula cuando la
salida se satura, aunque el error sea grande (aprendizaje lento cerca de 0 y 1); la BCE no tiene ese factor. Además la
gaussiana supone soporte no acotado y varianza constante. **La ablación (`ablacion`) mide si esto importa en estos datos**
— no lo den por hecho.

**Regresión lineal (referencia).** Mínimos cuadrados = máxima verosimilitud gaussiana con media lineal; Ridge añade
$\alpha\|w\|^2$ (equivale a un prior gaussiano sobre $w$).
**Logit fraccional (referencia).** $\hat y=\sigma(w^\top x+b)$ con la misma BCE: es la red sin capas ocultas. Separa el
efecto del enlace logístico del efecto de las capas ocultas. Se resuelve con L-BFGS (Liu y Nocedal, 1989 ⚠) porque el
problema es convexo.

## 5. Retropropagación y optimizador

Idénticos a la Parte 1 (ver su guía, §4): solo cambia el error inicial $\delta^L=\partial J/\partial z$ de la sección 4.
Código: `mlp_lgd.py::MLPLGD._loss_grads`; `tests/test_core.py` lo verifica con diferencias finitas para ambas pérdidas y
comprueba que con BCE el entrenamiento es idéntico al de la Parte 1. Regularización (L2, dropout, parada temprana) y
Adam/SGD con momento: las mismas ecuaciones de la Parte 1.

## 6. Métricas

$\text{MAE}=\frac1n\sum|y_i-\hat y_i|$, $\text{RMSE}=\sqrt{\frac1n\sum(y_i-\hat y_i)^2}$ (penaliza más los errores grandes).
Distribución predicha frente a observada: cuantiles, masa en los extremos, KS de dos muestras
$D=\sup_t|\hat F_{\text{pred}}(t)-\hat F_{\text{obs}}(t)|$ y Wasserstein-1 $W_1=\int|\hat F_{\text{pred}}(t)-\hat F_{\text{obs}}(t)|\,dt$.
Un modelo puede tener buen RMSE y una distribución mucho más estrecha que la observada (predice cerca de la media):
por eso el enunciado pide comparar distribuciones.

## 7. Pérdida esperada y EAD

$\mathbb E[L_i]=\mathrm{PD}_i\cdot\mathrm{LGD}_i\cdot\mathrm{EAD}_i$ con $\mathrm{EAD}_i=\texttt{funded\_amnt}_i$: la decisión se toma al recibir la
solicitud, cuando el saldo al incumplir no existe; es coherente con la LGD sobre el monto desembolsado ($\mathrm{LGD}\cdot\mathrm{EAD}$
= pérdida en dólares). Limitación: sobreestima la exposición de quien amortiza antes de incumplir.

**Comparar con la pérdida realizada** solo es posible en créditos con desenlace definitivo (pagó, o castigado con LGD
medida); `perdida_esperada_resumen.json` lista los excluidos. El bloque `diagnostico` separa la PD de la LGD. Dato real
de la Parte 1: en prueba la PD media de la red es **0.194** y la tasa de malos observada **0.260** (la tasa de malos por
año pasa de 0.137 en 2009 a 0.262 en 2014). Hipótesis a contrastar, no conclusión: los créditos recientes que ya tienen
desenlace a la fecha de corte están sesgados hacia el incumplimiento temprano (los buenos siguen `Current` y se excluyen).

## 8. Preguntas probables en la sustentación

¿Por qué la LGD solo con `Charged Off`? ¿Qué pasa con E[L] si incluyo `Default` (`--set lgd.estados_perdida_definitiva=[...]`)?
¿Por qué sigmoide y no lineal en la salida? Derive $\partial J/\partial z$ para BCE y para MSE. ¿Qué predice si cambio a MSE
(`--set red.perdida=mse`)? ¿Por qué el logit fraccional es una red? ¿Por qué EAD = `funded_amnt`? ¿Qué significa
E[L]/realizada = 0.8? ¿Por qué elegir la semilla representante con validación?
*Modificaciones en vivo típicas:* `--set final.sobreescribir.hidden=[64,32]`, `--set final.n_semillas=5`,
`--set red.perdida=mse`, `--set busqueda.n_pruebas=3`. **Predigan antes de ejecutar.**

## 9. Referencias candidatas (⚠ verifiquen existencia, año, páginas y DOI antes de citar)

Papke, L. E. y Wooldridge, J. M. (1996), *Journal of Applied Econometrics* 11(6), 619–632 ·
Gourieroux, C., Monfort, A. y Trognon, A. (1984), *Econometrica* 52(3), 681–700 ·
Liu, D. C. y Nocedal, J. (1989), *Mathematical Programming* 45, 503–528 ·
Bergstra, J. y Bengio, Y. (2012), *JMLR* 13, 281–305 · más las de la Parte 1.
