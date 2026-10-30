# Guía conceptual — Parte 1 (PD)

Úsenla para **entender**, no para copiar: en la sustentación les preguntan a cualquier integrante, y la IA (incluido quien escribió esto) puede equivocarse. Todo lo marcado con ⚠ **verifíquenlo** en una fuente primaria.

## 1. Mapa: tarea del enunciado → dónde está
| Tarea | Dónde |
|---|---|
| 1 Caso de uso | §2 de esta guía (lo redactan ustedes) |
| 2 Descriptivo + hipótesis | `python -m pdmodel describe` → `descriptive_bad_rate.csv/png` |
| 3 Regresión logística | `experiments.run_baseline` |
| 4 Red neuronal + búsqueda | `mlp.py`, `experiments.run_search` (búsqueda aleatoria; trial 0 = config de referencia) |
| 5 Evaluación | `metrics.py`, `experiments.run_final` (30 semillas), figuras ROC/KS/calibración |
| 6 Scorecard | `scorecard.py` (§5) |
| 7 Variables de riesgo | `interpret.py` (permutación) + curvas de puntos |

## 2. Caso de uso (borrador a adaptar)
Un analista de originación de un banco de consumo usa el modelo **en el momento en que llega la solicitud**, con la información del formulario y del reporte de buró (monto, plazo, ingreso, DTI, historial de moras, consultas, utilización, etc.) para decidir aprobar/rechazar/fijar precio. Consecuencias que deben defender:
- **Fuga de información:** `total_pymnt`, `recoveries`, `out_prncp`, `last_pymnt_d`, etc. ocurren *después*; usarlas daría AUC irreal. Se excluyen.
- **`int_rate`/`grade`:** las asigna Lending Club con su propio modelo de riesgo; un banco que construye *su* scorecard no las tendría. Por defecto se excluyen (configurable). ¿Qué pasaría con ellas? Es un buen experimento para el reporte.
- **Censura:** `Current` se excluye, pero eso sesga: los créditos recientes que ya terminaron (sobre todo los de 36 meses) no representan a los de 60. Documenten la limitación y el efecto en la tasa de malos por año.
- **Partición temporal:** entrenar con lo antiguo y evaluar con lo nuevo imita el uso real; la deriva (cambio de política, de economía) hace que el desempeño en test sea **menor** y que la calibración se deteriore: es un resultado, no un error.

## 3. Hipótesis sugeridas (confírmenlas o refútenlas con datos)
Plazo de 60 meses más riesgoso; DTI alto; alta utilización revolvente; ingreso bajo; más consultas recientes; moras o registros públicos previos; historia crediticia corta; propósito `small_business`; antigüedad laboral. Algo contraintuitivo que suele aparecer con *verification_status* ⚠ (verifíquenlo en sus datos, no lo asuman). Escriban las hipótesis **antes** de mirar el modelo.

## 4. Matemática que el reporte exige
**Neurona y MLP.** $z=\mathbf w^\top\mathbf x+b,\; h=g(z)$. MLP: $\mathbf a^0=\mathbf x$, $\mathbf z^l=\mathbf a^{l-1}W^l+\mathbf b^l$, $\mathbf h^l=g(\mathbf z^l)$, salida $\hat p=\sigma(\mathbf a^{L-1}\mathbf w+b)$.
Activaciones: ReLU $\max(0,z)$; Leaky-ReLU $z$ si $z>0$, $0.01z$ si no; tanh; sigmoide $\sigma(z)=1/(1+e^{-z})$ en la salida.

**Pérdida por máxima verosimilitud.** Supongan $y_i\mid\mathbf x_i\sim\text{Bernoulli}(p_i)$, $p_i=\sigma(f_\theta(\mathbf x_i))$. Verosimilitud $\prod p_i^{y_i}(1-p_i)^{1-y_i}$; minimizar $-\frac1n\log$ da la entropía cruzada binaria
$J=-\frac1n\sum[y_i\log p_i+(1-y_i)\log(1-p_i)]$. Con logit $z$: $\max(z,0)-zy+\log(1+e^{-|z|})$ (forma estable usada en el código). Resultado clave: $\partial J/\partial z_i=(p_i-y_i)/n$ — demuéstrenlo.

**Retropropagación** (lote de $n$, pérdida media): $\delta^{L}=(\hat p-y)/n$; $\nabla_{W^L}=(\mathbf a^{L-1})^\top\delta^L$; $\mathbf{da}^{l}=\delta^{l+1}(W^{l+1})^\top$; $\delta^l=\mathbf{da}^l\odot g'(\mathbf z^l)$ (con dropout: $\mathbf{da}^l\leftarrow\mathbf{da}^l\odot\mathbf m^l$); $\nabla_{W^l}=(\mathbf a^{l-1})^\top\delta^l$, $\nabla_{b^l}=\sum\delta^l$. En `mlp.py::_loss_grads`; `tests/test_core.py` lo verifica con diferencias finitas.

**Optimizadores (tal como están implementados).** SGD con momento: $v\leftarrow\mu v+g$, $\theta\leftarrow\theta-\eta v$. Adam: $m\leftarrow\beta_1m+(1-\beta_1)g$, $v\leftarrow\beta_2v+(1-\beta_2)g^2$, $\hat m=m/(1-\beta_1^t)$, $\hat v=v/(1-\beta_2^t)$, $\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ con $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$.

**Regularización.** *L2*: $J_{\text{tot}}=J+\frac\lambda2\sum_l\|W^l\|_F^2$ (sin sesgos) → gradiente $+\lambda W$. Ojo: con Adam, L2 en el gradiente **no** equivale a *weight decay* desacoplado (AdamW) ⚠ (Loshchilov & Hutter, 2019). *Dropout* invertido: en entrenamiento $\tilde h=h\odot m/(1-q)$, $m_j\sim\text{Bernoulli}(1-q)$; en inferencia no se aplica nada. *Parada temprana*: se detiene tras `patience` épocas sin mejorar la log-loss de validación y se restauran los mejores pesos.

**Métricas** (positivo = malo). AUC $=P(\hat p_{\text{malo}}>\hat p_{\text{bueno}})+\tfrac12P(=)$ (Mann–Whitney). Gini $=2\,\text{AUC}-1$. KS $=\max_t|F_{\text{malos}}(t)-F_{\text{buenos}}(t)|$. Brier $=\frac1n\sum(\hat p_i-y_i)^2$. Calibración: bins por cuantiles de $\hat p$, comparar media predicha vs tasa observada (el código también da ECE y log-loss).

## 5. Scorecard (derivación)
Odds de ser bueno $=(1-p)/p=e^{-\text{logit}(p)}$. Se exige que el puntaje sea lineal en $\ln(\text{odds})$, que valga $S_0$ cuando las odds valen $o_0$ y que sume PDO puntos al duplicar las odds:
$S=A+B\ln(\text{odds})$, $B\ln2=\text{PDO}\Rightarrow B=\text{PDO}/\ln2$, $A=S_0-B\ln o_0$ ⇒ $S=A-B\,\text{logit}(p)$.
Con 600 / 50:1 / 20: $B=28.85$, $A=487.12$.
**Puntos por variable.** Un MLP no es aditivo, así que se usa el *efecto parcial en el logit*: $\text{pts}_j(v)=-B\,(\mathbb E[\text{logit}\,f(x_j{=}v,x_{-j})]-\mathbb E[\text{logit}\,f(x)])$ y $S\approx S_{\text{base}}+\sum_j\text{pts}_j(x_j)$. Para la regresión logística coincide con $-B\beta_jx_j$ (centrado). Se reporta la **fidelidad** (correlación y error en puntos frente al puntaje exacto) en `scorecard_summary.json`. Limitación: promediar rompe la correlación entre variables (Friedman, 2001; ⚠ revísenlo) — y las interacciones quedan fuera de la aproximación aditiva.

## 6. Interpretabilidad (Tarea 7)
Implementado: importancia por permutación sobre las variables de solicitud (caída del AUC en test) y curvas de puntos (dependencia parcial). Limitaciones que deben discutir: variables correlacionadas, permutar crea combinaciones irreales. Pueden añadir SHAP (Lundberg & Lee, 2017 ⚠) si lo justifican.

## 7. Comparación justa (lo que más se evalúa)
Mismo preprocesamiento y particiones (`Data` en `experiments.py`); presupuesto explícito en `compute_budget.json` (n.º de configuraciones y segundos: LR ≈ 7, MLP = `n_trials`); 30 semillas con media/desv./mejor/peor; parámetros entrenables (LR: n_features+1). Con datos tabulares es habitual que la red **no** gane por mucho; reportar eso con evidencia es un buen resultado ⚠ (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022).

## 8. Cacería de la alucinación — cómo usarla bien
No inventen hallazgos (se penalizan más que un faltante). Hábitos que generan hallazgos reales: pedirle a la IA derivar la pérdida/scorecard y comparar con §4–5; revisar signos y definiciones (¿PD de malo o de bueno?, ¿PDO suma o resta?); recalcular una métrica a mano o con sklearn; pedir referencias y comprobar cada DOI/autor/año; preguntar el significado de variables y contrastarlo con el diccionario de Kaggle; mirar si el código ajusta el preprocesamiento con todos los datos. Para cada uno guarden: prompt literal, respuesta, sospecha, evidencia, corrección, lección.

## 9. Preguntas probables en la sustentación
¿Por qué excluyeron `Current`? ¿Qué pasa si quito la partición temporal? ¿Por qué BCE y no MSE? ¿Qué predice esta rampa de PDO si lo cambio a 40? ¿Por qué parar temprano con la validación y no con test? ¿Por qué la calibración empeora en test? ¿Qué hace dropout en inferencia? ¿Por qué no usaron pesos de clase? ¿La scorecard de la red es exacta? ¿Qué variable permutarías y qué esperas que pase con el AUC?
*Modificaciones en vivo típicas:* `--set scorecard.pdo=40`, `--set final.n_seeds=5`, `--set mlp_search.n_trials=3`, cambiar fechas de corte, `--set data.use_lc_pricing=true`. **Predigan antes de ejecutar.**

## 10. Referencias candidatas (⚠ verifiquen existencia, año, páginas y DOI antes de citar)
Rumelhart, Hinton & Williams (1986), *Nature* 323, 533–536 · Kingma & Ba (2015), Adam, ICLR · Srivastava et al. (2014), Dropout, *JMLR* 15, 1929–1958 · Bergstra & Bengio (2012), búsqueda aleatoria, *JMLR* 13, 281–305 · Breiman (2001), *Machine Learning* 45, 5–32 · Fisher, Rudin & Dominici (2019), *JMLR* 20(177) · Friedman (2001), *Annals of Statistics* 29(5) · Brier (1950), *Monthly Weather Review* 78(1) · Siddiqi (2006), *Credit Risk Scorecards*, Wiley · Grinsztajn, Oyallon & Varoquaux (2022), NeurIPS Datasets & Benchmarks · Shwartz-Ziv & Armon (2022), *Information Fusion* 81 · Loshchilov & Hutter (2019), ICLR · Lundberg & Lee (2017), NeurIPS.
