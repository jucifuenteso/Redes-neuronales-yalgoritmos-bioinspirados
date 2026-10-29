# Parte 2 — Fundamentos para el reporte (LGD y pérdida esperada)

Borrador de las secciones matemáticas y metodológicas de la Parte 2. Todo lo marcado **[verificar]** debe
contrastarse con la fuente primaria antes de entrar al reporte (la guía penaliza referencias o definiciones inexactas).

## 1. Variable objetivo: LGD

Para los créditos incumplidos con pérdida **definitiva** (`loan_status` ∈ {Charged Off, Default, Does not meet the credit
policy. Status:Charged Off}):

$$\mathrm{LGD}_i=\operatorname{clip}_{[0,1]}\!\left(1-\frac{\texttt{total\_rec\_prncp}_i+\max\{\texttt{recoveries}_i-\texttt{collection\_recovery\_fee}_i,\,0\}}{\texttt{funded\_amnt}_i}\right)$$

Implementada en `credit/data.py::build_lgd`.

| Campo | Significado (diccionario `LCDataDictionary.xlsx`, **[verificar texto literal]**) | Por qué se usa |
|---|---|---|
| `funded_amnt` | monto comprometido para el crédito | denominador: la exposición al originar (EAD) |
| `total_rec_prncp` | capital recibido a la fecha | lo ya recuperado vía pagos regulares antes de castigar |
| `recoveries` | recuperación bruta posterior al castigo | lo recuperado por cobranza tras el `Charged Off` |
| `collection_recovery_fee` | comisión de cobranza posterior al castigo | costo de recuperar: se resta para tener la recuperación **neta** |

Se **excluyen** `Late (31-120 days)` (es "malo" en la PD, pero su recuperación aún no ocurrió: la LGD no es observable)
y los intereses recibidos (`total_rec_int`): la LGD económica puede incluirlos, aquí se define sobre el capital.

### Limitaciones (para la discusión)

1. **Definición contable, no económica.** No se descuentan los flujos al costo de capital ni se incluyen costos
   internos; la recuperación está en valor nominal.
2. **Denominador = monto original, no saldo al incumplir.** Un crédito que pagó el 80 % y luego incumplió tiene LGD
   baja aunque sobre su saldo al incumplir haya perdido todo. Es coherente con `EAD = funded_amnt` (ver §5) pero no con la
   definición regulatoria de LGD sobre el saldo en el incumplimiento.
3. **Censura.** Solo créditos ya castigados tienen LGD; las recuperaciones posteriores a la fecha de corte del archivo no
   se ven. Los castigados recientes están sesgados hacia incumplimientos tempranos. Por eso la partición temporal
   (§4) mide el efecto real de este sesgo.
4. **Recorte a [0,1].** `recoveries` puede exceder la pérdida (reversiones, pagos posteriores), lo que daría LGD < 0.
5. **Distribución en forma de U / masa en los extremos.** Una parte de las LGD queda en ≈1 (nada recuperado) y otra
   cerca de 0; lo reportan `lgd_descriptive_summary.json` y `fig_lgd_descriptive.png`.

## 2. Neurona, MLP y activaciones

Neurona: $a=\phi(\mathbf w^\top\mathbf x+b)$. MLP de $L$ capas ocultas: $\mathbf h^{(0)}=\mathbf x$,
$\mathbf h^{(l)}=\phi_l(W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)})$, salida $z=\mathbf w_o^\top\mathbf h^{(L)}+b_o$ y
$\hat y=\sigma(z)$.

| Activación | Definición |
|---|---|
| ReLU | $\max(0,x)$ |
| tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ |
| ELU | $x$ si $x>0$; $e^x-1$ si $x\le 0$ |
| SiLU | $x\,\sigma(x)$ |
| sigmoide (salida) | $\sigma(z)=1/(1+e^{-z})\in(0,1)$ |

## 3. Función de pérdida y activación de salida

**Naturaleza del objetivo:** $y\in[0,1]$ continuo, acotado, con masa en los extremos y no gaussiano.

- **Salida sigmoide**: garantiza $\hat y\in(0,1)$, igual que el soporte de $y$. Una salida lineal produciría
  predicciones fuera de [0,1] (el modelo lineal de referencia necesita `clip`).
- **BCE como cuasi-verosimilitud de Bernoulli.** Se postula $\mathbb E[y\mid x]=p(x)=\sigma(z(x))$ y se maximiza
  $\ell(p;y)=y\log p+(1-y)\log(1-p)$, que acepta $y\in[0,1]$ (no solo $\{0,1\}$). La pérdida es
  $\mathcal L=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$. Como la densidad Bernoulli pertenece a la familia
  exponencial lineal, el estimador de cuasi-máxima verosimilitud es consistente para la media condicional si esta está bien
  especificada, aunque $y$ no sea Bernoulli (Gourieroux, Monfort & Trognon, 1984; Papke & Wooldridge, 1996) **[verificar]**.
  Es la derivación a presentar: *no* es una verosimilitud exacta, es cuasi-verosimilitud.
- **Gradiente.** Con $p=\sigma(z)$: $\partial\mathcal L/\partial z=p-y$ (BCE). Con MSE $\tfrac12(p-y)^2$:
  $\partial/\partial z=(p-y)\,p(1-p)$, que se anula cuando $p\to0$ o $1$ aunque el error sea grande: justo donde está la
  masa de la LGD. Esa es la razón de preferir BCE; `python run.py lgd-loss` lo contrasta empíricamente.
- **MSE** proviene de $y\mid x\sim\mathcal N(p(x),\sigma^2)$: $-\log\mathcal L\propto\sum(y-p)^2/(2\sigma^2)$. Es la
  opción "gaussiana" de comparación; asume soporte no acotado y varianza constante, que no se cumplen aquí.

## 4. Evaluación

- MAE $=\frac1n\sum|y_i-\hat y_i|$, RMSE $=\sqrt{\frac1n\sum(y_i-\hat y_i)^2}$.
- Referencia trivial: predecir la media de entrenamiento (`baseline_constante_media`). Un modelo que no la supera
  no aprendió nada útil; con LGD es habitual que la ganancia sea pequeña.
- Distribución: cuantiles, masa en {≤0.01, ≥0.99}, KS de dos muestras y distancia de Wasserstein-1 entre la LGD
  observada y la predicha (`distribution_test`). Un regresor a la media produce una distribución estrecha; que el MLP
  reproduzca la forma completa (U) es mucho más difícil que reducir el RMSE.
- Mismas particiones que la Parte 1: temporal por `issue_d` (train ≤ 2012-12, val ≤ 2013-06, test posterior).

## 5. Pérdida esperada y EAD

$$\mathbb E[L_i]=\mathrm{PD}_i\cdot\mathrm{LGD}_i\cdot\mathrm{EAD}_i,\qquad \mathrm{EAD}_i=\texttt{funded\_amnt}_i .$$

Justificación: la decisión de otorgar el crédito se toma al originar, cuando el saldo al incumplir aún no existe;
`funded_amnt` es la exposición máxima y es coherente con la LGD definida sobre `funded_amnt`, de modo que $\mathrm{LGD}\cdot\mathrm{EAD}$ es
exactamente la pérdida en dólares. Limitación: sobreestima la exposición de créditos que amortizan antes de incumplir.

**Comparación contra la pérdida realizada (sesgo a declarar):** solo es observable en créditos con desenlace definitivo.
En la ventana de prueba los créditos al día (`Current`) y los `Late (31-120 days)` quedan fuera, por lo que la tasa de
incumplimiento observable está inflada y $\sum\mathbb E[L]/\sum L_{real}$ tiende a quedar **por debajo** de 1 sin que el
modelo esté mal. Léase como cota, no como medida de acierto.

## 6. Búsqueda e hiperparámetros

Búsqueda aleatoria (Bergstra & Bengio, 2012, *JMLR* 13, 281–305) **[verificar]** con 30 pruebas, misma semilla de
entrenamiento, criterio RMSE de validación. Referencia lineal: rejilla de 5 valores de $\alpha$ en Ridge.
Ambos presupuestos y tiempos quedan en `outputs/lgd_summary.json` (`configs_trained*`, `seconds*`).
Parada temprana sobre la pérdida de validación con restauración de los mejores pesos. Penalización L2 vía `weight_decay`.

Para Adam, `weight_decay` suma $\lambda\mathbf w$ al gradiente (L2 clásica, que Adam reescala); AdamW lo desacopla
(Loshchilov & Hutter, 2019, *ICLR*) **[verificar]**. Que ambos no son equivalentes es un buen punto para la sustentación.
