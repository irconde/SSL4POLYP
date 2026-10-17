# Plan de análisis estadístico (SAP), campaña v3, versión v3.3

La versión v3.3 sustituye a la v3.2 (revisión c, SHA-256 `a387aed06f6901affca6c523a4f9ea3b1fca9b34f9e1c7ec69be3ebd36874d47`). Su único cambio es el método de inferencia de la sección 4 (puntos 2 y 3), decidido con una validación por simulación cuya regla de elección se fijó antes de ejecutarla (desviación 9.3). El cambio se hizo antes de cualquier análisis de los datos reales de la campaña. El resto del texto es el de la v3.2c.

Este plan se fija antes de lanzar las ejecuciones de la campaña v3. La versión v3.2 sustituye a la v3.1 (`plan_analisis_estadistico_v3_1.md`), que a su vez sustituyó a la v3 (`plan_analisis_estadistico_v3.md`). La v3.1 añadió la sección 10 y la desviación 9.1; la v3.2 cambia la construcción de las particiones, los subsets y los packs few-shot (sección 1), el diseño de E2 (sección 1.1), la selección de learning rate de SSL-imnet-matched (sección 10) y añade la desviación 9.2. Su versión final se guarda en el repositorio y su hash SHA-256 se registra en `RUNLOG.md` antes de la fase 1. Cualquier cambio posterior a la obtención de resultados se documenta como desviación (sección 9).

## 1. Diseño y unidades de replicación

- **Regímenes (4):** SUP-imnet, SSL-imnet, SSL-colon, SSL-imnet-matched.
- **Particiones de SUN (K = 5, mínimo aceptable 3):** cinco particiones independientes a nivel de caso, generadas con el constructor estratificado por morfología (`polypdb sun morph`, 33 frames por caso), con los mismos tamaños (70/15/15 de 100 casos positivos y 100 negativos) y un test con 5 casos planos y 10 polipoides.
  - k = 1 es el pack `sun_morphology` actual (semilla 47).
  - k = 2 a 5 usan, por orden, las 4 primeras semillas de la lista fija 101, 102, …, 120 cuyas particiones superen las aserciones del constructor. Las semillas descartadas y el motivo se registran.
  - Cada partición es el `sun_full` de esa partición: sus modelos son los padres de E2, E3 y E4.
  - Las particiones se generan de forma independiente, por lo que los tests de distintas particiones pueden compartir casos. Se declara como limitación; la variación entre particiones no procede de conjuntos de test disjuntos.
- **Subsets de E1:** `sun_subsets` anidados (5, 10, 25, 50 y 100 %) por partición, generados con `polypdb sun subsets-canonical` (semilla 13) sobre el train de cada partición; validación y test son los de la partición.
- **Semillas de entrenamiento:** 13, 29, 47 dentro de cada partición.
- **Packs few-shot:** una realización por combinación partición × semilla, con semilla de pack = 1000 × k + s (k = 1 a 5, s ∈ {13, 29, 47}); 15 packs por tamaño (50, 100, 200 y 500 frames).
- **Unidad de análisis principal:** la ejecución (partición × semilla), emparejada entre regímenes. Las diferencias entre regímenes se calculan dentro de cada ejecución emparejada.
- **Conjuntos de test de PolypGen:** fijos. La variación en E3 y E4 procede del modelo padre (partición × semilla) y, en E4, del pack.

### 1.1 E2 (morfología)

E2 no tiene entrenamiento propio: se evalúa por estrato (negativos + planos, negativos + polipoides) sobre el test de cada partición, con los modelos de `sun_full` de esa partición y su τ. H3 se mantiene sin cambios (AUROC del estrato plano, SSL-colon − SUP-imnet).

## 2. Definiciones de las métricas

- Métricas binarias de la clase positiva (pólipo) al umbral de operación τ de cada ejecución: precisión, sensibilidad, especificidad, F1, balanced accuracy.
- AUROC y average precision (AUPRC) calculados desde las probabilidades.
- AULC: integral trapezoidal sobre el eje lineal de presupuesto (5 a 100 %), normalizada por la anchura del eje.
- **Predictor degenerado:** ejecución cuya especificidad o sensibilidad en el test, al umbral de operación, es inferior a 0,10.
- **Calibración:** Brier score y error de calibración esperado (ECE) con 15 intervalos de igual masa.
- **Sensibilidad por pólipo (solo SUN):** proporción de casos positivos del test con al menos 1 frame, y con al menos 5 de sus 33 frames, por encima de τ.

## 3. Hipótesis principales

Todas son bilaterales, con α = 0,05 y corrección de Holm sobre H1 a H7.

| Id | Experimento | Contraste | Métrica | Condición |
|---|---|---|---|---|
| H1 | E1 | SSL-colon − SUP-imnet | AULC de AUPRC | Curva completa 5–100 % |
| H2 | E1 | SSL-colon − SSL-imnet-matched | AULC de AUPRC | Curva completa 5–100 % |
| H3 | E2 | SSL-colon − SUP-imnet | AUROC | Estrato plano |
| H4 | E3 | SSL-colon − SUP-imnet | AUROC | `polypgen_clean_test` |
| H5 | E3 | Negativos de secuencia − negativos aislados | AUROC, media de los 4 regímenes | Split de evaluación de 50 frames, positivos comunes |
| H6 | E4 | Brazo B (sin adaptar, τ recalibrado) − brazo A (sin adaptar, τ heredado) | Balanced accuracy, media de los 4 presupuestos y regímenes | n/a |
| H7 | E4 | Brazo C (adaptado, τ heredado) − brazo A | Balanced accuracy, media de los 4 presupuestos y regímenes | n/a |

H2 es la hipótesis de dominio. H5 a H7 sostienen el pilar del protocolo de evaluación.

## 4. Análisis principal

1. **Diferencias por ejecución.** Para cada hipótesis se calcula una diferencia d por ejecución emparejada (partición k, semilla s). En H5 a H7, d se promedia primero sobre regímenes y presupuestos dentro de cada (k, s).
2. **Modelo (v3.3).** El modelo de referencia es d_{k,s} = μ + u_k + e_{k,s}, con u_k ~ N(0, σ²_partición) y e_{k,s} ~ N(0, σ²_semilla). La inferencia sobre μ se hace siempre promediando las semillas dentro de cada partición y aplicando un t-test de una muestra (bilateral) sobre las K medias por partición, con gl = K − 1. La estimación principal es μ (la media de las K medias), con su IC del 95 % de la t y su valor p. Con todas las ejecuciones presentes (K particiones × 3 semillas) este procedimiento coincide exactamente con el ajuste REML de `lme4` no singular con grados de libertad de Satterthwaite o de Kenward–Roger.
3. **Sin regla de reserva (v3.3).** Al no depender del ajuste del modelo mixto, el procedimiento no necesita la regla de reserva de la v3.2 para ajustes singulares o que no convergen. Si faltan ejecuciones, cada partición contribuye con la media de sus semillas disponibles, con el mismo peso. Las desviaciones estándar entre particiones y entre semillas se informan de forma descriptiva (sección 7).
4. **Ajuste por multiplicidad.** Holm sobre los siete valores p.
5. **Regla de confirmación.** Una hipótesis se declara confirmada si se cumplen dos condiciones:
   - su valor p ajustado por Holm es < 0,05;
   - el IC del 95 % del bootstrap por clústeres (sección 5), calculado sobre la estimación promediada en todas las ejecuciones, excluye el cero.

   La doble condición exige que el efecto resista tanto la variabilidad del entrenamiento y de la partición como la del muestreo del test.

## 5. Bootstrap por clústeres

Se mantiene como medida de la incertidumbre del muestreo del test.

- B = 2000, semilla 12345.
- Una muestra por réplica, compartida entre semillas, particiones y regímenes cuando el conjunto de test es común (E3, E4); en SUN, una muestra por partición.
- Estratificado por clase; en E2, también por estrato.
- **Unidades:**
  - caso SUN en E1 y E2;
  - centro en E3;
  - en E4, positivos por centro y negativos por secuencia (por centro si no hay secuencia).
- **Estimación:** media de las diferencias por ejecución. **Intervalo:** percentil del 95 %.

## 6. Análisis secundarios (descriptivos, sin corrección por multiplicidad)

- Todas las comparaciones por presupuesto, estrato, métrica y comparador, con la estimación del modelo de la sección 4 y el IC bootstrap.
- E4 completo: los cuatro brazos por presupuesto, incluida la diferencia D − C (recalibración tras la adaptación).
- Número y proporción de predictores degenerados por régimen y presupuesto.
- Calibración (Brier, ECE y diagramas de fiabilidad) en sun_full, E3 y E4, por brazo.
- Sensibilidad por pólipo en E1 y E2.
- Brecha AUROC plano − polipoide por régimen y su diferencia entre regímenes.
- **Análisis de sensibilidad:**
  - leave-one-center-out en E3 y en la descomposición por origen de negativos;
  - leave-one-case-out en el estrato plano de E2, por partición;
  - H1 a H4 restringidas a la partición k = 1, para comparar con la versión anterior (con una sola partición, t-test de una muestra sobre las 3 semillas, gl = 2).
  - H6 y H7 repetidas excluyendo el presupuesto s500 (media de los presupuestos 50, 100 y 200), porque el split de evaluación de s500 no contiene negativos aislados y su prevalencia varía entre packs (revisión c).
- **AP en E4 (revisión c):** la AP de E4 solo se compara dentro de cada pack, es decir, entre brazos y entre regímenes sobre el mismo split de evaluación; nunca entre packs ni entre presupuestos, porque la prevalencia del split de evaluación cambia entre ellos.

## 7. Reglas de redacción

- Solo las hipótesis que cumplan la regla de la sección 4.5 se describen como confirmadas.
- Los resultados secundarios se describen como observados, con su estimación e intervalo, sin lenguaje de significación.
- Las estimaciones se informan como media sobre todas las ejecuciones emparejadas. La desviación estándar entre ejecuciones se informa por separado para las particiones y para las semillas.

## 8. Tamaño de la muestra

No se hizo cálculo de potencia. El número de réplicas (K particiones × 3 semillas) está limitado por el coste computacional y se declara como limitación. E3 está limitado por construcción a 6 centros.

## 9. Desviaciones

Cualquier cambio en hipótesis, métricas, modelo o reglas después de ver resultados de la campaña v3 se registra en `results_v3/SAP_deviations.md` con fecha, motivo y efecto, y se declara en el manuscrito.

### 9.1 Selección de learning rate (v3.1)

- **Fecha:** 2026-10-15.
- **Cambio:** se añade la sección 10. El learning rate del ajuste fino de cada régimen se elige con un barrido en la validación de la partición k = 1, en lugar de usar 1e-3 en todos.
- **Momento:** introducido tras la prueba de humo (fase 0) y antes de cualquier resultado de la campaña. La prueba de humo no forma parte de la campaña; ninguna partición nueva se ha generado ni ejecutado.
- **Motivo:** comprobar que el learning rate común no favorece a ningún régimen (equidad entre regímenes).
- **Efecto:** ninguno sobre hipótesis, métricas, modelo estadístico ni reglas de confirmación. Puede cambiar el learning rate de uno o más regímenes en toda la campaña.

### 9.2 Particiones, E2 y learning rate de SSL-imnet-matched (v3.2)

- **Fecha:** 2026-10-15.
- **Momento:** después del barrido de learning rate de la v3.1 y antes de generar ninguna partición nueva y de cualquier resultado de la campaña.
- **Cambios:**
  1. SSL-imnet-matched pasa por el mismo barrido de learning rate que los demás regímenes, con la misma regla; ya no hereda el de SSL-imnet (sección 10).
  2. Particiones: K = 5 particiones independientes con el constructor estratificado por morfología; k = 1 es el `sun_morphology` actual (semilla 47) y k = 2 a 5 usan las 4 primeras semillas de 101–120 que superen las aserciones. Los tests de distintas particiones pueden solaparse (sección 1).
  3. E2 se evalúa por estrato con los modelos de `sun_full` de cada partición y deja de tener entrenamiento propio. H3 se mantiene (sección 1.1).
  4. `sun_subsets` anidados por partición (`subsets-canonical`, semilla 13).
  5. Packs few-shot con semilla 1000 × k + s (15 packs por tamaño).
- **Desviación declarada:** el barrido de learning rate de la v3.1 se hizo sobre la validación del `sun_full` de v2 (semilla 47 del constructor no estratificado), que comparte casos con la nueva partición k = 1 (`sun_morphology`). La selección no usó ningún test, pero los casos de validación del barrido pueden aparecer en el train, la validación o el test de la nueva k = 1.
- **Efecto:** ninguno sobre las hipótesis H1 a H7, las métricas, el modelo estadístico ni las reglas de confirmación. Cambia el origen de las particiones y el diseño de E2.

## 10. Selección de learning rate

Se fija antes de lanzar el barrido y antes de la campaña.

- **Barrido:** `sun_full` de v2 (la partición k = 1 de la v3.1; ver desviación 9.2), regímenes SUP-imnet, SSL-imnet y SSL-colon; learning rates del ajuste fino 1e-4, 3e-4 y 1e-3 (el actual); semillas 13, 29 y 47 (27 ejecuciones). Todo lo demás es idéntico a la configuración v3 (`config/base.yaml`: AdamW, cosine con 5 épocas de warmup, parada temprana por pérdida de validación con paciencia 8).
- **Solo validación:** la selección usa únicamente la validación de ese `sun_full`. El test no se evalúa en ninguna ejecución del barrido (`--skip-test-eval`).
- **Métrica de selección:** AP (average precision desde las probabilidades) de la validación con los pesos del mejor epoch (mínima pérdida de validación), media sobre las 3 semillas.
- **Regla de decisión** (se aplica tal cual):
  1. Para cada régimen, el learning rate elegido es el de mayor AP media de validación.
  2. Si para los tres regímenes gana 1e-3, se mantiene el learning rate común y se documenta.
  3. Si no, cada régimen usa su learning rate elegido en toda la campaña v3.
  4. **SSL-imnet-matched (v3.2):** una vez preentrenado, pasa por el mismo barrido (1e-4, 3e-4 y 1e-3; semillas 13, 29 y 47; misma configuración y mismos datos que los otros tres regímenes; AP de validación; sin evaluar el test) y su learning rate es el de mayor AP media de validación. Sustituye a la regla de la v3.1, que le asignaba el learning rate de SSL-imnet.
- **Alcance:** el learning rate seleccionado es el del ajuste fino completo (parámetro `lr`), usado en sun_full y E1. E2 y E3 usan los modelos de sun_full sin entrenamiento adicional, de modo que heredan su learning rate; E4 adapta esos modelos con su propio calendario por etapas (learning rates por etapa en `config/exp/exp5c/budgets/`), que el barrido no modifica.
- **Registro:** resultados del barrido, learning rate elegido por régimen y diagnósticos (pérdida de la primera época, mejor epoch, paradas tempranas) en `results_v3/RUNLOG.md`.

### 9.3 Método de inferencia de la sección 4 (v3.3)

- **Fecha:** 2026-10-17.
- **Momento:** después de terminar la campaña de los tres regímenes y el preentrenamiento de SSL-imnet-matched, y **antes de cualquier análisis de los datos reales de la campaña**. Ningún resultado real se había calculado ni visto: los ficheros reales solo se habían leído para conocer sus nombres y columnas.
- **Cambio:** la inferencia sobre μ pasa a ser siempre el t-test de una muestra sobre las K medias por partición (gl = K − 1), en lugar de `lmer` con grados de libertad de Satterthwaite y regla de reserva (sección 4, puntos 2 y 3).
- **Motivo:** validación por simulación del análisis (`results_v3/verification/analysis_v3_variants/`) con campañas simuladas de la misma estructura que la real (4 regímenes, 5 particiones, 3 semillas): 200 repeticiones sin efecto y 50 con efectos conocidos, comparando tres variantes sobre los mismos datos: V1 (v3.2c), V2 (`lmer` con Kenward–Roger y la misma regla de reserva) y V3 (t-test sobre las medias por partición). La regla de elección se fijó y se registró (commit `0b6dd67`) antes de ejecutar la validación: se mantiene V1 si su tasa familiar de confirmaciones falsas tiene un límite superior del IC exacto del 95 % ≤ 0,08 y su cobertura del IC del 95 % es al menos 0,93 en todas las hipótesis; si no, la variante de mayor potencia media entre V2 y V3 que cumpla ambas condiciones; si ninguna las cumple, V3.
- **Resultado:** las tres variantes dieron las mismas decisiones en las 1.750 pruebas (250 repeticiones × 7 hipótesis); tasa familiar 7/200 = 0,035 (IC exacto 0,014–0,071), cobertura mínima 0,92 (H4 y H5 con efectos, 46/50), potencia media 1,00. Ninguna variante cumple la condición de cobertura y, por la regla, se elige V3.
- **Efecto:** ninguno sobre las hipótesis H1 a H7, las métricas, Holm, el bootstrap ni la regla de confirmación doble. Con la campaña completa (todas las ejecuciones presentes) las estimaciones, los IC y los valores p son los mismos que con el procedimiento de la v3.2c; el cambio solo afecta al caso de ejecuciones ausentes y elimina la dependencia de la comprobación de singularidad. Los análisis secundarios que usan el modelo de la sección 4 usan el mismo procedimiento.
