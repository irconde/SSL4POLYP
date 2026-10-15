# Plan de análisis estadístico (SAP), campaña v3, versión v3.1

Este plan se fija antes de lanzar las ejecuciones de la campaña v3. La versión v3.1 sustituye a la v3 (`plan_analisis_estadistico_v3.md`) y solo añade la sección 10 y la desviación 9.1. Su versión final se guarda en el repositorio y su hash SHA-256 se registra en `RUNLOG.md` antes de la fase 1. Cualquier cambio posterior a la obtención de resultados se documenta como desviación (sección 9).

## 1. Diseño y unidades de replicación

- **Regímenes (4):** SUP-imnet, SSL-imnet, SSL-colon, SSL-imnet-matched.
- **Particiones de SUN (K = 5, mínimo aceptable 3):** la partición k = 1 es la actual; k = 2 a 5 son particiones nuevas a nivel de caso con los mismos tamaños (70/15/15 de 100 casos positivos y 100 negativos). Cada partición tiene su propio `sun_morphology` equivalente (test con 5 casos planos y 10 polipoides) y sus propios `sun_subsets` anidados.
- **Semillas de entrenamiento:** 13, 29, 47 dentro de cada partición.
- **Packs few-shot:** una realización por combinación partición × semilla.
- **Unidad de análisis principal:** la ejecución (partición × semilla), emparejada entre regímenes. Las diferencias entre regímenes se calculan dentro de cada ejecución emparejada.
- **Conjuntos de test de PolypGen:** fijos. La variación en E3 y E4 procede del modelo padre (partición × semilla) y, en E4, del pack.

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
2. **Modelo.** d_{k,s} = μ + u_k + e_{k,s}, con u_k ~ N(0, σ²_partición) y e_{k,s} ~ N(0, σ²_semilla). Se ajusta por REML con `lme4` (R) y se infiere con `lmerTest`, grados de libertad de Satterthwaite. La estimación principal es μ, con su IC del 95 % y su valor p.
3. **Regla de reserva.** Si el ajuste es singular (σ²_partición estimada en 0) o no converge, se promedian las semillas dentro de cada partición y se aplica un t-test de una muestra sobre las K diferencias (gl = K − 1).
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
  - H1 a H4 restringidas a la partición k = 1, para comparar con la versión anterior.

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

## 10. Selección de learning rate

Se fija antes de lanzar el barrido y antes de la campaña.

- **Barrido:** sun_full de la partición k = 1, regímenes SUP-imnet, SSL-imnet y SSL-colon; learning rates del ajuste fino 1e-4, 3e-4 y 1e-3 (el actual); semillas 13, 29 y 47 (27 ejecuciones). Todo lo demás es idéntico a la configuración v3 (`config/base.yaml`: AdamW, cosine con 5 épocas de warmup, parada temprana por pérdida de validación con paciencia 8).
- **Solo validación:** la selección usa únicamente la validación de sun_full (k = 1). El test no se evalúa en ninguna ejecución del barrido (`--skip-test-eval`).
- **Métrica de selección:** AP (average precision desde las probabilidades) de la validación con los pesos del mejor epoch (mínima pérdida de validación), media sobre las 3 semillas.
- **Regla de decisión** (se aplica tal cual):
  1. Para cada régimen, el learning rate elegido es el de mayor AP media de validación.
  2. Si para los tres regímenes gana 1e-3, se mantiene el learning rate común y se documenta.
  3. Si no, cada régimen usa su learning rate elegido en toda la campaña v3, y el régimen SSL-imnet-matched usa el de SSL-imnet.
- **Alcance:** el learning rate seleccionado es el del ajuste fino completo (parámetro `lr`), usado en sun_full, E1 y E2. El programa de adaptación de E4 (learning rates por etapa en `config/exp/exp5c/budgets/`) no se modifica. E3 no entrena.
- **Registro:** resultados del barrido, learning rate elegido por régimen y diagnósticos (pérdida de la primera época, mejor epoch, paradas tempranas) en `results_v3/RUNLOG.md`.
