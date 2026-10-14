# Especificación de reentrenamiento v3.1 (SSL4POLYP, CMPB)

Objetivo: regenerar todos los resultados del artículo en una única versión de código, en la estación con 2 × RTX A6000, corrigiendo los defectos detectados en `analysis_v2` y añadiendo los análisis que necesita el manuscrito. Ningún resultado de la versión anterior (A40) debe mezclarse con los nuevos.

El análisis de resultados sigue `plan_analisis_estadistico_v3.md` (SAP). El SAP se guarda en el repositorio y su SHA-256 se registra en `RUNLOG.md` antes de la fase 1.

Cambios respecto a v3: particiones de SUN remuestreadas obligatorias (secciones 1.9 y 3.8), E3 y E4 sobre los padres de cada partición, corrección del brazo D de E4, métricas de calibración y sensibilidad por pólipo, y calendario y coste revisados.

## 0. Principios

1. Una sola versión de código para toda la campaña: crear la etiqueta git `v3-cmpb` antes de lanzar nada y registrar el hash del commit en cada ejecución.
2. Entorno fijado y registrado: Python, PyTorch (1.9.0+cu111 o superior, ver 1.8), CUDA, timm. Guardar `pip freeze` y `nvidia-smi` en `results_v3/environment/`.
3. Semillas de entrenamiento 13, 29 y 47 en todos los experimentos y en todas las particiones.
4. Una ejecución por GPU (`CUDA_VISIBLE_DEVICES`), sin entrenamiento distribuido, para que cada ejecución sea independiente y reproducible.
5. Salidas nuevas en `results_v3/` y `checkpoints_v3/`. No se borra ni se sobrescribe nada de la versión anterior.
6. Todo el análisis estadístico se hace con el módulo de `analysis_v2` (`metrics_v2.py`, `bootstrap_v2.py`), promovido a módulo del paquete.

## 1. Cambios de código obligatorios

1.1 **Evaluar con el mejor checkpoint.** Antes de evaluar el test en cualquier experimento, recargar el checkpoint del mejor epoch de validación (el código de `9e84437` ya lo hace; verificarlo). Añadir una aserción: el epoch del umbral (`thresholds.primary.epoch`) debe ser igual al epoch de los pesos evaluados; si no, la ejecución falla.

1.2 **Conservar los checkpoints.** Desactivar `clean_checkpoints.py` para esta campaña. Guardar por ejecución el checkpoint del mejor epoch y su SHA-256 en el JSON de la ejecución.

1.3 **Guardar puntuaciones por frame de todos los splits.** Para cada ejecución, escribir `*_val_outputs.csv` y `*_test_outputs.csv`; en E4, además, las puntuaciones del modelo sin adaptar y del adaptado sobre los frames de soporte (train y val del pack) y sobre el split de evaluación. Columnas mínimas: `frame_id, prob, label, pred, case_id, sequence_id, center_id, origin, morphology, negative_source` (`negative_source` ∈ {isolated, sequence, NA}), con la misma semántica en todos los experimentos.

1.4 **Métricas.** Un único módulo (el de `analysis_v2`) para todos los informes y para los JSON: métricas binarias de la clase positiva, average precision desde las probabilidades, precisión, sensibilidad, especificidad, F1, balanced accuracy y tp/fp/tn/fn. Eliminar los campos macro del JSON o renombrarlos con el sufijo `_macro`.

1.5 **Umbral F1-óptimo exacto.** Buscar sobre todas las puntuaciones únicas de validación, sin el submuestreo a 200 candidatos. Mantener el desempate actual (mayor recall, luego menor τ).

1.6 **Bootstrap.** Usar `bootstrap_v2.py` en todos los informes: B = 2000, semilla 12345, una muestra por réplica compartida entre semillas y modelos, estratificada por clase (E2 además por estrato), unidades: caso SUN en E1/E2, centro en E3, positivos por centro y negativos por secuencia o centro en E4. Estimación = media de las diferencias por semilla; IC percentil 95 %.

1.7 **Packs few-shot por partición y semilla.** Generar con la herramienta de `polyp-db-analysis` una realización de `polypgen_fewshot_s{50,100,200,500}` por combinación de partición (k = 1 a 5) y semilla (13, 29, 47), con la misma estrategia actual (positivos por frame, negativos "pool primero y luego secuencia", secuencias enteras a un lado). Guardar manifiestos y hashes. Cada ejecución de E4 usa el pack de su partición y semilla.

1.8 **Carga de checkpoints.** Llamar a `torch.load` con `weights_only=False` explícito donde se carguen checkpoints con objetos Python, para que el código funcione también con PyTorch ≥ 2.6.

1.9 **Particiones de SUN.** Generar 4 particiones nuevas de `sun_full` (k = 2 a 5) a nivel de caso, con los mismos tamaños que la actual (k = 1): 100 casos positivos y 100 negativos con reparto 70/15/15, y casos virtuales de 33 frames para los vídeos negativos largos. Para cada partición, generar `sun_subsets` anidados (5 a 100 %, mismo procedimiento que el actual) y un `sun_morphology` equivalente (test con 5 casos planos y 10 polipoides, estratificado por morfología). Semilla de generación fija por partición; guardar manifiestos y hashes. La partición k = 1 debe reproducir exactamente los ficheros actuales (comprobar hashes).

1.10 **Métricas adicionales en el módulo de análisis.** Brier score, ECE con 15 intervalos de igual masa, diagramas de fiabilidad y sensibilidad por pólipo (al menos 1 y al menos 5 de 33 frames por encima de τ), según el SAP. Se calculan a partir de los CSV; no requieren cambios en el entrenamiento.

## 2. Regímenes

| Régimen | Inicialización | Notas |
|---|---|---|
| SUP-imnet | AugReg ViT-B/16 (ImageNet-21k, ajuste en ImageNet-1k) | Igual que ahora |
| SSL-imnet | `mae_imagenet.pth` (MAE oficial de ImageNet-1k) | Documentar origen y hash |
| SSL-colon | `mae_hyperkvasir.pth` | Reutilizar, no repreentrenar |
| SSL-imnet-matched | MAE nuevo sobre 99.417 imágenes de ImageNet-1k | Ver sección 4 |

## 3. Experimentos y análisis

3.1 **sun_full** (línea base y padre de E3/E4): 4 regímenes × 5 particiones × 3 semillas.

3.2 **E1** (`sun_subsets`, 5 %, 10 %, 25 %, 50 %, 100 %): 4 regímenes × 5 presupuestos × 5 particiones × 3 semillas.

3.3 **E2** (`sun_morphology` de cada partición): 4 × 5 × 3.

3.4 **E3** (zero-shot sobre `polypgen_clean_test`, modelo y τ del padre sun_full de cada partición y semilla): solo evaluación, 4 × 5 × 3.

3.5 **E4** (few-shot, 50/100/200/500, pack de su partición y semilla, padre sun_full de la misma partición y semilla): 4 × 4 × 5 × 3, con cuatro brazos evaluados sobre el mismo split de evaluación:
   - A. Sin adaptar, τ heredado (referencia emparejada actual).
   - B. Sin adaptar, τ recalibrado: F1-óptimo sobre las puntuaciones de los frames de soporte (train + val del pack).
   - C. Adaptado, τ heredado (resultado actual de E4).
   - D. Adaptado, τ recalibrado con la misma regla pero **solo sobre la parte de validación del soporte** (4, 14, 40 y 100 frames), porque los frames de entrenamiento del soporte se usaron para adaptar el modelo. Si la validación contiene una sola clase, se conserva τ heredado y se registra.
   El brazo B es el experimento de recalibración que el Discussion presenta como hipótesis. No requiere entrenamiento adicional.

3.6 **Descomposición por origen de los negativos**: implementarla como análisis del repositorio, sobre las salidas zero-shot del split de evaluación de 50 frames (positivos comunes con `polypgen_clean_test`, negativos aislados frente a negativos de secuencia). Verificar recuentos.

3.7 **Sensibilidad**: E3 leave-one-center-out (también sobre la descomposición) y E2 leave-one-case-out en el estrato plano, con el código de `analysis_v2`.

3.8 **Particiones remuestreadas (obligatorio).** Todos los experimentos (3.1 a 3.5) se ejecutan sobre las 5 particiones de 1.9. Si el calendario obliga a recortar, el mínimo aceptable es K = 3 (k = 1 a 3) en todos los experimentos; no se admite reducir solo algunos experimentos.

3.9 **Análisis estadístico.** Según el SAP: hipótesis principales H1 a H7 con modelo de efectos mixtos y Holm, bootstrap por clústeres como segunda condición de confirmación, y análisis secundarios.

## 4. Régimen SSL-imnet-matched

4.1 Subconjunto reproducible de ImageNet-1k (train): 99.417 imágenes, estratificado por clase (99 por clase y una imagen adicional en 417 clases), semilla fija. Guardar la lista de rutas y su SHA-256.

4.2 Preentrenamiento con `main_pretrain.py` y exactamente los argumentos guardados en `mae_hyperkvasir.pth`: `mae_vit_base_patch16`, `batch_size 128`, `accum_iter 2`, 1 GPU, `epochs 400`, `warmup_epochs 40`, `blr 1e-3`, `weight_decay 0.05`, `mask_ratio 0.75`, `norm_pix_loss False`, `input_size 224`, misma augmentación. Sin enmascarado de endoscopia.

4.3 Registrar argumentos, logs completos con marcas de tiempo, duración real y SHA-256 del checkpoint.

4.4 Integrar el régimen en configs, `train_classification.py` e informes (lista de cambios de la Parte B.5 del informe de preentrenamiento). La comparación principal nueva es SSL-colon frente a SSL-imnet-matched.

## 5. Orden de ejecución y coste

Tiempos por ejecución medidos en la A40 (cota superior para la A6000): sun_full ≈ 20 min, E1 ≈ 11 min, E2 ≈ 16 min, E4 ≈ 1,5 min por brazo adaptado, E3 y brazos sin adaptar < 1 min.

| Fase | GPU 0 | GPU 1 | Horas de GPU (aprox.) |
|---|---|---|---|
| 0 | Prueba de humo: 1 ejecución de sun_full con SSL-colon y una de E4 a 50 frames con los cuatro brazos; comprobar 1.1 a 1.4 | Generar particiones (1.9), packs few-shot (1.7) y subconjunto de ImageNet (4.1); verificar hashes de k = 1 | < 2 |
| 1 | Preentrenamiento SSL-imnet-matched | sun_full y E3 de los 3 regímenes existentes, 5 particiones | 10–25 + 15 |
| 2 | (continúa) | E1 y E2 de los 3 regímenes, 5 particiones | 53 |
| 3 | E4 (cuatro brazos) de los 3 regímenes, 5 particiones | (continúa fase 2) | 5 |
| 4 | Cadena de SSL-imnet-matched, empezando por sun_full y E1 al 5 % de la partición 1 | Resto de la cadena de SSL-imnet-matched, repartida entre ambas GPU | 24 |
| 5 | Análisis completo según el SAP sobre `results_v3/` | | < 2 |

Total aproximado: 105–120 horas de GPU, unos 3 a 4 días de calendario con las dos A6000 si no hay incidencias. Con K = 3, unas 65–75 horas de GPU.

## 6. Criterios de verificación

- **Tras la prueba de humo:** el log contiene la recarga del mejor checkpoint; la aserción de 1.1 pasa; existen los CSV de validación y test con todas las columnas de 1.3; las métricas del JSON coinciden con las recalculadas desde el CSV (diferencia < 1e-6).
- **Tras la fase 0:** los ficheros de la partición k = 1 tienen los mismos hashes que los actuales; las particiones nuevas no comparten casos entre train, val y test y respetan los tamaños.
- **Tras cada fase:** número de ejecuciones en disco igual al esperado; ningún JSON con epoch de umbral distinto del epoch evaluado; hashes de los packs SUN y de `polypgen_clean_test` iguales a los de la versión anterior.
- **Al final:** resultados de H1 a H7 según el SAP (estimación, IC del modelo, p ajustada por Holm, IC bootstrap y veredicto); tablas equivalentes a las Tablas 3 a 9 del manuscrito más las nuevas (brazos B y D de E4, régimen igualado); `paired_differences.csv` completo; informe de deriva que compare cada valor nuevo con el de `analysis_v2` y señale los cambios de signo o de significación.

## 7. Entregables

`results_v3/` con salidas por ejecución, `checkpoints_v3/`, `results_v3/tables/`, `results_v3/paired_differences.csv`, `results_v3/sensitivity/`, `results_v3/environment/`, informe de deriva y un `RUNLOG.md` con el registro de cada fase (fecha, commit, GPU, duración, incidencias).
