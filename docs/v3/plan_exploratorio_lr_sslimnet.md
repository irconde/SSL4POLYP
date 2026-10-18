# Plan exploratorio: receta de preentrenamiento frente a learning rate en los presupuestos bajos de E1

## Naturaleza del análisis

**Análisis exploratorio, no registrado en el SAP, decidido el 2026-10-18 después de ver los resultados** del análisis
registrado de la campaña v3 (SAP v3.3, `v3-cmpb-analysis-results`) y del experimento exploratorio con learning rate
común (`v3-cmpb-exp-lr-results`). **Sus resultados no se usarán para confirmar ni refutar hipótesis**: sin corrección
de Holm ni veredictos de confirmación; se describen como observados, con su estimación e intervalo.

## Objetivo

Comprobar si la ventaja de SSL-colon y de SSL-imnet-matched sobre SSL-imnet en los presupuestos bajos de E1 (5 % y
10 %) se mantiene cuando SSL-imnet se ajusta con el mismo learning rate que esos dos regímenes (1e-4), es decir,
separar el efecto de la receta de preentrenamiento del efecto del learning rate de ajuste fino. En la campaña v3,
SSL-imnet usa 3e-4 (barrido de la sección 10), y SSL-colon y SSL-imnet-matched usan 1e-4.

## Diseño

- **Régimen:** SSL-imnet únicamente, con learning rate de ajuste fino 1e-4 como sobrescritura explícita
  `--override lr=1e-4` (registrada en el JSON de cada ejecución: `run.lr` = 0,0001, `run.lr_source` = `override`;
  `config/model/*.yaml` no se modifica).
- **Experimentos:** sun_full (15 ejecuciones) y E1 en los cinco presupuestos 5, 10, 25, 50 y 100 % (75 ejecuciones),
  para poder calcular el AULC. Particiones 1–5, semillas 13/29/47.
- **Todo lo demás idéntico a la campaña v3:** código de entrenamiento, entorno `ssl4polyp-ft`, packs v3, subsets
  anidados, umbral F1-óptimo en validación y selección del mejor epoch.
- **Ejecución:** raíz separada `checkpoints_v3/campaign_lr_sslimnet_1e-4/classification`, registros en
  `results_v3/campaign_lr_sslimnet_1e-4/`; `scripts/v3/run_campaign.py` fases `sun_full` y `e1` con
  `--regimes ssl_imnet --root … --lr-override 1e-4 --records …`; dos GPU, una ejecución por GPU; mismas
  comprobaciones de la sección 6 (`scripts/v3/verify_phase.py`, incluida la del learning rate). Planificación en seco:
  las 90 ejecuciones resuelven lr = 0,0001 con origen `override`, sin fallos. **No se hacen commits mientras corre la
  campaña.** Si alguna comprobación falla, se detiene y se informa.

## Análisis

- **Código:** `analysis/v3/exploratory_lr_sslimnet.py`, escrito y subido antes de lanzar; usa sin cambios el código de
  `analysis/v3` (carga, vistas, pesos del AULC, t-test sobre las 5 medias por partición y bootstrap por clústeres con
  B = 2000 y semilla 12345).
- **Contrastes:** SSL-colon − SSL-imnet y SSL-imnet-matched − SSL-imnet, en AUPRC y AUROC, en cada presupuesto de E1
  y en el AULC (5–100 %). SSL-colon y SSL-imnet-matched son las ejecuciones de la campaña v3.
- **Versiones:** con SSL-imnet a 1e-4 (este experimento) y con SSL-imnet a 3e-4 (campaña v3), sobre las mismas
  ejecuciones emparejadas (partición × semilla), y su diferencia (1e-4 − v3), que equivale a SSL-imnet(3e-4) −
  SSL-imnet(1e-4) y es la misma para ambos comparadores. Para cada una: estimación (media de las diferencias por
  ejecución), IC del 95 % del t-test sobre las 5 medias por partición e IC bootstrap (muestra de casos compartida, porque
  los tests de cada partición son los mismos en ambas campañas).
- **Salida:** `results_v3/analysis_exploratory_lr_sslimnet/`: tabla de contrastes (1e-4, v3 y diferencia, con IC) y
  niveles de SSL-imnet por presupuesto con ambos learning rates.

## Limitaciones

- Decidido después de ver los resultados: generador de hipótesis.
- 1e-4 no es el learning rate elegido para SSL-imnet por el barrido de la sección 10 (que eligió 3e-4 por su AP media de
  validación, con una diferencia de 0,0011 frente a 1e-4); el experimento iguala el learning rate entre regímenes, no
  optimiza ninguno.
- Solo E1 y sun_full; E2, E3 y E4 no se repiten.
