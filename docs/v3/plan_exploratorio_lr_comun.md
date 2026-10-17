# Plan exploratorio: learning rate común (1e-3) frente al learning rate por régimen

## Naturaleza del análisis

**Este análisis es exploratorio y no está registrado en el SAP.** Se decidió el 2026-10-17, **después de ver los
resultados de la campaña v3** (análisis SAP v3.3, etiqueta `v3-cmpb-analysis-results`), cuyo informe de deriva
mostró grandes cambios respecto al manuscrito anterior, que usaba un learning rate común de 1e-3. **Sus resultados
no se usarán para confirmar ni refutar ninguna hipótesis**: no se aplica la corrección de Holm ni la regla de
confirmación doble, y los resultados se describen como observados, con su estimación e intervalo, sin lenguaje de
significación. Las hipótesis H1 a H7 y sus veredictos son únicamente los del análisis registrado (SAP v3.3).

## Objetivo

Medir cuánto cambian las comparaciones entre regímenes si los tres regímenes de v2 se ajustan con el learning rate
común de 1e-3 que se usó en v2, frente al learning rate elegido por régimen en la campaña v3 (sección 10 del SAP:
SUP-imnet 3e-4, SSL-imnet 3e-4, SSL-colon 1e-4).

## Diseño

- **Regímenes:** SUP-imnet, SSL-imnet y SSL-colon (los de v2). SSL-imnet-matched no se incluye.
- **Learning rate:** 1e-3 para los tres en el ajuste fino completo de sun_full y E1, como sobrescritura explícita
  `--override lr=1e-3` de `train_classification` (queda registrada en el JSON de cada ejecución: `run.lr` = 0,001,
  `run.lr_source` = `override`). `config/model/*.yaml` no se modifica.
- **Todo lo demás idéntico a la campaña v3:** mismo código de entrenamiento, entorno `ssl4polyp-ft`, packs v3
  (`v3-cmpb-packs2`), particiones 1–5, semillas 13/29/47, política de umbral (F1-óptimo en validación), E2 y E3
  evaluados sobre los modelos de sun_full de esta campaña con su τ, E4 adaptando esos modelos con su calendario por
  etapas sin cambios (la sobrescritura no se aplica a E4) y los cuatro brazos.
- **Matriz:** sun_full 45, E2 45, E3 45, E1 225, E4 180 (540 ejecuciones), en el mismo orden que la campaña v3
  (sun_full; E2 y E3; E1; E4), una ejecución por GPU en las dos GPU, con las mismas comprobaciones de la sección 6
  de la especificación más la comprobación del learning rate (`override`, 0,001) en sun_full y E1.
- **Salidas:** raíz separada `checkpoints_v3/campaign_lr_common/classification` y registros en
  `results_v3/campaign_lr_common/<fase>/`. Nada se escribe en las raíces de las campañas registradas.
- **Lanzador:** `scripts/v3/run_campaign.py --root checkpoints_v3/campaign_lr_common/classification --lr-override 1e-3
  --records results_v3/campaign_lr_common --regimes sup_imnet ssl_imnet ssl_colon`; verificación con
  `scripts/v3/verify_phase.py` con los mismos argumentos. Los valores por defecto de ambos scripts reproducen la campaña
  registrada sin cambios (matriz idéntica comprobada). Planificación en seco: 540/540 ejecuciones resueltas sin fallos.
- **Si alguna comprobación falla, la campaña se detiene y se informa.** Cada fase se registra en `results_v3/RUNLOG.md`.

## Análisis

- **Código:** `analysis/v3/exploratory_lr_common.py`, escrito y subido antes de lanzar la campaña. Usa sin cambios el
  código de `analysis/v3` (carga, vistas, contrastes, t-test sobre las 5 medias por partición de la sección 4 del SAP
  v3.3 y bootstrap por clústeres de la sección 5: B = 2000, semilla 12345).
- **Contrastes:** los que el SAP define para H1, H3, H4, H5 y H7, y los secundarios entre regímenes (SSL-colon frente a
  SUP-imnet y frente a SSL-imnet) de E1 (por presupuesto y AULC), E2 (por estrato y brecha plano − polipoide), E3 y E4
  (por presupuesto y brazo), para todas las métricas del SAP, más los niveles de AULC y los contrastes de brazos B − A y
  C − A (AUROC y balanced accuracy).
- **Comparación:** para cada contraste, tres versiones sobre las mismas ejecuciones emparejadas (partición × semilla):
  learning rate común, campaña v3 y su diferencia (común − v3, por ejecución emparejada). Para cada una: estimación (media
  de las diferencias por ejecución), IC del 95 % del t-test sobre las 5 medias por partición e IC bootstrap. Ambas
  campañas usan los mismos conjuntos de test y packs, de modo que el bootstrap comparte la muestra de unidades entre las
  dos campañas en cada réplica.
- **Valores de v3:** se recalculan sobre los tres regímenes; H5 y H7, que promedian sobre regímenes, difieren por tanto
  de los valores registrados con cuatro regímenes.
- **Sin corrección de Holm y sin veredictos de confirmación.**
- **Salida:** `results_v3/analysis_exploratory_lr_common/`: tabla de contrastes (común, v3 y diferencia), tablas de
  niveles equivalentes a las tablas 3–9 (común frente a v3) y una figura que compara, para los contrastes de H1, H3 y H4,
  la estimación con learning rate común frente a la de v3.

## Limitaciones declaradas

- Decidido después de ver los resultados de v3; cualquier conclusión es generadora de hipótesis.
- 1e-3 no es el learning rate óptimo de ningún régimen según el barrido de la sección 10 (para SUP-imnet y SSL-imnet
  fue claramente peor en validación); el experimento mide la sensibilidad de las comparaciones a esa elección, no
  compara regímenes en condiciones óptimas.
- El entorno y la GPU difieren de v2 (torch 2.4.1 / RTX 6000 Ada frente a torch 1.9.0 / A40), por lo que este
  experimento no reproduce v2: solo aísla el efecto del learning rate dentro del diseño v3.
