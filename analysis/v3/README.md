# Campaign v3 analysis (SAP v3.3)

Implements `docs/v3/plan_analisis_estadistico_v3_3.md` (SHA-256 `fd6b0342…7ff56`, tag `v3-cmpb-sap33`, the last SAP
registered in `results_v3/RUNLOG.md`; v3.2c plus deviation 9.3). Environment: `ssl4polyp-analysis` (`env/environment-analysis.yml`, see `env/README.md`).

```bash
conda activate ssl4polyp-analysis
PYTHONPATH=. python -m analysis.v3.run_analysis --out results_v3/analysis            # real campaign (default roots)
PYTHONPATH=. python -m analysis.v3.run_analysis --roots <sim>/campaign/classification <sim>/campaign_matched/classification --out <dir>
PYTHONPATH=. python -m pytest analysis/v3/tests                                      # tests (simulated data only)
PYTHONPATH=. python -m analysis.v3.validate --out <dir>                              # Monte Carlo validation only
```

## Modules

| File | Content |
|---|---|
| `sap.py` | Constants fixed by the SAP (regimes, partitions, seeds, budgets, arms, B = 2000, seed 12345, α, H1–H7) and the SAP hash check |
| `metrics.py` | Loads `src/ssl4polyp/classification/metrics/binary.py` from its file (no torch) and evaluates metrics at τ, Brier, ECE (15 equal-mass bins), degenerate flag and per-polyp sensitivity |
| `loader.py` | Finds complete runs (`RUN_COMPLETE.json`) under one or more roots, reads τ from the JSONs and the per-frame CSVs; structural integrity checks (inherited τ = parent τ, paired runs score the same frames, E2 = sun_full test frames) |
| `design.py` | Views (run × rows × τ × bootstrap units), linear paired contrasts `d_{k,s} = Σ w·metric`, cluster bootstrap (section 5) |
| `contrasts.py` | H1–H7 exactly as in section 3; all secondary comparisons of section 6; k = 1 and no-s500 sensitivity contrasts |
| `R/fit_mixed.R`, `mixed.py` | Section 4 inference: one-sample t-test on the K partition means (v3.3); the v3.2c variants (`lmer` + Satterthwaite or Kenward–Roger, with fallback) are kept for the validation; Holm; SDs between partitions and seeds |
| `analysis.py` | Orchestration: metrics, contrasts, models, bootstrap per experiment, Holm + double confirmation rule, LOCO (E3 and negative-source decomposition) and leave-one-case-out (E2 flat) |
| `tables.py`, `figures.py`, `run_analysis.py` | Tables (CSV + Markdown), figures (PNG), report and CLI |
| `simulate.py` | Simulated campaigns with the real structure and known effects |
| `validate.py` | Monte Carlo validation (effects, null, zero partition variance) |
| `drift.py` | Drift report vs the previous manuscript (`analysis_v2/manuscript_values.py`) and `analysis_v2` (written before opening the real data) |
| `exploratory_lr_common.py` | Exploratory comparison, common lr 1e-3 vs campaign v3 (`docs/v3/plan_exploratorio_lr_comun.md`; not part of the SAP) |
| `exploratory_lr_sslimnet.py` | Exploratory comparison, SSL-imnet lr 1e-4 vs campaign v3 in E1 (`docs/v3/plan_exploratorio_lr_sslimnet.md`; not part of the SAP) |
| `validate_variants.py` | Validation of three inference variants (V1 v3.2c, V2 Kenward–Roger, V3 t-test) that led to SAP v3.3 |
| `tests/` | Automated tests |

## How the SAP is implemented

- **Per-run differences (4.1).** One `d_{k,s}` per paired run (partition × seed). H1/H2: AULC of AUPRC over
  5–100 % (trapezoid / axis width) of SSL-colon minus comparator. H3: AUROC of negatives + flat positives (E2,
  sun_full model and frozen τ). H4: AUROC on polypgen_clean_test (E3). H5: AUROC(common positives vs sequence
  negatives) − AUROC(common positives vs isolated negatives) on the E4 s50 zero-shot outputs (positives also in the
  E3 test of the same run), averaged over regimes. H6/H7: balanced accuracy of arm B (resp. C) − arm A, averaged
  over budgets and regimes.
- **Inference (4.2–4.3, v3.3).** One test per contrast: seeds averaged within each partition, one-sample t-test on
  the K partition means (df = K − 1); `method` is `partition_means_t`, `t_seeds_K1` (only one partition: k = 1
  sensitivity analysis) or `constant` (zero variance). With all runs present this equals the v3.2c `lmer` fit.
- **Holm (4.4).** Over the hypotheses that exist. Until SSL-imnet-matched results exist, H2 is reported as
  `pending`, Holm runs over the other six and their status is `provisional`; H5–H7 then average the three available
  regimes. Once the matched runs are found (under any root), H2 enters and Holm runs over all seven.
- **Confirmation (4.5).** Holm p < 0.05 and the cluster-bootstrap 95 % CI of the mean per-run difference excludes 0.
- **Bootstrap (5).** B = 2000, seed 12345, one sample of units per replicate per common test set: per partition for
  SUN (case units, class strata; E2 also morphology strata), one PolypGen sample (E3: centre units by class; E4 and
  H5: positives by centre, negatives by sequence or centre). One bootstrap per experiment family; all contrasts of a
  family share the replicates. Samples are drawn in the parent process, so results do not depend on `--workers`.
- **Secondary (6).** SSL-colon vs every comparator for every experiment, budget, stratum, arm and metric (binary
  metrics, Brier, ECE; per-polyp sensitivity in sun_full/E1/E2), with model estimate and bootstrap CI; AULC per
  metric; all E4 arm contrasts per budget, per regime and pooled (B−A, C−A, D−A, D−C, D−B, C−B; threshold-free
  metrics are skipped for B−A and D−C, which share probabilities); flat − polypoid AUROC gap and its difference
  between regimes; negative-source decomposition per regime; degenerate predictors; calibration and reliability
  diagrams (sun_full, E3, E4 per arm); per-polyp sensitivity (E1, E2).
- **Sensitivity (6).** E3 leave-one-centre-out; negative-source decomposition leave-one-centre-out (H5 and per
  regime); E2 flat leave-one-case-out per partition; H1–H4 on k = 1 only; H6/H7 without s500.
- **E4 AP (6, rev. c).** AP is never tabulated as a level for E4 or compared across packs or budgets: only within
  a pack (between arms and between regimes on the same evaluation split).
- Only τ is read from the JSONs; every metric is recomputed from the per-frame CSVs.

## Outputs

`report.md`, `summary.json`, `contrast_summary.csv` (one row per contrast: estimate, SDs, model, bootstrap),
`paired_differences.csv` (every `d_{k,s}`), `run_metrics.csv` (every view), `tables/` and `figures/`.
