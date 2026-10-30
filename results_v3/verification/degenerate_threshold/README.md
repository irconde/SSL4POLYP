# Degenerate predictors: sensitivity to the 0.10 threshold

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** It recounts the degenerate
predictors of the registered campaign with thresholds 0.05, 0.10 (registered) and 0.20. Analysis of existing data only:
nothing was trained and no registered result was modified.

    PYTHONPATH=. python scripts/v3/degenerate_threshold.py --out results_v3/verification/degenerate_threshold

`ssl4polyp-analysis`, run 2026-10-30 (`run_log.txt`).

## Registered definition

The 0.10 threshold is part of the registered plan from its first version, SAP v3. That version is
`docs/v3/plan_analisis_estadistico_v3.md`, SHA-256 `f2120a09…cc0474`, registered 2026-10-14 before phase 1 and added
in commit `795d1f0`. Section 2, "Métricas", line 19:

> - **Predictor degenerado:** ejecución cuya especificidad o sensibilidad en el test, al umbral de operación, es inferior a 0,10.

The sentence is identical in v3.1 (line 19), v3.2 (line 28) and v3.3 (line 30). Section 6 asks for the "Número y
proporción de predictores degenerados por régimen y presupuesto". The code applies it as a strict inequality: the
default threshold `DEGENERATE_THRESHOLD = 0.10` in `src/ssl4polyp/classification/metrics/binary.py`.

## Method

- **Source:** registered per-run metrics, `results_v3/analysis/run_metrics.csv`.
- **Runs:** 1,320 runs of the conditions of Table S18, in 88 conditions (condition × regime, 15 runs each):
  - E1 per budget: 300 runs;
  - E3: 60 runs;
  - E4 per arm and budget: 960 runs.
- **Metrics:** specificity and sensitivity are recomputed from the confusion counts. They equal the stored values (max
  |diff| 5e-11).
- **Rule:** a run is degenerate at threshold t when specificity < t or sensitivity < t.
- **Check:** the 0.10 count equals the registered per-run flag and `results_v3/analysis/tables/table_degenerate.csv` in
  all 88 conditions. The script stops otherwise.

## Files

| File | Content |
|---|---|
| `degenerate_by_condition.csv` | per condition and regime: runs and degenerate count at each threshold. For the 0.10 runs: how many fall below on specificity and on sensitivity, and the min, median and max of each |
| `degenerate_by_experiment.csv` | counts per experiment and regime |
| `degenerate_runs_0.10.csv` | the 66 runs degenerate at 0.10 (partition, seed, τ, confusion counts, specificity, sensitivity) |
| `summary.json`, `run_log.txt` | totals, checks, run information |

## Results

Each cell gives the degenerate runs out of 15 at thresholds 0.05 / 0.10 / 0.20.

| Condition | SUP-imnet | SSL-imnet | SSL-colon | SSL-imnet-matched |
|---|---|---|---|---|
| E1 5 % | 0 / 0 / 0 | 1 / 2 / 3 | 0 / 0 / 0 | 0 / 0 / 1 |
| E1 10, 25, 50, 100 % | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| E3 zero-shot | 0 / 4 / 10 | 0 / 0 / 8 | 4 / 6 / 12 | 2 / 5 / 8 |
| E4 A, 50 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| E4 B, 50 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 0 / 2 |
| E4 C, 50 | 0 / 1 / 3 | 1 / 3 / 3 | 1 / 1 / 3 | 4 / 7 / 7 |
| E4 D, 50 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 1 |
| E4 A, 100 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| E4 B, 100 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 2 |
| E4 C, 100 | 3 / 4 / 4 | 2 / 2 / 3 | 1 / 2 / 3 | 6 / 7 / 9 |
| E4 D, 100 | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 1 / 1 | 0 / 1 / 2 |
| E4 A, 200 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| E4 B, 200 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 0 / 1 |
| E4 C, 200 | 2 / 3 / 5 | 0 / 0 / 1 | 2 / 3 / 4 | 5 / 7 / 7 |
| E4 D, 200 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 1 / 3 |
| E4 A, 500 | 0 / 0 / 0 | 0 / 0 / 2 | 0 / 1 / 1 | 0 / 0 / 2 |
| E4 B, 500 | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 1 / 3 |
| E4 C, 500 | 0 / 1 / 3 | 0 / 0 / 0 | 1 / 1 / 1 | 2 / 2 / 5 |
| E4 D, 500 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Totals over the 1,320 runs: 37 / 66 / 135.

**Runs degenerate at 0.10 (66):** none falls below 0.10 on both metrics. 26 have low specificity and 40 low
sensitivity.

| Metric | Minimum | Median | Maximum |
|---|---|---|---|
| Specificity | 0.000 | 0.992 | 1.000 |
| Sensitivity | 0.000 | 0.058 | 1.000 |

- The low-specificity runs are those of E3 (15 runs), E1 5 % (2 runs) and arms A, B and D of E4 (5 runs), plus 4 runs
  of arm C. Their specificity is between 0.000 and 0.098 and their sensitivity is ≥ 0.886.
- The low-sensitivity runs are all in arm C of E4. Their sensitivity is between 0.000 and 0.086 and their specificity
  is ≥ 0.64, ≥ 0.95 in all but two.

**Effect of the threshold:**
- The conditions with degenerate runs are almost the same at all three thresholds: E3, E4 arm C, and sporadically
  E1 5 % and E4 arms A, B, D.
- In E3 and E4 arm C, SSL-imnet-matched has the most at all three thresholds.
- The count roughly doubles from one threshold to the next (37 → 66 → 135).
- E3 depends most on the threshold: its runs have specificity just below 0.10, so at 0.20 between 8 and 12 of 15 runs
  per regime are degenerate.

Sensitivity analysis: no Holm, no verdicts. Registered results are unchanged.
