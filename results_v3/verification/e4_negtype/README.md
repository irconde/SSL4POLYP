# E4 sensitivity: support size vs type of negatives — results

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** It follows the plan
registered before launching, `docs/v3/sensitivity_e4_negative_type_plan.md` (SHA-256 `7f9c9964…14da414`, tag
`v3-cmpb-sens-e4-negtype-plan`, commit `7a541ba`). No Holm, no verdicts; registered results are unchanged.

## Runs

- **Adaptations:** 60, the four regimes × 15 packs `data_packs/v3/polypgen_fewshot_s200seq/`, with the registered
  200-frame schedule. Each adapts the registered `sun_full` model of its partition, seed and regime.
  - Run on 2026-10-31, 09:25–09:36, GPUs 0 and 1, commit `7a541ba`.
  - 60/60 complete at the first attempt: 0 failures, 0 retries, 0.37 GPU-hours.
- **Checks:** the campaign section 6 checks (`verify_phase.py e4negtype`) passed 1,500/1,500, 0 problems
  (`../sens_e4_negtype_verification.json`):
  - per run: best-checkpoint reload, threshold epoch, checkpoint hash, partition and seed (60 each);
  - CSV columns and predictions: 360 each;
  - JSON = recomputation from the CSVs: 420;
  - τ: 120.
- **Records:** per-run records and the hashed manifest of every file were written before the weights were deleted
  (`../sens_e4_negtype_{runs,results_manifest}.csv`). The 60 checkpoint SHA-256 equal those in the metrics JSONs.
- **Weights:** the adaptation weights were then deleted (0 `.pth` left). Per-frame outputs and metrics JSONs are kept.
- **Driver:** `../sens_e4_negtype_driver.sh`.
- **Arm B:** not trained. Each adaptation run writes the four arms: B's τ is F1-optimal on the unadapted scores of
  the new support (train + val), as in the campaign.

## Analysis

    PYTHONPATH=. python scripts/v3/e4_negtype.py --seq-root checkpoints_v3/sens_e4_negtype/classification \
        --out results_v3/verification/e4_negtype

`ssl4polyp-analysis`, code as registered with the plan (unchanged), 176 s (`run_log.txt`).

- **Run check:** the 200seq runs are the 60 expected ones; A and C use the τ of their registered parent; the test is
  exactly the common set.
- **Reproduction check:** 200iso and 500 equal `../e4_common_eval/`, 168/168 quantities (max |diff| 5.0e-11;
  `common_eval_check.csv`).
- **Bootstrap:** the space equals the registered E4 space (6 positive units, 24 negative units).
- **Bootstrap replicates:** on the common set, the negatives of each pack come from 4 sequences. In 309 of the 2000
  bootstrap replicates some pack has no negatives, so balanced accuracy, specificity and AUROC are undefined for it. As
  in the registered analysis, those replicates are discarded: the bootstrap CIs of balanced accuracy, specificity and
  AUROC use 1691 replicates (column `boot_B` of the CSVs); sensitivity uses 2000.
- **D's τ:** no run kept the inherited τ for D, in any condition.

### Pairing

Every contrast is paired within a run: A, B, C and D of a given condition, regime, partition and seed come from the
same run (`pairing_check.csv`, 180 runs = 3 conditions × 60).

- The four arms of each run use its τ values, equal to its metrics JSON at CSV precision (relative diff ≤ 4.7e-10).
- A and B use its zero-shot outputs and C and D its adapted outputs, the same files hashed in its JSON.
- All four arms are evaluated on the same frames.
- The key contrasts difference two such within-run contrasts, from the runs of two conditions with the same regime,
  partition and seed.

### Arm A: new runs vs registered 200-frame runs

On the common set, per run, `armA_new_minus_registered200.csv` (60 pairs, same τ):

| Metric | Minimum | Median | Maximum |
|---|---|---|---|
| Balanced accuracy | 0 | 0 | 0 |
| AUROC | −2.2e-6 | 0 | 6.1e-6 |

The zero-shot probabilities differ from the registered `s500` ones by at most 0.0023 (inference non-determinism). This
changes no thresholded prediction and moves AUROC by at most 6.1e-6.

## Results

Each cell gives the estimate, then the t-test CI and the bootstrap CI (mean of the four regimes, 15 paired runs). All
conditions are evaluated on the common set.

**Balanced accuracy per condition:**

| Contrast | 200iso (registered) | 200seq (new) | 500 (registered) |
|---|---|---|---|
| B − A | −0.0089 [−0.0228, 0.0050; −0.0213, 0.0046] | 0.0123 [−0.0015, 0.0262; 0.0015, 0.0232] | 0.0000 [−0.0209, 0.0209; −0.0089, 0.0101] |
| C − A | −0.0948 [−0.1612, −0.0284; −0.1750, −0.0228] | 0.0970 [0.0655, 0.1284; 0.0526, 0.1333] | 0.0035 [−0.0534, 0.0604; −0.0704, 0.0722] |
| D − C | 0.0395 [−0.0066, 0.0856; 0.0202, 0.0533] | 0.0015 [−0.0083, 0.0113; −0.0057, 0.0124] | 0.0835 [0.0613, 0.1058; 0.0640, 0.0990] |
| D − A | −0.0553 [−0.0922, −0.0184; −0.1403, 0.0129] | 0.0985 [0.0676, 0.1293; 0.0611, 0.1324] | 0.0870 [0.0240, 0.1501; 0.0182, 0.1433] |

**Key contrasts:**

| Contrast | Balanced accuracy | Sensitivity | Specificity | AUROC |
|---|---|---|---|---|
| Type, D − A: [D − A](200seq) − [D − A](200iso) | **0.1538 [0.1363, 0.1713; 0.0955, 0.2148]** | 0.0583 [0.0236, 0.0929; −0.0048, 0.1069] | 0.2493 [0.1992, 0.2994; 0.1524, 0.3613] | 0.1720 [0.1565, 0.1875; 0.1031, 0.2436] |
| Size, D − A: [D − A](500) − [D − A](200seq) | **−0.0114 [−0.0513, 0.0285; −0.0594, 0.0337]** | −0.0133 [−0.0486, 0.0221; −0.0571, 0.0369] | −0.0096 [−0.1237, 0.1045; −0.0903, 0.0739] | −0.0178 [−0.0519, 0.0164; −0.0644, 0.0233] |
| Type, C − A | **0.1918 [0.1546, 0.2289; 0.1329, 0.2424]** | 0.3168 [0.2597, 0.3739; 0.2510, 0.3616] | 0.0668 [−0.0094, 0.1429; −0.0308, 0.1598] | 0.1720 [0.1565, 0.1875; 0.1031, 0.2436] |
| Size, C − A | **−0.0934 [−0.1193, −0.0676; −0.1394, −0.0442]** | −0.2448 [−0.2773, −0.2123; −0.2925, −0.1834] | 0.0579 [−0.0097, 0.1256; −0.0193, 0.1436] | −0.0178 [−0.0519, 0.0164; −0.0644, 0.0233] |

**Arm means, balanced accuracy / AUROC** (60 runs each; `arm_means.csv`):

| Condition | A | B | C | D |
|---|---|---|---|---|
| 200iso | 0.716 / 0.778 | 0.707 / 0.778 | 0.622 / 0.744 | 0.661 / 0.744 |
| 200seq | 0.716 / 0.778 | 0.729 / 0.778 | 0.813 / 0.916 | 0.815 / 0.916 |
| 500 | 0.716 / 0.778 | 0.716 / 0.778 | 0.720 / 0.899 | 0.803 / 0.899 |

## Classification by the fixed reading

The reading uses balanced accuracy and both 95 % intervals.

| | Type of negatives | Support size |
|---|---|---|
| **D − A (primary)** | **Type matters:** both intervals above 0 (+0.154) | **Not shown:** both intervals include 0 (−0.011) |
| C − A (secondary) | **Type matters:** both intervals above 0 (+0.192) | **Opposite direction:** both intervals below 0 (−0.093) |

- With 200 frames whose negatives are sequence negatives, the adapted arms reach the level of the 500-frame arm, or
  exceed it. D − A is +0.099 vs +0.087. C − A is +0.097 vs +0.004: at 500 frames, C (inherited τ) has low sensitivity
  (0.590), which D's recalibration recovers.
- With the same positives and schedule, replacing isolated negatives by sequence negatives turns D − A from −0.055 into
  +0.099. Going from 200seq to 500 adds nothing for D − A.
- The size contrast is not size alone: from 200seq to 500 the support changes in positives (100 → 250), negatives
  (100 sequence → 184 isolated + 66 sequence) and schedule (plan, "Notes on interpretation").

## Files

| File | Content |
|---|---|
| `contrasts_long.csv`, `contrasts.md` | every contrast: estimate, partition and seed SD, t-test CI, unadjusted p, bootstrap CI, B |
| `arm_means.csv`, `run_metrics.csv` | arm means per condition; per run and arm: τ, frames, the four metrics |
| `pairing_check.csv` | per run and condition: arms, τ vs metrics JSON, output files vs their SHA-256 |
| `armA_new_minus_registered200.csv` | arm A, new run − registered 200-frame run, per pack and regime |
| `common_eval_check.csv`, `summary.json`, `run_log.txt` | reproduction check and run information |
