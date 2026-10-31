# E4 arm contrasts: degenerate predictors excluded (a) and seed medians (b)

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** It asks how much of the E4
arm contrasts, H6 and H7 in particular, depends on degenerate predictors. Analysis of existing data only: nothing was
trained and no registered result was modified.

    PYTHONPATH=. python scripts/v3/degenerate_sensitivity.py --out results_v3/verification/degenerate_sensitivity

`ssl4polyp-analysis`, run 2026-10-31, 590 s (`run_log.txt`), registered campaign `checkpoints_v3/campaign/classification`.

## Contrasts and variants

All contrasts are in balanced accuracy, as the mean of the four regimes:

- H6 (B − A) and H7 (C − A), also averaged over the four budgets;
- B − A, C − A, D − C and D − A per budget (50, 100, 200, 500).

Per run (partition k, seed s), each contrast is the mean of its paired arm differences over its pairs (regime, budget),
as registered: 16 pairs per run for H6/H7, 4 for the per-budget contrasts.

- **Registered:** all pairs. Estimate = mean of the 15 per-run values.
- **(a) Exclusion:** a pair is left out when either of its two arms is a degenerate predictor with the registered rule
  (specificity or sensitivity < 0.10 at the operating threshold). The rule is applied per view, i.e. run × arm.
  - The per-run value is the mean over the remaining pairs. No run lost all its pairs, so all 15 runs remain.
  - The excluded pairs are fixed from the point estimates and kept fixed in the bootstrap replicates.
- **(b) Seed median:** nothing is excluded. Each partition's value is the median of its three per-run values instead of
  their mean. Estimate = mean of the five partition medians; the t-test is on the partition medians.

**Inference:** the registered code (`analysis/v3`), unchanged:

- t-test on the 5 partition values (`mixed`, variant `ttest`);
- cluster bootstrap of the E4 units (B = 2000, seed 12345).

Each contrast is built as 15 single-run contrasts over all the E4 views, so the bootstrap space and draws are those of
the registered e4 family (checked). Each replicate is then aggregated as registered, as (a) or as (b). A replicate in
which any averaged run is undefined is discarded, as registered.

**Valid replicates:** 1691 for H6, H7 and the 500-frame contrasts, 2000 for the others. In the other replicates some
pack's 500-frame evaluation split has no negatives (see `../e4_common_eval/README.md`).

## Checks

- **Degenerate flags:** recomputed per view, equal to the registered `results_v3/analysis/run_metrics.csv`. There are
  49 degenerate E4 views: arm A 1, B 1, C 44, D 3 (`degenerate_views.csv`).
- **Reproduction:** the registered aggregation reproduces `results_v3/analysis/contrast_summary.csv` (H6, H7 and the 16
  per-budget contrasts): 108/108 quantities equal, max |diff| 5.0e-11 (`registered_check.csv`). The quantities are
  estimate, t-test CI, bootstrap CI and B.

## Pairs excluded in (a)

From `excluded_pairs.csv`: pairs = (regime, budget, partition, seed), i.e. 240 for H6/H7 and 60 per budget.

| Contrast | 50 | 100 | 200 | 500 | H6 / H7 (all budgets) |
|---|---|---|---|---|---|
| B − A | 0 | 0 | 0 | 2 | H6: 2 of 240 |
| C − A | 12 | 15 | 13 | 5 | H7: 45 of 240 |
| D − C | 12 | 17 | 14 | 4 | |
| D − A | 0 | 2 | 1 | 1 | |

- **H7:** 44 of the 45 exclusions come from a degenerate arm C, 1 from a degenerate arm A.
- **By regime:** SSL-imnet-matched accounts for 23 of the 45 exclusions, SUP-imnet 9, SSL-colon 8, SSL-imnet 5.
- **Per budget:** C − A and D − C are excluded mostly because arm C is degenerate. Arm C keeps the inherited τ, and the
  degenerate C runs have sensitivity near 0.

## Results

Each cell gives the estimate, then the t-test CI and the bootstrap CI (`contrasts.csv`).

| Contrast | Registered | (a) Degenerate pairs excluded | (b) Seed median |
|---|---|---|---|
| **H6** B − A | −0.0064 [−0.0241, 0.0114; −0.0184, 0.0058] | −0.0062 [−0.0243, 0.0119; −0.0182, 0.0060] | −0.0044 [−0.0183, 0.0096; −0.0143, 0.0087] |
| **H7** C − A | −0.0748 [−0.1012, −0.0485; −0.1353, −0.0150] | −0.0423 [−0.0693, −0.0154; −0.1000, 0.0142] | −0.0817 [−0.1209, −0.0425; −0.1416, −0.0176] |
| 50: B − A | −0.0049 [−0.0279, 0.0182; −0.0217, 0.0101] | same (0 excluded) | 0.0010 [−0.0193, 0.0212; −0.0178, 0.0153] |
| 50: C − A | −0.0962 [−0.1234, −0.0690; −0.1447, −0.0440] | −0.0627 [−0.1133, −0.0121; −0.1053, −0.0192] | −0.0904 [−0.1153, −0.0655; −0.1398, −0.0398] |
| 50: D − C | 0.0355 [0.0181, 0.0529; 0.0213, 0.0457] | 0.0157 [−0.0135, 0.0449; 0.0042, 0.0243] | 0.0453 [0.0266, 0.0640; 0.0269, 0.0573] |
| 50: D − A | −0.0607 [−0.0947, −0.0267; −0.1101, −0.0111] | same (0 excluded) | −0.0561 [−0.0821, −0.0302; −0.1025, −0.0101] |
| 100: B − A | −0.0140 [−0.0390, 0.0110; −0.0295, 0.0020] | same (0 excluded) | −0.0079 [−0.0299, 0.0141; −0.0243, 0.0070] |
| 100: C − A | −0.1042 [−0.1255, −0.0829; −0.1683, −0.0336] | −0.0603 [−0.0844, −0.0362; −0.1197, 0.0047] | −0.1014 [−0.1196, −0.0832; −0.1512, −0.0437] |
| 100: D − C | 0.0527 [0.0136, 0.0917; 0.0383, 0.0640] | 0.0210 [0.0001, 0.0418; 0.0136, 0.0272] | 0.0706 [0.0260, 0.1152; 0.0481, 0.0827] |
| 100: D − A | −0.0515 [−0.0766, −0.0264; −0.1103, 0.0115] | −0.0469 [−0.0650, −0.0288; −0.1055, 0.0165] | −0.0410 [−0.0913, 0.0094; −0.0941, 0.0190] |
| 200: B − A | −0.0065 [−0.0143, 0.0012; −0.0191, 0.0067] | same (0 excluded) | −0.0062 [−0.0154, 0.0029; −0.0223, 0.0090] |
| 200: C − A | −0.1025 [−0.1352, −0.0698; −0.1730, −0.0232] | −0.0690 [−0.0992, −0.0388; −0.1391, 0.0064] | −0.0870 [−0.1274, −0.0466; −0.1584, −0.0142] |
| 200: D − C | 0.0426 [0.0021, 0.0830; 0.0254, 0.0544] | 0.0245 [−0.0225, 0.0714; 0.0114, 0.0345] | 0.0547 [0.0170, 0.0923; 0.0332, 0.0654] |
| 200: D − A | −0.0599 [−0.0924, −0.0275; −0.1323, 0.0177] | −0.0579 [−0.0878, −0.0279; −0.1297, 0.0188] | −0.0548 [−0.0976, −0.0119; −0.1238, 0.0211] |
| 500: B − A | 0.0000 [−0.0209, 0.0209; −0.0089, 0.0101] | 0.0030 [−0.0254, 0.0314; −0.0063, 0.0132] | −0.0031 [−0.0251, 0.0189; −0.0125, 0.0123] |
| 500: C − A | 0.0035 [−0.0534, 0.0604; −0.0704, 0.0722] | 0.0188 [−0.0381, 0.0757; −0.0538, 0.0862] | 0.0122 [−0.0754, 0.0997; −0.0722, 0.0975] |
| 500: D − C | 0.0835 [0.0613, 0.1058; 0.0640, 0.0990] | 0.0710 [0.0547, 0.0873; 0.0521, 0.0872] | 0.0677 [0.0415, 0.0939; 0.0480, 0.0979] |
| 500: D − A | 0.0870 [0.0240, 0.1501; 0.0182, 0.1433] | 0.0877 [0.0237, 0.1517; 0.0199, 0.1432] | 0.0829 [0.0044, 0.1613; 0.0129, 0.1484] |

## H7: share of the registered effect removed in (a)

Registered H7 = −0.07484; with the degenerate pairs excluded (a) = −0.04233. The share removed is
(−0.07484 − (−0.04233)) / −0.07484 = **0.434**. About 43 % of the registered drop in balanced accuracy of the adapted
arm with inherited τ (C − A) comes from the 45 pairs with a degenerate arm, 19 % of all pairs.

- **After exclusion:** the remaining drop, −0.042, is still negative with a t-test CI below 0. The bootstrap CI now
  includes 0 [−0.1000, 0.0142].
- **(b):** the seed median does not reduce H7 (−0.0817). The 44 degenerate C pairs are spread over 12 of the 15
  runs (2–5 pairs each), and over at least two of the three seeds of every partition. Each run averages 16 pairs, so
  no single seed carries them and the median does not remove them.

**Caution in reading (a).** The exclusion conditions on the outcome. A degenerate arm C has balanced accuracy near 0.5
by definition, so removing those pairs mechanically moves C − A towards 0. (a) is an upper bound on what the
degenerate runs contribute. It does not estimate the effect of adaptation in runs where adaptation works.

## Reading

- **H6 (B − A)** is unaffected: 2 pairs excluded; −0.0064 → −0.0062 (a), −0.0044 (b).
- **C − A per budget, 50–200:** shrinks by about a third to two fifths in (a): −0.096 → −0.063, −0.104 → −0.060,
  −0.103 → −0.069. It stays negative, with t-test CIs below 0.
- **D − C, 50–200:** about halves in (a), because the gains of recalibration come largely from repairing degenerate C
  runs.
- **D − A** barely changes in (a) or (b).
- **500 frames:** few exclusions, little change.
- **(b)** changes sign only in two contrasts close to 0 whose intervals include 0 in both versions: B − A at 50 and at
  500 frames. Which intervals exclude 0 changes only for D − A at 100 frames, whose t-test CI now includes 0.

Sensitivity analysis: no Holm, no verdicts. Registered results are unchanged.

## Files

| File | Content |
|---|---|
| `contrasts.csv`, `contrasts.md` | per contrast and variant: runs, partitions, estimate, t-test CI, unadjusted p, bootstrap CI, valid replicates |
| `excluded_pairs.csv` | per contrast: pairs, pairs excluded in (a) (by which arm and by regime), runs dropped |
| `degenerate_views.csv` | the 49 degenerate E4 views (budget, regime, partition, seed, arm, specificity, sensitivity, balanced accuracy) |
| `per_run.csv` | per contrast and variant, the per-run values |
| `registered_check.csv`, `summary.json`, `run_log.txt` | reproduction check and run information |
