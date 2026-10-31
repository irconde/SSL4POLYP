# E4 arm contrasts on the evaluation set common to the four budgets

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** The E4 evaluation split of a
pack depends on the budget, and the 500-frame split holds no isolated negatives. Comparisons across budgets therefore
mix the support size with the test composition. This analysis repeats the E4 arm contrasts on the frames common to
the four budgets. Analysis of existing data only: nothing was trained and no registered result was modified.

    PYTHONPATH=. python scripts/v3/e4_common_eval.py --out results_v3/verification/e4_common_eval

`ssl4polyp-analysis`, run 2026-10-31 (`run_log.txt`), registered campaign `checkpoints_v3/campaign/classification`.

## Method

- **Common set:** for each pack (partition k, seed s), the frames present in the evaluation splits of the four budgets
  (50, 100, 200, 500) and of the four regimes. The script checks that it is the same set for every budget and regime of
  a pack.
- **Metrics:** balanced accuracy, sensitivity, specificity and AUROC of the four arms. They are recomputed from the
  stored per-frame outputs (`*_zeroshot_outputs.csv` for A and B, `*_test_outputs.csv` for C and D). Each arm keeps the
  operating threshold of its run as fixed in the campaign (`*_last.metrics.json`, `e4_arms`); no threshold is
  re-estimated.
- **Contrasts:** B − A, C − A, D − C and D − A per budget, as the mean of the four regimes. AUROC is not reported for
  B − A and D − C: both arms share the same probabilities, so the difference is 0 by construction (as in the
  registered analysis).
- **Inference:** the registered code (`analysis/v3`), unchanged:
  - estimate: paired, the mean of the 15 per-run values (5 partitions × 3 seeds);
  - t-test CI: on the 5 partition means (`mixed`, variant `ttest`);
  - bootstrap CI: cluster bootstrap of the E4 units (B = 2000, seed 12345). The units are 6 centres for positives and
    24 sequences or centres for negatives.

  The original-split contrasts use every E4 view, so the bootstrap spaces and draws are those of the registered e4
  family. The common-set views are subsets of the same units and use the same draws.

## Check

The original-split contrasts reproduce the registered values (`results_v3/analysis/contrast_summary.csv`, ids
`arm|<budget>|pooled|<comparison>|<metric>`; Table S17): 336/336 quantities equal, max |diff| 5.0e-11
(`registered_check.csv`). The quantities are estimate, t-test CI, bootstrap CI and B. The script stops if any differs.

## Files

| File | Content |
|---|---|
| `common_set_by_pack.csv` | per pack: size and composition of the common set, size of each original split, common / original |
| `composition_by_pack_regime_budget.csv` | per pack, regime and budget: composition of the original split and of the common set |
| `run_metrics_original_vs_common.csv` | per run and arm: τ, frames, and the four metrics on the original split and on the common set |
| `arm_means.csv` | arm means per budget (60 runs: 15 × 4 regimes), original and common |
| `contrasts_original_vs_common.csv`, `.md` | one row per contrast: original and common estimate, t-test CI, bootstrap CI, common − original |
| `contrasts_long.csv` | one row per contrast and set, with partition and seed SD, unadjusted p and B |
| `registered_check.csv`, `summary.json`, `run_log.txt` | reproduction check and run information |

## Common set

**In all 15 packs, the common set is the whole 500-frame evaluation split.** That split is contained in the 50-, 100-
and 200-frame splits of the same pack. The set is the same in the four regimes.

| Split | Frames | Positives | Isolated negatives | Sequence negatives |
|---|---|---|---|---|
| 50 (all packs) | 4,578 | 1,229 | 159 | 3,190 |
| 100 (all packs) | 4,528 | 1,204 | 134 | 3,190 |
| 200 (all packs) | 4,428 | 1,154 | 84 | 3,190 |
| 500 = common set (per pack: min, median, max) | 1,306 / 1,714 / 2,035 | 1,004 | 0 | 302 / 710 / 1,031 |

- The common set holds 29–44 % of the 50-frame split, 29–45 % of the 100-frame split and 29–46 % of the 200-frame
  split.
- Compared with the 50-frame split, it loses 225 positives (18 %), all 159 isolated negatives and 68–91 % of the
  sequence negatives. Its prevalence of positives therefore rises from 27 % to 49–77 %, depending on the pack.

## Results

Each cell gives the estimate, then the t-test CI and the bootstrap CI (mean of the four regimes, 15 runs). At 500
frames the common set is the original split, so both columns are identical.

| Budget | Contrast | Metric | Original split (registered) | Common set |
|---|---|---|---|---|
| 50 | B − A | BA | −0.0049 [−0.0279, 0.0182; −0.0217, 0.0101] | −0.0033 [−0.0377, 0.0311; −0.0198, 0.0132] |
| 50 | B − A | Sens | −0.0686 [−0.1174, −0.0198; −0.0917, −0.0538] | −0.0681 [−0.1170, −0.0191; −0.0910, −0.0538] |
| 50 | B − A | Spec | 0.0589 [−0.0271, 0.1448; 0.0347, 0.0858] | 0.0615 [−0.0493, 0.1723; 0.0390, 0.0905] |
| 50 | C − A | BA | −0.0962 [−0.1234, −0.0690; −0.1447, −0.0440] | −0.0806 [−0.1189, −0.0423; −0.1353, −0.0308] |
| 50 | C − A | Sens | −0.4078 [−0.4815, −0.3342; −0.4434, −0.3824] | −0.4074 [−0.4789, −0.3359; −0.4427, −0.3810] |
| 50 | C − A | Spec | 0.2154 [0.1546, 0.2762; 0.1242, 0.3150] | 0.2463 [0.1446, 0.3479; 0.1463, 0.3416] |
| 50 | C − A | AUROC | −0.0319 [−0.0734, 0.0096; −0.0782, 0.0228] | −0.0099 [−0.0680, 0.0482; −0.0807, 0.0459] |
| 50 | D − C | BA | 0.0355 [0.0181, 0.0529; 0.0213, 0.0457] | 0.0299 [0.0055, 0.0543; 0.0093, 0.0549] |
| 50 | D − C | Sens | 0.0678 [−0.0137, 0.1492; 0.0439, 0.0842] | 0.0675 [−0.0119, 0.1470; 0.0434, 0.0832] |
| 50 | D − C | Spec | 0.0032 [−0.0470, 0.0535; −0.0109, 0.0149] | −0.0077 [−0.0605, 0.0451; −0.0411, 0.0374] |
| 50 | D − A | BA | −0.0607 [−0.0947, −0.0267; −0.1101, −0.0111] | −0.0507 [−0.0849, −0.0164; −0.0981, −0.0065] |
| 50 | D − A | Sens | −0.3401 [−0.4032, −0.2769; −0.3987, −0.3014] | −0.3399 [−0.4039, −0.2759; −0.3989, −0.3007] |
| 50 | D − A | Spec | 0.2187 [0.1978, 0.2395; 0.1381, 0.3090] | 0.2386 [0.1436, 0.3336; 0.1643, 0.3208] |
| 50 | D − A | AUROC | −0.0319 [−0.0734, 0.0096; −0.0782, 0.0228] | −0.0099 [−0.0680, 0.0482; −0.0807, 0.0459] |
| 100 | B − A | BA | −0.0140 [−0.0390, 0.0110; −0.0295, 0.0020] | −0.0103 [−0.0437, 0.0232; −0.0300, 0.0079] |
| 100 | B − A | Sens | −0.0348 [−0.1013, 0.0318; −0.0480, −0.0250] | −0.0338 [−0.0996, 0.0319; −0.0469, −0.0237] |
| 100 | B − A | Spec | 0.0067 [−0.0975, 0.1110; −0.0212, 0.0369] | 0.0133 [−0.1095, 0.1360; −0.0240, 0.0487] |
| 100 | C − A | BA | −0.1042 [−0.1255, −0.0829; −0.1683, −0.0336] | −0.0819 [−0.1337, −0.0300; −0.1597, −0.0120] |
| 100 | C − A | Sens | −0.4307 [−0.5383, −0.3232; −0.4717, −0.3909] | −0.4307 [−0.5358, −0.3255; −0.4721, −0.3898] |
| 100 | C − A | Spec | 0.2223 [0.1341, 0.3106; 0.1003, 0.3557] | 0.2669 [0.1148, 0.4190; 0.1169, 0.3976] |
| 100 | C − A | AUROC | −0.0491 [−0.0929, −0.0054; −0.1316, 0.0484] | −0.0139 [−0.0734, 0.0456; −0.1416, 0.0921] |
| 100 | D − C | BA | 0.0527 [0.0136, 0.0917; 0.0383, 0.0640] | 0.0524 [0.0055, 0.0992; 0.0334, 0.0699] |
| 100 | D − C | Sens | 0.2435 [0.1266, 0.3604; 0.2234, 0.2566] | 0.2426 [0.1263, 0.3590; 0.2214, 0.2565] |
| 100 | D − C | Spec | −0.1382 [−0.2175, −0.0588; −0.1585, −0.1205] | −0.1379 [−0.2397, −0.0361; −0.1678, −0.1048] |
| 100 | D − A | BA | −0.0515 [−0.0766, −0.0264; −0.1103, 0.0115] | −0.0295 [−0.0782, 0.0192; −0.1013, 0.0303] |
| 100 | D − A | Sens | −0.1872 [−0.2477, −0.1267; −0.2412, −0.1442] | −0.1880 [−0.2499, −0.1262; −0.2429, −0.1439] |
| 100 | D − A | Spec | 0.0842 [0.0364, 0.1319; −0.0219, 0.2005] | 0.1290 [−0.0137, 0.2717; 0.0017, 0.2394] |
| 100 | D − A | AUROC | −0.0491 [−0.0929, −0.0054; −0.1316, 0.0484] | −0.0139 [−0.0734, 0.0456; −0.1416, 0.0921] |
| 200 | B − A | BA | −0.0065 [−0.0143, 0.0012; −0.0191, 0.0067] | −0.0089 [−0.0228, 0.0050; −0.0213, 0.0046] |
| 200 | B − A | Sens | −0.0281 [−0.0841, 0.0279; −0.0426, −0.0185] | −0.0280 [−0.0839, 0.0279; −0.0426, −0.0183] |
| 200 | B − A | Spec | 0.0150 [−0.0507, 0.0807; −0.0058, 0.0388] | 0.0101 [−0.0639, 0.0842; −0.0116, 0.0341] |
| 200 | C − A | BA | −0.1025 [−0.1352, −0.0698; −0.1730, −0.0232] | −0.0948 [−0.1612, −0.0284; −0.1750, −0.0228] |
| 200 | C − A | Sens | −0.3948 [−0.5056, −0.2840; −0.4506, −0.3332] | −0.3944 [−0.5047, −0.2841; −0.4501, −0.3322] |
| 200 | C − A | Spec | 0.1898 [0.1004, 0.2793; 0.0593, 0.3346] | 0.2048 [0.0609, 0.3486; 0.0583, 0.3344] |
| 200 | C − A | AUROC | −0.0528 [−0.0917, −0.0140; −0.1601, 0.0672] | −0.0332 [−0.1018, 0.0353; −0.1769, 0.0866] |
| 200 | D − C | BA | 0.0426 [0.0021, 0.0830; 0.0254, 0.0544] | 0.0395 [−0.0066, 0.0856; 0.0202, 0.0533] |
| 200 | D − C | Sens | 0.2817 [0.1982, 0.3653; 0.2545, 0.2960] | 0.2816 [0.1969, 0.3663; 0.2543, 0.2959] |
| 200 | D − C | Spec | −0.1966 [−0.3133, −0.0799; −0.2169, −0.1790] | −0.2026 [−0.3303, −0.0749; −0.2300, −0.1802] |
| 200 | D − A | BA | −0.0599 [−0.0924, −0.0275; −0.1323, 0.0177] | −0.0553 [−0.0922, −0.0184; −0.1403, 0.0129] |
| 200 | D − A | Sens | −0.1131 [−0.1788, −0.0474; −0.1863, −0.0500] | −0.1127 [−0.1802, −0.0453; −0.1868, −0.0481] |
| 200 | D − A | Spec | −0.0068 [−0.0968, 0.0832; −0.1320, 0.1353] | 0.0021 [−0.0179, 0.0222; −0.1470, 0.1254] |
| 200 | D − A | AUROC | −0.0528 [−0.0917, −0.0140; −0.1601, 0.0672] | −0.0332 [−0.1018, 0.0353; −0.1769, 0.0866] |
| 500 | B − A | BA | 0.0000 [−0.0209, 0.0209; −0.0089, 0.0101] | same |
| 500 | B − A | Sens | −0.0050 [−0.0594, 0.0494; −0.0148, 0.0013] | same |
| 500 | B − A | Spec | 0.0050 [−0.0835, 0.0936; −0.0105, 0.0240] | same |
| 500 | C − A | BA | 0.0035 [−0.0534, 0.0604; −0.0704, 0.0722] | same |
| 500 | C − A | Sens | −0.3224 [−0.4069, −0.2379; −0.3949, −0.2564] | same |
| 500 | C − A | Spec | 0.3295 [0.2019, 0.4570; 0.2023, 0.4487] | same |
| 500 | C − A | AUROC | 0.1210 [0.0273, 0.2147; 0.0002, 0.2198] | same |
| 500 | D − C | BA | 0.0835 [0.0613, 0.1058; 0.0640, 0.0990] | same |
| 500 | D − C | Sens | 0.2546 [0.2050, 0.3043; 0.2383, 0.2664] | same |
| 500 | D − C | Spec | −0.0876 [−0.1573, −0.0179; −0.1209, −0.0587] | same |
| 500 | D − A | BA | 0.0870 [0.0240, 0.1501; 0.0182, 0.1433] | same |
| 500 | D − A | Sens | −0.0677 [−0.1252, −0.0103; −0.1433, −0.0108] | same |
| 500 | D − A | Spec | 0.2418 [0.0854, 0.3982; 0.1265, 0.3425] | same |
| 500 | D − A | AUROC | 0.1210 [0.0273, 0.2147; 0.0002, 0.2198] | same |

BA = balanced accuracy; Sens = sensitivity; Spec = specificity.

**Arm means, common set** (60 runs per cell; from `arm_means.csv`):

| Budget | A: BA / AUROC | C: BA / AUROC | D: BA / AUROC |
|---|---|---|---|
| 50 | 0.716 / 0.778 | 0.636 / 0.768 | 0.666 / 0.768 |
| 100 | 0.716 / 0.778 | 0.634 / 0.764 | 0.687 / 0.764 |
| 200 | 0.716 / 0.778 | 0.622 / 0.744 | 0.661 / 0.744 |
| 500 | 0.716 / 0.778 | 0.720 / 0.899 | 0.803 / 0.899 |

## Reading

- **Arm A is constant across budgets on the common set** (BA 0.716, AUROC 0.778): the zero-shot model and τ do not
  depend on the budget, so with the same frames it gives the same values. On the original splits it varied with the
  split (BA 0.728–0.732 at 50–200, 0.716 at 500).
- **At 50–200 frames the contrasts barely change.** Sensitivity differences change by ≤ 0.001. The largest shifts
  are in specificity, and in C − A and D − A for AUROC and BA:
  - C − A, AUROC moves towards 0: −0.032 → −0.010 (50), −0.049 → −0.014 (100), −0.053 → −0.033 (200);
  - C − A, BA becomes less negative by 0.008–0.022;
  - D − C is practically unchanged.

  No contrast changes sign, except small ones whose intervals include 0 (D − C specificity at 50, D − A specificity
  at 200). Most intervals widen, because the common set has fewer negatives.
- **The change at 500 frames is not explained by the composition of the test.** With the same test frames, adapting
  (C − A) goes from −0.081, −0.082 and −0.095 in BA at 50, 100 and 200 to +0.004 at 500. AUROC goes from −0.010,
  −0.014 and −0.033 to +0.121. D − A goes from −0.051, −0.030 and −0.055 to +0.087. The jump at 500 is therefore
  associated with the support size (or with what else changes in the 500-frame packs), not with the evaluation frames.

Sensitivity analysis: no Holm, no verdicts. Registered results are unchanged.
