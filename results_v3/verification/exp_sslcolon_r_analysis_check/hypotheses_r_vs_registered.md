# Exploratory: H1–H7 with SSL-colon-r in place of SSL-colon

Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run values (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345). r: SSL-colon-r replaces SSL-colon in every hypothesis (H5–H7 are means over the four regimes); registered: as registered; diff = r − registered.

### H1–H7

| hypothesis | definition | r_estimate | r_t_ci_lo | r_t_ci_hi | r_boot_lo | r_boot_hi | registered_estimate | registered_t_ci_lo | registered_t_ci_hi | registered_boot_lo | registered_boot_hi | diff_estimate | diff_t_ci_lo | diff_t_ci_hi | diff_boot_lo | diff_boot_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | E1, SSL-colon − SUP-imnet, AULC of AUPRC (5–100 %) | 0.0475 | 0.0286 | 0.0663 | 0.0355 | 0.0592 | 0.0475 | 0.0286 | 0.0663 | 0.0355 | 0.0592 | 5.55e-18 | -9.86e-18 | 2.10e-17 | -1.48e-17 | 1.48e-17 |
| H2 | E1, SSL-colon − SSL-imnet-matched, AULC of AUPRC (5–100 %) | -0.0138 | -0.0285 | 8.67e-04 | -0.0214 | -0.0064 | -0.0138 | -0.0285 | 8.67e-04 | -0.0214 | -0.0064 | -7.40e-18 | -2.00e-17 | 5.18e-18 | -1.48e-17 | 1.48e-17 |
| H3 | E2, SSL-colon − SUP-imnet, AUROC, flat stratum | 0.0380 | 0.0122 | 0.0639 | 0.0202 | 0.0593 | 0.0380 | 0.0122 | 0.0639 | 0.0202 | 0.0593 | 0.0000 |  |  | 0.0000 | 0.0000 |
| H4 | E3, SSL-colon − SUP-imnet, AUROC, polypgen_clean_test | -0.0720 | -0.0985 | -0.0455 | -0.3447 | 0.0944 | -0.0720 | -0.0985 | -0.0455 | -0.3447 | 0.0944 | 0.0000 |  |  | 0.0000 | 0.0000 |
| H5 | E3 decomposition, sequence − isolated negatives, AUROC, mean of regimes, 50-frame split, common positives | 0.1548 | 0.1202 | 0.1895 | -0.0310 | 0.3093 | 0.1548 | 0.1202 | 0.1895 | -0.0310 | 0.3093 | -7.40e-18 | -2.44e-17 | 9.64e-18 | -1.48e-17 | 1.57e-17 |
| H6 | E4, arm B − arm A, balanced accuracy, mean of budgets and regimes | -0.0064 | -0.0241 | 0.0114 | -0.0184 | 0.0058 | -0.0064 | -0.0241 | 0.0114 | -0.0184 | 0.0058 | 2.31e-18 | -4.42e-18 | 9.05e-18 | -5.09e-18 | 5.09e-18 |
| H7 | E4, arm C − arm A, balanced accuracy, mean of budgets and regimes | -0.0748 | -0.1012 | -0.0485 | -0.1353 | -0.0150 | -0.0748 | -0.1012 | -0.0485 | -0.1353 | -0.0150 | 1.39e-18 | -1.06e-17 | 1.33e-17 | -1.22e-17 | 1.20e-17 |

### Pretraining mean train loss (selected epochs)

| epoch | train_loss_matched | train_loss_r | r_minus_matched |
|---|---|---|---|
| 0.0000 | 1.2130 | 1.2130 | 0.0000 |
| 50 | 0.3238 | 0.3238 | 0.0000 |
| 100 | 0.3110 | 0.3110 | 0.0000 |
| 150 | 0.3054 | 0.3054 | 0.0000 |
| 200 | 0.2998 | 0.2998 | 0.0000 |
| 250 | 0.2941 | 0.2941 | 0.0000 |
| 300 | 0.2877 | 0.2877 | 0.0000 |
| 350 | 0.2821 | 0.2821 | 0.0000 |
| 399 | 0.2803 | 0.2803 | 0.0000 |

Registered values recomputed: 84/84 equal to results_v3/analysis/contrast_summary.csv (max |diff| 4.72e-11).
