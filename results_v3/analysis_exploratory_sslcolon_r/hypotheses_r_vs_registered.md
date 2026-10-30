# Exploratory: H1–H7 with SSL-colon-r in place of SSL-colon

Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run values (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345). r: SSL-colon-r replaces SSL-colon in every hypothesis (H5–H7 are means over the four regimes); registered: as registered; diff = r − registered.

### H1–H7

| hypothesis | definition | r_estimate | r_t_ci_lo | r_t_ci_hi | r_boot_lo | r_boot_hi | registered_estimate | registered_t_ci_lo | registered_t_ci_hi | registered_boot_lo | registered_boot_hi | diff_estimate | diff_t_ci_lo | diff_t_ci_hi | diff_boot_lo | diff_boot_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | E1, SSL-colon − SUP-imnet, AULC of AUPRC (5–100 %) | 0.0563 | 0.0390 | 0.0737 | 0.0439 | 0.0688 | 0.0475 | 0.0286 | 0.0663 | 0.0355 | 0.0592 | 0.0089 | -0.0033 | 0.0211 | 0.0026 | 0.0152 |
| H2 | E1, SSL-colon − SSL-imnet-matched, AULC of AUPRC (5–100 %) | -0.0050 | -0.0176 | 0.0076 | -0.0121 | 0.0019 | -0.0138 | -0.0285 | 8.67e-04 | -0.0214 | -0.0064 | 0.0089 | -0.0033 | 0.0211 | 0.0026 | 0.0152 |
| H3 | E2, SSL-colon − SUP-imnet, AUROC, flat stratum | 0.0406 | 0.0095 | 0.0717 | 0.0203 | 0.0648 | 0.0380 | 0.0122 | 0.0639 | 0.0202 | 0.0593 | 0.0026 | -0.0081 | 0.0133 | -0.0049 | 0.0156 |
| H4 | E3, SSL-colon − SUP-imnet, AUROC, polypgen_clean_test | -0.0614 | -0.0758 | -0.0471 | -0.2859 | 0.0837 | -0.0720 | -0.0985 | -0.0455 | -0.3447 | 0.0944 | 0.0106 | -0.0099 | 0.0310 | -0.0429 | 0.0722 |
| H5 | E3 decomposition, sequence − isolated negatives, AUROC, mean of regimes, 50-frame split, common positives | 0.1431 | 0.1123 | 0.1740 | -0.0379 | 0.3004 | 0.1548 | 0.1202 | 0.1895 | -0.0310 | 0.3093 | -0.0117 | -0.0240 | 5.12e-04 | -0.0286 | 0.0035 |
| H6 | E4, arm B − arm A, balanced accuracy, mean of budgets and regimes | -0.0080 | -0.0228 | 0.0069 | -0.0205 | 0.0039 | -0.0064 | -0.0241 | 0.0114 | -0.0184 | 0.0058 | -0.0016 | -0.0073 | 0.0041 | -0.0040 | 4.08e-04 |
| H7 | E4, arm C − arm A, balanced accuracy, mean of budgets and regimes | -0.0786 | -0.0986 | -0.0586 | -0.1381 | -0.0188 | -0.0748 | -0.1012 | -0.0485 | -0.1353 | -0.0150 | -0.0037 | -0.0163 | 0.0089 | -0.0071 | -8.10e-05 |

### Pretraining mean train loss (selected epochs)

| epoch | train_loss_matched | train_loss_r | r_minus_matched |
|---|---|---|---|
| 0.0000 | 1.2130 | 0.7961 | -0.4169 |
| 50 | 0.3238 | 0.1097 | -0.2141 |
| 100 | 0.3110 | 0.1026 | -0.2084 |
| 150 | 0.3054 | 0.0992 | -0.2063 |
| 200 | 0.2998 | 0.0967 | -0.2031 |
| 250 | 0.2941 | 0.0937 | -0.2004 |
| 300 | 0.2877 | 0.0906 | -0.1970 |
| 350 | 0.2821 | 0.0880 | -0.1941 |
| 399 | 0.2803 | 0.0872 | -0.1932 |

Registered values recomputed: 42/42 equal to results_v3/analysis/contrast_summary.csv (max |diff| 4.72e-11).
