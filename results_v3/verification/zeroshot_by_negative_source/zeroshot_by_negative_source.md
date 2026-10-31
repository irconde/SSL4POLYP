# Unadapted transfer by negative source (AUROC, E4 50-frame split, common positives)

Post hoc sensitivity analysis, not registered. Estimate = mean of the paired per-run values (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, the registered draws).

### Composition per pack (same frames in every regime)

| k | seed | positives | neg_sequence | neg_isolated |
|---|---|---|---|---|
| 1 | 13 | 1229 | 3190 | 159 |
| 1 | 29 | 1229 | 3190 | 159 |
| 1 | 47 | 1229 | 3190 | 159 |
| 2 | 13 | 1229 | 3190 | 159 |
| 2 | 29 | 1229 | 3190 | 159 |
| 2 | 47 | 1229 | 3190 | 159 |
| 3 | 13 | 1229 | 3190 | 159 |
| 3 | 29 | 1229 | 3190 | 159 |
| 3 | 47 | 1229 | 3190 | 159 |
| 4 | 13 | 1229 | 3190 | 159 |
| 4 | 29 | 1229 | 3190 | 159 |
| 4 | 47 | 1229 | 3190 | 159 |
| 5 | 13 | 1229 | 3190 | 159 |
| 5 | 29 | 1229 | 3190 | 159 |
| 5 | 47 | 1229 | 3190 | 159 |

### AUROC levels and contrasts

| kind | population | comparison | estimate | t_ci_lo | t_ci_hi | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|
| level | sequence | SUP-imnet | 0.8828 | 0.8526 | 0.9130 | 0.7912 | 0.9572 | 2000 |
| level | sequence | SSL-imnet | 0.8258 | 0.7890 | 0.8626 | 0.7225 | 0.9103 | 2000 |
| level | sequence | SSL-colon | 0.7754 | 0.7287 | 0.8221 | 0.6181 | 0.9035 | 2000 |
| level | sequence | SSL-imnet-matched | 0.7574 | 0.7105 | 0.8043 | 0.5907 | 0.8926 | 2000 |
| contrast | sequence | ssl_colon - sup_imnet | -0.1074 | -0.1337 | -0.0812 | -0.1864 | -0.0427 | 2000 |
| contrast | sequence | ssl_colon - ssl_imnet | -0.0504 | -0.0870 | -0.0137 | -0.1187 | 0.0057 | 2000 |
| contrast | sequence | ssl_colon - ssl_imnet_matched | 0.0180 | 0.0021 | 0.0339 | -0.0240 | 0.0688 | 2000 |
| contrast | sequence | ssl_imnet - sup_imnet | -0.0571 | -0.0747 | -0.0394 | -0.0824 | -0.0340 | 2000 |
| level | isolated | SUP-imnet | 0.6886 | 0.6723 | 0.7050 | 0.5039 | 0.8781 | 1997 |
| level | isolated | SSL-imnet | 0.6958 | 0.6776 | 0.7139 | 0.5501 | 0.8301 | 1997 |
| level | isolated | SSL-colon | 0.6187 | 0.5920 | 0.6454 | 0.4037 | 0.7519 | 1997 |
| level | isolated | SSL-imnet-matched | 0.6190 | 0.6085 | 0.6294 | 0.5258 | 0.7169 | 1997 |
| contrast | isolated | ssl_colon - sup_imnet | -0.0699 | -0.0954 | -0.0444 | -0.3542 | 0.1108 | 1997 |
| contrast | isolated | ssl_colon - ssl_imnet | -0.0770 | -0.0954 | -0.0586 | -0.2972 | 0.0522 | 1997 |
| contrast | isolated | ssl_colon - ssl_imnet_matched | -2.35e-04 | -0.0214 | 0.0210 | -0.1936 | 0.1003 | 1997 |
| contrast | isolated | ssl_imnet - sup_imnet | 0.0071 | -0.0101 | 0.0244 | -0.0694 | 0.0614 | 1997 |
| level | all | SUP-imnet | 0.8736 | 0.8442 | 0.9030 | 0.7846 | 0.9451 | 2000 |
| level | all | SSL-imnet | 0.8196 | 0.7848 | 0.8544 | 0.7202 | 0.9012 | 2000 |
| level | all | SSL-colon | 0.7680 | 0.7232 | 0.8128 | 0.6154 | 0.8909 | 2000 |
| level | all | SSL-imnet-matched | 0.7509 | 0.7061 | 0.7956 | 0.5919 | 0.8766 | 2000 |
| contrast | all | ssl_colon - sup_imnet | -0.1056 | -0.1308 | -0.0805 | -0.1829 | -0.0412 | 2000 |
| contrast | all | ssl_colon - ssl_imnet | -0.0516 | -0.0869 | -0.0164 | -0.1177 | 0.0030 | 2000 |
| contrast | all | ssl_colon - ssl_imnet_matched | 0.0171 | 0.0021 | 0.0321 | -0.0265 | 0.0689 | 2000 |
| contrast | all | ssl_imnet - sup_imnet | -0.0540 | -0.0706 | -0.0374 | -0.0786 | -0.0301 | 2000 |

Registered SSL-colon contrasts in (a) and (b) recomputed: 36/36 quantities equal to results_v3/analysis/contrast_summary.csv (max |diff| 4.56e-11).
