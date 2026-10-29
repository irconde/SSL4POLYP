# Exploratory: learning rate selected per E1 budget vs registered

Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run differences (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345). selected: each regime at the learning rate selected for each budget on validation AP; registered: each regime at its registered learning rate; diff = selected − registered.

### Mean validation AP (best-epoch weights) by regime, budget and learning rate

| regime | budget | lr | val_ap_mean | val_ap_sd | n_runs | mark |
|---|---|---|---|---|---|---|
| ssl_colon | 5 | 1e-3 | 0.7851 | 0.1013 | 15 |  |
| ssl_colon | 5 | 1e-4 | 0.8370 | 0.0828 | 15 | selected (registered) |
| ssl_colon | 5 | 3e-4 | 0.8171 | 0.0857 | 15 |  |
| ssl_colon | 10 | 1e-3 | 0.8017 | 0.0577 | 15 |  |
| ssl_colon | 10 | 1e-4 | 0.8625 | 0.0503 | 15 | selected (registered) |
| ssl_colon | 10 | 3e-4 | 0.8425 | 0.0541 | 15 |  |
| ssl_colon | 25 | 1e-3 | 0.8739 | 0.0595 | 15 |  |
| ssl_colon | 25 | 1e-4 | 0.9046 | 0.0542 | 15 | selected (registered) |
| ssl_colon | 25 | 3e-4 | 0.8995 | 0.0579 | 15 |  |
| ssl_colon | 50 | 1e-3 | 0.9365 | 0.0650 | 15 |  |
| ssl_colon | 50 | 1e-4 | 0.9711 | 0.0267 | 15 | selected (registered) |
| ssl_colon | 50 | 3e-4 | 0.9392 | 0.0607 | 15 |  |
| ssl_colon | 100 | 1e-3 | 0.9845 | 0.0090 | 15 |  |
| ssl_colon | 100 | 1e-4 | 0.9956 | 0.0036 | 15 | selected (registered) |
| ssl_colon | 100 | 3e-4 | 0.9933 | 0.0038 | 15 |  |
| ssl_imnet | 5 | 1e-3 | 0.6645 | 0.0958 | 15 |  |
| ssl_imnet | 5 | 1e-4 | 0.7357 | 0.1091 | 15 | selected |
| ssl_imnet | 5 | 3e-4 | 0.7165 | 0.0836 | 15 | (registered) |
| ssl_imnet | 10 | 1e-3 | 0.6626 | 0.0935 | 15 |  |
| ssl_imnet | 10 | 1e-4 | 0.8885 | 0.0701 | 15 | selected |
| ssl_imnet | 10 | 3e-4 | 0.7926 | 0.0721 | 15 | (registered) |
| ssl_imnet | 25 | 1e-3 | 0.7853 | 0.0671 | 15 |  |
| ssl_imnet | 25 | 1e-4 | 0.9517 | 0.0275 | 15 | selected |
| ssl_imnet | 25 | 3e-4 | 0.9372 | 0.0383 | 15 | (registered) |
| ssl_imnet | 50 | 1e-3 | 0.7999 | 0.0634 | 15 |  |
| ssl_imnet | 50 | 1e-4 | 0.9883 | 0.0109 | 15 |  |
| ssl_imnet | 50 | 3e-4 | 0.9900 | 0.0081 | 15 | selected (registered) |
| ssl_imnet | 100 | 1e-3 | 0.8816 | 0.0926 | 15 |  |
| ssl_imnet | 100 | 1e-4 | 0.9969 | 0.0026 | 15 |  |
| ssl_imnet | 100 | 3e-4 | 0.9986 | 0.0021 | 15 | selected (registered) |
| ssl_imnet_matched | 5 | 1e-3 | 0.7608 | 0.1018 | 15 |  |
| ssl_imnet_matched | 5 | 1e-4 | 0.8257 | 0.0947 | 15 | selected (registered) |
| ssl_imnet_matched | 5 | 3e-4 | 0.7968 | 0.1156 | 15 |  |
| ssl_imnet_matched | 10 | 1e-3 | 0.8024 | 0.0510 | 15 |  |
| ssl_imnet_matched | 10 | 1e-4 | 0.9111 | 0.0401 | 15 | selected (registered) |
| ssl_imnet_matched | 10 | 3e-4 | 0.8581 | 0.0505 | 15 |  |
| ssl_imnet_matched | 25 | 1e-3 | 0.8657 | 0.0721 | 15 |  |
| ssl_imnet_matched | 25 | 1e-4 | 0.9473 | 0.0365 | 15 | selected (registered) |
| ssl_imnet_matched | 25 | 3e-4 | 0.9447 | 0.0341 | 15 |  |
| ssl_imnet_matched | 50 | 1e-3 | 0.9392 | 0.0447 | 15 |  |
| ssl_imnet_matched | 50 | 1e-4 | 0.9875 | 0.0135 | 15 | (registered) |
| ssl_imnet_matched | 50 | 3e-4 | 0.9887 | 0.0117 | 15 | selected |
| ssl_imnet_matched | 100 | 1e-3 | 0.9942 | 0.0052 | 15 |  |
| ssl_imnet_matched | 100 | 1e-4 | 0.9987 | 0.0013 | 15 | selected (registered) |
| ssl_imnet_matched | 100 | 3e-4 | 0.9964 | 0.0043 | 15 |  |
| sup_imnet | 5 | 1e-3 | 0.6281 | 0.1284 | 15 |  |
| sup_imnet | 5 | 1e-4 | 0.7590 | 0.0960 | 15 |  |
| sup_imnet | 5 | 3e-4 | 0.7683 | 0.0968 | 15 | selected (registered) |
| sup_imnet | 10 | 1e-3 | 0.6783 | 0.1088 | 15 |  |
| sup_imnet | 10 | 1e-4 | 0.8441 | 0.0656 | 15 | selected |
| sup_imnet | 10 | 3e-4 | 0.7726 | 0.0686 | 15 | (registered) |
| sup_imnet | 25 | 1e-3 | 0.8021 | 0.0675 | 15 |  |
| sup_imnet | 25 | 1e-4 | 0.9297 | 0.0473 | 15 | selected |
| sup_imnet | 25 | 3e-4 | 0.8659 | 0.0547 | 15 | (registered) |
| sup_imnet | 50 | 1e-3 | 0.8246 | 0.0750 | 15 |  |
| sup_imnet | 50 | 1e-4 | 0.9452 | 0.0368 | 15 | selected |
| sup_imnet | 50 | 3e-4 | 0.9426 | 0.0330 | 15 | (registered) |
| sup_imnet | 100 | 1e-3 | 0.9016 | 0.0872 | 15 |  |
| sup_imnet | 100 | 1e-4 | 0.9676 | 0.0252 | 15 |  |
| sup_imnet | 100 | 3e-4 | 0.9774 | 0.0192 | 15 | selected (registered) |

### Contrasts (AUPRC on the SUN test split)

| comparison | budget | metric | selected_estimate | selected_t_ci_lo | selected_t_ci_hi | selected_boot_lo | selected_boot_hi | registered_estimate | registered_t_ci_lo | registered_t_ci_hi | diff_estimate | diff_t_ci_lo | diff_t_ci_hi | diff_boot_lo | diff_boot_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssl_colon - sup_imnet | AULC | AULC(auprc) | 0.0287 | 0.0197 | 0.0376 | 0.0155 | 0.0434 | 0.0475 | 0.0286 | 0.0663 | -0.0188 | -0.0308 | -0.0068 | -0.0297 | -0.0065 |
| ssl_colon - sup_imnet | p5 | auprc | 0.1197 | 0.0216 | 0.2178 | 0.0794 | 0.1529 | 0.1197 | 0.0216 | 0.2178 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - sup_imnet | p10 | auprc | 0.0437 | -0.0096 | 0.0970 | 0.0107 | 0.0765 | 0.1379 | 0.0635 | 0.2122 | -0.0942 | -0.1828 | -0.0055 | -0.1314 | -0.0473 |
| ssl_colon - sup_imnet | p25 | auprc | 0.0042 | -0.0182 | 0.0266 | -0.0194 | 0.0268 | 0.0453 | 0.0086 | 0.0820 | -0.0411 | -0.0787 | -0.0035 | -0.0691 | -0.0124 |
| ssl_colon - sup_imnet | p50 | auprc | 0.0377 | 0.0208 | 0.0547 | 0.0220 | 0.0555 | 0.0384 | 0.0138 | 0.0630 | -6.37e-04 | -0.0176 | 0.0163 | -0.0130 | 0.0129 |
| ssl_colon - sup_imnet | p100 | auprc | 0.0195 | 0.0057 | 0.0332 | 0.0111 | 0.0290 | 0.0195 | 0.0057 | 0.0332 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | AULC | AULC(auprc) | -0.0011 | -0.0163 | 0.0141 | -0.0102 | 0.0086 | 0.0104 | -0.0038 | 0.0246 | -0.0115 | -0.0167 | -0.0063 | -0.0160 | -0.0070 |
| ssl_colon - ssl_imnet | p5 | auprc | 0.1247 | 0.0164 | 0.2331 | 0.0875 | 0.1565 | 0.1660 | 0.1203 | 0.2118 | -0.0413 | -0.1496 | 0.0670 | -0.0747 | -0.0104 |
| ssl_colon - ssl_imnet | p10 | auprc | 0.0271 | -0.0078 | 0.0620 | -0.0010 | 0.0551 | 0.1002 | 0.0416 | 0.1589 | -0.0731 | -0.1184 | -0.0278 | -0.0976 | -0.0505 |
| ssl_colon - ssl_imnet | p25 | auprc | -0.0224 | -0.0661 | 0.0213 | -0.0459 | -0.0016 | -0.0095 | -0.0375 | 0.0186 | -0.0129 | -0.0412 | 0.0153 | -0.0258 | -1.03e-04 |
| ssl_colon - ssl_imnet | p50 | auprc | -0.0052 | -0.0234 | 0.0131 | -0.0169 | 0.0065 | -0.0052 | -0.0234 | 0.0131 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | p100 | auprc | -0.0020 | -0.0073 | 0.0033 | -0.0063 | 0.0014 | -0.0020 | -0.0073 | 0.0033 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | AULC | AULC(auprc) | -0.0146 | -0.0282 | -0.0011 | -0.0220 | -0.0076 | -0.0138 | -0.0285 | 8.67e-04 | -8.02e-04 | -0.0040 | 0.0024 | -0.0026 | 7.31e-04 |
| ssl_colon - ssl_imnet_matched | p5 | auprc | 0.0383 | -0.0062 | 0.0828 | 0.0146 | 0.0641 | 0.0383 | -0.0062 | 0.0828 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p10 | auprc | -0.0148 | -0.0507 | 0.0211 | -0.0340 | 0.0060 | -0.0148 | -0.0507 | 0.0211 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p25 | auprc | -0.0355 | -0.0825 | 0.0116 | -0.0577 | -0.0155 | -0.0355 | -0.0825 | 0.0116 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p50 | auprc | -0.0152 | -0.0287 | -0.0017 | -0.0251 | -0.0070 | -0.0132 | -0.0269 | 4.99e-04 | -0.0020 | -0.0101 | 0.0060 | -0.0065 | 0.0019 |
| ssl_colon - ssl_imnet_matched | p100 | auprc | -0.0024 | -0.0074 | 0.0026 | -0.0053 | -2.01e-05 | -0.0024 | -0.0074 | 0.0026 | 0.0000 |  |  | 0.0000 | 0.0000 |

Registered values recomputed: 108/108 equal to results_v3/analysis/contrast_summary.csv (max |diff| 3.94e-11).
