# Exploratory: learning rate selected per E1 budget vs registered

Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run differences (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345). selected: each regime at the learning rate selected for each budget on validation AP; registered: each regime at its registered learning rate; diff = selected − registered.

### Mean validation AP (best-epoch weights) by regime, budget and learning rate

| regime | budget | lr | val_ap_mean | val_ap_sd | n_runs | mark |
|---|---|---|---|---|---|---|
| ssl_colon | 5 | 1e-4 | 0.8370 | 0.0828 | 15 | selected (registered) |
| ssl_colon | 10 | 1e-4 | 0.8625 | 0.0503 | 15 | selected (registered) |
| ssl_colon | 25 | 1e-4 | 0.9046 | 0.0542 | 15 | selected (registered) |
| ssl_colon | 50 | 1e-4 | 0.9711 | 0.0267 | 15 | selected (registered) |
| ssl_colon | 100 | 1e-4 | 0.9956 | 0.0036 | 15 | selected (registered) |
| ssl_imnet | 5 | 3e-4 | 0.7165 | 0.0836 | 15 | selected (registered) |
| ssl_imnet | 10 | 3e-4 | 0.7926 | 0.0721 | 15 | selected (registered) |
| ssl_imnet | 25 | 3e-4 | 0.9372 | 0.0383 | 15 | selected (registered) |
| ssl_imnet | 50 | 3e-4 | 0.9900 | 0.0081 | 15 | selected (registered) |
| ssl_imnet | 100 | 3e-4 | 0.9986 | 0.0021 | 15 | selected (registered) |
| ssl_imnet_matched | 5 | 1e-4 | 0.8257 | 0.0947 | 15 | selected (registered) |
| ssl_imnet_matched | 10 | 1e-4 | 0.9111 | 0.0401 | 15 | selected (registered) |
| ssl_imnet_matched | 25 | 1e-4 | 0.9473 | 0.0365 | 15 | selected (registered) |
| ssl_imnet_matched | 50 | 1e-4 | 0.9875 | 0.0135 | 15 | selected (registered) |
| ssl_imnet_matched | 100 | 1e-4 | 0.9987 | 0.0013 | 15 | selected (registered) |
| sup_imnet | 5 | 3e-4 | 0.7683 | 0.0968 | 15 | selected (registered) |
| sup_imnet | 10 | 3e-4 | 0.7726 | 0.0686 | 15 | selected (registered) |
| sup_imnet | 25 | 3e-4 | 0.8659 | 0.0547 | 15 | selected (registered) |
| sup_imnet | 50 | 3e-4 | 0.9426 | 0.0330 | 15 | selected (registered) |
| sup_imnet | 100 | 3e-4 | 0.9774 | 0.0192 | 15 | selected (registered) |

### Contrasts (AUPRC on the SUN test split)

| comparison | budget | metric | selected_estimate | selected_t_ci_lo | selected_t_ci_hi | selected_boot_lo | selected_boot_hi | registered_estimate | registered_t_ci_lo | registered_t_ci_hi | diff_estimate | diff_t_ci_lo | diff_t_ci_hi | diff_boot_lo | diff_boot_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssl_colon - sup_imnet | AULC | AULC(auprc) | 0.0475 | 0.0286 | 0.0663 | 0.0355 | 0.0592 | 0.0475 | 0.0286 | 0.0663 | 5.55e-18 | -9.86e-18 | 2.10e-17 | -1.48e-17 | 1.48e-17 |
| ssl_colon - sup_imnet | p5 | auprc | 0.1197 | 0.0216 | 0.2178 | 0.0794 | 0.1529 | 0.1197 | 0.0216 | 0.2178 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - sup_imnet | p10 | auprc | 0.1379 | 0.0635 | 0.2122 | 0.1005 | 0.1691 | 0.1379 | 0.0635 | 0.2122 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - sup_imnet | p25 | auprc | 0.0453 | 0.0086 | 0.0820 | 0.0241 | 0.0646 | 0.0453 | 0.0086 | 0.0820 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - sup_imnet | p50 | auprc | 0.0384 | 0.0138 | 0.0630 | 0.0236 | 0.0543 | 0.0384 | 0.0138 | 0.0630 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - sup_imnet | p100 | auprc | 0.0195 | 0.0057 | 0.0332 | 0.0111 | 0.0290 | 0.0195 | 0.0057 | 0.0332 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | AULC | AULC(auprc) | 0.0104 | -0.0038 | 0.0246 | 0.0013 | 0.0197 | 0.0104 | -0.0038 | 0.0246 | 3.70e-18 | -6.57e-18 | 1.40e-17 | -1.48e-17 | 1.48e-17 |
| ssl_colon - ssl_imnet | p5 | auprc | 0.1660 | 0.1203 | 0.2118 | 0.1291 | 0.1989 | 0.1660 | 0.1203 | 0.2118 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | p10 | auprc | 0.1002 | 0.0416 | 0.1589 | 0.0703 | 0.1291 | 0.1002 | 0.0416 | 0.1589 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | p25 | auprc | -0.0095 | -0.0375 | 0.0186 | -0.0321 | 0.0113 | -0.0095 | -0.0375 | 0.0186 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | p50 | auprc | -0.0052 | -0.0234 | 0.0131 | -0.0169 | 0.0065 | -0.0052 | -0.0234 | 0.0131 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet | p100 | auprc | -0.0020 | -0.0073 | 0.0033 | -0.0063 | 0.0014 | -0.0020 | -0.0073 | 0.0033 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | AULC | AULC(auprc) | -0.0138 | -0.0285 | 8.67e-04 | -0.0214 | -0.0064 | -0.0138 | -0.0285 | 8.67e-04 | -7.40e-18 | -2.00e-17 | 5.18e-18 | -1.48e-17 | 1.48e-17 |
| ssl_colon - ssl_imnet_matched | p5 | auprc | 0.0383 | -0.0062 | 0.0828 | 0.0146 | 0.0641 | 0.0383 | -0.0062 | 0.0828 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p10 | auprc | -0.0148 | -0.0507 | 0.0211 | -0.0340 | 0.0060 | -0.0148 | -0.0507 | 0.0211 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p25 | auprc | -0.0355 | -0.0825 | 0.0116 | -0.0577 | -0.0155 | -0.0355 | -0.0825 | 0.0116 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p50 | auprc | -0.0132 | -0.0269 | 4.99e-04 | -0.0220 | -0.0057 | -0.0132 | -0.0269 | 4.99e-04 | 0.0000 |  |  | 0.0000 | 0.0000 |
| ssl_colon - ssl_imnet_matched | p100 | auprc | -0.0024 | -0.0074 | 0.0026 | -0.0053 | -2.01e-05 | -0.0024 | -0.0074 | 0.0026 | 0.0000 |  |  | 0.0000 | 0.0000 |

Registered values recomputed: 108/108 equal to results_v3/analysis/contrast_summary.csv (max |diff| 3.94e-11).
