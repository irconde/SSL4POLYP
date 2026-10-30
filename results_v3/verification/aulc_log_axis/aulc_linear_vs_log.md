# E1 AULC on a linear (registered) vs logarithmic budget axis

Post hoc sensitivity analysis, not registered. Estimate = mean of the paired per-run values (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, the registered draws).

### Weights per budget

| budget_percent | weight_linear | weight_log |
|---|---|---|
| 5 | 0.0263 | 0.1157 |
| 10 | 0.1053 | 0.2686 |
| 25 | 0.2105 | 0.2686 |
| 50 | 0.3947 | 0.2314 |
| 100 | 0.2632 | 0.1157 |

### AULC differences

| comparison | metric | registered_as | linear_estimate | linear_t_ci_lo | linear_t_ci_hi | linear_boot_lo | linear_boot_hi | log_estimate | log_t_ci_lo | log_t_ci_hi | log_boot_lo | log_boot_hi | log_minus_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssl_colon - sup_imnet | AULC(auprc) | H1 | 0.0475 | 0.0286 | 0.0663 | 0.0355 | 0.0592 | 0.0742 | 0.0408 | 0.1075 | 0.0554 | 0.0908 | 0.0267 |
| ssl_colon - ssl_imnet | AULC(auprc) | aulc/ssl_colon-ssl_imnet/auprc | 0.0104 | -0.0038 | 0.0246 | 0.0013 | 0.0197 | 0.0422 | 0.0156 | 0.0688 | 0.0280 | 0.0561 | 0.0318 |
| ssl_colon - ssl_imnet_matched | AULC(auprc) | H2 | -0.0138 | -0.0285 | 8.67e-04 | -0.0214 | -0.0064 | -0.0124 | -0.0332 | 0.0084 | -0.0237 | -0.0011 | 0.0014 |
| ssl_colon - sup_imnet | AULC(auroc) | aulc/ssl_colon-sup_imnet/auroc | 0.0435 | 0.0277 | 0.0593 | 0.0315 | 0.0569 | 0.0578 | 0.0340 | 0.0817 | 0.0408 | 0.0762 | 0.0143 |
| ssl_colon - ssl_imnet | AULC(auroc) | aulc/ssl_colon-ssl_imnet/auroc | 0.0089 | -0.0031 | 0.0209 | 5.54e-05 | 0.0186 | 0.0365 | 0.0145 | 0.0585 | 0.0234 | 0.0512 | 0.0276 |
| ssl_colon - ssl_imnet_matched | AULC(auroc) | aulc/ssl_colon-ssl_imnet_matched/auroc | -0.0150 | -0.0321 | 0.0020 | -0.0234 | -0.0071 | -0.0147 | -0.0406 | 0.0111 | -0.0270 | -0.0017 | 3.04e-04 |

Linear AULC recomputed: 48/48 quantities equal to results_v3/analysis/contrast_summary.csv (max |diff| 4.3e-12).
