### Sensitivity: H1–H4 on partition k = 1 only; H6 and H7 without s500

| id | analysis | hypothesis | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1/H1 | k = 1 only | H1 | ssl_colon - sup_imnet | AULC(auprc) | 0.0375 |  | 0.0210 | 3 | 1 | t_seeds_K1 | 0.0375 | -0.0147 | 0.0897 | 0.0907 | 0.0184 | 0.0570 | 2000 |
| k1/H2 | k = 1 only | H2 | ssl_colon - ssl_imnet_matched | AULC(auprc) | -0.0273 |  | 0.0021 | 3 | 1 | t_seeds_K1 | -0.0273 | -0.0325 | -0.0221 | 0.0019 | -0.0513 | -0.0101 | 2000 |
| k1/H3 | k = 1 only | H3 | ssl_colon - sup_imnet | auroc | 0.0042 |  | 0.0164 | 3 | 1 | t_seeds_K1 | 0.0042 | -0.0367 | 0.0450 | 0.7033 | -0.0125 | 0.0151 | 2000 |
| k1/H4 | k = 1 only | H4 | ssl_colon - sup_imnet | auroc | -0.0630 |  | 0.0153 | 3 | 1 | t_seeds_K1 | -0.0630 | -0.1011 | -0.0249 | 0.0192 | -0.3547 | 0.1148 | 2000 |
| no_s500/H6 | without s500 | H6 | arm B - arm A | balanced_accuracy | -0.0085 | 0.0144 | 0.0221 | 15 | 5 | partition_means_t | -0.0085 | -0.0264 | 0.0095 | 0.2600 | -0.0230 | 0.0059 | 2000 |
| no_s500/H7 | without s500 | H7 | arm C - arm A | balanced_accuracy | -0.1010 | 0.0197 | 0.0525 | 15 | 5 | partition_means_t | -0.1010 | -0.1254 | -0.0765 | 3.30e-04 | -0.1608 | -0.0338 | 2000 |
