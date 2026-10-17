### Sensitivity: H1–H4 on partition k = 1 only; H6 and H7 without s500

| id | analysis | hypothesis | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1/H1 | k = 1 only | H1 | ssl_colon - sup_imnet | AULC(auprc) | 0.0434 |  | 0.0140 | 3 | 1 | t_seeds_K1 | 0.0434 | 0.0087 | 0.0782 | 0.0328 | 0.0277 | 0.0613 | 2000 |
| k1/H2 | k = 1 only | H2 | ssl_colon - ssl_imnet_matched | AULC(auprc) | 0.0379 |  | 0.0025 | 3 | 1 | t_seeds_K1 | 0.0379 | 0.0316 | 0.0442 | 0.0015 | 0.0222 | 0.0540 | 2000 |
| k1/H3 | k = 1 only | H3 | ssl_colon - sup_imnet | auroc | 0.0928 |  | 0.0048 | 3 | 1 | t_seeds_K1 | 0.0928 | 0.0809 | 0.1046 | 8.83e-04 | 0.0650 | 0.1200 | 2000 |
| k1/H4 | k = 1 only | H4 | ssl_colon - sup_imnet | auroc | 0.0531 |  | 0.0211 | 3 | 1 | t_seeds_K1 | 0.0531 | 6.26e-04 | 0.1056 | 0.0489 | 0.0375 | 0.0690 | 2000 |
| no_s500/H6 | without s500 | H6 | arm B - arm A | balanced_accuracy | 0.0466 | 0.0137 | 0.0071 | 15 | 5 | partition_means_t | 0.0466 | 0.0295 | 0.0636 | 0.0016 | 0.0434 | 0.0499 | 2000 |
| no_s500/H7 | without s500 | H7 | arm C - arm A | balanced_accuracy | 0.0817 | 0.0091 | 0.0100 | 15 | 5 | partition_means_t | 0.0817 | 0.0703 | 0.0930 | 3.68e-05 | 0.0790 | 0.0856 | 2000 |
