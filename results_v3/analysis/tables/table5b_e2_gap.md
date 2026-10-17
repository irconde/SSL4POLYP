### E2: AUROC gap flat − polypoid per regime and between regimes

| regime | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sup_imnet | flat - polypoid | auroc | -0.0328 | 0.0305 | 0.0229 | 15 | 5 | partition_means_t | -0.0328 | -0.0706 | 0.0051 | 0.0742 | -0.0598 | -0.0115 | 2000 |
| ssl_imnet | flat - polypoid | auroc | -0.0023 | 0.0023 | 0.0017 | 15 | 5 | partition_means_t | -0.0023 | -0.0051 | 5.73e-04 | 0.0907 | -0.0056 | 3.65e-04 | 2000 |
| ssl_colon | flat - polypoid | auroc | -0.0087 | 0.0165 | 0.0078 | 15 | 5 | partition_means_t | -0.0087 | -0.0292 | 0.0118 | 0.3030 | -0.0228 | -6.16e-04 | 2000 |
| ssl_imnet_matched | flat - polypoid | auroc | -0.0048 | 0.0053 | 0.0044 | 15 | 5 | partition_means_t | -0.0048 | -0.0114 | 0.0018 | 0.1127 | -0.0119 | -9.77e-04 | 2000 |
|  | gap ssl_colon - gap sup_imnet | auroc | 0.0241 | 0.0184 | 0.0239 | 15 | 5 | partition_means_t | 0.0241 | 0.0012 | 0.0469 | 0.0431 | 0.0069 | 0.0437 | 2000 |
|  | gap ssl_colon - gap ssl_imnet | auroc | -0.0064 | 0.0147 | 0.0088 | 15 | 5 | partition_means_t | -0.0064 | -0.0247 | 0.0118 | 0.3828 | -0.0186 | 0.0011 | 2000 |
|  | gap ssl_colon - gap ssl_imnet_matched | auroc | -0.0039 | 0.0123 | 0.0081 | 15 | 5 | partition_means_t | -0.0039 | -0.0192 | 0.0114 | 0.5164 | -0.0124 | 0.0020 | 2000 |
