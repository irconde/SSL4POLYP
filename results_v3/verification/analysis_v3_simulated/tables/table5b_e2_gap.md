### E2: AUROC gap flat − polypoid per regime and between regimes

| regime | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sup_imnet | flat - polypoid | auroc | -0.1306 | 0.0301 | 0.0305 | 15 | 5 | partition_means_t | -0.1306 | -0.1680 | -0.0931 | 6.35e-04 | -0.1673 | -0.0938 | 2000 |
| ssl_imnet | flat - polypoid | auroc | -0.0833 | 0.0119 | 0.0234 | 15 | 5 | partition_means_t | -0.0833 | -0.0981 | -0.0685 | 9.80e-05 | -0.1123 | -0.0544 | 2000 |
| ssl_colon | flat - polypoid | auroc | -0.0689 | 0.0140 | 0.0209 | 15 | 5 | partition_means_t | -0.0689 | -0.0864 | -0.0515 | 3.90e-04 | -0.0943 | -0.0438 | 2000 |
| ssl_imnet_matched | flat - polypoid | auroc | -0.0887 | 0.0278 | 0.0260 | 15 | 5 | partition_means_t | -0.0887 | -0.1231 | -0.0542 | 0.0020 | -0.1195 | -0.0586 | 2000 |
|  | gap ssl_colon - gap sup_imnet | auroc | 0.0616 | 0.0187 | 0.0275 | 15 | 5 | partition_means_t | 0.0616 | 0.0384 | 0.0849 | 0.0018 | 0.0478 | 0.0763 | 2000 |
|  | gap ssl_colon - gap ssl_imnet | auroc | 0.0143 | 0.0135 | 0.0190 | 15 | 5 | partition_means_t | 0.0143 | -0.0024 | 0.0311 | 0.0761 | 0.0063 | 0.0230 | 2000 |
|  | gap ssl_colon - gap ssl_imnet_matched | auroc | 0.0197 | 0.0206 | 0.0235 | 15 | 5 | partition_means_t | 0.0197 | -0.0058 | 0.0453 | 0.0984 | 0.0101 | 0.0295 | 2000 |
