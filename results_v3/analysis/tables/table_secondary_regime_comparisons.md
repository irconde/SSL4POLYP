### Secondary: SSL-colon vs each comparator by experiment, budget, stratum, arm and metric

Descriptive (no multiplicity correction). Markdown shows a subset of metrics; the CSV has all.

| experiment | variant | slot | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sun_full |  | all | ssl_colon - sup_imnet | auroc | 0.0220 | 0.0117 | 0.0171 | 15 | 5 | partition_means_t | 0.0220 | 0.0074 | 0.0365 | 0.0138 | 0.0127 | 0.0328 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | auprc | 0.0195 | 0.0111 | 0.0153 | 15 | 5 | partition_means_t | 0.0195 | 0.0057 | 0.0332 | 0.0171 | 0.0111 | 0.0290 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0509 | 0.0375 | 0.0382 | 15 | 5 | partition_means_t | 0.0509 | 0.0044 | 0.0974 | 0.0385 | 0.0304 | 0.0722 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | brier | -0.0405 | 0.0184 | 0.0275 | 15 | 5 | partition_means_t | -0.0405 | -0.0634 | -0.0177 | 0.0079 | -0.0552 | -0.0268 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | ece | -0.0249 | 0.0091 | 0.0283 | 15 | 5 | partition_means_t | -0.0249 | -0.0362 | -0.0136 | 0.0036 | -0.0410 | -0.0144 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0178 | 0.0186 | 0.0298 | 15 | 5 | partition_means_t | 0.0178 | -0.0053 | 0.0409 | 0.0993 | 0.0044 | 0.0356 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | auroc | -0.0025 | 0.0053 | 0.0099 | 15 | 5 | partition_means_t | -0.0025 | -0.0091 | 0.0041 | 0.3515 | -0.0073 | 0.0013 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | auprc | -0.0020 | 0.0043 | 0.0092 | 15 | 5 | partition_means_t | -0.0020 | -0.0073 | 0.0033 | 0.3499 | -0.0063 | 0.0014 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | balanced_accuracy | 6.73e-05 | 0.0203 | 0.0222 | 15 | 5 | partition_means_t | 6.73e-05 | -0.0252 | 0.0253 | 0.9944 | -0.0150 | 0.0170 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | brier | 5.64e-04 | 0.0187 | 0.0216 | 15 | 5 | partition_means_t | 5.64e-04 | -0.0227 | 0.0238 | 0.9495 | -0.0126 | 0.0119 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | ece | -0.0047 | 0.0259 | 0.0222 | 15 | 5 | partition_means_t | -0.0047 | -0.0368 | 0.0275 | 0.7082 | -0.0187 | 0.0117 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | auroc | -0.0029 | 0.0045 | 0.0065 | 15 | 5 | partition_means_t | -0.0029 | -0.0085 | 0.0027 | 0.2249 | -0.0065 | -2.43e-04 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | auprc | -0.0024 | 0.0040 | 0.0058 | 15 | 5 | partition_means_t | -0.0024 | -0.0074 | 0.0026 | 0.2591 | -0.0053 | -2.01e-05 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0083 | 0.0224 | 0.0229 | 15 | 5 | partition_means_t | -0.0083 | -0.0361 | 0.0195 | 0.4542 | -0.0185 | 0.0011 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | brier | 0.0095 | 0.0156 | 0.0186 | 15 | 5 | partition_means_t | 0.0095 | -0.0099 | 0.0289 | 0.2446 | 0.0026 | 0.0167 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | ece | 0.0087 | 0.0100 | 0.0253 | 15 | 5 | partition_means_t | 0.0087 | -0.0037 | 0.0212 | 0.1242 | 0.0013 | 0.0167 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | auroc | 0.0797 | 0.0549 | 0.0521 | 15 | 5 | partition_means_t | 0.0797 | 0.0115 | 0.1478 | 0.0315 | 0.0450 | 0.1155 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | auprc | 0.1197 | 0.0790 | 0.0534 | 15 | 5 | partition_means_t | 0.1197 | 0.0216 | 0.2178 | 0.0275 | 0.0794 | 0.1529 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0399 | 0.0451 | 0.0679 | 15 | 5 | partition_means_t | 0.0399 | -0.0161 | 0.0958 | 0.1190 | 0.0082 | 0.0696 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | brier | -0.0370 | 0.0218 | 0.0299 | 15 | 5 | partition_means_t | -0.0370 | -0.0641 | -0.0099 | 0.0192 | -0.0514 | -0.0215 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | ece | -0.0099 | 0.0401 | 0.0442 | 15 | 5 | partition_means_t | -0.0099 | -0.0597 | 0.0399 | 0.6104 | -0.0348 | 0.0061 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | -0.0044 | 0.0099 | 0.0172 | 15 | 5 | partition_means_t | -0.0044 | -0.0168 | 0.0079 | 0.3739 | -0.0133 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | auroc | 0.1417 | 0.0432 | 0.0923 | 15 | 5 | partition_means_t | 0.1417 | 0.0880 | 0.1954 | 0.0018 | 0.1085 | 0.1769 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | auprc | 0.1660 | 0.0368 | 0.1080 | 15 | 5 | partition_means_t | 0.1660 | 0.1203 | 0.2118 | 5.45e-04 | 0.1291 | 0.1989 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.1164 | 0.0676 | 0.0837 | 15 | 5 | partition_means_t | 0.1164 | 0.0324 | 0.2003 | 0.0183 | 0.0922 | 0.1416 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | brier | -0.0627 | 0.0239 | 0.0116 | 15 | 5 | partition_means_t | -0.0627 | -0.0924 | -0.0331 | 0.0042 | -0.0746 | -0.0504 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | ece | -0.0169 | 0.0661 | 0.0673 | 15 | 5 | partition_means_t | -0.0169 | -0.0989 | 0.0652 | 0.5982 | -0.0421 | -0.0043 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | -0.0044 | 0.0099 | 0.0172 | 15 | 5 | partition_means_t | -0.0044 | -0.0168 | 0.0079 | 0.3739 | -0.0133 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | auroc | 0.0302 | 0.0405 | 0.0468 | 15 | 5 | partition_means_t | 0.0302 | -0.0201 | 0.0806 | 0.1705 | 0.0080 | 0.0547 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | auprc | 0.0383 | 0.0359 | 0.0493 | 15 | 5 | partition_means_t | 0.0383 | -0.0062 | 0.0828 | 0.0754 | 0.0146 | 0.0641 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0139 | 0.0787 | 0.0647 | 15 | 5 | partition_means_t | 0.0139 | -0.0839 | 0.1116 | 0.7136 | -0.0100 | 0.0370 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | brier | -0.0160 | 0.0199 | 0.0197 | 15 | 5 | partition_means_t | -0.0160 | -0.0407 | 0.0087 | 0.1460 | -0.0268 | -0.0053 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | ece | -0.0036 | 0.0401 | 0.0412 | 15 | 5 | partition_means_t | -0.0036 | -0.0534 | 0.0463 | 0.8514 | -0.0250 | 0.0117 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0089 | 0.0199 | 0.0172 | 15 | 5 | partition_means_t | 0.0089 | -0.0158 | 0.0336 | 0.3739 | 0.0000 | 0.0222 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | auroc | 0.0954 | 0.0443 | 0.0801 | 15 | 5 | partition_means_t | 0.0954 | 0.0403 | 0.1504 | 0.0086 | 0.0639 | 0.1287 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | auprc | 0.1379 | 0.0599 | 0.0921 | 15 | 5 | partition_means_t | 0.1379 | 0.0635 | 0.2122 | 0.0068 | 0.1005 | 0.1691 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0685 | 0.0547 | 0.0667 | 15 | 5 | partition_means_t | 0.0685 | 5.89e-04 | 0.1364 | 0.0488 | 0.0397 | 0.0983 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | brier | -0.0509 | 0.0124 | 0.0575 | 15 | 5 | partition_means_t | -0.0509 | -0.0663 | -0.0355 | 7.84e-04 | -0.0676 | -0.0335 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | ece | -0.0210 | 0.0640 | 0.0668 | 15 | 5 | partition_means_t | -0.0210 | -0.1004 | 0.0584 | 0.5037 | -0.0501 | -0.0017 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0157 | 0.0385 | 15 | 5 | partition_means_t | 0.0000 | -0.0195 | 0.0195 | 1 | -0.0222 | 0.0222 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | auroc | 0.0869 | 0.0411 | 0.0340 | 15 | 5 | partition_means_t | 0.0869 | 0.0359 | 0.1380 | 0.0091 | 0.0603 | 0.1164 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | auprc | 0.1002 | 0.0472 | 0.0375 | 15 | 5 | partition_means_t | 0.1002 | 0.0416 | 0.1589 | 0.0090 | 0.0703 | 0.1291 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0750 | 0.0418 | 0.0503 | 15 | 5 | partition_means_t | 0.0750 | 0.0232 | 0.1269 | 0.0159 | 0.0532 | 0.0979 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | brier | -0.0467 | 0.0196 | 0.0274 | 15 | 5 | partition_means_t | -0.0467 | -0.0710 | -0.0223 | 0.0060 | -0.0604 | -0.0328 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | ece | -0.0238 | 0.0311 | 0.0651 | 15 | 5 | partition_means_t | -0.0238 | -0.0624 | 0.0148 | 0.1624 | -0.0435 | -0.0062 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0089 | 0.0122 | 0.0385 | 15 | 5 | partition_means_t | 0.0089 | -0.0062 | 0.0240 | 0.1778 | -0.0089 | 0.0268 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | auroc | -0.0207 | 0.0293 | 0.0447 | 15 | 5 | partition_means_t | -0.0207 | -0.0571 | 0.0157 | 0.1893 | -0.0423 | 0.0023 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | auprc | -0.0148 | 0.0289 | 0.0488 | 15 | 5 | partition_means_t | -0.0148 | -0.0507 | 0.0211 | 0.3159 | -0.0340 | 0.0060 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0424 | 0.0364 | 0.0539 | 15 | 5 | partition_means_t | -0.0424 | -0.0875 | 0.0028 | 0.0598 | -0.0699 | -0.0141 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | brier | 0.0168 | 0.0150 | 0.0183 | 15 | 5 | partition_means_t | 0.0168 | -0.0018 | 0.0354 | 0.0658 | -0.0014 | 0.0330 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | ece | 0.0086 | 0.0425 | 0.0329 | 15 | 5 | partition_means_t | 0.0086 | -0.0442 | 0.0614 | 0.6742 | -0.0107 | 0.0279 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0311 | 0.0696 | 0.0344 | 15 | 5 | partition_means_t | 0.0311 | -0.0553 | 0.1175 | 0.3739 | 0.0000 | 0.0623 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0356 | 0.0951 | 0.0243 | 15 | 5 | partition_means_t | 0.0356 | -0.0825 | 0.1536 | 0.4500 | -0.0044 | 0.0844 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | auroc | 0.0376 | 0.0268 | 0.0367 | 15 | 5 | partition_means_t | 0.0376 | 0.0043 | 0.0709 | 0.0351 | 0.0186 | 0.0575 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | auprc | 0.0453 | 0.0296 | 0.0486 | 15 | 5 | partition_means_t | 0.0453 | 0.0086 | 0.0820 | 0.0267 | 0.0241 | 0.0646 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0406 | 0.0263 | 0.0499 | 15 | 5 | partition_means_t | 0.0406 | 0.0080 | 0.0733 | 0.0260 | 0.0176 | 0.0642 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | brier | -0.0087 | 0.0143 | 0.0249 | 15 | 5 | partition_means_t | -0.0087 | -0.0265 | 0.0090 | 0.2434 | -0.0247 | 0.0073 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | ece | 0.0380 | 0.0169 | 0.0542 | 15 | 5 | partition_means_t | 0.0380 | 0.0170 | 0.0590 | 0.0073 | 0.0121 | 0.0525 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0044 | 0.0099 | 0.0172 | 15 | 5 | partition_means_t | 0.0044 | -0.0079 | 0.0168 | 0.3739 | 0.0000 | 0.0133 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0157 | 0.0385 | 15 | 5 | partition_means_t | 0.0000 | -0.0195 | 0.0195 | 1 | -0.0178 | 0.0178 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | auroc | -0.0070 | 0.0221 | 0.0346 | 15 | 5 | partition_means_t | -0.0070 | -0.0344 | 0.0204 | 0.5163 | -0.0280 | 0.0122 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | auprc | -0.0095 | 0.0226 | 0.0362 | 15 | 5 | partition_means_t | -0.0095 | -0.0375 | 0.0186 | 0.4019 | -0.0321 | 0.0113 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | balanced_accuracy | -0.0168 | 0.0277 | 0.0447 | 15 | 5 | partition_means_t | -0.0168 | -0.0512 | 0.0176 | 0.2474 | -0.0418 | 0.0071 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | brier | 0.0132 | 0.0093 | 0.0460 | 15 | 5 | partition_means_t | 0.0132 | 0.0016 | 0.0248 | 0.0343 | -0.0035 | 0.0296 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | ece | 0.0252 | 0.0391 | 0.0704 | 15 | 5 | partition_means_t | 0.0252 | -0.0234 | 0.0737 | 0.2239 | 0.0047 | 0.0454 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0089 | 0.0122 | 0.0243 | 15 | 5 | partition_means_t | 0.0089 | -0.0062 | 0.0240 | 0.1778 | 0.0000 | 0.0222 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0222 | 0.0385 | 0.0710 | 15 | 5 | partition_means_t | 0.0222 | -0.0256 | 0.0700 | 0.2663 | 0.0000 | 0.0489 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | auroc | -0.0329 | 0.0440 | 0.0271 | 15 | 5 | partition_means_t | -0.0329 | -0.0876 | 0.0218 | 0.1704 | -0.0549 | -0.0115 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | auprc | -0.0355 | 0.0379 | 0.0253 | 15 | 5 | partition_means_t | -0.0355 | -0.0825 | 0.0116 | 0.1045 | -0.0577 | -0.0155 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0527 | 0.0634 | 0.0363 | 15 | 5 | partition_means_t | -0.0527 | -0.1313 | 0.0260 | 0.1367 | -0.0784 | -0.0268 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | brier | 0.0429 | 0.0379 | 0.0179 | 15 | 5 | partition_means_t | 0.0429 | -0.0041 | 0.0899 | 0.0643 | 0.0221 | 0.0632 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | ece | 0.0427 | 0.0158 | 0.0239 | 15 | 5 | partition_means_t | 0.0427 | 0.0231 | 0.0623 | 0.0038 | 0.0151 | 0.0667 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0356 | 0.0678 | 0.0385 | 15 | 5 | partition_means_t | 0.0356 | -0.0486 | 0.1197 | 0.3058 | 0.0044 | 0.0711 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0356 | 0.0713 | 0.0516 | 15 | 5 | partition_means_t | 0.0356 | -0.0530 | 0.1241 | 0.3274 | 0.0044 | 0.0711 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | auroc | 0.0447 | 0.0195 | 0.0401 | 15 | 5 | partition_means_t | 0.0447 | 0.0206 | 0.0689 | 0.0068 | 0.0287 | 0.0621 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | auprc | 0.0384 | 0.0198 | 0.0361 | 15 | 5 | partition_means_t | 0.0384 | 0.0138 | 0.0630 | 0.0123 | 0.0236 | 0.0543 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0706 | 0.0387 | 0.0493 | 15 | 5 | partition_means_t | 0.0706 | 0.0225 | 0.1186 | 0.0151 | 0.0458 | 0.0939 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | brier | -0.0525 | 0.0269 | 0.0362 | 15 | 5 | partition_means_t | -0.0525 | -0.0859 | -0.0191 | 0.0121 | -0.0703 | -0.0343 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | ece | -0.0420 | 0.0261 | 0.0407 | 15 | 5 | partition_means_t | -0.0420 | -0.0744 | -0.0096 | 0.0228 | -0.0594 | -0.0236 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | -0.0044 | 0.0099 | 0.0172 | 15 | 5 | partition_means_t | -0.0044 | -0.0168 | 0.0079 | 0.3739 | -0.0133 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0157 | 0.0385 | 15 | 5 | partition_means_t | 0.0000 | -0.0195 | 0.0195 | 1 | -0.0178 | 0.0178 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | auroc | -0.0046 | 0.0163 | 0.0138 | 15 | 5 | partition_means_t | -0.0046 | -0.0249 | 0.0156 | 0.5578 | -0.0173 | 0.0080 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | auprc | -0.0052 | 0.0147 | 0.0126 | 15 | 5 | partition_means_t | -0.0052 | -0.0234 | 0.0131 | 0.4763 | -0.0169 | 0.0065 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | balanced_accuracy | -0.0146 | 0.0354 | 0.0365 | 15 | 5 | partition_means_t | -0.0146 | -0.0586 | 0.0294 | 0.4086 | -0.0382 | 0.0106 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | brier | 0.0089 | 0.0242 | 0.0217 | 15 | 5 | partition_means_t | 0.0089 | -0.0212 | 0.0389 | 0.4586 | -0.0100 | 0.0272 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | ece | 0.0024 | 0.0305 | 0.0273 | 15 | 5 | partition_means_t | 0.0024 | -0.0354 | 0.0402 | 0.8678 | -0.0167 | 0.0206 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0044 | 0.0243 | 0.0385 | 15 | 5 | partition_means_t | 0.0044 | -0.0258 | 0.0347 | 0.7040 | -0.0089 | 0.0178 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0044 | 0.0243 | 0.0487 | 15 | 5 | partition_means_t | 0.0044 | -0.0258 | 0.0347 | 0.7040 | -0.0089 | 0.0267 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | auroc | -0.0151 | 0.0117 | 0.0104 | 15 | 5 | partition_means_t | -0.0151 | -0.0296 | -6.45e-04 | 0.0441 | -0.0254 | -0.0068 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | auprc | -0.0132 | 0.0110 | 0.0098 | 15 | 5 | partition_means_t | -0.0132 | -0.0269 | 4.99e-04 | 0.0555 | -0.0220 | -0.0057 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0298 | 0.0275 | 0.0291 | 15 | 5 | partition_means_t | -0.0298 | -0.0639 | 0.0044 | 0.0728 | -0.0468 | -0.0139 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | brier | 0.0232 | 0.0198 | 0.0210 | 15 | 5 | partition_means_t | 0.0232 | -0.0014 | 0.0478 | 0.0588 | 0.0110 | 0.0361 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | ece | 0.0091 | 0.0211 | 0.0211 | 15 | 5 | partition_means_t | 0.0091 | -0.0171 | 0.0354 | 0.3888 | -0.0018 | 0.0259 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | -0.0044 | 0.0099 | 0.0172 | 15 | 5 | partition_means_t | -0.0044 | -0.0168 | 0.0079 | 0.3739 | -0.0133 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | -0.0133 | 0.0122 | 0.0298 | 15 | 5 | partition_means_t | -0.0133 | -0.0284 | 0.0018 | 0.0705 | -0.0311 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | auroc | 0.0220 | 0.0117 | 0.0171 | 15 | 5 | partition_means_t | 0.0220 | 0.0074 | 0.0365 | 0.0138 | 0.0127 | 0.0328 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | auprc | 0.0195 | 0.0111 | 0.0153 | 15 | 5 | partition_means_t | 0.0195 | 0.0057 | 0.0332 | 0.0171 | 0.0111 | 0.0290 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0509 | 0.0375 | 0.0382 | 15 | 5 | partition_means_t | 0.0509 | 0.0044 | 0.0974 | 0.0385 | 0.0304 | 0.0722 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | brier | -0.0405 | 0.0184 | 0.0275 | 15 | 5 | partition_means_t | -0.0405 | -0.0634 | -0.0177 | 0.0079 | -0.0552 | -0.0268 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | ece | -0.0249 | 0.0091 | 0.0283 | 15 | 5 | partition_means_t | -0.0249 | -0.0362 | -0.0136 | 0.0036 | -0.0410 | -0.0144 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0178 | 0.0186 | 0.0298 | 15 | 5 | partition_means_t | 0.0178 | -0.0053 | 0.0409 | 0.0993 | 0.0044 | 0.0356 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | auroc | -0.0025 | 0.0053 | 0.0099 | 15 | 5 | partition_means_t | -0.0025 | -0.0091 | 0.0041 | 0.3515 | -0.0073 | 0.0013 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | auprc | -0.0020 | 0.0043 | 0.0092 | 15 | 5 | partition_means_t | -0.0020 | -0.0073 | 0.0033 | 0.3499 | -0.0063 | 0.0014 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | balanced_accuracy | 6.73e-05 | 0.0203 | 0.0222 | 15 | 5 | partition_means_t | 6.73e-05 | -0.0252 | 0.0253 | 0.9944 | -0.0150 | 0.0170 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | brier | 5.64e-04 | 0.0187 | 0.0216 | 15 | 5 | partition_means_t | 5.64e-04 | -0.0227 | 0.0238 | 0.9495 | -0.0126 | 0.0119 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | ece | -0.0047 | 0.0259 | 0.0222 | 15 | 5 | partition_means_t | -0.0047 | -0.0368 | 0.0275 | 0.7082 | -0.0187 | 0.0117 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | auroc | -0.0029 | 0.0045 | 0.0065 | 15 | 5 | partition_means_t | -0.0029 | -0.0085 | 0.0027 | 0.2249 | -0.0065 | -2.43e-04 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | auprc | -0.0024 | 0.0040 | 0.0058 | 15 | 5 | partition_means_t | -0.0024 | -0.0074 | 0.0026 | 0.2591 | -0.0053 | -2.01e-05 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0083 | 0.0224 | 0.0229 | 15 | 5 | partition_means_t | -0.0083 | -0.0361 | 0.0195 | 0.4542 | -0.0185 | 0.0011 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | brier | 0.0095 | 0.0156 | 0.0186 | 15 | 5 | partition_means_t | 0.0095 | -0.0099 | 0.0289 | 0.2446 | 0.0026 | 0.0167 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | ece | 0.0087 | 0.0100 | 0.0253 | 15 | 5 | partition_means_t | 0.0087 | -0.0037 | 0.0212 | 0.1242 | 0.0013 | 0.0167 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | auroc | 0.0220 | 0.0117 | 0.0171 | 15 | 5 | partition_means_t | 0.0220 | 0.0074 | 0.0365 | 0.0138 | 0.0131 | 0.0319 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | auprc | 0.0195 | 0.0111 | 0.0153 | 15 | 5 | partition_means_t | 0.0195 | 0.0057 | 0.0332 | 0.0171 | 0.0113 | 0.0285 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | balanced_accuracy | 0.0509 | 0.0375 | 0.0382 | 15 | 5 | partition_means_t | 0.0509 | 0.0044 | 0.0974 | 0.0385 | 0.0308 | 0.0706 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | brier | -0.0405 | 0.0184 | 0.0275 | 15 | 5 | partition_means_t | -0.0405 | -0.0634 | -0.0177 | 0.0079 | -0.0548 | -0.0270 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | ece | -0.0249 | 0.0091 | 0.0283 | 15 | 5 | partition_means_t | -0.0249 | -0.0362 | -0.0136 | 0.0036 | -0.0408 | -0.0142 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0178 | 0.0186 | 0.0298 | 15 | 5 | partition_means_t | 0.0178 | -0.0053 | 0.0409 | 0.0993 | 0.0044 | 0.0356 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | auroc | -0.0025 | 0.0053 | 0.0099 | 15 | 5 | partition_means_t | -0.0025 | -0.0091 | 0.0041 | 0.3515 | -0.0071 | 0.0012 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | auprc | -0.0020 | 0.0043 | 0.0092 | 15 | 5 | partition_means_t | -0.0020 | -0.0073 | 0.0033 | 0.3499 | -0.0060 | 0.0011 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | balanced_accuracy | 6.73e-05 | 0.0203 | 0.0222 | 15 | 5 | partition_means_t | 6.73e-05 | -0.0252 | 0.0253 | 0.9944 | -0.0149 | 0.0173 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | brier | 5.64e-04 | 0.0187 | 0.0216 | 15 | 5 | partition_means_t | 5.64e-04 | -0.0227 | 0.0238 | 0.9495 | -0.0124 | 0.0120 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | ece | -0.0047 | 0.0259 | 0.0222 | 15 | 5 | partition_means_t | -0.0047 | -0.0368 | 0.0275 | 0.7082 | -0.0187 | 0.0107 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | auroc | -0.0029 | 0.0045 | 0.0065 | 15 | 5 | partition_means_t | -0.0029 | -0.0085 | 0.0027 | 0.2249 | -0.0064 | -4.38e-04 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | auprc | -0.0024 | 0.0040 | 0.0058 | 15 | 5 | partition_means_t | -0.0024 | -0.0074 | 0.0026 | 0.2591 | -0.0052 | -2.28e-04 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0083 | 0.0224 | 0.0229 | 15 | 5 | partition_means_t | -0.0083 | -0.0361 | 0.0195 | 0.4542 | -0.0190 | 9.43e-04 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | brier | 0.0095 | 0.0156 | 0.0186 | 15 | 5 | partition_means_t | 0.0095 | -0.0099 | 0.0289 | 0.2446 | 0.0027 | 0.0169 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | ece | 0.0087 | 0.0100 | 0.0253 | 15 | 5 | partition_means_t | 0.0087 | -0.0037 | 0.0212 | 0.1242 | 8.32e-04 | 0.0168 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | auroc | 0.0380 | 0.0208 | 0.0303 | 15 | 5 | partition_means_t | 0.0380 | 0.0122 | 0.0639 | 0.0151 | 0.0202 | 0.0593 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | auprc | 0.0781 | 0.0479 | 0.0603 | 15 | 5 | partition_means_t | 0.0781 | 0.0186 | 0.1375 | 0.0218 | 0.0406 | 0.1220 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | balanced_accuracy | 0.0729 | 0.0499 | 0.0565 | 15 | 5 | partition_means_t | 0.0729 | 0.0109 | 0.1349 | 0.0310 | 0.0445 | 0.1042 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | brier | -0.0510 | 0.0206 | 0.0382 | 15 | 5 | partition_means_t | -0.0510 | -0.0767 | -0.0254 | 0.0052 | -0.0721 | -0.0313 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | ece | -0.0432 | 0.0108 | 0.0438 | 15 | 5 | partition_means_t | -0.0432 | -0.0566 | -0.0298 | 8.66e-04 | -0.0684 | -0.0266 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0533 | 0.0558 | 0.0894 | 15 | 5 | partition_means_t | 0.0533 | -0.0159 | 0.1226 | 0.0993 | 0.0133 | 0.1067 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | auroc | -0.0068 | 0.0150 | 0.0140 | 15 | 5 | partition_means_t | -0.0068 | -0.0255 | 0.0119 | 0.3696 | -0.0189 | 0.0011 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | auprc | -0.0121 | 0.0257 | 0.0302 | 15 | 5 | partition_means_t | -0.0121 | -0.0441 | 0.0198 | 0.3519 | -0.0354 | 0.0029 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | balanced_accuracy | -0.0034 | 0.0311 | 0.0365 | 15 | 5 | partition_means_t | -0.0034 | -0.0420 | 0.0352 | 0.8205 | -0.0204 | 0.0147 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | brier | -0.0041 | 0.0329 | 0.0313 | 15 | 5 | partition_means_t | -0.0041 | -0.0450 | 0.0368 | 0.7947 | -0.0229 | 0.0119 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | ece | -0.0090 | 0.0378 | 0.0366 | 15 | 5 | partition_means_t | -0.0090 | -0.0560 | 0.0380 | 0.6242 | -0.0298 | 0.0122 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | auroc | -0.0055 | 0.0117 | 0.0112 | 15 | 5 | partition_means_t | -0.0055 | -0.0201 | 0.0091 | 0.3532 | -0.0143 | 3.24e-04 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | auprc | -0.0082 | 0.0222 | 0.0227 | 15 | 5 | partition_means_t | -0.0082 | -0.0358 | 0.0194 | 0.4574 | -0.0228 | 0.0030 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0082 | 0.0330 | 0.0290 | 15 | 5 | partition_means_t | -0.0082 | -0.0492 | 0.0327 | 0.6072 | -0.0242 | 0.0060 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | brier | 0.0067 | 0.0189 | 0.0264 | 15 | 5 | partition_means_t | 0.0067 | -0.0168 | 0.0301 | 0.4730 | -0.0028 | 0.0168 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | ece | 0.0046 | 0.0172 | 0.0382 | 15 | 5 | partition_means_t | 0.0046 | -0.0167 | 0.0260 | 0.5781 | -0.0053 | 0.0172 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | auroc | 0.0140 | 0.0101 | 0.0130 | 15 | 5 | partition_means_t | 0.0140 | 0.0014 | 0.0265 | 0.0366 | 0.0072 | 0.0221 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | auprc | 0.0172 | 0.0131 | 0.0158 | 15 | 5 | partition_means_t | 0.0172 | 9.23e-04 | 0.0335 | 0.0427 | 0.0086 | 0.0273 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | balanced_accuracy | 0.0399 | 0.0340 | 0.0316 | 15 | 5 | partition_means_t | 0.0399 | -0.0023 | 0.0822 | 0.0585 | 0.0197 | 0.0592 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | brier | -0.0348 | 0.0150 | 0.0258 | 15 | 5 | partition_means_t | -0.0348 | -0.0535 | -0.0161 | 0.0066 | -0.0506 | -0.0197 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | ece | -0.0315 | 0.0166 | 0.0351 | 15 | 5 | partition_means_t | -0.0315 | -0.0522 | -0.0109 | 0.0133 | -0.0501 | -0.0166 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | auroc | -3.55e-04 | 0.0011 | 0.0085 | 15 | 5 | partition_means_t | -3.55e-04 | -0.0017 | 9.98e-04 | 0.5070 | -0.0033 | 0.0023 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | auprc | -4.18e-04 | 0.0015 | 0.0116 | 15 | 5 | partition_means_t | -4.18e-04 | -0.0022 | 0.0014 | 0.5578 | -0.0041 | 0.0029 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | balanced_accuracy | 0.0018 | 0.0153 | 0.0211 | 15 | 5 | partition_means_t | 0.0018 | -0.0172 | 0.0208 | 0.8070 | -0.0137 | 0.0191 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | brier | -4.44e-04 | 0.0208 | 0.0274 | 15 | 5 | partition_means_t | -4.44e-04 | -0.0262 | 0.0253 | 0.9641 | -0.0160 | 0.0127 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | ece | 7.14e-04 | 0.0356 | 0.0290 | 15 | 5 | partition_means_t | 7.14e-04 | -0.0435 | 0.0449 | 0.9663 | -0.0178 | 0.0165 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | auroc | -0.0016 | 0.0035 | 0.0047 | 15 | 5 | partition_means_t | -0.0016 | -0.0060 | 0.0028 | 0.3709 | -0.0042 | 1.47e-04 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | auprc | -0.0021 | 0.0047 | 0.0061 | 15 | 5 | partition_means_t | -0.0021 | -0.0079 | 0.0037 | 0.3761 | -0.0053 | 2.04e-04 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0083 | 0.0217 | 0.0230 | 15 | 5 | partition_means_t | -0.0083 | -0.0353 | 0.0187 | 0.4403 | -0.0180 | -9.85e-05 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | brier | 0.0079 | 0.0170 | 0.0203 | 15 | 5 | partition_means_t | 0.0079 | -0.0132 | 0.0291 | 0.3558 | 4.45e-04 | 0.0158 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | ece | 0.0056 | 0.0133 | 0.0271 | 15 | 5 | partition_means_t | 0.0056 | -0.0109 | 0.0221 | 0.4026 | -0.0028 | 0.0154 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | auroc | -0.0720 | 0.0214 | 0.0361 | 15 | 5 | partition_means_t | -0.0720 | -0.0985 | -0.0455 | 0.0017 | -0.3447 | 0.0944 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | auprc | -0.0170 | 0.0071 | 0.0080 | 15 | 5 | partition_means_t | -0.0170 | -0.0258 | -0.0083 | 0.0057 | -0.0856 | 0.0406 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | balanced_accuracy | -0.0314 | 0.0113 | 0.0347 | 15 | 5 | partition_means_t | -0.0314 | -0.0454 | -0.0174 | 0.0034 | -0.0801 | 0.0195 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | brier | 0.0278 | 0.0108 | 0.0165 | 15 | 5 | partition_means_t | 0.0278 | 0.0144 | 0.0412 | 0.0045 | 0.0043 | 0.0554 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | ece | 0.0350 | 0.0162 | 0.0233 | 15 | 5 | partition_means_t | 0.0350 | 0.0149 | 0.0551 | 0.0084 | 0.0044 | 0.0693 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | auroc | -0.0786 | 0.0163 | 0.0271 | 15 | 5 | partition_means_t | -0.0786 | -0.0989 | -0.0583 | 4.23e-04 | -0.2837 | 0.0421 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | auprc | -0.0223 | 0.0052 | 0.0077 | 15 | 5 | partition_means_t | -0.0223 | -0.0287 | -0.0158 | 6.60e-04 | -0.0796 | 0.0193 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | balanced_accuracy | -0.0412 | 0.0192 | 0.0470 | 15 | 5 | partition_means_t | -0.0412 | -0.0650 | -0.0175 | 0.0086 | -0.0894 | 0.0072 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | brier | 0.0142 | 0.0230 | 0.0179 | 15 | 5 | partition_means_t | 0.0142 | -0.0144 | 0.0428 | 0.2405 | -0.0097 | 0.0452 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | ece | 0.0177 | 0.0261 | 0.0230 | 15 | 5 | partition_means_t | 0.0177 | -0.0148 | 0.0501 | 0.2054 | -0.0102 | 0.0555 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | auroc | 1.52e-05 | 0.0186 | 0.0170 | 15 | 5 | partition_means_t | 1.52e-05 | -0.0231 | 0.0231 | 0.9986 | -0.1673 | 0.0999 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | auprc | -5.03e-05 | 0.0075 | 0.0048 | 15 | 5 | partition_means_t | -5.03e-05 | -0.0094 | 0.0093 | 0.9888 | -0.0453 | 0.0475 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0092 | 0.0175 | 0.0268 | 15 | 5 | partition_means_t | -0.0092 | -0.0309 | 0.0124 | 0.3026 | -0.0518 | 0.0072 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | brier | -0.0057 | 0.0153 | 0.0184 | 15 | 5 | partition_means_t | -0.0057 | -0.0247 | 0.0133 | 0.4531 | -0.0170 | 0.0064 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | ece | -0.0088 | 0.0169 | 0.0208 | 15 | 5 | partition_means_t | -0.0088 | -0.0298 | 0.0122 | 0.3090 | -0.0221 | 0.0123 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | auroc | -0.1056 | 0.0202 | 0.0501 | 15 | 5 | partition_means_t | -0.1056 | -0.1308 | -0.0805 | 3.07e-04 | -0.1829 | -0.0412 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | auprc | -0.2101 | 0.0354 | 0.1087 | 15 | 5 | partition_means_t | -0.2101 | -0.2540 | -0.1662 | 1.86e-04 | -0.2978 | -0.0401 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | balanced_accuracy | -0.0704 | 0.0106 | 0.0396 | 15 | 5 | partition_means_t | -0.0704 | -0.0836 | -0.0572 | 1.22e-04 | -0.1112 | -0.0387 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | brier | 0.0794 | 0.0372 | 0.0465 | 15 | 5 | partition_means_t | 0.0794 | 0.0332 | 0.1257 | 0.0088 | 0.0454 | 0.1221 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | ece | 0.0619 | 0.0409 | 0.0455 | 15 | 5 | partition_means_t | 0.0619 | 0.0112 | 0.1127 | 0.0276 | 0.0293 | 0.1020 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | auroc | -0.0516 | 0.0284 | 0.0293 | 15 | 5 | partition_means_t | -0.0516 | -0.0869 | -0.0164 | 0.0153 | -0.1177 | 0.0030 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | auprc | -0.1459 | 0.0705 | 0.0532 | 15 | 5 | partition_means_t | -0.1459 | -0.2334 | -0.0583 | 0.0098 | -0.2077 | -0.0086 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | balanced_accuracy | -0.0011 | 0.0299 | 0.0319 | 15 | 5 | partition_means_t | -0.0011 | -0.0382 | 0.0360 | 0.9384 | -0.0323 | 0.0262 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | brier | 0.0105 | 0.0669 | 0.0559 | 15 | 5 | partition_means_t | 0.0105 | -0.0726 | 0.0935 | 0.7442 | -0.0251 | 0.0491 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | ece | -0.0013 | 0.0766 | 0.0579 | 15 | 5 | partition_means_t | -0.0013 | -0.0964 | 0.0938 | 0.9715 | -0.0362 | 0.0358 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0171 | 0.0121 | 0.0306 | 15 | 5 | partition_means_t | 0.0171 | 0.0021 | 0.0321 | 0.0339 | -0.0265 | 0.0689 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0396 | 0.0165 | 0.0590 | 15 | 5 | partition_means_t | 0.0396 | 0.0191 | 0.0600 | 0.0058 | -0.0347 | 0.1455 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 3.09e-04 | 0.0134 | 0.0408 | 15 | 5 | partition_means_t | 3.09e-04 | -0.0163 | 0.0169 | 0.9613 | -0.0234 | 0.0210 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | brier | -4.07e-04 | 0.0329 | 0.0734 | 15 | 5 | partition_means_t | -4.07e-04 | -0.0412 | 0.0404 | 0.9792 | -0.0215 | 0.0220 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | ece | 0.0014 | 0.0387 | 0.0798 | 15 | 5 | partition_means_t | 0.0014 | -0.0466 | 0.0494 | 0.9386 | -0.0209 | 0.0241 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | auroc | -0.1056 | 0.0202 | 0.0501 | 15 | 5 | partition_means_t | -0.1056 | -0.1308 | -0.0805 | 3.07e-04 | -0.1829 | -0.0412 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | auprc | -0.2101 | 0.0354 | 0.1087 | 15 | 5 | partition_means_t | -0.2101 | -0.2540 | -0.1662 | 1.86e-04 | -0.2978 | -0.0401 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | balanced_accuracy | -0.0666 | 0.0379 | 0.0695 | 15 | 5 | partition_means_t | -0.0666 | -0.1137 | -0.0195 | 0.0172 | -0.1051 | -0.0307 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | brier | 0.0794 | 0.0372 | 0.0465 | 15 | 5 | partition_means_t | 0.0794 | 0.0332 | 0.1257 | 0.0088 | 0.0454 | 0.1221 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | ece | 0.0619 | 0.0409 | 0.0455 | 15 | 5 | partition_means_t | 0.0619 | 0.0112 | 0.1127 | 0.0276 | 0.0293 | 0.1020 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | auroc | -0.0516 | 0.0284 | 0.0293 | 15 | 5 | partition_means_t | -0.0516 | -0.0869 | -0.0164 | 0.0153 | -0.1177 | 0.0030 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | auprc | -0.1459 | 0.0705 | 0.0532 | 15 | 5 | partition_means_t | -0.1459 | -0.2334 | -0.0583 | 0.0098 | -0.2077 | -0.0086 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | balanced_accuracy | -0.0320 | 0.0430 | 0.0475 | 15 | 5 | partition_means_t | -0.0320 | -0.0854 | 0.0213 | 0.1706 | -0.0660 | 0.0014 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | brier | 0.0105 | 0.0669 | 0.0559 | 15 | 5 | partition_means_t | 0.0105 | -0.0726 | 0.0935 | 0.7442 | -0.0251 | 0.0491 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | ece | -0.0013 | 0.0766 | 0.0579 | 15 | 5 | partition_means_t | -0.0013 | -0.0964 | 0.0938 | 0.9715 | -0.0362 | 0.0358 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0171 | 0.0121 | 0.0306 | 15 | 5 | partition_means_t | 0.0171 | 0.0021 | 0.0321 | 0.0339 | -0.0265 | 0.0689 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0396 | 0.0165 | 0.0590 | 15 | 5 | partition_means_t | 0.0396 | 0.0191 | 0.0600 | 0.0058 | -0.0347 | 0.1455 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0178 | 0.0232 | 0.0437 | 15 | 5 | partition_means_t | 0.0178 | -0.0110 | 0.0465 | 0.1612 | -0.0075 | 0.0458 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | brier | -4.07e-04 | 0.0329 | 0.0734 | 15 | 5 | partition_means_t | -4.07e-04 | -0.0412 | 0.0404 | 0.9792 | -0.0215 | 0.0220 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | ece | 0.0014 | 0.0387 | 0.0798 | 15 | 5 | partition_means_t | 0.0014 | -0.0466 | 0.0494 | 0.9386 | -0.0209 | 0.0241 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | auroc | -0.1013 | 0.0514 | 0.0940 | 15 | 5 | partition_means_t | -0.1013 | -0.1651 | -0.0375 | 0.0116 | -0.1490 | -0.0577 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | auprc | -0.1658 | 0.0783 | 0.1339 | 15 | 5 | partition_means_t | -0.1658 | -0.2630 | -0.0686 | 0.0091 | -0.2422 | -0.0640 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | balanced_accuracy | -0.0363 | 0.0660 | 0.1269 | 15 | 5 | partition_means_t | -0.0363 | -0.1183 | 0.0457 | 0.2865 | -0.0682 | -0.0028 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | brier | 0.0464 | 0.0289 | 0.0581 | 15 | 5 | partition_means_t | 0.0464 | 0.0105 | 0.0823 | 0.0229 | 0.0270 | 0.0663 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | ece | 0.0433 | 0.0317 | 0.0868 | 15 | 5 | partition_means_t | 0.0433 | 0.0039 | 0.0826 | 0.0380 | 0.0040 | 0.0850 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | auroc | -0.0037 | 0.0296 | 0.0734 | 15 | 5 | partition_means_t | -0.0037 | -0.0404 | 0.0331 | 0.7961 | -0.0472 | 0.0304 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | auprc | -0.0465 | 0.0961 | 0.0736 | 15 | 5 | partition_means_t | -0.0465 | -0.1659 | 0.0728 | 0.3401 | -0.1045 | 0.0261 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | balanced_accuracy | 0.0039 | 0.0599 | 0.1531 | 15 | 5 | partition_means_t | 0.0039 | -0.0705 | 0.0784 | 0.8905 | -0.0233 | 0.0306 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | brier | -0.0050 | 0.0117 | 0.0413 | 15 | 5 | partition_means_t | -0.0050 | -0.0196 | 0.0095 | 0.3914 | -0.0208 | 0.0143 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | ece | -0.0298 | 0.0292 | 0.0756 | 15 | 5 | partition_means_t | -0.0298 | -0.0661 | 0.0065 | 0.0850 | -0.0559 | -0.0025 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0895 | 0.0579 | 0.1256 | 15 | 5 | partition_means_t | 0.0895 | 0.0176 | 0.1615 | 0.0259 | 0.0482 | 0.1192 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | auprc | 0.1006 | 0.0331 | 0.1172 | 15 | 5 | partition_means_t | 0.1006 | 0.0595 | 0.1417 | 0.0025 | 0.0309 | 0.1627 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0818 | 0.0732 | 0.1291 | 15 | 5 | partition_means_t | 0.0818 | -0.0090 | 0.1727 | 0.0668 | 0.0498 | 0.1131 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | brier | -0.0396 | 0.0324 | 0.0621 | 15 | 5 | partition_means_t | -0.0396 | -0.0798 | 5.34e-04 | 0.0519 | -0.0528 | -0.0248 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | ece | -0.0533 | 0.0401 | 0.0967 | 15 | 5 | partition_means_t | -0.0533 | -0.1031 | -0.0034 | 0.0412 | -0.0701 | -0.0306 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | auroc | -0.1013 | 0.0514 | 0.0940 | 15 | 5 | partition_means_t | -0.1013 | -0.1651 | -0.0375 | 0.0116 | -0.1490 | -0.0577 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | auprc | -0.1658 | 0.0783 | 0.1339 | 15 | 5 | partition_means_t | -0.1658 | -0.2630 | -0.0686 | 0.0091 | -0.2422 | -0.0640 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | balanced_accuracy | -0.0727 | 0.0453 | 0.0906 | 15 | 5 | partition_means_t | -0.0727 | -0.1290 | -0.0165 | 0.0230 | -0.1127 | -0.0302 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | brier | 0.0464 | 0.0289 | 0.0581 | 15 | 5 | partition_means_t | 0.0464 | 0.0105 | 0.0823 | 0.0229 | 0.0270 | 0.0663 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | ece | 0.0433 | 0.0317 | 0.0868 | 15 | 5 | partition_means_t | 0.0433 | 0.0039 | 0.0826 | 0.0380 | 0.0040 | 0.0850 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | auroc | -0.0037 | 0.0296 | 0.0734 | 15 | 5 | partition_means_t | -0.0037 | -0.0404 | 0.0331 | 0.7961 | -0.0472 | 0.0304 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | auprc | -0.0465 | 0.0961 | 0.0736 | 15 | 5 | partition_means_t | -0.0465 | -0.1659 | 0.0728 | 0.3401 | -0.1045 | 0.0261 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | balanced_accuracy | -0.0132 | 0.0211 | 0.0570 | 15 | 5 | partition_means_t | -0.0132 | -0.0394 | 0.0131 | 0.2355 | -0.0515 | 0.0217 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | brier | -0.0050 | 0.0117 | 0.0413 | 15 | 5 | partition_means_t | -0.0050 | -0.0196 | 0.0095 | 0.3914 | -0.0208 | 0.0143 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | ece | -0.0298 | 0.0292 | 0.0756 | 15 | 5 | partition_means_t | -0.0298 | -0.0661 | 0.0065 | 0.0850 | -0.0559 | -0.0025 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0895 | 0.0579 | 0.1256 | 15 | 5 | partition_means_t | 0.0895 | 0.0176 | 0.1615 | 0.0259 | 0.0482 | 0.1192 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | auprc | 0.1006 | 0.0331 | 0.1172 | 15 | 5 | partition_means_t | 0.1006 | 0.0595 | 0.1417 | 0.0025 | 0.0309 | 0.1627 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0698 | 0.0407 | 0.1207 | 15 | 5 | partition_means_t | 0.0698 | 0.0193 | 0.1203 | 0.0185 | 0.0221 | 0.1113 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | brier | -0.0396 | 0.0324 | 0.0621 | 15 | 5 | partition_means_t | -0.0396 | -0.0798 | 5.34e-04 | 0.0519 | -0.0528 | -0.0248 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | ece | -0.0533 | 0.0401 | 0.0967 | 15 | 5 | partition_means_t | -0.0533 | -0.1031 | -0.0034 | 0.0412 | -0.0701 | -0.0306 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | auroc | -0.1058 | 0.0205 | 0.0505 | 15 | 5 | partition_means_t | -0.1058 | -0.1312 | -0.0804 | 3.19e-04 | -0.1827 | -0.0411 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | auprc | -0.2128 | 0.0364 | 0.1109 | 15 | 5 | partition_means_t | -0.2128 | -0.2579 | -0.1676 | 1.97e-04 | -0.2989 | -0.0433 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | balanced_accuracy | -0.0708 | 0.0108 | 0.0399 | 15 | 5 | partition_means_t | -0.0708 | -0.0842 | -0.0574 | 1.25e-04 | -0.1118 | -0.0391 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | brier | 0.0802 | 0.0375 | 0.0470 | 15 | 5 | partition_means_t | 0.0802 | 0.0336 | 0.1268 | 0.0088 | 0.0457 | 0.1228 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | ece | 0.0625 | 0.0409 | 0.0461 | 15 | 5 | partition_means_t | 0.0625 | 0.0117 | 0.1134 | 0.0269 | 0.0296 | 0.1029 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | auroc | -0.0515 | 0.0285 | 0.0290 | 15 | 5 | partition_means_t | -0.0515 | -0.0869 | -0.0162 | 0.0155 | -0.1181 | 0.0035 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | auprc | -0.1470 | 0.0718 | 0.0531 | 15 | 5 | partition_means_t | -0.1470 | -0.2361 | -0.0580 | 0.0102 | -0.2089 | -0.0080 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | balanced_accuracy | -0.0010 | 0.0305 | 0.0320 | 15 | 5 | partition_means_t | -0.0010 | -0.0389 | 0.0368 | 0.9431 | -0.0323 | 0.0264 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | brier | 0.0102 | 0.0673 | 0.0564 | 15 | 5 | partition_means_t | 0.0102 | -0.0734 | 0.0938 | 0.7527 | -0.0262 | 0.0493 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | ece | -0.0018 | 0.0774 | 0.0587 | 15 | 5 | partition_means_t | -0.0018 | -0.0980 | 0.0943 | 0.9605 | -0.0370 | 0.0359 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0173 | 0.0122 | 0.0311 | 15 | 5 | partition_means_t | 0.0173 | 0.0021 | 0.0324 | 0.0339 | -0.0259 | 0.0686 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0401 | 0.0168 | 0.0596 | 15 | 5 | partition_means_t | 0.0401 | 0.0192 | 0.0610 | 0.0059 | -0.0341 | 0.1470 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 3.64e-04 | 0.0137 | 0.0414 | 15 | 5 | partition_means_t | 3.64e-04 | -0.0166 | 0.0173 | 0.9553 | -0.0235 | 0.0212 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | brier | -2.80e-04 | 0.0331 | 0.0741 | 15 | 5 | partition_means_t | -2.80e-04 | -0.0413 | 0.0408 | 0.9858 | -0.0216 | 0.0223 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | ece | 0.0018 | 0.0384 | 0.0803 | 15 | 5 | partition_means_t | 0.0018 | -0.0459 | 0.0495 | 0.9222 | -0.0212 | 0.0243 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | auroc | -0.1058 | 0.0205 | 0.0505 | 15 | 5 | partition_means_t | -0.1058 | -0.1312 | -0.0804 | 3.19e-04 | -0.1827 | -0.0411 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | auprc | -0.2128 | 0.0364 | 0.1109 | 15 | 5 | partition_means_t | -0.2128 | -0.2579 | -0.1676 | 1.97e-04 | -0.2989 | -0.0433 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | balanced_accuracy | -0.0960 | 0.0539 | 0.0751 | 15 | 5 | partition_means_t | -0.0960 | -0.1629 | -0.0290 | 0.0164 | -0.1353 | -0.0581 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | brier | 0.0802 | 0.0375 | 0.0470 | 15 | 5 | partition_means_t | 0.0802 | 0.0336 | 0.1268 | 0.0088 | 0.0457 | 0.1228 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | ece | 0.0625 | 0.0409 | 0.0461 | 15 | 5 | partition_means_t | 0.0625 | 0.0117 | 0.1134 | 0.0269 | 0.0296 | 0.1029 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | auroc | -0.0515 | 0.0285 | 0.0290 | 15 | 5 | partition_means_t | -0.0515 | -0.0869 | -0.0162 | 0.0155 | -0.1181 | 0.0035 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | auprc | -0.1470 | 0.0718 | 0.0531 | 15 | 5 | partition_means_t | -0.1470 | -0.2361 | -0.0580 | 0.0102 | -0.2089 | -0.0080 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | balanced_accuracy | -0.0257 | 0.0574 | 0.0598 | 15 | 5 | partition_means_t | -0.0257 | -0.0970 | 0.0455 | 0.3724 | -0.0573 | 0.0032 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | brier | 0.0102 | 0.0673 | 0.0564 | 15 | 5 | partition_means_t | 0.0102 | -0.0734 | 0.0938 | 0.7527 | -0.0262 | 0.0493 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | ece | -0.0018 | 0.0774 | 0.0587 | 15 | 5 | partition_means_t | -0.0018 | -0.0980 | 0.0943 | 0.9605 | -0.0370 | 0.0359 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0173 | 0.0122 | 0.0311 | 15 | 5 | partition_means_t | 0.0173 | 0.0021 | 0.0324 | 0.0339 | -0.0259 | 0.0686 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0401 | 0.0168 | 0.0596 | 15 | 5 | partition_means_t | 0.0401 | 0.0192 | 0.0610 | 0.0059 | -0.0341 | 0.1470 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0154 | 0.0236 | 0.0631 | 15 | 5 | partition_means_t | 0.0154 | -0.0139 | 0.0448 | 0.2182 | -0.0072 | 0.0392 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | brier | -2.80e-04 | 0.0331 | 0.0741 | 15 | 5 | partition_means_t | -2.80e-04 | -0.0413 | 0.0408 | 0.9858 | -0.0216 | 0.0223 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | ece | 0.0018 | 0.0384 | 0.0803 | 15 | 5 | partition_means_t | 0.0018 | -0.0459 | 0.0495 | 0.9222 | -0.0212 | 0.0243 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | auroc | -0.1562 | 0.1012 | 0.1635 | 15 | 5 | partition_means_t | -0.1562 | -0.2819 | -0.0305 | 0.0260 | -0.2189 | -0.0880 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | auprc | -0.2108 | 0.1145 | 0.1879 | 15 | 5 | partition_means_t | -0.2108 | -0.3530 | -0.0687 | 0.0146 | -0.3003 | -0.1016 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | balanced_accuracy | -0.0177 | 0.1089 | 0.1504 | 15 | 5 | partition_means_t | -0.0177 | -0.1529 | 0.1175 | 0.7350 | -0.0597 | 0.0268 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | brier | 0.1124 | 0.0380 | 0.1117 | 15 | 5 | partition_means_t | 0.1124 | 0.0652 | 0.1595 | 0.0027 | 0.0750 | 0.1490 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | ece | 0.1335 | 0.0529 | 0.1511 | 15 | 5 | partition_means_t | 0.1335 | 0.0678 | 0.1991 | 0.0049 | 0.0871 | 0.1749 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | auroc | -0.1122 | 0.0731 | 0.1561 | 15 | 5 | partition_means_t | -0.1122 | -0.2029 | -0.0215 | 0.0264 | -0.1634 | -0.0541 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | auprc | -0.1655 | 0.0902 | 0.1947 | 15 | 5 | partition_means_t | -0.1655 | -0.2776 | -0.0535 | 0.0148 | -0.2389 | -0.0730 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | balanced_accuracy | -0.0561 | 0.0686 | 0.1772 | 15 | 5 | partition_means_t | -0.0561 | -0.1413 | 0.0292 | 0.1418 | -0.0955 | -0.0151 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | brier | 0.0757 | 0.0191 | 0.1014 | 15 | 5 | partition_means_t | 0.0757 | 0.0519 | 0.0994 | 9.02e-04 | 0.0501 | 0.1029 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | ece | 0.0508 | 0.0360 | 0.1127 | 15 | 5 | partition_means_t | 0.0508 | 0.0061 | 0.0955 | 0.0343 | 0.0305 | 0.0800 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0527 | 0.1492 | 0.1944 | 15 | 5 | partition_means_t | 0.0527 | -0.1326 | 0.2379 | 0.4740 | 0.0162 | 0.0877 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | auprc | 0.0642 | 0.1410 | 0.1871 | 15 | 5 | partition_means_t | 0.0642 | -0.1109 | 0.2393 | 0.3662 | -0.0052 | 0.1244 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0774 | 0.0936 | 0.1234 | 15 | 5 | partition_means_t | 0.0774 | -0.0388 | 0.1936 | 0.1382 | 0.0384 | 0.1136 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | brier | 3.23e-04 | 0.0707 | 0.1035 | 15 | 5 | partition_means_t | 3.23e-04 | -0.0874 | 0.0881 | 0.9923 | -0.0226 | 0.0252 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | ece | -0.0058 | 0.0662 | 0.1115 | 15 | 5 | partition_means_t | -0.0058 | -0.0880 | 0.0763 | 0.8532 | -0.0307 | 0.0214 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | auroc | -0.1562 | 0.1012 | 0.1635 | 15 | 5 | partition_means_t | -0.1562 | -0.2819 | -0.0305 | 0.0260 | -0.2189 | -0.0880 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | auprc | -0.2108 | 0.1145 | 0.1879 | 15 | 5 | partition_means_t | -0.2108 | -0.3530 | -0.0687 | 0.0146 | -0.3003 | -0.1016 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | balanced_accuracy | -0.1199 | 0.0600 | 0.0946 | 15 | 5 | partition_means_t | -0.1199 | -0.1944 | -0.0454 | 0.0111 | -0.1611 | -0.0745 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | brier | 0.1124 | 0.0380 | 0.1117 | 15 | 5 | partition_means_t | 0.1124 | 0.0652 | 0.1595 | 0.0027 | 0.0750 | 0.1490 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | ece | 0.1335 | 0.0529 | 0.1511 | 15 | 5 | partition_means_t | 0.1335 | 0.0678 | 0.1991 | 0.0049 | 0.0871 | 0.1749 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | auroc | -0.1122 | 0.0731 | 0.1561 | 15 | 5 | partition_means_t | -0.1122 | -0.2029 | -0.0215 | 0.0264 | -0.1634 | -0.0541 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | auprc | -0.1655 | 0.0902 | 0.1947 | 15 | 5 | partition_means_t | -0.1655 | -0.2776 | -0.0535 | 0.0148 | -0.2389 | -0.0730 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | balanced_accuracy | -0.0683 | 0.0364 | 0.0950 | 15 | 5 | partition_means_t | -0.0683 | -0.1136 | -0.0231 | 0.0137 | -0.0994 | -0.0322 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | brier | 0.0757 | 0.0191 | 0.1014 | 15 | 5 | partition_means_t | 0.0757 | 0.0519 | 0.0994 | 9.02e-04 | 0.0501 | 0.1029 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | ece | 0.0508 | 0.0360 | 0.1127 | 15 | 5 | partition_means_t | 0.0508 | 0.0061 | 0.0955 | 0.0343 | 0.0305 | 0.0800 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0527 | 0.1492 | 0.1944 | 15 | 5 | partition_means_t | 0.0527 | -0.1326 | 0.2379 | 0.4740 | 0.0162 | 0.0877 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | auprc | 0.0642 | 0.1410 | 0.1871 | 15 | 5 | partition_means_t | 0.0642 | -0.1109 | 0.2393 | 0.3662 | -0.0052 | 0.1244 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0244 | 0.0771 | 0.1175 | 15 | 5 | partition_means_t | 0.0244 | -0.0713 | 0.1201 | 0.5184 | -5.57e-04 | 0.0503 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | brier | 3.23e-04 | 0.0707 | 0.1035 | 15 | 5 | partition_means_t | 3.23e-04 | -0.0874 | 0.0881 | 0.9923 | -0.0226 | 0.0252 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | ece | -0.0058 | 0.0662 | 0.1115 | 15 | 5 | partition_means_t | -0.0058 | -0.0880 | 0.0763 | 0.8532 | -0.0307 | 0.0214 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | auroc | -0.1068 | 0.0209 | 0.0508 | 15 | 5 | partition_means_t | -0.1068 | -0.1328 | -0.0808 | 3.37e-04 | -0.1842 | -0.0425 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | auprc | -0.2177 | 0.0379 | 0.1143 | 15 | 5 | partition_means_t | -0.2177 | -0.2647 | -0.1706 | 2.11e-04 | -0.3045 | -0.0484 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | balanced_accuracy | -0.0718 | 0.0111 | 0.0405 | 15 | 5 | partition_means_t | -0.0718 | -0.0856 | -0.0580 | 1.33e-04 | -0.1137 | -0.0396 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | brier | 0.0818 | 0.0382 | 0.0481 | 15 | 5 | partition_means_t | 0.0818 | 0.0344 | 0.1292 | 0.0087 | 0.0469 | 0.1245 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | ece | 0.0640 | 0.0416 | 0.0470 | 15 | 5 | partition_means_t | 0.0640 | 0.0123 | 0.1156 | 0.0263 | 0.0303 | 0.1051 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | auroc | -0.0510 | 0.0292 | 0.0293 | 15 | 5 | partition_means_t | -0.0510 | -0.0873 | -0.0148 | 0.0174 | -0.1188 | 0.0041 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | auprc | -0.1486 | 0.0735 | 0.0541 | 15 | 5 | partition_means_t | -0.1486 | -0.2399 | -0.0573 | 0.0107 | -0.2103 | -0.0086 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | balanced_accuracy | -3.11e-04 | 0.0315 | 0.0324 | 15 | 5 | partition_means_t | -3.11e-04 | -0.0394 | 0.0388 | 0.9835 | -0.0320 | 0.0276 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | brier | 0.0096 | 0.0682 | 0.0573 | 15 | 5 | partition_means_t | 0.0096 | -0.0751 | 0.0944 | 0.7680 | -0.0281 | 0.0499 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | ece | -0.0026 | 0.0784 | 0.0594 | 15 | 5 | partition_means_t | -0.0026 | -0.0999 | 0.0947 | 0.9440 | -0.0386 | 0.0362 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0172 | 0.0122 | 0.0312 | 15 | 5 | partition_means_t | 0.0172 | 0.0021 | 0.0324 | 0.0340 | -0.0255 | 0.0687 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0415 | 0.0170 | 0.0611 | 15 | 5 | partition_means_t | 0.0415 | 0.0204 | 0.0626 | 0.0055 | -0.0311 | 0.1509 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 3.63e-04 | 0.0141 | 0.0417 | 15 | 5 | partition_means_t | 3.63e-04 | -0.0172 | 0.0179 | 0.9569 | -0.0238 | 0.0218 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | brier | 5.80e-05 | 0.0333 | 0.0753 | 15 | 5 | partition_means_t | 5.80e-05 | -0.0413 | 0.0414 | 0.9971 | -0.0217 | 0.0233 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | ece | 0.0020 | 0.0384 | 0.0815 | 15 | 5 | partition_means_t | 0.0020 | -0.0456 | 0.0497 | 0.9113 | -0.0211 | 0.0251 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | auroc | -0.1068 | 0.0209 | 0.0508 | 15 | 5 | partition_means_t | -0.1068 | -0.1328 | -0.0808 | 3.37e-04 | -0.1842 | -0.0425 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | auprc | -0.2177 | 0.0379 | 0.1143 | 15 | 5 | partition_means_t | -0.2177 | -0.2647 | -0.1706 | 2.11e-04 | -0.3045 | -0.0484 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | balanced_accuracy | -0.0867 | 0.0474 | 0.0911 | 15 | 5 | partition_means_t | -0.0867 | -0.1456 | -0.0278 | 0.0150 | -0.1326 | -0.0445 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | brier | 0.0818 | 0.0382 | 0.0481 | 15 | 5 | partition_means_t | 0.0818 | 0.0344 | 0.1292 | 0.0087 | 0.0469 | 0.1245 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | ece | 0.0640 | 0.0416 | 0.0470 | 15 | 5 | partition_means_t | 0.0640 | 0.0123 | 0.1156 | 0.0263 | 0.0303 | 0.1051 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | auroc | -0.0510 | 0.0292 | 0.0293 | 15 | 5 | partition_means_t | -0.0510 | -0.0873 | -0.0148 | 0.0174 | -0.1188 | 0.0041 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | auprc | -0.1486 | 0.0735 | 0.0541 | 15 | 5 | partition_means_t | -0.1486 | -0.2399 | -0.0573 | 0.0107 | -0.2103 | -0.0086 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | balanced_accuracy | -0.0316 | 0.0425 | 0.0706 | 15 | 5 | partition_means_t | -0.0316 | -0.0844 | 0.0212 | 0.1723 | -0.0640 | -0.0011 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | brier | 0.0096 | 0.0682 | 0.0573 | 15 | 5 | partition_means_t | 0.0096 | -0.0751 | 0.0944 | 0.7680 | -0.0281 | 0.0499 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | ece | -0.0026 | 0.0784 | 0.0594 | 15 | 5 | partition_means_t | -0.0026 | -0.0999 | 0.0947 | 0.9440 | -0.0386 | 0.0362 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0172 | 0.0122 | 0.0312 | 15 | 5 | partition_means_t | 0.0172 | 0.0021 | 0.0324 | 0.0340 | -0.0255 | 0.0687 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0415 | 0.0170 | 0.0611 | 15 | 5 | partition_means_t | 0.0415 | 0.0204 | 0.0626 | 0.0055 | -0.0311 | 0.1509 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0085 | 0.0343 | 0.0705 | 15 | 5 | partition_means_t | 0.0085 | -0.0342 | 0.0511 | 0.6102 | -0.0103 | 0.0286 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | brier | 5.80e-05 | 0.0333 | 0.0753 | 15 | 5 | partition_means_t | 5.80e-05 | -0.0413 | 0.0414 | 0.9971 | -0.0217 | 0.0233 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | ece | 0.0020 | 0.0384 | 0.0815 | 15 | 5 | partition_means_t | 0.0020 | -0.0456 | 0.0497 | 0.9113 | -0.0211 | 0.0251 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | auroc | -0.1529 | 0.0919 | 0.1625 | 15 | 5 | partition_means_t | -0.1529 | -0.2670 | -0.0388 | 0.0205 | -0.2351 | -0.0713 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | auprc | -0.2384 | 0.0751 | 0.2338 | 15 | 5 | partition_means_t | -0.2384 | -0.3316 | -0.1452 | 0.0021 | -0.3231 | -0.1181 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | balanced_accuracy | -0.0339 | 0.0534 | 0.1778 | 15 | 5 | partition_means_t | -0.0339 | -0.1003 | 0.0324 | 0.2285 | -0.0768 | 0.0070 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | brier | 0.0840 | 0.0481 | 0.0908 | 15 | 5 | partition_means_t | 0.0840 | 0.0244 | 0.1437 | 0.0174 | 0.0431 | 0.1268 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | ece | 0.0782 | 0.0749 | 0.1190 | 15 | 5 | partition_means_t | 0.0782 | -0.0148 | 0.1712 | 0.0799 | 0.0391 | 0.1260 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | auroc | -0.0764 | 0.0838 | 0.1936 | 15 | 5 | partition_means_t | -0.0764 | -0.1805 | 0.0277 | 0.1112 | -0.1376 | -0.0150 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | auprc | -0.1542 | 0.0798 | 0.2489 | 15 | 5 | partition_means_t | -0.1542 | -0.2533 | -0.0552 | 0.0124 | -0.2157 | -0.0616 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | balanced_accuracy | -0.0642 | 0.0639 | 0.1655 | 15 | 5 | partition_means_t | -0.0642 | -0.1435 | 0.0151 | 0.0880 | -0.1013 | -0.0275 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | brier | 0.0316 | 0.0502 | 0.1072 | 15 | 5 | partition_means_t | 0.0316 | -0.0307 | 0.0939 | 0.2322 | 0.0044 | 0.0629 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | ece | 0.0094 | 0.0537 | 0.1529 | 15 | 5 | partition_means_t | 0.0094 | -0.0573 | 0.0761 | 0.7155 | -0.0131 | 0.0466 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0405 | 0.0594 | 0.1294 | 15 | 5 | partition_means_t | 0.0405 | -0.0333 | 0.1143 | 0.2023 | 0.0072 | 0.0755 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | auprc | 0.0104 | 0.0713 | 0.1402 | 15 | 5 | partition_means_t | 0.0104 | -0.0781 | 0.0989 | 0.7608 | -0.0327 | 0.0610 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0648 | 0.0488 | 0.0980 | 15 | 5 | partition_means_t | 0.0648 | 0.0042 | 0.1255 | 0.0411 | 0.0388 | 0.0873 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | brier | -0.0182 | 0.0274 | 0.0828 | 15 | 5 | partition_means_t | -0.0182 | -0.0523 | 0.0158 | 0.2112 | -0.0366 | -7.37e-04 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | ece | -0.0442 | 0.0301 | 0.1104 | 15 | 5 | partition_means_t | -0.0442 | -0.0816 | -0.0069 | 0.0303 | -0.0637 | -0.0062 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | auroc | -0.1529 | 0.0919 | 0.1625 | 15 | 5 | partition_means_t | -0.1529 | -0.2670 | -0.0388 | 0.0205 | -0.2351 | -0.0713 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | auprc | -0.2384 | 0.0751 | 0.2338 | 15 | 5 | partition_means_t | -0.2384 | -0.3316 | -0.1452 | 0.0021 | -0.3231 | -0.1181 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | balanced_accuracy | -0.0779 | 0.0616 | 0.1397 | 15 | 5 | partition_means_t | -0.0779 | -0.1544 | -0.0015 | 0.0474 | -0.1359 | -0.0226 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | brier | 0.0840 | 0.0481 | 0.0908 | 15 | 5 | partition_means_t | 0.0840 | 0.0244 | 0.1437 | 0.0174 | 0.0431 | 0.1268 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | ece | 0.0782 | 0.0749 | 0.1190 | 15 | 5 | partition_means_t | 0.0782 | -0.0148 | 0.1712 | 0.0799 | 0.0391 | 0.1260 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | auroc | -0.0764 | 0.0838 | 0.1936 | 15 | 5 | partition_means_t | -0.0764 | -0.1805 | 0.0277 | 0.1112 | -0.1376 | -0.0150 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | auprc | -0.1542 | 0.0798 | 0.2489 | 15 | 5 | partition_means_t | -0.1542 | -0.2533 | -0.0552 | 0.0124 | -0.2157 | -0.0616 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | balanced_accuracy | -0.0282 | 0.0313 | 0.1196 | 15 | 5 | partition_means_t | -0.0282 | -0.0671 | 0.0107 | 0.1147 | -0.0699 | 0.0137 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | brier | 0.0316 | 0.0502 | 0.1072 | 15 | 5 | partition_means_t | 0.0316 | -0.0307 | 0.0939 | 0.2322 | 0.0044 | 0.0629 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | ece | 0.0094 | 0.0537 | 0.1529 | 15 | 5 | partition_means_t | 0.0094 | -0.0573 | 0.0761 | 0.7155 | -0.0131 | 0.0466 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0405 | 0.0594 | 0.1294 | 15 | 5 | partition_means_t | 0.0405 | -0.0333 | 0.1143 | 0.2023 | 0.0072 | 0.0755 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | auprc | 0.0104 | 0.0713 | 0.1402 | 15 | 5 | partition_means_t | 0.0104 | -0.0781 | 0.0989 | 0.7608 | -0.0327 | 0.0610 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0679 | 0.0413 | 0.1012 | 15 | 5 | partition_means_t | 0.0679 | 0.0166 | 0.1192 | 0.0213 | 0.0421 | 0.0928 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | brier | -0.0182 | 0.0274 | 0.0828 | 15 | 5 | partition_means_t | -0.0182 | -0.0523 | 0.0158 | 0.2112 | -0.0366 | -7.37e-04 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | ece | -0.0442 | 0.0301 | 0.1104 | 15 | 5 | partition_means_t | -0.0442 | -0.0816 | -0.0069 | 0.0303 | -0.0637 | -0.0062 | 2000 |
| e4 | 500 | A | ssl_colon - sup_imnet | auroc | -0.1219 | 0.0515 | 0.0996 | 15 | 5 | partition_means_t | -0.1219 | -0.1858 | -0.0579 | 0.0061 | -0.1907 | -0.0435 | 1691 |
| e4 | 500 | A | ssl_colon - sup_imnet | auprc | -0.1170 | 0.0615 | 0.1214 | 15 | 5 | partition_means_t | -0.1170 | -0.1934 | -0.0407 | 0.0131 | -0.1693 | -0.0193 | 1691 |
| e4 | 500 | A | ssl_colon - sup_imnet | balanced_accuracy | -0.0693 | 0.0124 | 0.0515 | 15 | 5 | partition_means_t | -0.0693 | -0.0847 | -0.0539 | 2.38e-04 | -0.1231 | -0.0365 | 1691 |
| e4 | 500 | A | ssl_colon - sup_imnet | brier | 0.0608 | 0.0189 | 0.0458 | 15 | 5 | partition_means_t | 0.0608 | 0.0373 | 0.0843 | 0.0020 | 0.0355 | 0.0928 | 2000 |
| e4 | 500 | A | ssl_colon - sup_imnet | ece | 0.0567 | 0.0220 | 0.0454 | 15 | 5 | partition_means_t | 0.0567 | 0.0293 | 0.0840 | 0.0045 | 0.0333 | 0.0865 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet | auroc | -0.0633 | 0.0685 | 0.0942 | 15 | 5 | partition_means_t | -0.0633 | -0.1484 | 0.0218 | 0.1079 | -0.1169 | 0.0044 | 1691 |
| e4 | 500 | A | ssl_colon - ssl_imnet | auprc | -0.0698 | 0.0711 | 0.0990 | 15 | 5 | partition_means_t | -0.0698 | -0.1581 | 0.0185 | 0.0931 | -0.1055 | -6.34e-04 | 1691 |
| e4 | 500 | A | ssl_colon - ssl_imnet | balanced_accuracy | 0.0020 | 0.0423 | 0.0329 | 15 | 5 | partition_means_t | 0.0020 | -0.0505 | 0.0545 | 0.9198 | -0.0274 | 0.0273 | 1691 |
| e4 | 500 | A | ssl_colon - ssl_imnet | brier | 0.0150 | 0.0409 | 0.0306 | 15 | 5 | partition_means_t | 0.0150 | -0.0357 | 0.0658 | 0.4569 | -0.0114 | 0.0435 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet | ece | 0.0164 | 0.0432 | 0.0337 | 15 | 5 | partition_means_t | 0.0164 | -0.0373 | 0.0701 | 0.4436 | -0.0098 | 0.0473 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0261 | 0.0393 | 0.0489 | 15 | 5 | partition_means_t | 0.0261 | -0.0227 | 0.0749 | 0.2111 | -0.0188 | 0.0743 | 1691 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0244 | 0.0268 | 0.0400 | 15 | 5 | partition_means_t | 0.0244 | -0.0089 | 0.0577 | 0.1116 | -0.0133 | 0.0606 | 1691 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0052 | 0.0278 | 0.0477 | 15 | 5 | partition_means_t | 0.0052 | -0.0293 | 0.0398 | 0.6959 | -0.0214 | 0.0227 | 1691 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | brier | 0.0016 | 0.0280 | 0.0524 | 15 | 5 | partition_means_t | 0.0016 | -0.0332 | 0.0363 | 0.9064 | -0.0112 | 0.0158 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | ece | 0.0038 | 0.0360 | 0.0636 | 15 | 5 | partition_means_t | 0.0038 | -0.0408 | 0.0485 | 0.8233 | -0.0104 | 0.0194 | 2000 |
| e4 | 500 | B | ssl_colon - sup_imnet | auroc | -0.1219 | 0.0515 | 0.0996 | 15 | 5 | partition_means_t | -0.1219 | -0.1858 | -0.0579 | 0.0061 | -0.1907 | -0.0435 | 1691 |
| e4 | 500 | B | ssl_colon - sup_imnet | auprc | -0.1170 | 0.0615 | 0.1214 | 15 | 5 | partition_means_t | -0.1170 | -0.1934 | -0.0407 | 0.0131 | -0.1693 | -0.0193 | 1691 |
| e4 | 500 | B | ssl_colon - sup_imnet | balanced_accuracy | -0.0931 | 0.0321 | 0.0809 | 15 | 5 | partition_means_t | -0.0931 | -0.1330 | -0.0532 | 0.0029 | -0.1595 | -0.0511 | 1691 |
| e4 | 500 | B | ssl_colon - sup_imnet | brier | 0.0608 | 0.0189 | 0.0458 | 15 | 5 | partition_means_t | 0.0608 | 0.0373 | 0.0843 | 0.0020 | 0.0355 | 0.0928 | 2000 |
| e4 | 500 | B | ssl_colon - sup_imnet | ece | 0.0567 | 0.0220 | 0.0454 | 15 | 5 | partition_means_t | 0.0567 | 0.0293 | 0.0840 | 0.0045 | 0.0333 | 0.0865 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet | auroc | -0.0633 | 0.0685 | 0.0942 | 15 | 5 | partition_means_t | -0.0633 | -0.1484 | 0.0218 | 0.1079 | -0.1169 | 0.0044 | 1691 |
| e4 | 500 | B | ssl_colon - ssl_imnet | auprc | -0.0698 | 0.0711 | 0.0990 | 15 | 5 | partition_means_t | -0.0698 | -0.1581 | 0.0185 | 0.0931 | -0.1055 | -6.34e-04 | 1691 |
| e4 | 500 | B | ssl_colon - ssl_imnet | balanced_accuracy | -0.0281 | 0.0603 | 0.0778 | 15 | 5 | partition_means_t | -0.0281 | -0.1029 | 0.0467 | 0.3561 | -0.0712 | 0.0025 | 1691 |
| e4 | 500 | B | ssl_colon - ssl_imnet | brier | 0.0150 | 0.0409 | 0.0306 | 15 | 5 | partition_means_t | 0.0150 | -0.0357 | 0.0658 | 0.4569 | -0.0114 | 0.0435 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet | ece | 0.0164 | 0.0432 | 0.0337 | 15 | 5 | partition_means_t | 0.0164 | -0.0373 | 0.0701 | 0.4436 | -0.0098 | 0.0473 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0261 | 0.0393 | 0.0489 | 15 | 5 | partition_means_t | 0.0261 | -0.0227 | 0.0749 | 0.2111 | -0.0188 | 0.0743 | 1691 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0244 | 0.0268 | 0.0400 | 15 | 5 | partition_means_t | 0.0244 | -0.0089 | 0.0577 | 0.1116 | -0.0133 | 0.0606 | 1691 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0265 | 0.0411 | 0.0806 | 15 | 5 | partition_means_t | 0.0265 | -0.0245 | 0.0776 | 0.2221 | -0.0092 | 0.0568 | 1691 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | brier | 0.0016 | 0.0280 | 0.0524 | 15 | 5 | partition_means_t | 0.0016 | -0.0332 | 0.0363 | 0.9064 | -0.0112 | 0.0158 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | ece | 0.0038 | 0.0360 | 0.0636 | 15 | 5 | partition_means_t | 0.0038 | -0.0408 | 0.0485 | 0.8233 | -0.0104 | 0.0194 | 2000 |
| e4 | 500 | C | ssl_colon - sup_imnet | auroc | -0.0235 | 0.0317 | 0.0673 | 15 | 5 | partition_means_t | -0.0235 | -0.0628 | 0.0159 | 0.1731 | -0.0828 | 0.0163 | 1691 |
| e4 | 500 | C | ssl_colon - sup_imnet | auprc | -0.0241 | 0.0306 | 0.0574 | 15 | 5 | partition_means_t | -0.0241 | -0.0621 | 0.0139 | 0.1527 | -0.0652 | 0.0075 | 1691 |
| e4 | 500 | C | ssl_colon - sup_imnet | balanced_accuracy | 0.0240 | 0.1012 | 0.1302 | 15 | 5 | partition_means_t | 0.0240 | -0.1017 | 0.1497 | 0.6242 | -0.0268 | 0.0565 | 1691 |
| e4 | 500 | C | ssl_colon - sup_imnet | brier | 0.0246 | 0.0170 | 0.0355 | 15 | 5 | partition_means_t | 0.0246 | 0.0035 | 0.0458 | 0.0318 | -0.0016 | 0.0521 | 2000 |
| e4 | 500 | C | ssl_colon - sup_imnet | ece | 0.0210 | 0.0242 | 0.0486 | 15 | 5 | partition_means_t | 0.0210 | -0.0090 | 0.0510 | 0.1240 | -0.0111 | 0.0496 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet | auroc | -0.0169 | 0.0174 | 0.0724 | 15 | 5 | partition_means_t | -0.0169 | -0.0385 | 0.0046 | 0.0947 | -0.0857 | 0.0332 | 1691 |
| e4 | 500 | C | ssl_colon - ssl_imnet | auprc | -0.0025 | 0.0264 | 0.1022 | 15 | 5 | partition_means_t | -0.0025 | -0.0353 | 0.0303 | 0.8407 | -0.0509 | 0.0348 | 1691 |
| e4 | 500 | C | ssl_colon - ssl_imnet | balanced_accuracy | -0.0600 | 0.0739 | 0.1109 | 15 | 5 | partition_means_t | -0.0600 | -0.1518 | 0.0319 | 0.1440 | -0.1121 | -0.0196 | 1691 |
| e4 | 500 | C | ssl_colon - ssl_imnet | brier | 0.0224 | 0.0192 | 0.0374 | 15 | 5 | partition_means_t | 0.0224 | -0.0014 | 0.0462 | 0.0591 | -0.0102 | 0.0563 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet | ece | 0.0034 | 0.0216 | 0.0475 | 15 | 5 | partition_means_t | 0.0034 | -0.0234 | 0.0303 | 0.7411 | -0.0227 | 0.0423 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | auroc | -0.0139 | 0.0146 | 0.0581 | 15 | 5 | partition_means_t | -0.0139 | -0.0320 | 0.0042 | 0.1008 | -0.0405 | 0.0070 | 1691 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | auprc | -0.0156 | 0.0150 | 0.0483 | 15 | 5 | partition_means_t | -0.0156 | -0.0343 | 0.0031 | 0.0812 | -0.0390 | 0.0035 | 1691 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0317 | 0.1307 | 0.1140 | 15 | 5 | partition_means_t | 0.0317 | -0.1306 | 0.1940 | 0.6163 | -0.0077 | 0.0586 | 1691 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | brier | 0.0159 | 0.0117 | 0.0296 | 15 | 5 | partition_means_t | 0.0159 | 0.0014 | 0.0304 | 0.0386 | 0.0021 | 0.0304 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | ece | 0.0198 | 0.0230 | 0.0371 | 15 | 5 | partition_means_t | 0.0198 | -0.0088 | 0.0484 | 0.1265 | -0.0036 | 0.0352 | 2000 |
| e4 | 500 | D | ssl_colon - sup_imnet | auroc | -0.0235 | 0.0317 | 0.0673 | 15 | 5 | partition_means_t | -0.0235 | -0.0628 | 0.0159 | 0.1731 | -0.0828 | 0.0163 | 1691 |
| e4 | 500 | D | ssl_colon - sup_imnet | auprc | -0.0241 | 0.0306 | 0.0574 | 15 | 5 | partition_means_t | -0.0241 | -0.0621 | 0.0139 | 0.1527 | -0.0652 | 0.0075 | 1691 |
| e4 | 500 | D | ssl_colon - sup_imnet | balanced_accuracy | -0.0382 | 0.0323 | 0.0811 | 15 | 5 | partition_means_t | -0.0382 | -0.0783 | 0.0019 | 0.0574 | -0.0946 | 0.0047 | 1691 |
| e4 | 500 | D | ssl_colon - sup_imnet | brier | 0.0246 | 0.0170 | 0.0355 | 15 | 5 | partition_means_t | 0.0246 | 0.0035 | 0.0458 | 0.0318 | -0.0016 | 0.0521 | 2000 |
| e4 | 500 | D | ssl_colon - sup_imnet | ece | 0.0210 | 0.0242 | 0.0486 | 15 | 5 | partition_means_t | 0.0210 | -0.0090 | 0.0510 | 0.1240 | -0.0111 | 0.0496 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet | auroc | -0.0169 | 0.0174 | 0.0724 | 15 | 5 | partition_means_t | -0.0169 | -0.0385 | 0.0046 | 0.0947 | -0.0857 | 0.0332 | 1691 |
| e4 | 500 | D | ssl_colon - ssl_imnet | auprc | -0.0025 | 0.0264 | 0.1022 | 15 | 5 | partition_means_t | -0.0025 | -0.0353 | 0.0303 | 0.8407 | -0.0509 | 0.0348 | 1691 |
| e4 | 500 | D | ssl_colon - ssl_imnet | balanced_accuracy | -0.0464 | 0.0382 | 0.0589 | 15 | 5 | partition_means_t | -0.0464 | -0.0939 | 0.0010 | 0.0532 | -0.1085 | -0.0025 | 1691 |
| e4 | 500 | D | ssl_colon - ssl_imnet | brier | 0.0224 | 0.0192 | 0.0374 | 15 | 5 | partition_means_t | 0.0224 | -0.0014 | 0.0462 | 0.0591 | -0.0102 | 0.0563 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet | ece | 0.0034 | 0.0216 | 0.0475 | 15 | 5 | partition_means_t | 0.0034 | -0.0234 | 0.0303 | 0.7411 | -0.0227 | 0.0423 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | auroc | -0.0139 | 0.0146 | 0.0581 | 15 | 5 | partition_means_t | -0.0139 | -0.0320 | 0.0042 | 0.1008 | -0.0405 | 0.0070 | 1691 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | auprc | -0.0156 | 0.0150 | 0.0483 | 15 | 5 | partition_means_t | -0.0156 | -0.0343 | 0.0031 | 0.0812 | -0.0390 | 0.0035 | 1691 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0037 | 0.0472 | 0.0916 | 15 | 5 | partition_means_t | 0.0037 | -0.0549 | 0.0623 | 0.8699 | -0.0266 | 0.0303 | 1691 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | brier | 0.0159 | 0.0117 | 0.0296 | 15 | 5 | partition_means_t | 0.0159 | 0.0014 | 0.0304 | 0.0386 | 0.0021 | 0.0304 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | ece | 0.0198 | 0.0230 | 0.0371 | 15 | 5 | partition_means_t | 0.0198 | -0.0088 | 0.0484 | 0.1265 | -0.0036 | 0.0352 | 2000 |
| neg | 50 | isolated | ssl_colon - sup_imnet | auroc | -0.0699 | 0.0205 | 0.0383 | 15 | 5 | partition_means_t | -0.0699 | -0.0954 | -0.0444 | 0.0016 | -0.3542 | 0.1108 | 1997 |
| neg | 50 | isolated | ssl_colon - sup_imnet | auprc | -0.0147 | 0.0055 | 0.0081 | 15 | 5 | partition_means_t | -0.0147 | -0.0216 | -0.0078 | 0.0040 | -0.0795 | 0.0348 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet | auroc | -0.0770 | 0.0148 | 0.0296 | 15 | 5 | partition_means_t | -0.0770 | -0.0954 | -0.0586 | 3.14e-04 | -0.2972 | 0.0522 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet | auprc | -0.0197 | 0.0047 | 0.0079 | 15 | 5 | partition_means_t | -0.0197 | -0.0255 | -0.0139 | 7.06e-04 | -0.0753 | 0.0166 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet_matched | auroc | -2.35e-04 | 0.0171 | 0.0196 | 15 | 5 | partition_means_t | -2.35e-04 | -0.0214 | 0.0210 | 0.9769 | -0.1936 | 0.1003 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet_matched | auprc | -2.51e-04 | 0.0064 | 0.0052 | 15 | 5 | partition_means_t | -2.51e-04 | -0.0082 | 0.0077 | 0.9346 | -0.0408 | 0.0455 | 1997 |
| neg | 50 | sequence | ssl_colon - sup_imnet | auroc | -0.1074 | 0.0211 | 0.0516 | 15 | 5 | partition_means_t | -0.1074 | -0.1337 | -0.0812 | 3.42e-04 | -0.1864 | -0.0427 | 2000 |
| neg | 50 | sequence | ssl_colon - sup_imnet | auprc | -0.2228 | 0.0387 | 0.1162 | 15 | 5 | partition_means_t | -0.2228 | -0.2708 | -0.1748 | 2.09e-04 | -0.3092 | -0.0606 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet | auroc | -0.0504 | 0.0295 | 0.0302 | 15 | 5 | partition_means_t | -0.0504 | -0.0870 | -0.0137 | 0.0189 | -0.1187 | 0.0057 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet | auprc | -0.1490 | 0.0743 | 0.0548 | 15 | 5 | partition_means_t | -0.1490 | -0.2413 | -0.0567 | 0.0110 | -0.2112 | -0.0056 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet_matched | auroc | 0.0180 | 0.0128 | 0.0316 | 15 | 5 | partition_means_t | 0.0180 | 0.0021 | 0.0339 | 0.0349 | -0.0240 | 0.0688 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet_matched | auprc | 0.0417 | 0.0174 | 0.0616 | 15 | 5 | partition_means_t | 0.0417 | 0.0201 | 0.0633 | 0.0058 | -0.0300 | 0.1480 | 2000 |
