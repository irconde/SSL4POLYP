### Table 4: E1 AULC (trapezoid over 5–100 %, normalised by the axis width)

| regime | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B | comparison |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sup_imnet | AULC(auroc) | 0.8719 | 0.0092 | 0.0080 | 15 | 5 | partition_means_t | 0.8719 | 0.8605 | 0.8833 | 2.94e-09 | 0.8486 | 0.8938 | 2000 |  |
| ssl_imnet | AULC(auroc) | 0.9013 | 0.0064 | 0.0071 | 15 | 5 | partition_means_t | 0.9013 | 0.8933 | 0.9092 | 6.05e-10 | 0.8821 | 0.9193 | 2000 |  |
| ssl_colon | AULC(auroc) | 0.9089 | 0.0052 | 0.0077 | 15 | 5 | partition_means_t | 0.9089 | 0.9024 | 0.9154 | 2.63e-10 | 0.8903 | 0.9259 | 2000 |  |
| ssl_imnet_matched | AULC(auroc) | 0.8751 | 0.0103 | 0.0049 | 15 | 5 | partition_means_t | 0.8751 | 0.8623 | 0.8879 | 4.60e-09 | 0.8522 | 0.8968 | 2000 |  |
| sup_imnet | AULC(auprc) | 0.8705 | 0.0087 | 0.0088 | 15 | 5 | partition_means_t | 0.8705 | 0.8597 | 0.8813 | 2.38e-09 | 0.8447 | 0.8944 | 2000 |  |
| ssl_imnet | AULC(auprc) | 0.9004 | 0.0040 | 0.0081 | 15 | 5 | partition_means_t | 0.9004 | 0.8955 | 0.9053 | 8.99e-11 | 0.8793 | 0.9205 | 2000 |  |
| ssl_colon | AULC(auprc) | 0.9079 | 0.0050 | 0.0068 | 15 | 5 | partition_means_t | 0.9079 | 0.9018 | 0.9141 | 2.16e-10 | 0.8882 | 0.9265 | 2000 |  |
| ssl_imnet_matched | AULC(auprc) | 0.8733 | 0.0091 | 0.0043 | 15 | 5 | partition_means_t | 0.8733 | 0.8620 | 0.8846 | 2.86e-09 | 0.8481 | 0.8974 | 2000 |  |
| sup_imnet | AULC(f1) | 0.7967 | 0.0058 | 0.0080 | 15 | 5 | partition_means_t | 0.7967 | 0.7895 | 0.8040 | 6.78e-10 | 0.7777 | 0.8161 | 2000 |  |
| ssl_imnet | AULC(f1) | 0.8231 | 0.0083 | 0.0086 | 15 | 5 | partition_means_t | 0.8231 | 0.8129 | 0.8334 | 2.46e-09 | 0.8057 | 0.8409 | 2000 |  |
| ssl_colon | AULC(f1) | 0.8288 | 0.0061 | 0.0086 | 15 | 5 | partition_means_t | 0.8288 | 0.8213 | 0.8364 | 7.02e-10 | 0.8107 | 0.8467 | 2000 |  |
| ssl_imnet_matched | AULC(f1) | 0.7963 | 0.0104 | 0.0076 | 15 | 5 | partition_means_t | 0.7963 | 0.7834 | 0.8092 | 6.91e-09 | 0.7769 | 0.8156 | 2000 |  |
| sup_imnet | AULC(precision) | 0.7561 | 0.0248 | 0.0153 | 15 | 5 | partition_means_t | 0.7561 | 0.7253 | 0.7869 | 2.78e-07 | 0.7312 | 0.7826 | 2000 |  |
| ssl_imnet | AULC(precision) | 0.7858 | 0.0223 | 0.0147 | 15 | 5 | partition_means_t | 0.7858 | 0.7582 | 0.8135 | 1.54e-07 | 0.7619 | 0.8112 | 2000 |  |
| ssl_colon | AULC(precision) | 0.8006 | 0.0242 | 0.0163 | 15 | 5 | partition_means_t | 0.8006 | 0.7706 | 0.8306 | 1.99e-07 | 0.7756 | 0.8268 | 2000 |  |
| ssl_imnet_matched | AULC(precision) | 0.7559 | 0.0247 | 0.0154 | 15 | 5 | partition_means_t | 0.7559 | 0.7253 | 0.7866 | 2.72e-07 | 0.7315 | 0.7818 | 2000 |  |
| sup_imnet | AULC(recall) | 0.8456 | 0.0232 | 0.0202 | 15 | 5 | partition_means_t | 0.8456 | 0.8167 | 0.8744 | 1.37e-07 | 0.8210 | 0.8693 | 2000 |  |
| ssl_imnet | AULC(recall) | 0.8669 | 0.0160 | 0.0170 | 15 | 5 | partition_means_t | 0.8669 | 0.8471 | 0.8868 | 2.79e-08 | 0.8447 | 0.8887 | 2000 |  |
| ssl_colon | AULC(recall) | 0.8624 | 0.0207 | 0.0186 | 15 | 5 | partition_means_t | 0.8624 | 0.8367 | 0.8882 | 8.01e-08 | 0.8398 | 0.8842 | 2000 |  |
| ssl_imnet_matched | AULC(recall) | 0.8456 | 0.0201 | 0.0226 | 15 | 5 | partition_means_t | 0.8456 | 0.8206 | 0.8706 | 7.70e-08 | 0.8198 | 0.8698 | 2000 |  |
| sup_imnet | AULC(specificity) | 0.7224 | 0.0422 | 0.0262 | 15 | 5 | partition_means_t | 0.7224 | 0.6700 | 0.7749 | 2.79e-06 | 0.6838 | 0.7595 | 2000 |  |
| ssl_imnet | AULC(specificity) | 0.7595 | 0.0356 | 0.0251 | 15 | 5 | partition_means_t | 0.7595 | 0.7152 | 0.8037 | 1.16e-06 | 0.7228 | 0.7942 | 2000 |  |
| ssl_colon | AULC(specificity) | 0.7803 | 0.0363 | 0.0234 | 15 | 5 | partition_means_t | 0.7803 | 0.7352 | 0.8254 | 1.12e-06 | 0.7447 | 0.8143 | 2000 |  |
| ssl_imnet_matched | AULC(specificity) | 0.7209 | 0.0438 | 0.0281 | 15 | 5 | partition_means_t | 0.7209 | 0.6665 | 0.7753 | 3.27e-06 | 0.6823 | 0.7575 | 2000 |  |
| sup_imnet | AULC(balanced_accuracy) | 0.7840 | 0.0111 | 0.0091 | 15 | 5 | partition_means_t | 0.7840 | 0.7702 | 0.7978 | 9.59e-09 | 0.7617 | 0.8058 | 2000 |  |
| ssl_imnet | AULC(balanced_accuracy) | 0.8132 | 0.0122 | 0.0102 | 15 | 5 | partition_means_t | 0.8132 | 0.7980 | 0.8284 | 1.23e-08 | 0.7930 | 0.8327 | 2000 |  |
| ssl_colon | AULC(balanced_accuracy) | 0.8214 | 0.0097 | 0.0094 | 15 | 5 | partition_means_t | 0.8214 | 0.8093 | 0.8334 | 4.64e-09 | 0.8009 | 0.8408 | 2000 |  |
| ssl_imnet_matched | AULC(balanced_accuracy) | 0.7833 | 0.0157 | 0.0082 | 15 | 5 | partition_means_t | 0.7833 | 0.7638 | 0.8027 | 3.84e-08 | 0.7611 | 0.8045 | 2000 |  |
| sup_imnet | AULC(brier) | 0.1516 | 0.0043 | 0.0063 | 15 | 5 | partition_means_t | 0.1516 | 0.1463 | 0.1569 | 1.50e-07 | 0.1403 | 0.1629 | 2000 |  |
| ssl_imnet | AULC(brier) | 0.1380 | 0.0045 | 0.0046 | 15 | 5 | partition_means_t | 0.1380 | 0.1324 | 0.1436 | 2.79e-07 | 0.1274 | 0.1489 | 2000 |  |
| ssl_colon | AULC(brier) | 0.1343 | 0.0029 | 0.0073 | 15 | 5 | partition_means_t | 0.1343 | 0.1306 | 0.1379 | 5.36e-08 | 0.1238 | 0.1454 | 2000 |  |
| ssl_imnet_matched | AULC(brier) | 0.1512 | 0.0048 | 0.0049 | 15 | 5 | partition_means_t | 0.1512 | 0.1453 | 0.1571 | 2.35e-07 | 0.1399 | 0.1627 | 2000 |  |
| sup_imnet | AULC(ece) | 0.0765 | 0.0077 | 0.0125 | 15 | 5 | partition_means_t | 0.0765 | 0.0669 | 0.0861 | 2.47e-05 | 0.0803 | 0.0962 | 2000 |  |
| ssl_imnet | AULC(ece) | 0.0952 | 0.0113 | 0.0097 | 15 | 5 | partition_means_t | 0.0952 | 0.0812 | 0.1092 | 4.66e-05 | 0.0951 | 0.1118 | 2000 |  |
| ssl_colon | AULC(ece) | 0.0994 | 0.0044 | 0.0091 | 15 | 5 | partition_means_t | 0.0994 | 0.0939 | 0.1049 | 9.43e-07 | 0.0998 | 0.1152 | 2000 |  |
| ssl_imnet_matched | AULC(ece) | 0.0827 | 0.0101 | 0.0101 | 15 | 5 | partition_means_t | 0.0827 | 0.0701 | 0.0953 | 5.30e-05 | 0.0848 | 0.0998 | 2000 |  |
| sup_imnet | AULC(per_polyp_sensitivity_ge1) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| ssl_imnet | AULC(per_polyp_sensitivity_ge1) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| ssl_colon | AULC(per_polyp_sensitivity_ge1) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| ssl_imnet_matched | AULC(per_polyp_sensitivity_ge1) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| sup_imnet | AULC(per_polyp_sensitivity_ge5) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| ssl_imnet | AULC(per_polyp_sensitivity_ge5) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| ssl_colon | AULC(per_polyp_sensitivity_ge5) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
| ssl_imnet_matched | AULC(per_polyp_sensitivity_ge5) | 1 | 0.0000 | 0.0000 | 15 | 5 | constant | 1 |  |  |  | 1 | 1 | 2000 |  |
|  | AULC(auroc) | 0.0370 | 0.0096 | 0.0121 | 15 | 5 | partition_means_t | 0.0370 | 0.0251 | 0.0490 | 0.0010 | 0.0313 | 0.0433 | 2000 | ssl_colon - sup_imnet |
|  | AULC(auroc) | 0.0077 | 0.0024 | 0.0083 | 15 | 5 | partition_means_t | 0.0077 | 0.0047 | 0.0107 | 0.0021 | 0.0045 | 0.0110 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(auroc) | 0.0338 | 0.0095 | 0.0064 | 15 | 5 | partition_means_t | 0.0338 | 0.0220 | 0.0457 | 0.0014 | 0.0281 | 0.0397 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(auprc) | 0.0375 | 0.0109 | 0.0112 | 15 | 5 | partition_means_t | 0.0375 | 0.0239 | 0.0510 | 0.0016 | 0.0309 | 0.0443 | 2000 | ssl_colon - sup_imnet |
|  | AULC(auprc) | 0.0075 | 0.0020 | 0.0082 | 15 | 5 | partition_means_t | 0.0075 | 0.0050 | 0.0100 | 0.0011 | 0.0038 | 0.0113 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(auprc) | 0.0346 | 0.0089 | 0.0065 | 15 | 5 | partition_means_t | 0.0346 | 0.0236 | 0.0457 | 9.67e-04 | 0.0283 | 0.0413 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(f1) | 0.0321 | 0.0099 | 0.0123 | 15 | 5 | partition_means_t | 0.0321 | 0.0198 | 0.0444 | 0.0019 | 0.0276 | 0.0365 | 2000 | ssl_colon - sup_imnet |
|  | AULC(f1) | 0.0057 | 0.0059 | 0.0114 | 15 | 5 | partition_means_t | 0.0057 | -0.0017 | 0.0131 | 0.0988 | 0.0020 | 0.0094 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(f1) | 0.0325 | 0.0071 | 0.0099 | 15 | 5 | partition_means_t | 0.0325 | 0.0237 | 0.0413 | 5.18e-04 | 0.0282 | 0.0369 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(precision) | 0.0445 | 0.0113 | 0.0210 | 15 | 5 | partition_means_t | 0.0445 | 0.0305 | 0.0586 | 9.17e-04 | 0.0391 | 0.0498 | 2000 | ssl_colon - sup_imnet |
|  | AULC(precision) | 0.0148 | 0.0088 | 0.0115 | 15 | 5 | partition_means_t | 0.0148 | 0.0038 | 0.0257 | 0.0199 | 0.0106 | 0.0193 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(precision) | 0.0446 | 0.0203 | 0.0182 | 15 | 5 | partition_means_t | 0.0446 | 0.0195 | 0.0698 | 0.0079 | 0.0391 | 0.0501 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(recall) | 0.0168 | 0.0111 | 0.0326 | 15 | 5 | partition_means_t | 0.0168 | 0.0031 | 0.0306 | 0.0272 | 0.0110 | 0.0223 | 2000 | ssl_colon - sup_imnet |
|  | AULC(recall) | -0.0045 | 0.0116 | 0.0245 | 15 | 5 | partition_means_t | -0.0045 | -0.0189 | 0.0099 | 0.4321 | -0.0100 | 7.77e-04 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(recall) | 0.0168 | 0.0147 | 0.0352 | 15 | 5 | partition_means_t | 0.0168 | -0.0015 | 0.0350 | 0.0631 | 0.0110 | 0.0223 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(specificity) | 0.0579 | 0.0154 | 0.0328 | 15 | 5 | partition_means_t | 0.0579 | 0.0388 | 0.0769 | 0.0011 | 0.0502 | 0.0659 | 2000 | ssl_colon - sup_imnet |
|  | AULC(specificity) | 0.0208 | 0.0136 | 0.0197 | 15 | 5 | partition_means_t | 0.0208 | 0.0040 | 0.0377 | 0.0265 | 0.0153 | 0.0269 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(specificity) | 0.0594 | 0.0324 | 0.0336 | 15 | 5 | partition_means_t | 0.0594 | 0.0192 | 0.0996 | 0.0149 | 0.0523 | 0.0669 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(balanced_accuracy) | 0.0374 | 0.0108 | 0.0120 | 15 | 5 | partition_means_t | 0.0374 | 0.0240 | 0.0507 | 0.0015 | 0.0326 | 0.0422 | 2000 | ssl_colon - sup_imnet |
|  | AULC(balanced_accuracy) | 0.0082 | 0.0065 | 0.0107 | 15 | 5 | partition_means_t | 0.0082 | 1.42e-04 | 0.0162 | 0.0476 | 0.0043 | 0.0121 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(balanced_accuracy) | 0.0381 | 0.0111 | 0.0077 | 15 | 5 | partition_means_t | 0.0381 | 0.0243 | 0.0519 | 0.0016 | 0.0335 | 0.0429 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(brier) | -0.0173 | 0.0033 | 0.0114 | 15 | 5 | partition_means_t | -0.0173 | -0.0214 | -0.0133 | 2.96e-04 | -0.0199 | -0.0149 | 2000 | ssl_colon - sup_imnet |
|  | AULC(brier) | -0.0037 | 0.0046 | 0.0086 | 15 | 5 | partition_means_t | -0.0037 | -0.0094 | 0.0019 | 0.1424 | -0.0056 | -0.0019 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(brier) | -0.0170 | 0.0063 | 0.0057 | 15 | 5 | partition_means_t | -0.0170 | -0.0248 | -0.0092 | 0.0038 | -0.0193 | -0.0146 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(ece) | 0.0229 | 0.0096 | 0.0170 | 15 | 5 | partition_means_t | 0.0229 | 0.0110 | 0.0348 | 0.0059 | 0.0146 | 0.0237 | 2000 | ssl_colon - sup_imnet |
|  | AULC(ece) | 0.0042 | 0.0099 | 0.0166 | 15 | 5 | partition_means_t | 0.0042 | -0.0081 | 0.0165 | 0.3958 | 1.32e-05 | 0.0080 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(ece) | 0.0167 | 0.0061 | 0.0115 | 15 | 5 | partition_means_t | 0.0167 | 0.0091 | 0.0243 | 0.0036 | 0.0109 | 0.0192 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(per_polyp_sensitivity_ge1) | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 | ssl_colon - sup_imnet |
|  | AULC(per_polyp_sensitivity_ge1) | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(per_polyp_sensitivity_ge1) | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(per_polyp_sensitivity_ge5) | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 | ssl_colon - sup_imnet |
|  | AULC(per_polyp_sensitivity_ge5) | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(per_polyp_sensitivity_ge5) | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 | ssl_colon - ssl_imnet_matched |
