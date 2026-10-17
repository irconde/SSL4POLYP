### Table 9: E4 arm contrasts per budget (C − A = adapted vs zero-shot; D − C = recalibration after adaptation)

| variant | regime | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | sup_imnet | arm B - arm A | recall | -0.0777 | 0.0471 | 0.1176 | 15 | 5 | partition_means_t | -0.0777 | -0.1362 | -0.0191 | 0.0211 | -0.1068 | -0.0582 | 2000 |
| 50 | ssl_imnet | arm B - arm A | recall | -0.1233 | 0.0572 | 0.1065 | 15 | 5 | partition_means_t | -0.1233 | -0.1944 | -0.0522 | 0.0085 | -0.1677 | -0.0934 | 2000 |
| 50 | ssl_colon | arm B - arm A | recall | -0.0707 | 0.0601 | 0.1097 | 15 | 5 | partition_means_t | -0.0707 | -0.1454 | 0.0039 | 0.0581 | -0.0946 | -0.0477 | 2000 |
| 50 | ssl_imnet_matched | arm B - arm A | recall | -0.0028 | 0.0725 | 0.1537 | 15 | 5 | partition_means_t | -0.0028 | -0.0928 | 0.0873 | 0.9361 | -0.0170 | 0.0068 | 2000 |
| 50 | mean of regimes | arm B - arm A | recall | -0.0686 | 0.0393 | 0.0734 | 15 | 5 | partition_means_t | -0.0686 | -0.1174 | -0.0198 | 0.0175 | -0.0917 | -0.0538 | 2000 |
| 50 | sup_imnet | arm B - arm A | specificity | 0.0556 | 0.0500 | 0.2334 | 15 | 5 | partition_means_t | 0.0556 | -0.0065 | 0.1176 | 0.0677 | 0.0194 | 0.0926 | 2000 |
| 50 | ssl_imnet | arm B - arm A | specificity | 0.1706 | 0.0800 | 0.1313 | 15 | 5 | partition_means_t | 0.1706 | 0.0712 | 0.2700 | 0.0089 | 0.1351 | 0.2102 | 2000 |
| 50 | ssl_colon | arm B - arm A | specificity | 0.0561 | 0.0967 | 0.1383 | 15 | 5 | partition_means_t | 0.0561 | -0.0639 | 0.1762 | 0.2640 | 0.0313 | 0.0832 | 2000 |
| 50 | ssl_imnet_matched | arm B - arm A | specificity | -0.0468 | 0.0988 | 0.2020 | 15 | 5 | partition_means_t | -0.0468 | -0.1694 | 0.0759 | 0.3494 | -0.0676 | -0.0240 | 2000 |
| 50 | mean of regimes | arm B - arm A | specificity | 0.0589 | 0.0692 | 0.1187 | 15 | 5 | partition_means_t | 0.0589 | -0.0271 | 0.1448 | 0.1299 | 0.0347 | 0.0858 | 2000 |
| 50 | sup_imnet | arm B - arm A | balanced_accuracy | -0.0110 | 0.0291 | 0.0654 | 15 | 5 | partition_means_t | -0.0110 | -0.0472 | 0.0251 | 0.4443 | -0.0352 | 0.0108 | 2000 |
| 50 | ssl_imnet | arm B - arm A | balanced_accuracy | 0.0237 | 0.0148 | 0.0240 | 15 | 5 | partition_means_t | 0.0237 | 0.0052 | 0.0421 | 0.0235 | -0.0044 | 0.0489 | 2000 |
| 50 | ssl_colon | arm B - arm A | balanced_accuracy | -0.0073 | 0.0209 | 0.0315 | 15 | 5 | partition_means_t | -0.0073 | -0.0333 | 0.0187 | 0.4785 | -0.0242 | 0.0106 | 2000 |
| 50 | ssl_imnet_matched | arm B - arm A | balanced_accuracy | -0.0248 | 0.0292 | 0.0376 | 15 | 5 | partition_means_t | -0.0248 | -0.0610 | 0.0115 | 0.1310 | -0.0380 | -0.0123 | 2000 |
| 50 | mean of regimes | arm B - arm A | balanced_accuracy | -0.0049 | 0.0185 | 0.0279 | 15 | 5 | partition_means_t | -0.0049 | -0.0279 | 0.0182 | 0.5889 | -0.0217 | 0.0101 | 2000 |
| 50 | sup_imnet | arm C - arm A | auroc | -0.0050 | 0.0173 | 0.0383 | 15 | 5 | partition_means_t | -0.0050 | -0.0265 | 0.0164 | 0.5498 | -0.0242 | 0.0157 | 2000 |
| 50 | ssl_imnet | arm C - arm A | auroc | -0.0487 | 0.0279 | 0.1191 | 15 | 5 | partition_means_t | -0.0487 | -0.0833 | -0.0140 | 0.0175 | -0.0790 | -0.0149 | 2000 |
| 50 | ssl_colon | arm C - arm A | auroc | -7.01e-04 | 0.0384 | 0.0708 | 15 | 5 | partition_means_t | -7.01e-04 | -0.0484 | 0.0470 | 0.9694 | -0.0645 | 0.0681 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm A | auroc | -0.0731 | 0.0726 | 0.1839 | 15 | 5 | partition_means_t | -0.0731 | -0.1633 | 0.0171 | 0.0875 | -0.1575 | 0.0289 | 2000 |
| 50 | mean of regimes | arm C - arm A | auroc | -0.0319 | 0.0334 | 0.0876 | 15 | 5 | partition_means_t | -0.0319 | -0.0734 | 0.0096 | 0.1000 | -0.0782 | 0.0228 | 2000 |
| 50 | sup_imnet | arm C - arm A | auprc | 0.0146 | 0.0411 | 0.0933 | 15 | 5 | partition_means_t | 0.0146 | -0.0364 | 0.0655 | 0.4720 | -0.0418 | 0.0572 | 2000 |
| 50 | ssl_imnet | arm C - arm A | auprc | -0.0405 | 0.0320 | 0.1447 | 15 | 5 | partition_means_t | -0.0405 | -0.0802 | -7.95e-04 | 0.0472 | -0.0932 | 0.0108 | 2000 |
| 50 | ssl_colon | arm C - arm A | auprc | 0.0589 | 0.0564 | 0.1089 | 15 | 5 | partition_means_t | 0.0589 | -0.0112 | 0.1289 | 0.0800 | -0.0907 | 0.1277 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm A | auprc | -0.0021 | 0.0652 | 0.1367 | 15 | 5 | partition_means_t | -0.0021 | -0.0831 | 0.0788 | 0.9449 | -0.1551 | 0.0720 | 2000 |
| 50 | mean of regimes | arm C - arm A | auprc | 0.0077 | 0.0439 | 0.0953 | 15 | 5 | partition_means_t | 0.0077 | -0.0469 | 0.0623 | 0.7151 | -0.0920 | 0.0631 | 2000 |
| 50 | sup_imnet | arm C - arm A | recall | -0.4603 | 0.1668 | 0.2072 | 15 | 5 | partition_means_t | -0.4603 | -0.6674 | -0.2532 | 0.0035 | -0.5028 | -0.4260 | 2000 |
| 50 | ssl_imnet | arm C - arm A | recall | -0.3340 | 0.1025 | 0.3271 | 15 | 5 | partition_means_t | -0.3340 | -0.4612 | -0.2067 | 0.0019 | -0.3801 | -0.3058 | 2000 |
| 50 | ssl_colon | arm C - arm A | recall | -0.3254 | 0.2353 | 0.2896 | 15 | 5 | partition_means_t | -0.3254 | -0.6176 | -0.0332 | 0.0365 | -0.3644 | -0.2848 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm A | recall | -0.5117 | 0.1671 | 0.3351 | 15 | 5 | partition_means_t | -0.5117 | -0.7192 | -0.3042 | 0.0024 | -0.5414 | -0.4793 | 2000 |
| 50 | mean of regimes | arm C - arm A | recall | -0.4078 | 0.0593 | 0.1491 | 15 | 5 | partition_means_t | -0.4078 | -0.4815 | -0.3342 | 1.05e-04 | -0.4434 | -0.3824 | 2000 |
| 50 | sup_imnet | arm C - arm A | specificity | 0.2600 | 0.0590 | 0.1107 | 15 | 5 | partition_means_t | 0.2600 | 0.1867 | 0.3333 | 5.96e-04 | 0.1600 | 0.3756 | 2000 |
| 50 | ssl_imnet | arm C - arm A | specificity | 0.1918 | 0.1191 | 0.3287 | 15 | 5 | partition_means_t | 0.1918 | 0.0439 | 0.3397 | 0.0227 | 0.1004 | 0.2882 | 2000 |
| 50 | ssl_colon | arm C - arm A | specificity | 0.1933 | 0.1651 | 0.1978 | 15 | 5 | partition_means_t | 0.1933 | -0.0117 | 0.3982 | 0.0589 | 0.1228 | 0.2740 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm A | specificity | 0.2166 | 0.1450 | 0.2766 | 15 | 5 | partition_means_t | 0.2166 | 0.0365 | 0.3966 | 0.0288 | 0.0971 | 0.3442 | 2000 |
| 50 | mean of regimes | arm C - arm A | specificity | 0.2154 | 0.0490 | 0.1195 | 15 | 5 | partition_means_t | 0.2154 | 0.1546 | 0.2762 | 6.00e-04 | 0.1242 | 0.3150 | 2000 |
| 50 | sup_imnet | arm C - arm A | balanced_accuracy | -0.1001 | 0.0689 | 0.0730 | 15 | 5 | partition_means_t | -0.1001 | -0.1856 | -0.0146 | 0.0313 | -0.1545 | -0.0397 | 2000 |
| 50 | ssl_imnet | arm C - arm A | balanced_accuracy | -0.0711 | 0.0349 | 0.1270 | 15 | 5 | partition_means_t | -0.0711 | -0.1144 | -0.0277 | 0.0104 | -0.1225 | -0.0221 | 2000 |
| 50 | ssl_colon | arm C - arm A | balanced_accuracy | -0.0661 | 0.0546 | 0.0841 | 15 | 5 | partition_means_t | -0.0661 | -0.1339 | 0.0018 | 0.0539 | -0.1062 | -0.0222 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm A | balanced_accuracy | -0.1476 | 0.0399 | 0.1252 | 15 | 5 | partition_means_t | -0.1476 | -0.1971 | -0.0980 | 0.0012 | -0.2088 | -0.0808 | 2000 |
| 50 | mean of regimes | arm C - arm A | balanced_accuracy | -0.0962 | 0.0219 | 0.0666 | 15 | 5 | partition_means_t | -0.0962 | -0.1234 | -0.0690 | 6.02e-04 | -0.1447 | -0.0440 | 2000 |
| 50 | sup_imnet | arm D - arm A | auroc | -0.0050 | 0.0173 | 0.0383 | 15 | 5 | partition_means_t | -0.0050 | -0.0265 | 0.0164 | 0.5498 | -0.0242 | 0.0157 | 2000 |
| 50 | ssl_imnet | arm D - arm A | auroc | -0.0487 | 0.0279 | 0.1191 | 15 | 5 | partition_means_t | -0.0487 | -0.0833 | -0.0140 | 0.0175 | -0.0790 | -0.0149 | 2000 |
| 50 | ssl_colon | arm D - arm A | auroc | -7.01e-04 | 0.0384 | 0.0708 | 15 | 5 | partition_means_t | -7.01e-04 | -0.0484 | 0.0470 | 0.9694 | -0.0645 | 0.0681 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm A | auroc | -0.0731 | 0.0726 | 0.1839 | 15 | 5 | partition_means_t | -0.0731 | -0.1633 | 0.0171 | 0.0875 | -0.1575 | 0.0289 | 2000 |
| 50 | mean of regimes | arm D - arm A | auroc | -0.0319 | 0.0334 | 0.0876 | 15 | 5 | partition_means_t | -0.0319 | -0.0734 | 0.0096 | 0.1000 | -0.0782 | 0.0228 | 2000 |
| 50 | sup_imnet | arm D - arm A | auprc | 0.0146 | 0.0411 | 0.0933 | 15 | 5 | partition_means_t | 0.0146 | -0.0364 | 0.0655 | 0.4720 | -0.0418 | 0.0572 | 2000 |
| 50 | ssl_imnet | arm D - arm A | auprc | -0.0405 | 0.0320 | 0.1447 | 15 | 5 | partition_means_t | -0.0405 | -0.0802 | -7.95e-04 | 0.0472 | -0.0932 | 0.0108 | 2000 |
| 50 | ssl_colon | arm D - arm A | auprc | 0.0589 | 0.0564 | 0.1089 | 15 | 5 | partition_means_t | 0.0589 | -0.0112 | 0.1289 | 0.0800 | -0.0907 | 0.1277 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm A | auprc | -0.0021 | 0.0652 | 0.1367 | 15 | 5 | partition_means_t | -0.0021 | -0.0831 | 0.0788 | 0.9449 | -0.1551 | 0.0720 | 2000 |
| 50 | mean of regimes | arm D - arm A | auprc | 0.0077 | 0.0439 | 0.0953 | 15 | 5 | partition_means_t | 0.0077 | -0.0469 | 0.0623 | 0.7151 | -0.0920 | 0.0631 | 2000 |
| 50 | sup_imnet | arm D - arm A | recall | -0.3132 | 0.0336 | 0.2055 | 15 | 5 | partition_means_t | -0.3132 | -0.3550 | -0.2714 | 3.15e-05 | -0.3645 | -0.2778 | 2000 |
| 50 | ssl_imnet | arm D - arm A | recall | -0.3764 | 0.0694 | 0.2247 | 15 | 5 | partition_means_t | -0.3764 | -0.4626 | -0.2902 | 2.66e-04 | -0.4340 | -0.3297 | 2000 |
| 50 | ssl_colon | arm D - arm A | recall | -0.3248 | 0.0769 | 0.2070 | 15 | 5 | partition_means_t | -0.3248 | -0.4203 | -0.2293 | 7.00e-04 | -0.4165 | -0.2651 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm A | recall | -0.3459 | 0.1247 | 0.2251 | 15 | 5 | partition_means_t | -0.3459 | -0.5007 | -0.1910 | 0.0034 | -0.3987 | -0.3002 | 2000 |
| 50 | mean of regimes | arm D - arm A | recall | -0.3401 | 0.0508 | 0.1202 | 15 | 5 | partition_means_t | -0.3401 | -0.4032 | -0.2769 | 1.16e-04 | -0.3987 | -0.3014 | 2000 |
| 50 | sup_imnet | arm D - arm A | specificity | 0.2240 | 0.0945 | 0.1402 | 15 | 5 | partition_means_t | 0.2240 | 0.1067 | 0.3413 | 0.0061 | 0.1376 | 0.3202 | 2000 |
| 50 | ssl_imnet | arm D - arm A | specificity | 0.3067 | 0.0755 | 0.1736 | 15 | 5 | partition_means_t | 0.3067 | 0.2130 | 0.4004 | 8.13e-04 | 0.2216 | 0.3941 | 2000 |
| 50 | ssl_colon | arm D - arm A | specificity | 0.2309 | 0.0342 | 0.1742 | 15 | 5 | partition_means_t | 0.2309 | 0.1884 | 0.2734 | 1.13e-04 | 0.1655 | 0.3030 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm A | specificity | 0.1130 | 0.1279 | 0.2905 | 15 | 5 | partition_means_t | 0.1130 | -0.0458 | 0.2719 | 0.1194 | -0.0066 | 0.2406 | 2000 |
| 50 | mean of regimes | arm D - arm A | specificity | 0.2187 | 0.0168 | 0.1264 | 15 | 5 | partition_means_t | 0.2187 | 0.1978 | 0.2395 | 8.21e-06 | 0.1381 | 0.3090 | 2000 |
| 50 | sup_imnet | arm D - arm A | balanced_accuracy | -0.0446 | 0.0449 | 0.0697 | 15 | 5 | partition_means_t | -0.0446 | -0.1003 | 0.0111 | 0.0905 | -0.0965 | 0.0074 | 2000 |
| 50 | ssl_imnet | arm D - arm A | balanced_accuracy | -0.0349 | 0.0150 | 0.0704 | 15 | 5 | partition_means_t | -0.0349 | -0.0535 | -0.0163 | 0.0065 | -0.0873 | 0.0147 | 2000 |
| 50 | ssl_colon | arm D - arm A | balanced_accuracy | -0.0470 | 0.0226 | 0.0496 | 15 | 5 | partition_means_t | -0.0470 | -0.0750 | -0.0189 | 0.0097 | -0.1011 | 0.0013 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm A | balanced_accuracy | -0.1164 | 0.0561 | 0.1255 | 15 | 5 | partition_means_t | -0.1164 | -0.1861 | -0.0468 | 0.0097 | -0.1799 | -0.0459 | 2000 |
| 50 | mean of regimes | arm D - arm A | balanced_accuracy | -0.0607 | 0.0274 | 0.0457 | 15 | 5 | partition_means_t | -0.0607 | -0.0947 | -0.0267 | 0.0077 | -0.1101 | -0.0111 | 2000 |
| 50 | sup_imnet | arm D - arm C | recall | 0.1471 | 0.1740 | 0.2403 | 15 | 5 | partition_means_t | 0.1471 | -0.0690 | 0.3631 | 0.1318 | 0.1363 | 0.1540 | 2000 |
| 50 | ssl_imnet | arm D - arm C | recall | -0.0424 | 0.0726 | 0.3913 | 15 | 5 | partition_means_t | -0.0424 | -0.1326 | 0.0478 | 0.2616 | -0.0618 | -0.0179 | 2000 |
| 50 | ssl_colon | arm D - arm C | recall | 5.97e-04 | 0.2402 | 0.3802 | 15 | 5 | partition_means_t | 5.97e-04 | -0.2977 | 0.2989 | 0.9958 | -0.0575 | 0.0347 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm C | recall | 0.1658 | 0.2626 | 0.3472 | 15 | 5 | partition_means_t | 0.1658 | -0.1603 | 0.4919 | 0.2308 | 0.1386 | 0.1882 | 2000 |
| 50 | mean of regimes | arm D - arm C | recall | 0.0678 | 0.0656 | 0.1575 | 15 | 5 | partition_means_t | 0.0678 | -0.0137 | 0.1492 | 0.0820 | 0.0439 | 0.0842 | 2000 |
| 50 | sup_imnet | arm D - arm C | specificity | -0.0360 | 0.1171 | 0.1395 | 15 | 5 | partition_means_t | -0.0360 | -0.1814 | 0.1094 | 0.5296 | -0.0569 | -0.0181 | 2000 |
| 50 | ssl_imnet | arm D - arm C | specificity | 0.1149 | 0.0799 | 0.2804 | 15 | 5 | partition_means_t | 0.1149 | 0.0157 | 0.2140 | 0.0324 | 0.0986 | 0.1295 | 2000 |
| 50 | ssl_colon | arm D - arm C | specificity | 0.0376 | 0.1771 | 0.2796 | 15 | 5 | partition_means_t | 0.0376 | -0.1822 | 0.2575 | 0.6595 | 0.0103 | 0.0594 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm C | specificity | -0.1036 | 0.1759 | 0.2480 | 15 | 5 | partition_means_t | -0.1036 | -0.3220 | 0.1149 | 0.2584 | -0.1143 | -0.0935 | 2000 |
| 50 | mean of regimes | arm D - arm C | specificity | 0.0032 | 0.0405 | 0.0799 | 15 | 5 | partition_means_t | 0.0032 | -0.0470 | 0.0535 | 0.8671 | -0.0109 | 0.0149 | 2000 |
| 50 | sup_imnet | arm D - arm C | balanced_accuracy | 0.0555 | 0.0318 | 0.0917 | 15 | 5 | partition_means_t | 0.0555 | 0.0160 | 0.0950 | 0.0175 | 0.0436 | 0.0652 | 2000 |
| 50 | ssl_imnet | arm D - arm C | balanced_accuracy | 0.0362 | 0.0435 | 0.1057 | 15 | 5 | partition_means_t | 0.0362 | -0.0178 | 0.0902 | 0.1362 | 0.0234 | 0.0503 | 2000 |
| 50 | ssl_colon | arm D - arm C | balanced_accuracy | 0.0191 | 0.0451 | 0.0980 | 15 | 5 | partition_means_t | 0.0191 | -0.0369 | 0.0751 | 0.3971 | -0.0117 | 0.0400 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm C | balanced_accuracy | 0.0311 | 0.0706 | 0.0689 | 15 | 5 | partition_means_t | 0.0311 | -0.0565 | 0.1188 | 0.3799 | 0.0157 | 0.0437 | 2000 |
| 50 | mean of regimes | arm D - arm C | balanced_accuracy | 0.0355 | 0.0140 | 0.0496 | 15 | 5 | partition_means_t | 0.0355 | 0.0181 | 0.0529 | 0.0048 | 0.0213 | 0.0457 | 2000 |
| 50 | sup_imnet | arm D - arm B | auroc | -0.0050 | 0.0173 | 0.0383 | 15 | 5 | partition_means_t | -0.0050 | -0.0265 | 0.0164 | 0.5498 | -0.0242 | 0.0157 | 2000 |
| 50 | ssl_imnet | arm D - arm B | auroc | -0.0487 | 0.0279 | 0.1191 | 15 | 5 | partition_means_t | -0.0487 | -0.0833 | -0.0140 | 0.0175 | -0.0790 | -0.0149 | 2000 |
| 50 | ssl_colon | arm D - arm B | auroc | -7.01e-04 | 0.0384 | 0.0708 | 15 | 5 | partition_means_t | -7.01e-04 | -0.0484 | 0.0470 | 0.9694 | -0.0645 | 0.0681 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm B | auroc | -0.0731 | 0.0726 | 0.1839 | 15 | 5 | partition_means_t | -0.0731 | -0.1633 | 0.0171 | 0.0875 | -0.1575 | 0.0289 | 2000 |
| 50 | mean of regimes | arm D - arm B | auroc | -0.0319 | 0.0334 | 0.0876 | 15 | 5 | partition_means_t | -0.0319 | -0.0734 | 0.0096 | 0.1000 | -0.0782 | 0.0228 | 2000 |
| 50 | sup_imnet | arm D - arm B | auprc | 0.0146 | 0.0411 | 0.0933 | 15 | 5 | partition_means_t | 0.0146 | -0.0364 | 0.0655 | 0.4720 | -0.0418 | 0.0572 | 2000 |
| 50 | ssl_imnet | arm D - arm B | auprc | -0.0405 | 0.0320 | 0.1447 | 15 | 5 | partition_means_t | -0.0405 | -0.0802 | -7.95e-04 | 0.0472 | -0.0932 | 0.0108 | 2000 |
| 50 | ssl_colon | arm D - arm B | auprc | 0.0589 | 0.0564 | 0.1089 | 15 | 5 | partition_means_t | 0.0589 | -0.0112 | 0.1289 | 0.0800 | -0.0907 | 0.1277 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm B | auprc | -0.0021 | 0.0652 | 0.1367 | 15 | 5 | partition_means_t | -0.0021 | -0.0831 | 0.0788 | 0.9449 | -0.1551 | 0.0720 | 2000 |
| 50 | mean of regimes | arm D - arm B | auprc | 0.0077 | 0.0439 | 0.0953 | 15 | 5 | partition_means_t | 0.0077 | -0.0469 | 0.0623 | 0.7151 | -0.0920 | 0.0631 | 2000 |
| 50 | sup_imnet | arm D - arm B | recall | -0.2355 | 0.0769 | 0.2006 | 15 | 5 | partition_means_t | -0.2355 | -0.3310 | -0.1401 | 0.0024 | -0.2568 | -0.2174 | 2000 |
| 50 | ssl_imnet | arm D - arm B | recall | -0.2531 | 0.1076 | 0.2224 | 15 | 5 | partition_means_t | -0.2531 | -0.3867 | -0.1195 | 0.0063 | -0.2802 | -0.2244 | 2000 |
| 50 | ssl_colon | arm D - arm B | recall | -0.2541 | 0.0536 | 0.2222 | 15 | 5 | partition_means_t | -0.2541 | -0.3206 | -0.1876 | 4.47e-04 | -0.3232 | -0.2109 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm B | recall | -0.3431 | 0.1058 | 0.2629 | 15 | 5 | partition_means_t | -0.3431 | -0.4744 | -0.2118 | 0.0019 | -0.3902 | -0.2967 | 2000 |
| 50 | mean of regimes | arm D - arm B | recall | -0.2715 | 0.0509 | 0.1288 | 15 | 5 | partition_means_t | -0.2715 | -0.3347 | -0.2082 | 2.84e-04 | -0.3109 | -0.2392 | 2000 |
| 50 | sup_imnet | arm D - arm B | specificity | 0.1684 | 0.1195 | 0.2518 | 15 | 5 | partition_means_t | 0.1684 | 0.0200 | 0.3168 | 0.0345 | 0.1130 | 0.2328 | 2000 |
| 50 | ssl_imnet | arm D - arm B | specificity | 0.1361 | 0.1496 | 0.1714 | 15 | 5 | partition_means_t | 0.1361 | -0.0497 | 0.3218 | 0.1117 | 0.0761 | 0.1965 | 2000 |
| 50 | ssl_colon | arm D - arm B | specificity | 0.1748 | 0.0799 | 0.2081 | 15 | 5 | partition_means_t | 0.1748 | 0.0756 | 0.2740 | 0.0081 | 0.1229 | 0.2352 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm B | specificity | 0.1598 | 0.1418 | 0.2803 | 15 | 5 | partition_means_t | 0.1598 | -0.0163 | 0.3359 | 0.0654 | 0.0587 | 0.2665 | 2000 |
| 50 | mean of regimes | arm D - arm B | specificity | 0.1598 | 0.0716 | 0.1448 | 15 | 5 | partition_means_t | 0.1598 | 0.0708 | 0.2487 | 0.0076 | 0.0961 | 0.2314 | 2000 |
| 50 | sup_imnet | arm D - arm B | balanced_accuracy | -0.0336 | 0.0550 | 0.0914 | 15 | 5 | partition_means_t | -0.0336 | -0.1019 | 0.0348 | 0.2444 | -0.0640 | 5.12e-04 | 2000 |
| 50 | ssl_imnet | arm D - arm B | balanced_accuracy | -0.0585 | 0.0240 | 0.0872 | 15 | 5 | partition_means_t | -0.0585 | -0.0884 | -0.0287 | 0.0055 | -0.0918 | -0.0240 | 2000 |
| 50 | ssl_colon | arm D - arm B | balanced_accuracy | -0.0397 | 0.0357 | 0.0368 | 15 | 5 | partition_means_t | -0.0397 | -0.0840 | 0.0047 | 0.0680 | -0.0840 | -5.44e-04 | 2000 |
| 50 | ssl_imnet_matched | arm D - arm B | balanced_accuracy | -0.0917 | 0.0824 | 0.1225 | 15 | 5 | partition_means_t | -0.0917 | -0.1940 | 0.0107 | 0.0677 | -0.1446 | -0.0310 | 2000 |
| 50 | mean of regimes | arm D - arm B | balanced_accuracy | -0.0558 | 0.0433 | 0.0489 | 15 | 5 | partition_means_t | -0.0558 | -0.1096 | -0.0021 | 0.0448 | -0.0928 | -0.0173 | 2000 |
| 50 | sup_imnet | arm C - arm B | auroc | -0.0050 | 0.0173 | 0.0383 | 15 | 5 | partition_means_t | -0.0050 | -0.0265 | 0.0164 | 0.5498 | -0.0242 | 0.0157 | 2000 |
| 50 | ssl_imnet | arm C - arm B | auroc | -0.0487 | 0.0279 | 0.1191 | 15 | 5 | partition_means_t | -0.0487 | -0.0833 | -0.0140 | 0.0175 | -0.0790 | -0.0149 | 2000 |
| 50 | ssl_colon | arm C - arm B | auroc | -7.01e-04 | 0.0384 | 0.0708 | 15 | 5 | partition_means_t | -7.01e-04 | -0.0484 | 0.0470 | 0.9694 | -0.0645 | 0.0681 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm B | auroc | -0.0731 | 0.0726 | 0.1839 | 15 | 5 | partition_means_t | -0.0731 | -0.1633 | 0.0171 | 0.0875 | -0.1575 | 0.0289 | 2000 |
| 50 | mean of regimes | arm C - arm B | auroc | -0.0319 | 0.0334 | 0.0876 | 15 | 5 | partition_means_t | -0.0319 | -0.0734 | 0.0096 | 0.1000 | -0.0782 | 0.0228 | 2000 |
| 50 | sup_imnet | arm C - arm B | auprc | 0.0146 | 0.0411 | 0.0933 | 15 | 5 | partition_means_t | 0.0146 | -0.0364 | 0.0655 | 0.4720 | -0.0418 | 0.0572 | 2000 |
| 50 | ssl_imnet | arm C - arm B | auprc | -0.0405 | 0.0320 | 0.1447 | 15 | 5 | partition_means_t | -0.0405 | -0.0802 | -7.95e-04 | 0.0472 | -0.0932 | 0.0108 | 2000 |
| 50 | ssl_colon | arm C - arm B | auprc | 0.0589 | 0.0564 | 0.1089 | 15 | 5 | partition_means_t | 0.0589 | -0.0112 | 0.1289 | 0.0800 | -0.0907 | 0.1277 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm B | auprc | -0.0021 | 0.0652 | 0.1367 | 15 | 5 | partition_means_t | -0.0021 | -0.0831 | 0.0788 | 0.9449 | -0.1551 | 0.0720 | 2000 |
| 50 | mean of regimes | arm C - arm B | auprc | 0.0077 | 0.0439 | 0.0953 | 15 | 5 | partition_means_t | 0.0077 | -0.0469 | 0.0623 | 0.7151 | -0.0920 | 0.0631 | 2000 |
| 50 | sup_imnet | arm C - arm B | recall | -0.3826 | 0.1458 | 0.2436 | 15 | 5 | partition_means_t | -0.3826 | -0.5637 | -0.2015 | 0.0042 | -0.4040 | -0.3632 | 2000 |
| 50 | ssl_imnet | arm C - arm B | recall | -0.2107 | 0.1528 | 0.3565 | 15 | 5 | partition_means_t | -0.2107 | -0.4004 | -0.0210 | 0.0368 | -0.2321 | -0.1865 | 2000 |
| 50 | ssl_colon | arm C - arm B | recall | -0.2547 | 0.2164 | 0.3594 | 15 | 5 | partition_means_t | -0.2547 | -0.5234 | 0.0140 | 0.0581 | -0.2840 | -0.2184 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm B | recall | -0.5089 | 0.2170 | 0.4288 | 15 | 5 | partition_means_t | -0.5089 | -0.7784 | -0.2395 | 0.0063 | -0.5303 | -0.4770 | 2000 |
| 50 | mean of regimes | arm C - arm B | recall | -0.3392 | 0.0332 | 0.1914 | 15 | 5 | partition_means_t | -0.3392 | -0.3804 | -0.2980 | 2.16e-05 | -0.3608 | -0.3143 | 2000 |
| 50 | sup_imnet | arm C - arm B | specificity | 0.2044 | 0.0396 | 0.2888 | 15 | 5 | partition_means_t | 0.2044 | 0.1552 | 0.2537 | 3.23e-04 | 0.1334 | 0.2871 | 2000 |
| 50 | ssl_imnet | arm C - arm B | specificity | 0.0212 | 0.1836 | 0.3589 | 15 | 5 | partition_means_t | 0.0212 | -0.2067 | 0.2491 | 0.8089 | -0.0482 | 0.0963 | 2000 |
| 50 | ssl_colon | arm C - arm B | specificity | 0.1372 | 0.1667 | 0.2594 | 15 | 5 | partition_means_t | 0.1372 | -0.0699 | 0.3442 | 0.1397 | 0.0756 | 0.2058 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm B | specificity | 0.2633 | 0.2233 | 0.3640 | 15 | 5 | partition_means_t | 0.2633 | -0.0139 | 0.5406 | 0.0578 | 0.1624 | 0.3719 | 2000 |
| 50 | mean of regimes | arm C - arm B | specificity | 0.1565 | 0.0761 | 0.1418 | 15 | 5 | partition_means_t | 0.1565 | 0.0620 | 0.2511 | 0.0101 | 0.0838 | 0.2373 | 2000 |
| 50 | sup_imnet | arm C - arm B | balanced_accuracy | -0.0891 | 0.0790 | 0.0850 | 15 | 5 | partition_means_t | -0.0891 | -0.1871 | 0.0090 | 0.0652 | -0.1255 | -0.0457 | 2000 |
| 50 | ssl_imnet | arm C - arm B | balanced_accuracy | -0.0947 | 0.0426 | 0.1294 | 15 | 5 | partition_means_t | -0.0947 | -0.1476 | -0.0419 | 0.0076 | -0.1314 | -0.0559 | 2000 |
| 50 | ssl_colon | arm C - arm B | balanced_accuracy | -0.0588 | 0.0749 | 0.0918 | 15 | 5 | partition_means_t | -0.0588 | -0.1518 | 0.0343 | 0.1543 | -0.0931 | -0.0187 | 2000 |
| 50 | ssl_imnet_matched | arm C - arm B | balanced_accuracy | -0.1228 | 0.0519 | 0.1122 | 15 | 5 | partition_means_t | -0.1228 | -0.1872 | -0.0584 | 0.0061 | -0.1738 | -0.0654 | 2000 |
| 50 | mean of regimes | arm C - arm B | balanced_accuracy | -0.0913 | 0.0391 | 0.0655 | 15 | 5 | partition_means_t | -0.0913 | -0.1399 | -0.0428 | 0.0064 | -0.1285 | -0.0481 | 2000 |
| 100 | sup_imnet | arm B - arm A | recall | -0.0860 | 0.0593 | 0.1154 | 15 | 5 | partition_means_t | -0.0860 | -0.1597 | -0.0124 | 0.0316 | -0.1205 | -0.0642 | 2000 |
| 100 | ssl_imnet | arm B - arm A | recall | -0.0824 | 0.0844 | 0.1003 | 15 | 5 | partition_means_t | -0.0824 | -0.1872 | 0.0224 | 0.0944 | -0.1084 | -0.0642 | 2000 |
| 100 | ssl_colon | arm B - arm A | recall | -0.0332 | 0.0747 | 0.1271 | 15 | 5 | partition_means_t | -0.0332 | -0.1260 | 0.0596 | 0.3765 | -0.0429 | -0.0236 | 2000 |
| 100 | ssl_imnet_matched | arm B - arm A | recall | 0.0626 | 0.1006 | 0.1021 | 15 | 5 | partition_means_t | 0.0626 | -0.0624 | 0.1875 | 0.2368 | 0.0482 | 0.0852 | 2000 |
| 100 | mean of regimes | arm B - arm A | recall | -0.0348 | 0.0536 | 0.0761 | 15 | 5 | partition_means_t | -0.0348 | -0.1013 | 0.0318 | 0.2205 | -0.0480 | -0.0250 | 2000 |
| 100 | sup_imnet | arm B - arm A | specificity | 0.0910 | 0.0730 | 0.1874 | 15 | 5 | partition_means_t | 0.0910 | 2.86e-04 | 0.1816 | 0.0496 | 0.0568 | 0.1265 | 2000 |
| 100 | ssl_imnet | arm B - arm A | specificity | 0.0864 | 0.1592 | 0.1228 | 15 | 5 | partition_means_t | 0.0864 | -0.1113 | 0.2841 | 0.2919 | 0.0577 | 0.1198 | 2000 |
| 100 | ssl_colon | arm B - arm A | specificity | -0.0122 | 0.1134 | 0.1712 | 15 | 5 | partition_means_t | -0.0122 | -0.1530 | 0.1286 | 0.8215 | -0.0328 | 0.0110 | 2000 |
| 100 | ssl_imnet_matched | arm B - arm A | specificity | -0.1381 | 0.1362 | 0.1571 | 15 | 5 | partition_means_t | -0.1381 | -0.3073 | 0.0310 | 0.0860 | -0.1807 | -0.0917 | 2000 |
| 100 | mean of regimes | arm B - arm A | specificity | 0.0067 | 0.0840 | 0.1168 | 15 | 5 | partition_means_t | 0.0067 | -0.0975 | 0.1110 | 0.8662 | -0.0212 | 0.0369 | 2000 |
| 100 | sup_imnet | arm B - arm A | balanced_accuracy | 0.0025 | 0.0330 | 0.0498 | 15 | 5 | partition_means_t | 0.0025 | -0.0385 | 0.0434 | 0.8760 | -0.0217 | 0.0232 | 2000 |
| 100 | ssl_imnet | arm B - arm A | balanced_accuracy | 0.0020 | 0.0419 | 0.0304 | 15 | 5 | partition_means_t | 0.0020 | -0.0500 | 0.0540 | 0.9207 | -0.0179 | 0.0214 | 2000 |
| 100 | ssl_colon | arm B - arm A | balanced_accuracy | -0.0227 | 0.0199 | 0.0442 | 15 | 5 | partition_means_t | -0.0227 | -0.0475 | 0.0020 | 0.0635 | -0.0337 | -0.0103 | 2000 |
| 100 | ssl_imnet_matched | arm B - arm A | balanced_accuracy | -0.0378 | 0.0234 | 0.0352 | 15 | 5 | partition_means_t | -0.0378 | -0.0668 | -0.0087 | 0.0225 | -0.0613 | -0.0115 | 2000 |
| 100 | mean of regimes | arm B - arm A | balanced_accuracy | -0.0140 | 0.0202 | 0.0264 | 15 | 5 | partition_means_t | -0.0140 | -0.0390 | 0.0110 | 0.1949 | -0.0295 | 0.0020 | 2000 |
| 100 | sup_imnet | arm C - arm A | auroc | -0.0177 | 0.0262 | 0.0612 | 15 | 5 | partition_means_t | -0.0177 | -0.0502 | 0.0148 | 0.2054 | -0.0551 | 0.0242 | 2000 |
| 100 | ssl_imnet | arm C - arm A | auroc | -0.0074 | 0.0477 | 0.0949 | 15 | 5 | partition_means_t | -0.0074 | -0.0667 | 0.0519 | 0.7472 | -0.0681 | 0.0561 | 2000 |
| 100 | ssl_colon | arm C - arm A | auroc | -0.0680 | 0.0994 | 0.1478 | 15 | 5 | partition_means_t | -0.0680 | -0.1915 | 0.0554 | 0.2006 | -0.1911 | 0.0756 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm A | auroc | -0.1035 | 0.0852 | 0.1779 | 15 | 5 | partition_means_t | -0.1035 | -0.2092 | 0.0023 | 0.0532 | -0.2225 | 0.0413 | 2000 |
| 100 | mean of regimes | arm C - arm A | auroc | -0.0491 | 0.0352 | 0.0967 | 15 | 5 | partition_means_t | -0.0491 | -0.0929 | -0.0054 | 0.0356 | -0.1316 | 0.0484 | 2000 |
| 100 | sup_imnet | arm C - arm A | auprc | 0.0180 | 0.0592 | 0.1125 | 15 | 5 | partition_means_t | 0.0180 | -0.0555 | 0.0916 | 0.5337 | -0.0811 | 0.0871 | 2000 |
| 100 | ssl_imnet | arm C - arm A | auprc | 0.0384 | 0.0781 | 0.1579 | 15 | 5 | partition_means_t | 0.0384 | -0.0586 | 0.1355 | 0.3330 | -0.0745 | 0.1226 | 2000 |
| 100 | ssl_colon | arm C - arm A | auprc | 0.0200 | 0.1163 | 0.1593 | 15 | 5 | partition_means_t | 0.0200 | -0.1244 | 0.1644 | 0.7204 | -0.2302 | 0.1599 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm A | auprc | -0.0041 | 0.0995 | 0.1634 | 15 | 5 | partition_means_t | -0.0041 | -0.1277 | 0.1194 | 0.9305 | -0.2243 | 0.1095 | 2000 |
| 100 | mean of regimes | arm C - arm A | auprc | 0.0181 | 0.0603 | 0.1173 | 15 | 5 | partition_means_t | 0.0181 | -0.0568 | 0.0929 | 0.5392 | -0.1487 | 0.1185 | 2000 |
| 100 | sup_imnet | arm C - arm A | recall | -0.5826 | 0.1803 | 0.2324 | 15 | 5 | partition_means_t | -0.5826 | -0.8064 | -0.3588 | 0.0019 | -0.6114 | -0.5634 | 2000 |
| 100 | ssl_imnet | arm C - arm A | recall | -0.3349 | 0.1160 | 0.2862 | 15 | 5 | partition_means_t | -0.3349 | -0.4790 | -0.1909 | 0.0030 | -0.3823 | -0.3058 | 2000 |
| 100 | ssl_colon | arm C - arm A | recall | -0.2729 | 0.2281 | 0.2639 | 15 | 5 | partition_means_t | -0.2729 | -0.5561 | 0.0103 | 0.0555 | -0.3386 | -0.1952 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm A | recall | -0.5325 | 0.2366 | 0.2695 | 15 | 5 | partition_means_t | -0.5325 | -0.8263 | -0.2387 | 0.0073 | -0.5686 | -0.4857 | 2000 |
| 100 | mean of regimes | arm C - arm A | recall | -0.4307 | 0.0866 | 0.1243 | 15 | 5 | partition_means_t | -0.4307 | -0.5383 | -0.3232 | 3.72e-04 | -0.4717 | -0.3909 | 2000 |
| 100 | sup_imnet | arm C - arm A | specificity | 0.3055 | 0.0385 | 0.0512 | 15 | 5 | partition_means_t | 0.3055 | 0.2577 | 0.3534 | 5.93e-05 | 0.1846 | 0.4458 | 2000 |
| 100 | ssl_imnet | arm C - arm A | specificity | 0.2741 | 0.1151 | 0.2580 | 15 | 5 | partition_means_t | 0.2741 | 0.1312 | 0.4171 | 0.0060 | 0.1608 | 0.3875 | 2000 |
| 100 | ssl_colon | arm C - arm A | specificity | 0.1021 | 0.1810 | 0.2204 | 15 | 5 | partition_means_t | 0.1021 | -0.1227 | 0.3268 | 0.2759 | -0.0350 | 0.2560 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm A | specificity | 0.2076 | 0.1587 | 0.3440 | 15 | 5 | partition_means_t | 0.2076 | 0.0105 | 0.4047 | 0.0430 | 0.0714 | 0.3524 | 2000 |
| 100 | mean of regimes | arm C - arm A | specificity | 0.2223 | 0.0711 | 0.1530 | 15 | 5 | partition_means_t | 0.2223 | 0.1341 | 0.3106 | 0.0022 | 0.1003 | 0.3557 | 2000 |
| 100 | sup_imnet | arm C - arm A | balanced_accuracy | -0.1385 | 0.0922 | 0.0983 | 15 | 5 | partition_means_t | -0.1385 | -0.2530 | -0.0240 | 0.0283 | -0.2011 | -0.0676 | 2000 |
| 100 | ssl_imnet | arm C - arm A | balanced_accuracy | -0.0304 | 0.0436 | 0.1186 | 15 | 5 | partition_means_t | -0.0304 | -0.0846 | 0.0238 | 0.1944 | -0.0914 | 0.0295 | 2000 |
| 100 | ssl_colon | arm C - arm A | balanced_accuracy | -0.0854 | 0.0423 | 0.1247 | 15 | 5 | partition_means_t | -0.0854 | -0.1379 | -0.0329 | 0.0107 | -0.1610 | -5.73e-06 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm A | balanced_accuracy | -0.1624 | 0.0611 | 0.1080 | 15 | 5 | partition_means_t | -0.1624 | -0.2383 | -0.0866 | 0.0040 | -0.2322 | -0.0829 | 2000 |
| 100 | mean of regimes | arm C - arm A | balanced_accuracy | -0.1042 | 0.0172 | 0.0682 | 15 | 5 | partition_means_t | -0.1042 | -0.1255 | -0.0829 | 1.71e-04 | -0.1683 | -0.0336 | 2000 |
| 100 | sup_imnet | arm D - arm A | auroc | -0.0177 | 0.0262 | 0.0612 | 15 | 5 | partition_means_t | -0.0177 | -0.0502 | 0.0148 | 0.2054 | -0.0551 | 0.0242 | 2000 |
| 100 | ssl_imnet | arm D - arm A | auroc | -0.0074 | 0.0477 | 0.0949 | 15 | 5 | partition_means_t | -0.0074 | -0.0667 | 0.0519 | 0.7472 | -0.0681 | 0.0561 | 2000 |
| 100 | ssl_colon | arm D - arm A | auroc | -0.0680 | 0.0994 | 0.1478 | 15 | 5 | partition_means_t | -0.0680 | -0.1915 | 0.0554 | 0.2006 | -0.1911 | 0.0756 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm A | auroc | -0.1035 | 0.0852 | 0.1779 | 15 | 5 | partition_means_t | -0.1035 | -0.2092 | 0.0023 | 0.0532 | -0.2225 | 0.0413 | 2000 |
| 100 | mean of regimes | arm D - arm A | auroc | -0.0491 | 0.0352 | 0.0967 | 15 | 5 | partition_means_t | -0.0491 | -0.0929 | -0.0054 | 0.0356 | -0.1316 | 0.0484 | 2000 |
| 100 | sup_imnet | arm D - arm A | auprc | 0.0180 | 0.0592 | 0.1125 | 15 | 5 | partition_means_t | 0.0180 | -0.0555 | 0.0916 | 0.5337 | -0.0811 | 0.0871 | 2000 |
| 100 | ssl_imnet | arm D - arm A | auprc | 0.0384 | 0.0781 | 0.1579 | 15 | 5 | partition_means_t | 0.0384 | -0.0586 | 0.1355 | 0.3330 | -0.0745 | 0.1226 | 2000 |
| 100 | ssl_colon | arm D - arm A | auprc | 0.0200 | 0.1163 | 0.1593 | 15 | 5 | partition_means_t | 0.0200 | -0.1244 | 0.1644 | 0.7204 | -0.2302 | 0.1599 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm A | auprc | -0.0041 | 0.0995 | 0.1634 | 15 | 5 | partition_means_t | -0.0041 | -0.1277 | 0.1194 | 0.9305 | -0.2243 | 0.1095 | 2000 |
| 100 | mean of regimes | arm D - arm A | auprc | 0.0181 | 0.0603 | 0.1173 | 15 | 5 | partition_means_t | 0.0181 | -0.0568 | 0.0929 | 0.5392 | -0.1487 | 0.1185 | 2000 |
| 100 | sup_imnet | arm D - arm A | recall | -0.1632 | 0.0425 | 0.1808 | 15 | 5 | partition_means_t | -0.1632 | -0.2160 | -0.1103 | 0.0010 | -0.1992 | -0.1420 | 2000 |
| 100 | ssl_imnet | arm D - arm A | recall | -0.2612 | 0.1275 | 0.1483 | 15 | 5 | partition_means_t | -0.2612 | -0.4195 | -0.1029 | 0.0102 | -0.3254 | -0.2278 | 2000 |
| 100 | ssl_colon | arm D - arm A | recall | -0.2053 | 0.0892 | 0.1958 | 15 | 5 | partition_means_t | -0.2053 | -0.3160 | -0.0945 | 0.0068 | -0.2865 | -0.1305 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm A | recall | -0.1192 | 0.0808 | 0.1901 | 15 | 5 | partition_means_t | -0.1192 | -0.2195 | -0.0189 | 0.0300 | -0.1750 | -0.0599 | 2000 |
| 100 | mean of regimes | arm D - arm A | recall | -0.1872 | 0.0487 | 0.1177 | 15 | 5 | partition_means_t | -0.1872 | -0.2477 | -0.1267 | 0.0010 | -0.2412 | -0.1442 | 2000 |
| 100 | sup_imnet | arm D - arm A | specificity | 0.1122 | 0.0683 | 0.1901 | 15 | 5 | partition_means_t | 0.1122 | 0.0274 | 0.1969 | 0.0213 | 0.0257 | 0.2082 | 2000 |
| 100 | ssl_imnet | arm D - arm A | specificity | 0.2466 | 0.1080 | 0.2744 | 15 | 5 | partition_means_t | 0.2466 | 0.1124 | 0.3807 | 0.0070 | 0.1361 | 0.3605 | 2000 |
| 100 | ssl_colon | arm D - arm A | specificity | 0.0560 | 0.1263 | 0.2449 | 15 | 5 | partition_means_t | 0.0560 | -0.1008 | 0.2128 | 0.3774 | -0.0616 | 0.1910 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm A | specificity | -0.0781 | 0.1581 | 0.2863 | 15 | 5 | partition_means_t | -0.0781 | -0.2743 | 0.1182 | 0.3314 | -0.2033 | 0.0619 | 2000 |
| 100 | mean of regimes | arm D - arm A | specificity | 0.0842 | 0.0385 | 0.1734 | 15 | 5 | partition_means_t | 0.0842 | 0.0364 | 0.1319 | 0.0081 | -0.0219 | 0.2005 | 2000 |
| 100 | sup_imnet | arm D - arm A | balanced_accuracy | -0.0255 | 0.0206 | 0.0669 | 15 | 5 | partition_means_t | -0.0255 | -0.0510 | 2.14e-05 | 0.0501 | -0.0725 | 0.0236 | 2000 |
| 100 | ssl_imnet | arm D - arm A | balanced_accuracy | -0.0073 | 0.0311 | 0.0925 | 15 | 5 | partition_means_t | -0.0073 | -0.0460 | 0.0314 | 0.6275 | -0.0690 | 0.0525 | 2000 |
| 100 | ssl_colon | arm D - arm A | balanced_accuracy | -0.0746 | 0.0503 | 0.1086 | 15 | 5 | partition_means_t | -0.0746 | -0.1371 | -0.0121 | 0.0295 | -0.1458 | 0.0031 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm A | balanced_accuracy | -0.0986 | 0.0445 | 0.1136 | 15 | 5 | partition_means_t | -0.0986 | -0.1539 | -0.0434 | 0.0077 | -0.1677 | -0.0211 | 2000 |
| 100 | mean of regimes | arm D - arm A | balanced_accuracy | -0.0515 | 0.0202 | 0.0804 | 15 | 5 | partition_means_t | -0.0515 | -0.0766 | -0.0264 | 0.0047 | -0.1103 | 0.0115 | 2000 |
| 100 | sup_imnet | arm D - arm C | recall | 0.4194 | 0.1915 | 0.3293 | 15 | 5 | partition_means_t | 0.4194 | 0.1817 | 0.6572 | 0.0081 | 0.4088 | 0.4268 | 2000 |
| 100 | ssl_imnet | arm D - arm C | recall | 0.0738 | 0.2148 | 0.3728 | 15 | 5 | partition_means_t | 0.0738 | -0.1929 | 0.3404 | 0.4854 | 0.0569 | 0.0832 | 2000 |
| 100 | ssl_colon | arm D - arm C | recall | 0.0677 | 0.2332 | 0.3464 | 15 | 5 | partition_means_t | 0.0677 | -0.2219 | 0.3573 | 0.5519 | 0.0319 | 0.0958 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm C | recall | 0.4133 | 0.2809 | 0.4162 | 15 | 5 | partition_means_t | 0.4133 | 0.0645 | 0.7621 | 0.0302 | 0.3884 | 0.4365 | 2000 |
| 100 | mean of regimes | arm D - arm C | recall | 0.2435 | 0.0941 | 0.2261 | 15 | 5 | partition_means_t | 0.2435 | 0.1266 | 0.3604 | 0.0044 | 0.2234 | 0.2566 | 2000 |
| 100 | sup_imnet | arm D - arm C | specificity | -0.1934 | 0.0520 | 0.1855 | 15 | 5 | partition_means_t | -0.1934 | -0.2579 | -0.1288 | 0.0011 | -0.2438 | -0.1516 | 2000 |
| 100 | ssl_imnet | arm D - arm C | specificity | -0.0276 | 0.1540 | 0.3625 | 15 | 5 | partition_means_t | -0.0276 | -0.2188 | 0.1636 | 0.7093 | -0.0329 | -0.0217 | 2000 |
| 100 | ssl_colon | arm D - arm C | specificity | -0.0460 | 0.2299 | 0.2812 | 15 | 5 | partition_means_t | -0.0460 | -0.3315 | 0.2394 | 0.6774 | -0.0728 | -0.0196 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm C | specificity | -0.2857 | 0.2810 | 0.4099 | 15 | 5 | partition_means_t | -0.2857 | -0.6346 | 0.0632 | 0.0854 | -0.3131 | -0.2608 | 2000 |
| 100 | mean of regimes | arm D - arm C | specificity | -0.1382 | 0.0639 | 0.1858 | 15 | 5 | partition_means_t | -0.1382 | -0.2175 | -0.0588 | 0.0084 | -0.1585 | -0.1205 | 2000 |
| 100 | sup_imnet | arm D - arm C | balanced_accuracy | 0.1130 | 0.0831 | 0.1072 | 15 | 5 | partition_means_t | 0.1130 | 0.0098 | 0.2162 | 0.0384 | 0.0869 | 0.1334 | 2000 |
| 100 | ssl_imnet | arm D - arm C | balanced_accuracy | 0.0231 | 0.0395 | 0.1216 | 15 | 5 | partition_means_t | 0.0231 | -0.0259 | 0.0721 | 0.2611 | 0.0139 | 0.0288 | 2000 |
| 100 | ssl_colon | arm D - arm C | balanced_accuracy | 0.0108 | 0.0202 | 0.0638 | 15 | 5 | partition_means_t | 0.0108 | -0.0143 | 0.0359 | 0.2973 | -0.0118 | 0.0301 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm C | balanced_accuracy | 0.0638 | 0.0555 | 0.0562 | 15 | 5 | partition_means_t | 0.0638 | -0.0051 | 0.1327 | 0.0620 | 0.0446 | 0.0808 | 2000 |
| 100 | mean of regimes | arm D - arm C | balanced_accuracy | 0.0527 | 0.0315 | 0.0499 | 15 | 5 | partition_means_t | 0.0527 | 0.0136 | 0.0917 | 0.0200 | 0.0383 | 0.0640 | 2000 |
| 100 | sup_imnet | arm D - arm B | auroc | -0.0177 | 0.0262 | 0.0612 | 15 | 5 | partition_means_t | -0.0177 | -0.0502 | 0.0148 | 0.2054 | -0.0551 | 0.0242 | 2000 |
| 100 | ssl_imnet | arm D - arm B | auroc | -0.0074 | 0.0477 | 0.0949 | 15 | 5 | partition_means_t | -0.0074 | -0.0667 | 0.0519 | 0.7472 | -0.0681 | 0.0561 | 2000 |
| 100 | ssl_colon | arm D - arm B | auroc | -0.0680 | 0.0994 | 0.1478 | 15 | 5 | partition_means_t | -0.0680 | -0.1915 | 0.0554 | 0.2006 | -0.1911 | 0.0756 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm B | auroc | -0.1035 | 0.0852 | 0.1779 | 15 | 5 | partition_means_t | -0.1035 | -0.2092 | 0.0023 | 0.0532 | -0.2225 | 0.0413 | 2000 |
| 100 | mean of regimes | arm D - arm B | auroc | -0.0491 | 0.0352 | 0.0967 | 15 | 5 | partition_means_t | -0.0491 | -0.0929 | -0.0054 | 0.0356 | -0.1316 | 0.0484 | 2000 |
| 100 | sup_imnet | arm D - arm B | auprc | 0.0180 | 0.0592 | 0.1125 | 15 | 5 | partition_means_t | 0.0180 | -0.0555 | 0.0916 | 0.5337 | -0.0811 | 0.0871 | 2000 |
| 100 | ssl_imnet | arm D - arm B | auprc | 0.0384 | 0.0781 | 0.1579 | 15 | 5 | partition_means_t | 0.0384 | -0.0586 | 0.1355 | 0.3330 | -0.0745 | 0.1226 | 2000 |
| 100 | ssl_colon | arm D - arm B | auprc | 0.0200 | 0.1163 | 0.1593 | 15 | 5 | partition_means_t | 0.0200 | -0.1244 | 0.1644 | 0.7204 | -0.2302 | 0.1599 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm B | auprc | -0.0041 | 0.0995 | 0.1634 | 15 | 5 | partition_means_t | -0.0041 | -0.1277 | 0.1194 | 0.9305 | -0.2243 | 0.1095 | 2000 |
| 100 | mean of regimes | arm D - arm B | auprc | 0.0181 | 0.0603 | 0.1173 | 15 | 5 | partition_means_t | 0.0181 | -0.0568 | 0.0929 | 0.5392 | -0.1487 | 0.1185 | 2000 |
| 100 | sup_imnet | arm D - arm B | recall | -0.0771 | 0.0781 | 0.2160 | 15 | 5 | partition_means_t | -0.0771 | -0.1742 | 0.0199 | 0.0919 | -0.1005 | -0.0475 | 2000 |
| 100 | ssl_imnet | arm D - arm B | recall | -0.1788 | 0.1597 | 0.1806 | 15 | 5 | partition_means_t | -0.1788 | -0.3771 | 0.0196 | 0.0666 | -0.2235 | -0.1526 | 2000 |
| 100 | ssl_colon | arm D - arm B | recall | -0.1720 | 0.1028 | 0.1911 | 15 | 5 | partition_means_t | -0.1720 | -0.2996 | -0.0444 | 0.0201 | -0.2451 | -0.1046 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm B | recall | -0.1818 | 0.0490 | 0.2074 | 15 | 5 | partition_means_t | -0.1818 | -0.2426 | -0.1209 | 0.0012 | -0.2351 | -0.1340 | 2000 |
| 100 | mean of regimes | arm D - arm B | recall | -0.1524 | 0.0591 | 0.1449 | 15 | 5 | partition_means_t | -0.1524 | -0.2258 | -0.0790 | 0.0045 | -0.1993 | -0.1112 | 2000 |
| 100 | sup_imnet | arm D - arm B | specificity | 0.0212 | 0.0769 | 0.2884 | 15 | 5 | partition_means_t | 0.0212 | -0.0743 | 0.1167 | 0.5712 | -0.0446 | 0.0957 | 2000 |
| 100 | ssl_imnet | arm D - arm B | specificity | 0.1602 | 0.2281 | 0.2918 | 15 | 5 | partition_means_t | 0.1602 | -0.1230 | 0.4434 | 0.1913 | 0.0732 | 0.2524 | 2000 |
| 100 | ssl_colon | arm D - arm B | specificity | 0.0682 | 0.1366 | 0.2606 | 15 | 5 | partition_means_t | 0.0682 | -0.1014 | 0.2379 | 0.3266 | -0.0364 | 0.1879 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm B | specificity | 0.0601 | 0.0462 | 0.2817 | 15 | 5 | partition_means_t | 0.0601 | 0.0027 | 0.1174 | 0.0437 | -0.0343 | 0.1592 | 2000 |
| 100 | mean of regimes | arm D - arm B | specificity | 0.0774 | 0.0531 | 0.2087 | 15 | 5 | partition_means_t | 0.0774 | 0.0115 | 0.1434 | 0.0311 | -0.0050 | 0.1705 | 2000 |
| 100 | sup_imnet | arm D - arm B | balanced_accuracy | -0.0280 | 0.0416 | 0.0762 | 15 | 5 | partition_means_t | -0.0280 | -0.0796 | 0.0237 | 0.2071 | -0.0640 | 0.0112 | 2000 |
| 100 | ssl_imnet | arm D - arm B | balanced_accuracy | -0.0093 | 0.0638 | 0.0929 | 15 | 5 | partition_means_t | -0.0093 | -0.0886 | 0.0700 | 0.7611 | -0.0596 | 0.0399 | 2000 |
| 100 | ssl_colon | arm D - arm B | balanced_accuracy | -0.0519 | 0.0514 | 0.0985 | 15 | 5 | partition_means_t | -0.0519 | -0.1157 | 0.0119 | 0.0867 | -0.1150 | 0.0169 | 2000 |
| 100 | ssl_imnet_matched | arm D - arm B | balanced_accuracy | -0.0609 | 0.0239 | 0.0982 | 15 | 5 | partition_means_t | -0.0609 | -0.0905 | -0.0312 | 0.0047 | -0.1142 | -0.0047 | 2000 |
| 100 | mean of regimes | arm D - arm B | balanced_accuracy | -0.0375 | 0.0320 | 0.0744 | 15 | 5 | partition_means_t | -0.0375 | -0.0772 | 0.0022 | 0.0586 | -0.0839 | 0.0131 | 2000 |
| 100 | sup_imnet | arm C - arm B | auroc | -0.0177 | 0.0262 | 0.0612 | 15 | 5 | partition_means_t | -0.0177 | -0.0502 | 0.0148 | 0.2054 | -0.0551 | 0.0242 | 2000 |
| 100 | ssl_imnet | arm C - arm B | auroc | -0.0074 | 0.0477 | 0.0949 | 15 | 5 | partition_means_t | -0.0074 | -0.0667 | 0.0519 | 0.7472 | -0.0681 | 0.0561 | 2000 |
| 100 | ssl_colon | arm C - arm B | auroc | -0.0680 | 0.0994 | 0.1478 | 15 | 5 | partition_means_t | -0.0680 | -0.1915 | 0.0554 | 0.2006 | -0.1911 | 0.0756 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm B | auroc | -0.1035 | 0.0852 | 0.1779 | 15 | 5 | partition_means_t | -0.1035 | -0.2092 | 0.0023 | 0.0532 | -0.2225 | 0.0413 | 2000 |
| 100 | mean of regimes | arm C - arm B | auroc | -0.0491 | 0.0352 | 0.0967 | 15 | 5 | partition_means_t | -0.0491 | -0.0929 | -0.0054 | 0.0356 | -0.1316 | 0.0484 | 2000 |
| 100 | sup_imnet | arm C - arm B | auprc | 0.0180 | 0.0592 | 0.1125 | 15 | 5 | partition_means_t | 0.0180 | -0.0555 | 0.0916 | 0.5337 | -0.0811 | 0.0871 | 2000 |
| 100 | ssl_imnet | arm C - arm B | auprc | 0.0384 | 0.0781 | 0.1579 | 15 | 5 | partition_means_t | 0.0384 | -0.0586 | 0.1355 | 0.3330 | -0.0745 | 0.1226 | 2000 |
| 100 | ssl_colon | arm C - arm B | auprc | 0.0200 | 0.1163 | 0.1593 | 15 | 5 | partition_means_t | 0.0200 | -0.1244 | 0.1644 | 0.7204 | -0.2302 | 0.1599 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm B | auprc | -0.0041 | 0.0995 | 0.1634 | 15 | 5 | partition_means_t | -0.0041 | -0.1277 | 0.1194 | 0.9305 | -0.2243 | 0.1095 | 2000 |
| 100 | mean of regimes | arm C - arm B | auprc | 0.0181 | 0.0603 | 0.1173 | 15 | 5 | partition_means_t | 0.0181 | -0.0568 | 0.0929 | 0.5392 | -0.1487 | 0.1185 | 2000 |
| 100 | sup_imnet | arm C - arm B | recall | -0.4966 | 0.2062 | 0.2624 | 15 | 5 | partition_means_t | -0.4966 | -0.7526 | -0.2405 | 0.0057 | -0.5141 | -0.4712 | 2000 |
| 100 | ssl_imnet | arm C - arm B | recall | -0.2525 | 0.1742 | 0.3225 | 15 | 5 | partition_means_t | -0.2525 | -0.4689 | -0.0362 | 0.0316 | -0.2817 | -0.2270 | 2000 |
| 100 | ssl_colon | arm C - arm B | recall | -0.2397 | 0.2328 | 0.3256 | 15 | 5 | partition_means_t | -0.2397 | -0.5287 | 0.0493 | 0.0827 | -0.3022 | -0.1656 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm B | recall | -0.5951 | 0.2757 | 0.2990 | 15 | 5 | partition_means_t | -0.5951 | -0.9374 | -0.2528 | 0.0085 | -0.6264 | -0.5628 | 2000 |
| 100 | mean of regimes | arm C - arm B | recall | -0.3960 | 0.0747 | 0.1673 | 15 | 5 | partition_means_t | -0.3960 | -0.4887 | -0.3033 | 2.89e-04 | -0.4301 | -0.3579 | 2000 |
| 100 | sup_imnet | arm C - arm B | specificity | 0.2146 | 0.0783 | 0.2028 | 15 | 5 | partition_means_t | 0.2146 | 0.1173 | 0.3118 | 0.0036 | 0.1160 | 0.3293 | 2000 |
| 100 | ssl_imnet | arm C - arm B | specificity | 0.1878 | 0.2568 | 0.2917 | 15 | 5 | partition_means_t | 0.1878 | -0.1311 | 0.5066 | 0.1774 | 0.0995 | 0.2820 | 2000 |
| 100 | ssl_colon | arm C - arm B | specificity | 0.1143 | 0.1853 | 0.2428 | 15 | 5 | partition_means_t | 0.1143 | -0.1158 | 0.3444 | 0.2400 | -0.0103 | 0.2555 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm B | specificity | 0.3457 | 0.2529 | 0.3379 | 15 | 5 | partition_means_t | 0.3457 | 0.0317 | 0.6598 | 0.0378 | 0.2378 | 0.4599 | 2000 |
| 100 | mean of regimes | arm C - arm B | specificity | 0.2156 | 0.1027 | 0.1833 | 15 | 5 | partition_means_t | 0.2156 | 0.0880 | 0.3432 | 0.0094 | 0.1166 | 0.3258 | 2000 |
| 100 | sup_imnet | arm C - arm B | balanced_accuracy | -0.1410 | 0.0927 | 0.1005 | 15 | 5 | partition_means_t | -0.1410 | -0.2561 | -0.0259 | 0.0272 | -0.1897 | -0.0822 | 2000 |
| 100 | ssl_imnet | arm C - arm B | balanced_accuracy | -0.0324 | 0.0682 | 0.1173 | 15 | 5 | partition_means_t | -0.0324 | -0.1171 | 0.0523 | 0.3485 | -0.0799 | 0.0153 | 2000 |
| 100 | ssl_colon | arm C - arm B | balanced_accuracy | -0.0627 | 0.0470 | 0.1255 | 15 | 5 | partition_means_t | -0.0627 | -0.1210 | -0.0044 | 0.0405 | -0.1323 | 0.0168 | 2000 |
| 100 | ssl_imnet_matched | arm C - arm B | balanced_accuracy | -0.1247 | 0.0506 | 0.0967 | 15 | 5 | partition_means_t | -0.1247 | -0.1875 | -0.0618 | 0.0053 | -0.1796 | -0.0648 | 2000 |
| 100 | mean of regimes | arm C - arm B | balanced_accuracy | -0.0902 | 0.0304 | 0.0634 | 15 | 5 | partition_means_t | -0.0902 | -0.1279 | -0.0525 | 0.0027 | -0.1414 | -0.0325 | 2000 |
| 200 | sup_imnet | arm B - arm A | recall | -0.0860 | 0.0493 | 0.0833 | 15 | 5 | partition_means_t | -0.0860 | -0.1472 | -0.0247 | 0.0176 | -0.1170 | -0.0647 | 2000 |
| 200 | ssl_imnet | arm B - arm A | recall | -0.0757 | 0.0410 | 0.0764 | 15 | 5 | partition_means_t | -0.0757 | -0.1266 | -0.0249 | 0.0144 | -0.1049 | -0.0564 | 2000 |
| 200 | ssl_colon | arm B - arm A | recall | -0.0013 | 0.0525 | 0.0921 | 15 | 5 | partition_means_t | -0.0013 | -0.0664 | 0.0639 | 0.9594 | -0.0071 | 0.0054 | 2000 |
| 200 | ssl_imnet_matched | arm B - arm A | recall | 0.0507 | 0.1009 | 0.1129 | 15 | 5 | partition_means_t | 0.0507 | -0.0746 | 0.1760 | 0.3244 | 0.0355 | 0.0685 | 2000 |
| 200 | mean of regimes | arm B - arm A | recall | -0.0281 | 0.0451 | 0.0511 | 15 | 5 | partition_means_t | -0.0281 | -0.0841 | 0.0279 | 0.2362 | -0.0426 | -0.0185 | 2000 |
| 200 | sup_imnet | arm B - arm A | specificity | 0.0837 | 0.0691 | 0.1759 | 15 | 5 | partition_means_t | 0.0837 | -0.0021 | 0.1696 | 0.0536 | 0.0516 | 0.1201 | 2000 |
| 200 | ssl_imnet | arm B - arm A | specificity | 0.1061 | 0.0669 | 0.1368 | 15 | 5 | partition_means_t | 0.1061 | 0.0231 | 0.1892 | 0.0238 | 0.0821 | 0.1350 | 2000 |
| 200 | ssl_colon | arm B - arm A | specificity | -0.0308 | 0.0802 | 0.1487 | 15 | 5 | partition_means_t | -0.0308 | -0.1303 | 0.0687 | 0.4382 | -0.0467 | -0.0131 | 2000 |
| 200 | ssl_imnet_matched | arm B - arm A | specificity | -0.0990 | 0.1326 | 0.1774 | 15 | 5 | partition_means_t | -0.0990 | -0.2637 | 0.0657 | 0.1704 | -0.1332 | -0.0624 | 2000 |
| 200 | mean of regimes | arm B - arm A | specificity | 0.0150 | 0.0529 | 0.0831 | 15 | 5 | partition_means_t | 0.0150 | -0.0507 | 0.0807 | 0.5602 | -0.0058 | 0.0388 | 2000 |
| 200 | sup_imnet | arm B - arm A | balanced_accuracy | -0.0011 | 0.0284 | 0.0566 | 15 | 5 | partition_means_t | -0.0011 | -0.0363 | 0.0341 | 0.9341 | -0.0241 | 0.0201 | 2000 |
| 200 | ssl_imnet | arm B - arm A | balanced_accuracy | 0.0152 | 0.0156 | 0.0363 | 15 | 5 | partition_means_t | 0.0152 | -0.0041 | 0.0346 | 0.0946 | -0.0045 | 0.0336 | 2000 |
| 200 | ssl_colon | arm B - arm A | balanced_accuracy | -0.0160 | 0.0177 | 0.0464 | 15 | 5 | partition_means_t | -0.0160 | -0.0380 | 0.0059 | 0.1120 | -0.0248 | -0.0067 | 2000 |
| 200 | ssl_imnet_matched | arm B - arm A | balanced_accuracy | -0.0242 | 0.0192 | 0.0376 | 15 | 5 | partition_means_t | -0.0242 | -0.0480 | -3.55e-04 | 0.0479 | -0.0435 | -0.0030 | 2000 |
| 200 | mean of regimes | arm B - arm A | balanced_accuracy | -0.0065 | 0.0062 | 0.0210 | 15 | 5 | partition_means_t | -0.0065 | -0.0143 | 0.0012 | 0.0792 | -0.0191 | 0.0067 | 2000 |
| 200 | sup_imnet | arm C - arm A | auroc | -0.0188 | 0.0230 | 0.0549 | 15 | 5 | partition_means_t | -0.0188 | -0.0474 | 0.0098 | 0.1421 | -0.0697 | 0.0346 | 2000 |
| 200 | ssl_imnet | arm C - arm A | auroc | -0.0395 | 0.0401 | 0.1453 | 15 | 5 | partition_means_t | -0.0395 | -0.0893 | 0.0103 | 0.0927 | -0.1145 | 0.0435 | 2000 |
| 200 | ssl_colon | arm C - arm A | auroc | -0.0649 | 0.0828 | 0.1181 | 15 | 5 | partition_means_t | -0.0649 | -0.1677 | 0.0379 | 0.1546 | -0.2120 | 0.1016 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm A | auroc | -0.0881 | 0.0623 | 0.1235 | 15 | 5 | partition_means_t | -0.0881 | -0.1655 | -0.0108 | 0.0341 | -0.2580 | 0.1060 | 2000 |
| 200 | mean of regimes | arm C - arm A | auroc | -0.0528 | 0.0313 | 0.0736 | 15 | 5 | partition_means_t | -0.0528 | -0.0917 | -0.0140 | 0.0195 | -0.1601 | 0.0672 | 2000 |
| 200 | sup_imnet | arm C - arm A | auprc | 0.0280 | 0.0276 | 0.0907 | 15 | 5 | partition_means_t | 0.0280 | -0.0062 | 0.0622 | 0.0854 | -0.1046 | 0.1222 | 2000 |
| 200 | ssl_imnet | arm C - arm A | auprc | 0.0129 | 0.1052 | 0.1954 | 15 | 5 | partition_means_t | 0.0129 | -0.1177 | 0.1435 | 0.7974 | -0.1421 | 0.1204 | 2000 |
| 200 | ssl_colon | arm C - arm A | auprc | 0.0073 | 0.0784 | 0.1740 | 15 | 5 | partition_means_t | 0.0073 | -0.0901 | 0.1047 | 0.8457 | -0.2720 | 0.1773 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm A | auprc | 0.0384 | 0.0565 | 0.1405 | 15 | 5 | partition_means_t | 0.0384 | -0.0317 | 0.1085 | 0.2031 | -0.2901 | 0.2268 | 2000 |
| 200 | mean of regimes | arm C - arm A | auprc | 0.0217 | 0.0492 | 0.1005 | 15 | 5 | partition_means_t | 0.0217 | -0.0394 | 0.0827 | 0.3808 | -0.1999 | 0.1598 | 2000 |
| 200 | sup_imnet | arm C - arm A | recall | -0.5033 | 0.2239 | 0.2851 | 15 | 5 | partition_means_t | -0.5033 | -0.7812 | -0.2253 | 0.0073 | -0.5367 | -0.4659 | 2000 |
| 200 | ssl_imnet | arm C - arm A | recall | -0.2756 | 0.0687 | 0.2564 | 15 | 5 | partition_means_t | -0.2756 | -0.3608 | -0.1903 | 8.53e-04 | -0.3286 | -0.2281 | 2000 |
| 200 | ssl_colon | arm C - arm A | recall | -0.3112 | 0.1698 | 0.3427 | 15 | 5 | partition_means_t | -0.3112 | -0.5221 | -0.1003 | 0.0149 | -0.3903 | -0.2241 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm A | recall | -0.4892 | 0.2799 | 0.2451 | 15 | 5 | partition_means_t | -0.4892 | -0.8367 | -0.1417 | 0.0174 | -0.5561 | -0.4064 | 2000 |
| 200 | mean of regimes | arm C - arm A | recall | -0.3948 | 0.0892 | 0.1574 | 15 | 5 | partition_means_t | -0.3948 | -0.5056 | -0.2840 | 5.86e-04 | -0.4506 | -0.3332 | 2000 |
| 200 | sup_imnet | arm C - arm A | specificity | 0.2418 | 0.0980 | 0.1154 | 15 | 5 | partition_means_t | 0.2418 | 0.1202 | 0.3635 | 0.0053 | 0.1244 | 0.3776 | 2000 |
| 200 | ssl_imnet | arm C - arm A | specificity | 0.2176 | 0.1418 | 0.2199 | 15 | 5 | partition_means_t | 0.2176 | 0.0414 | 0.3937 | 0.0265 | 0.1046 | 0.3338 | 2000 |
| 200 | ssl_colon | arm C - arm A | specificity | 0.1255 | 0.1646 | 0.2987 | 15 | 5 | partition_means_t | 0.1255 | -0.0789 | 0.3298 | 0.1634 | -0.0194 | 0.2859 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm A | specificity | 0.1745 | 0.2560 | 0.2462 | 15 | 5 | partition_means_t | 0.1745 | -0.1434 | 0.4924 | 0.2021 | 0.0176 | 0.3504 | 2000 |
| 200 | mean of regimes | arm C - arm A | specificity | 0.1898 | 0.0720 | 0.1038 | 15 | 5 | partition_means_t | 0.1898 | 0.1004 | 0.2793 | 0.0041 | 0.0593 | 0.3346 | 2000 |
| 200 | sup_imnet | arm C - arm A | balanced_accuracy | -0.1307 | 0.0791 | 0.1076 | 15 | 5 | partition_means_t | -0.1307 | -0.2289 | -0.0325 | 0.0209 | -0.1919 | -0.0614 | 2000 |
| 200 | ssl_imnet | arm C - arm A | balanced_accuracy | -0.0290 | 0.0508 | 0.1169 | 15 | 5 | partition_means_t | -0.0290 | -0.0921 | 0.0341 | 0.2709 | -0.0906 | 0.0363 | 2000 |
| 200 | ssl_colon | arm C - arm A | balanced_accuracy | -0.0929 | 0.0365 | 0.1094 | 15 | 5 | partition_means_t | -0.0929 | -0.1382 | -0.0476 | 0.0047 | -0.1767 | -0.0011 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm A | balanced_accuracy | -0.1573 | 0.0359 | 0.0851 | 15 | 5 | partition_means_t | -0.1573 | -0.2019 | -0.1128 | 6.07e-04 | -0.2433 | -0.0582 | 2000 |
| 200 | mean of regimes | arm C - arm A | balanced_accuracy | -0.1025 | 0.0263 | 0.0607 | 15 | 5 | partition_means_t | -0.1025 | -0.1352 | -0.0698 | 9.58e-04 | -0.1730 | -0.0232 | 2000 |
| 200 | sup_imnet | arm D - arm A | auroc | -0.0188 | 0.0230 | 0.0549 | 15 | 5 | partition_means_t | -0.0188 | -0.0474 | 0.0098 | 0.1421 | -0.0697 | 0.0346 | 2000 |
| 200 | ssl_imnet | arm D - arm A | auroc | -0.0395 | 0.0401 | 0.1453 | 15 | 5 | partition_means_t | -0.0395 | -0.0893 | 0.0103 | 0.0927 | -0.1145 | 0.0435 | 2000 |
| 200 | ssl_colon | arm D - arm A | auroc | -0.0649 | 0.0828 | 0.1181 | 15 | 5 | partition_means_t | -0.0649 | -0.1677 | 0.0379 | 0.1546 | -0.2120 | 0.1016 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm A | auroc | -0.0881 | 0.0623 | 0.1235 | 15 | 5 | partition_means_t | -0.0881 | -0.1655 | -0.0108 | 0.0341 | -0.2580 | 0.1060 | 2000 |
| 200 | mean of regimes | arm D - arm A | auroc | -0.0528 | 0.0313 | 0.0736 | 15 | 5 | partition_means_t | -0.0528 | -0.0917 | -0.0140 | 0.0195 | -0.1601 | 0.0672 | 2000 |
| 200 | sup_imnet | arm D - arm A | auprc | 0.0280 | 0.0276 | 0.0907 | 15 | 5 | partition_means_t | 0.0280 | -0.0062 | 0.0622 | 0.0854 | -0.1046 | 0.1222 | 2000 |
| 200 | ssl_imnet | arm D - arm A | auprc | 0.0129 | 0.1052 | 0.1954 | 15 | 5 | partition_means_t | 0.0129 | -0.1177 | 0.1435 | 0.7974 | -0.1421 | 0.1204 | 2000 |
| 200 | ssl_colon | arm D - arm A | auprc | 0.0073 | 0.0784 | 0.1740 | 15 | 5 | partition_means_t | 0.0073 | -0.0901 | 0.1047 | 0.8457 | -0.2720 | 0.1773 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm A | auprc | 0.0384 | 0.0565 | 0.1405 | 15 | 5 | partition_means_t | 0.0384 | -0.0317 | 0.1085 | 0.2031 | -0.2901 | 0.2268 | 2000 |
| 200 | mean of regimes | arm D - arm A | auprc | 0.0217 | 0.0492 | 0.1005 | 15 | 5 | partition_means_t | 0.0217 | -0.0394 | 0.0827 | 0.3808 | -0.1999 | 0.1598 | 2000 |
| 200 | sup_imnet | arm D - arm A | recall | -0.1222 | 0.0729 | 0.1303 | 15 | 5 | partition_means_t | -0.1222 | -0.2127 | -0.0317 | 0.0200 | -0.1658 | -0.0907 | 2000 |
| 200 | ssl_imnet | arm D - arm A | recall | -0.1215 | 0.0354 | 0.1089 | 15 | 5 | partition_means_t | -0.1215 | -0.1654 | -0.0776 | 0.0015 | -0.1848 | -0.0859 | 2000 |
| 200 | ssl_colon | arm D - arm A | recall | -0.1083 | 0.0600 | 0.1207 | 15 | 5 | partition_means_t | -0.1083 | -0.1828 | -0.0339 | 0.0156 | -0.2139 | -0.0155 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm A | recall | -0.1003 | 0.1361 | 0.1650 | 15 | 5 | partition_means_t | -0.1003 | -0.2693 | 0.0687 | 0.1748 | -0.1996 | -0.0027 | 2000 |
| 200 | mean of regimes | arm D - arm A | recall | -0.1131 | 0.0529 | 0.0760 | 15 | 5 | partition_means_t | -0.1131 | -0.1788 | -0.0474 | 0.0088 | -0.1863 | -0.0500 | 2000 |
| 200 | sup_imnet | arm D - arm A | specificity | 0.0314 | 0.0887 | 0.2364 | 15 | 5 | partition_means_t | 0.0314 | -0.0787 | 0.1415 | 0.4728 | -0.0689 | 0.1389 | 2000 |
| 200 | ssl_imnet | arm D - arm A | specificity | 0.0741 | 0.0583 | 0.2713 | 15 | 5 | partition_means_t | 0.0741 | 0.0017 | 0.1465 | 0.0467 | -0.0461 | 0.2007 | 2000 |
| 200 | ssl_colon | arm D - arm A | specificity | 0.0052 | 0.0989 | 0.1948 | 15 | 5 | partition_means_t | 0.0052 | -0.1176 | 0.1281 | 0.9115 | -0.1312 | 0.1573 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm A | specificity | -0.1379 | 0.1794 | 0.1722 | 15 | 5 | partition_means_t | -0.1379 | -0.3606 | 0.0848 | 0.1607 | -0.3056 | 0.0524 | 2000 |
| 200 | mean of regimes | arm D - arm A | specificity | -0.0068 | 0.0725 | 0.1277 | 15 | 5 | partition_means_t | -0.0068 | -0.0968 | 0.0832 | 0.8446 | -0.1320 | 0.1353 | 2000 |
| 200 | sup_imnet | arm D - arm A | balanced_accuracy | -0.0454 | 0.0291 | 0.0757 | 15 | 5 | partition_means_t | -0.0454 | -0.0815 | -0.0093 | 0.0252 | -0.1013 | 0.0109 | 2000 |
| 200 | ssl_imnet | arm D - arm A | balanced_accuracy | -0.0237 | 0.0269 | 0.1028 | 15 | 5 | partition_means_t | -0.0237 | -0.0571 | 0.0097 | 0.1204 | -0.0910 | 0.0415 | 2000 |
| 200 | ssl_colon | arm D - arm A | balanced_accuracy | -0.0515 | 0.0402 | 0.0792 | 15 | 5 | partition_means_t | -0.0515 | -0.1014 | -0.0016 | 0.0456 | -0.1383 | 0.0384 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm A | balanced_accuracy | -0.1191 | 0.0618 | 0.0864 | 15 | 5 | partition_means_t | -0.1191 | -0.1958 | -0.0423 | 0.0126 | -0.2164 | -0.0097 | 2000 |
| 200 | mean of regimes | arm D - arm A | balanced_accuracy | -0.0599 | 0.0261 | 0.0535 | 15 | 5 | partition_means_t | -0.0599 | -0.0924 | -0.0275 | 0.0069 | -0.1323 | 0.0177 | 2000 |
| 200 | sup_imnet | arm D - arm C | recall | 0.3811 | 0.1930 | 0.3545 | 15 | 5 | partition_means_t | 0.3811 | 0.1414 | 0.6208 | 0.0116 | 0.3612 | 0.3927 | 2000 |
| 200 | ssl_imnet | arm D - arm C | recall | 0.1541 | 0.1003 | 0.2965 | 15 | 5 | partition_means_t | 0.1541 | 0.0296 | 0.2786 | 0.0264 | 0.1282 | 0.1726 | 2000 |
| 200 | ssl_colon | arm D - arm C | recall | 0.2029 | 0.1329 | 0.3857 | 15 | 5 | partition_means_t | 0.2029 | 0.0379 | 0.3679 | 0.0269 | 0.1687 | 0.2175 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm C | recall | 0.3889 | 0.2996 | 0.1881 | 15 | 5 | partition_means_t | 0.3889 | 0.0168 | 0.7610 | 0.0440 | 0.3506 | 0.4105 | 2000 |
| 200 | mean of regimes | arm D - arm C | recall | 0.2817 | 0.0673 | 0.1887 | 15 | 5 | partition_means_t | 0.2817 | 0.1982 | 0.3653 | 7.25e-04 | 0.2545 | 0.2960 | 2000 |
| 200 | sup_imnet | arm D - arm C | specificity | -0.2104 | 0.1644 | 0.2620 | 15 | 5 | partition_means_t | -0.2104 | -0.4146 | -0.0063 | 0.0458 | -0.2569 | -0.1714 | 2000 |
| 200 | ssl_imnet | arm D - arm C | specificity | -0.1434 | 0.1643 | 0.2883 | 15 | 5 | partition_means_t | -0.1434 | -0.3474 | 0.0606 | 0.1227 | -0.1617 | -0.1250 | 2000 |
| 200 | ssl_colon | arm D - arm C | specificity | -0.1202 | 0.1446 | 0.3070 | 15 | 5 | partition_means_t | -0.1202 | -0.2998 | 0.0594 | 0.1366 | -0.1395 | -0.0993 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm C | specificity | -0.3124 | 0.2226 | 0.1750 | 15 | 5 | partition_means_t | -0.3124 | -0.5887 | -0.0361 | 0.0349 | -0.3458 | -0.2778 | 2000 |
| 200 | mean of regimes | arm D - arm C | specificity | -0.1966 | 0.0940 | 0.1466 | 15 | 5 | partition_means_t | -0.1966 | -0.3133 | -0.0799 | 0.0095 | -0.2169 | -0.1790 | 2000 |
| 200 | sup_imnet | arm D - arm C | balanced_accuracy | 0.0853 | 0.0759 | 0.0762 | 15 | 5 | partition_means_t | 0.0853 | -0.0089 | 0.1795 | 0.0657 | 0.0602 | 0.1056 | 2000 |
| 200 | ssl_imnet | arm D - arm C | balanced_accuracy | 0.0053 | 0.0548 | 0.0773 | 15 | 5 | partition_means_t | 0.0053 | -0.0628 | 0.0734 | 0.8386 | -0.0110 | 0.0185 | 2000 |
| 200 | ssl_colon | arm D - arm C | balanced_accuracy | 0.0413 | 0.0303 | 0.0764 | 15 | 5 | partition_means_t | 0.0413 | 0.0037 | 0.0790 | 0.0382 | 0.0224 | 0.0552 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm C | balanced_accuracy | 0.0383 | 0.0698 | 0.0402 | 15 | 5 | partition_means_t | 0.0383 | -0.0485 | 0.1250 | 0.2878 | 0.0127 | 0.0592 | 2000 |
| 200 | mean of regimes | arm D - arm C | balanced_accuracy | 0.0426 | 0.0326 | 0.0397 | 15 | 5 | partition_means_t | 0.0426 | 0.0021 | 0.0830 | 0.0433 | 0.0254 | 0.0544 | 2000 |
| 200 | sup_imnet | arm D - arm B | auroc | -0.0188 | 0.0230 | 0.0549 | 15 | 5 | partition_means_t | -0.0188 | -0.0474 | 0.0098 | 0.1421 | -0.0697 | 0.0346 | 2000 |
| 200 | ssl_imnet | arm D - arm B | auroc | -0.0395 | 0.0401 | 0.1453 | 15 | 5 | partition_means_t | -0.0395 | -0.0893 | 0.0103 | 0.0927 | -0.1145 | 0.0435 | 2000 |
| 200 | ssl_colon | arm D - arm B | auroc | -0.0649 | 0.0828 | 0.1181 | 15 | 5 | partition_means_t | -0.0649 | -0.1677 | 0.0379 | 0.1546 | -0.2120 | 0.1016 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm B | auroc | -0.0881 | 0.0623 | 0.1235 | 15 | 5 | partition_means_t | -0.0881 | -0.1655 | -0.0108 | 0.0341 | -0.2580 | 0.1060 | 2000 |
| 200 | mean of regimes | arm D - arm B | auroc | -0.0528 | 0.0313 | 0.0736 | 15 | 5 | partition_means_t | -0.0528 | -0.0917 | -0.0140 | 0.0195 | -0.1601 | 0.0672 | 2000 |
| 200 | sup_imnet | arm D - arm B | auprc | 0.0280 | 0.0276 | 0.0907 | 15 | 5 | partition_means_t | 0.0280 | -0.0062 | 0.0622 | 0.0854 | -0.1046 | 0.1222 | 2000 |
| 200 | ssl_imnet | arm D - arm B | auprc | 0.0129 | 0.1052 | 0.1954 | 15 | 5 | partition_means_t | 0.0129 | -0.1177 | 0.1435 | 0.7974 | -0.1421 | 0.1204 | 2000 |
| 200 | ssl_colon | arm D - arm B | auprc | 0.0073 | 0.0784 | 0.1740 | 15 | 5 | partition_means_t | 0.0073 | -0.0901 | 0.1047 | 0.8457 | -0.2720 | 0.1773 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm B | auprc | 0.0384 | 0.0565 | 0.1405 | 15 | 5 | partition_means_t | 0.0384 | -0.0317 | 0.1085 | 0.2031 | -0.2901 | 0.2268 | 2000 |
| 200 | mean of regimes | arm D - arm B | auprc | 0.0217 | 0.0492 | 0.1005 | 15 | 5 | partition_means_t | 0.0217 | -0.0394 | 0.0827 | 0.3808 | -0.1999 | 0.1598 | 2000 |
| 200 | sup_imnet | arm D - arm B | recall | -0.0362 | 0.0340 | 0.1188 | 15 | 5 | partition_means_t | -0.0362 | -0.0785 | 0.0060 | 0.0759 | -0.0689 | 0.0036 | 2000 |
| 200 | ssl_imnet | arm D - arm B | recall | -0.0458 | 0.0322 | 0.1296 | 15 | 5 | partition_means_t | -0.0458 | -0.0857 | -0.0058 | 0.0336 | -0.0880 | -0.0149 | 2000 |
| 200 | ssl_colon | arm D - arm B | recall | -0.1070 | 0.0282 | 0.0732 | 15 | 5 | partition_means_t | -0.1070 | -0.1420 | -0.0721 | 0.0010 | -0.2059 | -0.0178 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm B | recall | -0.1510 | 0.0507 | 0.1576 | 15 | 5 | partition_means_t | -0.1510 | -0.2140 | -0.0879 | 0.0027 | -0.2426 | -0.0584 | 2000 |
| 200 | mean of regimes | arm D - arm B | recall | -0.0850 | 0.0164 | 0.0585 | 15 | 5 | partition_means_t | -0.0850 | -0.1054 | -0.0646 | 3.18e-04 | -0.1512 | -0.0242 | 2000 |
| 200 | sup_imnet | arm D - arm B | specificity | -0.0523 | 0.0416 | 0.1895 | 15 | 5 | partition_means_t | -0.0523 | -0.1040 | -6.58e-04 | 0.0482 | -0.1434 | 0.0448 | 2000 |
| 200 | ssl_imnet | arm D - arm B | specificity | -0.0320 | 0.0246 | 0.2350 | 15 | 5 | partition_means_t | -0.0320 | -0.0626 | -0.0014 | 0.0438 | -0.1394 | 0.0856 | 2000 |
| 200 | ssl_colon | arm D - arm B | specificity | 0.0361 | 0.1089 | 0.1745 | 15 | 5 | partition_means_t | 0.0361 | -0.0992 | 0.1713 | 0.5003 | -0.0909 | 0.1767 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm B | specificity | -0.0389 | 0.1486 | 0.2210 | 15 | 5 | partition_means_t | -0.0389 | -0.2234 | 0.1456 | 0.5898 | -0.1774 | 0.1196 | 2000 |
| 200 | mean of regimes | arm D - arm B | specificity | -0.0218 | 0.0652 | 0.1052 | 15 | 5 | partition_means_t | -0.0218 | -0.1027 | 0.0592 | 0.4963 | -0.1304 | 0.1036 | 2000 |
| 200 | sup_imnet | arm D - arm B | balanced_accuracy | -0.0443 | 0.0156 | 0.0520 | 15 | 5 | partition_means_t | -0.0443 | -0.0637 | -0.0249 | 0.0032 | -0.0933 | 0.0066 | 2000 |
| 200 | ssl_imnet | arm D - arm B | balanced_accuracy | -0.0389 | 0.0171 | 0.0917 | 15 | 5 | partition_means_t | -0.0389 | -0.0601 | -0.0177 | 0.0071 | -0.0971 | 0.0209 | 2000 |
| 200 | ssl_colon | arm D - arm B | balanced_accuracy | -0.0355 | 0.0489 | 0.0791 | 15 | 5 | partition_means_t | -0.0355 | -0.0962 | 0.0252 | 0.1800 | -0.1178 | 0.0481 | 2000 |
| 200 | ssl_imnet_matched | arm D - arm B | balanced_accuracy | -0.0949 | 0.0627 | 0.0999 | 15 | 5 | partition_means_t | -0.0949 | -0.1728 | -0.0171 | 0.0276 | -0.1804 | -0.0014 | 2000 |
| 200 | mean of regimes | arm D - arm B | balanced_accuracy | -0.0534 | 0.0307 | 0.0536 | 15 | 5 | partition_means_t | -0.0534 | -0.0915 | -0.0153 | 0.0176 | -0.1177 | 0.0156 | 2000 |
| 200 | sup_imnet | arm C - arm B | auroc | -0.0188 | 0.0230 | 0.0549 | 15 | 5 | partition_means_t | -0.0188 | -0.0474 | 0.0098 | 0.1421 | -0.0697 | 0.0346 | 2000 |
| 200 | ssl_imnet | arm C - arm B | auroc | -0.0395 | 0.0401 | 0.1453 | 15 | 5 | partition_means_t | -0.0395 | -0.0893 | 0.0103 | 0.0927 | -0.1145 | 0.0435 | 2000 |
| 200 | ssl_colon | arm C - arm B | auroc | -0.0649 | 0.0828 | 0.1181 | 15 | 5 | partition_means_t | -0.0649 | -0.1677 | 0.0379 | 0.1546 | -0.2120 | 0.1016 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm B | auroc | -0.0881 | 0.0623 | 0.1235 | 15 | 5 | partition_means_t | -0.0881 | -0.1655 | -0.0108 | 0.0341 | -0.2580 | 0.1060 | 2000 |
| 200 | mean of regimes | arm C - arm B | auroc | -0.0528 | 0.0313 | 0.0736 | 15 | 5 | partition_means_t | -0.0528 | -0.0917 | -0.0140 | 0.0195 | -0.1601 | 0.0672 | 2000 |
| 200 | sup_imnet | arm C - arm B | auprc | 0.0280 | 0.0276 | 0.0907 | 15 | 5 | partition_means_t | 0.0280 | -0.0062 | 0.0622 | 0.0854 | -0.1046 | 0.1222 | 2000 |
| 200 | ssl_imnet | arm C - arm B | auprc | 0.0129 | 0.1052 | 0.1954 | 15 | 5 | partition_means_t | 0.0129 | -0.1177 | 0.1435 | 0.7974 | -0.1421 | 0.1204 | 2000 |
| 200 | ssl_colon | arm C - arm B | auprc | 0.0073 | 0.0784 | 0.1740 | 15 | 5 | partition_means_t | 0.0073 | -0.0901 | 0.1047 | 0.8457 | -0.2720 | 0.1773 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm B | auprc | 0.0384 | 0.0565 | 0.1405 | 15 | 5 | partition_means_t | 0.0384 | -0.0317 | 0.1085 | 0.2031 | -0.2901 | 0.2268 | 2000 |
| 200 | mean of regimes | arm C - arm B | auprc | 0.0217 | 0.0492 | 0.1005 | 15 | 5 | partition_means_t | 0.0217 | -0.0394 | 0.0827 | 0.3808 | -0.1999 | 0.1598 | 2000 |
| 200 | sup_imnet | arm C - arm B | recall | -0.4173 | 0.2184 | 0.3093 | 15 | 5 | partition_means_t | -0.4173 | -0.6886 | -0.1461 | 0.0129 | -0.4508 | -0.3682 | 2000 |
| 200 | ssl_imnet | arm C - arm B | recall | -0.1998 | 0.0979 | 0.2821 | 15 | 5 | partition_means_t | -0.1998 | -0.3214 | -0.0782 | 0.0103 | -0.2371 | -0.1553 | 2000 |
| 200 | ssl_colon | arm C - arm B | recall | -0.3099 | 0.1552 | 0.4006 | 15 | 5 | partition_means_t | -0.3099 | -0.5027 | -0.1172 | 0.0111 | -0.3847 | -0.2290 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm B | recall | -0.5399 | 0.3127 | 0.2548 | 15 | 5 | partition_means_t | -0.5399 | -0.9282 | -0.1516 | 0.0181 | -0.6029 | -0.4651 | 2000 |
| 200 | mean of regimes | arm C - arm B | recall | -0.3667 | 0.0790 | 0.1903 | 15 | 5 | partition_means_t | -0.3667 | -0.4648 | -0.2687 | 4.86e-04 | -0.4189 | -0.3069 | 2000 |
| 200 | sup_imnet | arm C - arm B | specificity | 0.1581 | 0.1538 | 0.2170 | 15 | 5 | partition_means_t | 0.1581 | -0.0329 | 0.3491 | 0.0831 | 0.0546 | 0.2764 | 2000 |
| 200 | ssl_imnet | arm C - arm B | specificity | 0.1114 | 0.1511 | 0.2111 | 15 | 5 | partition_means_t | 0.1114 | -0.0762 | 0.2990 | 0.1745 | 0.0117 | 0.2164 | 2000 |
| 200 | ssl_colon | arm C - arm B | specificity | 0.1563 | 0.1780 | 0.3639 | 15 | 5 | partition_means_t | 0.1563 | -0.0647 | 0.3773 | 0.1211 | 0.0211 | 0.3039 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm B | specificity | 0.2735 | 0.3072 | 0.2654 | 15 | 5 | partition_means_t | 0.2735 | -0.1080 | 0.6550 | 0.1174 | 0.1424 | 0.4195 | 2000 |
| 200 | mean of regimes | arm C - arm B | specificity | 0.1748 | 0.0918 | 0.1404 | 15 | 5 | partition_means_t | 0.1748 | 0.0608 | 0.2889 | 0.0131 | 0.0622 | 0.3025 | 2000 |
| 200 | sup_imnet | arm C - arm B | balanced_accuracy | -0.1296 | 0.0803 | 0.0977 | 15 | 5 | partition_means_t | -0.1296 | -0.2293 | -0.0299 | 0.0226 | -0.1832 | -0.0641 | 2000 |
| 200 | ssl_imnet | arm C - arm B | balanced_accuracy | -0.0442 | 0.0435 | 0.1049 | 15 | 5 | partition_means_t | -0.0442 | -0.0982 | 0.0098 | 0.0855 | -0.0969 | 0.0140 | 2000 |
| 200 | ssl_colon | arm C - arm B | balanced_accuracy | -0.0768 | 0.0511 | 0.1144 | 15 | 5 | partition_means_t | -0.0768 | -0.1402 | -0.0134 | 0.0282 | -0.1547 | 0.0070 | 2000 |
| 200 | ssl_imnet_matched | arm C - arm B | balanced_accuracy | -0.1332 | 0.0505 | 0.0916 | 15 | 5 | partition_means_t | -0.1332 | -0.1959 | -0.0705 | 0.0041 | -0.2069 | -0.0474 | 2000 |
| 200 | mean of regimes | arm C - arm B | balanced_accuracy | -0.0960 | 0.0287 | 0.0529 | 15 | 5 | partition_means_t | -0.0960 | -0.1316 | -0.0603 | 0.0017 | -0.1574 | -0.0263 | 2000 |
| 500 | sup_imnet | arm B - arm A | recall | -0.0468 | 0.0404 | 0.0593 | 15 | 5 | partition_means_t | -0.0468 | -0.0970 | 0.0034 | 0.0608 | -0.0705 | -0.0317 | 2000 |
| 500 | ssl_imnet | arm B - arm A | recall | -0.0519 | 0.0406 | 0.0651 | 15 | 5 | partition_means_t | -0.0519 | -0.1023 | -0.0014 | 0.0462 | -0.0725 | -0.0377 | 2000 |
| 500 | ssl_colon | arm B - arm A | recall | 0.0116 | 0.0500 | 0.0643 | 15 | 5 | partition_means_t | 0.0116 | -0.0505 | 0.0736 | 0.6326 | 0.0044 | 0.0203 | 2000 |
| 500 | ssl_imnet_matched | arm B - arm A | recall | 0.0671 | 0.0920 | 0.0786 | 15 | 5 | partition_means_t | 0.0671 | -0.0472 | 0.1813 | 0.1786 | 0.0530 | 0.0844 | 2000 |
| 500 | mean of regimes | arm B - arm A | recall | -0.0050 | 0.0438 | 0.0364 | 15 | 5 | partition_means_t | -0.0050 | -0.0594 | 0.0494 | 0.8107 | -0.0148 | 0.0013 | 2000 |
| 500 | sup_imnet | arm B - arm A | specificity | 0.0781 | 0.0690 | 0.0867 | 15 | 5 | partition_means_t | 0.0781 | -0.0075 | 0.1638 | 0.0645 | 0.0536 | 0.1083 | 1691 |
| 500 | ssl_imnet | arm B - arm A | specificity | 0.0958 | 0.0952 | 0.1310 | 15 | 5 | partition_means_t | 0.0958 | -0.0224 | 0.2140 | 0.0876 | 0.0725 | 0.1426 | 1691 |
| 500 | ssl_colon | arm B - arm A | specificity | -0.0278 | 0.0918 | 0.1265 | 15 | 5 | partition_means_t | -0.0278 | -0.1418 | 0.0861 | 0.5346 | -0.0567 | -0.0025 | 1691 |
| 500 | ssl_imnet_matched | arm B - arm A | specificity | -0.1260 | 0.1034 | 0.1509 | 15 | 5 | partition_means_t | -0.1260 | -0.2543 | 0.0024 | 0.0527 | -0.1719 | -0.0729 | 1691 |
| 500 | mean of regimes | arm B - arm A | specificity | 0.0050 | 0.0713 | 0.0673 | 15 | 5 | partition_means_t | 0.0050 | -0.0835 | 0.0936 | 0.8824 | -0.0105 | 0.0240 | 1691 |
| 500 | sup_imnet | arm B - arm A | balanced_accuracy | 0.0157 | 0.0194 | 0.0292 | 15 | 5 | partition_means_t | 0.0157 | -0.0085 | 0.0398 | 0.1459 | -0.0013 | 0.0324 | 1691 |
| 500 | ssl_imnet | arm B - arm A | balanced_accuracy | 0.0220 | 0.0315 | 0.0386 | 15 | 5 | partition_means_t | 0.0220 | -0.0171 | 0.0611 | 0.1937 | 0.0070 | 0.0457 | 1691 |
| 500 | ssl_colon | arm B - arm A | balanced_accuracy | -0.0081 | 0.0236 | 0.0343 | 15 | 5 | partition_means_t | -0.0081 | -0.0374 | 0.0211 | 0.4829 | -0.0227 | 0.0049 | 1691 |
| 500 | ssl_imnet_matched | arm B - arm A | balanced_accuracy | -0.0295 | 0.0190 | 0.0443 | 15 | 5 | partition_means_t | -0.0295 | -0.0530 | -0.0059 | 0.0256 | -0.0538 | -0.0021 | 1691 |
| 500 | mean of regimes | arm B - arm A | balanced_accuracy | 7.31e-06 | 0.0168 | 0.0194 | 15 | 5 | partition_means_t | 7.31e-06 | -0.0209 | 0.0209 | 0.9993 | -0.0089 | 0.0101 | 1691 |
| 500 | sup_imnet | arm C - arm A | auroc | 0.0488 | 0.0381 | 0.0558 | 15 | 5 | partition_means_t | 0.0488 | 0.0015 | 0.0960 | 0.0457 | -0.0300 | 0.1148 | 1691 |
| 500 | ssl_imnet | arm C - arm A | auroc | 0.1008 | 0.0679 | 0.0785 | 15 | 5 | partition_means_t | 0.1008 | 0.0165 | 0.1851 | 0.0294 | 0.0231 | 0.1754 | 1691 |
| 500 | ssl_colon | arm C - arm A | auroc | 0.1472 | 0.0965 | 0.1214 | 15 | 5 | partition_means_t | 0.1472 | 0.0273 | 0.2670 | 0.0271 | -0.0219 | 0.2791 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm A | auroc | 0.1872 | 0.1190 | 0.1498 | 15 | 5 | partition_means_t | 0.1872 | 0.0394 | 0.3349 | 0.0245 | 0.0160 | 0.3289 | 1691 |
| 500 | mean of regimes | arm C - arm A | auroc | 0.1210 | 0.0755 | 0.0809 | 15 | 5 | partition_means_t | 0.1210 | 0.0273 | 0.2147 | 0.0231 | 1.58e-04 | 0.2198 | 1691 |
| 500 | sup_imnet | arm C - arm A | auprc | 0.0541 | 0.0400 | 0.0779 | 15 | 5 | partition_means_t | 0.0541 | 0.0045 | 0.1038 | 0.0388 | -0.0151 | 0.1122 | 1691 |
| 500 | ssl_imnet | arm C - arm A | auprc | 0.0797 | 0.0666 | 0.0993 | 15 | 5 | partition_means_t | 0.0797 | -0.0029 | 0.1624 | 0.0553 | 0.0068 | 0.1433 | 1691 |
| 500 | ssl_colon | arm C - arm A | auprc | 0.1470 | 0.0903 | 0.1513 | 15 | 5 | partition_means_t | 0.1470 | 0.0349 | 0.2591 | 0.0219 | -0.0153 | 0.2525 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm A | auprc | 0.1870 | 0.1104 | 0.1733 | 15 | 5 | partition_means_t | 0.1870 | 0.0499 | 0.3242 | 0.0193 | 0.0039 | 0.3004 | 1691 |
| 500 | mean of regimes | arm C - arm A | auprc | 0.1170 | 0.0727 | 0.1027 | 15 | 5 | partition_means_t | 0.1170 | 0.0267 | 0.2072 | 0.0228 | -0.0013 | 0.1989 | 1691 |
| 500 | sup_imnet | arm C - arm A | recall | -0.4329 | 0.2050 | 0.2535 | 15 | 5 | partition_means_t | -0.4329 | -0.6874 | -0.1783 | 0.0092 | -0.4805 | -0.3898 | 2000 |
| 500 | ssl_imnet | arm C - arm A | recall | -0.1936 | 0.0823 | 0.1814 | 15 | 5 | partition_means_t | -0.1936 | -0.2958 | -0.0914 | 0.0063 | -0.2595 | -0.1608 | 2000 |
| 500 | ssl_colon | arm C - arm A | recall | -0.2776 | 0.2112 | 0.2829 | 15 | 5 | partition_means_t | -0.2776 | -0.5399 | -0.0154 | 0.0424 | -0.3758 | -0.1831 | 2000 |
| 500 | ssl_imnet_matched | arm C - arm A | recall | -0.3855 | 0.2192 | 0.2459 | 15 | 5 | partition_means_t | -0.3855 | -0.6578 | -0.1133 | 0.0171 | -0.4763 | -0.2841 | 2000 |
| 500 | mean of regimes | arm C - arm A | recall | -0.3224 | 0.0680 | 0.1347 | 15 | 5 | partition_means_t | -0.3224 | -0.4069 | -0.2379 | 4.49e-04 | -0.3949 | -0.2564 | 2000 |
| 500 | sup_imnet | arm C - arm A | specificity | 0.2823 | 0.1084 | 0.1729 | 15 | 5 | partition_means_t | 0.2823 | 0.1477 | 0.4169 | 0.0043 | 0.1361 | 0.4209 | 1691 |
| 500 | ssl_imnet | arm C - arm A | specificity | 0.3535 | 0.1670 | 0.2453 | 15 | 5 | partition_means_t | 0.3535 | 0.1461 | 0.5608 | 0.0091 | 0.2533 | 0.4570 | 1691 |
| 500 | ssl_colon | arm C - arm A | specificity | 0.3136 | 0.1334 | 0.1944 | 15 | 5 | partition_means_t | 0.3136 | 0.1479 | 0.4792 | 0.0063 | 0.1641 | 0.4439 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm A | specificity | 0.3685 | 0.1346 | 0.2458 | 15 | 5 | partition_means_t | 0.3685 | 0.2014 | 0.5357 | 0.0036 | 0.2172 | 0.4934 | 1691 |
| 500 | mean of regimes | arm C - arm A | specificity | 0.3295 | 0.1027 | 0.1366 | 15 | 5 | partition_means_t | 0.3295 | 0.2019 | 0.4570 | 0.0020 | 0.2023 | 0.4487 | 1691 |
| 500 | sup_imnet | arm C - arm A | balanced_accuracy | -0.0753 | 0.0923 | 0.1060 | 15 | 5 | partition_means_t | -0.0753 | -0.1899 | 0.0393 | 0.1422 | -0.1547 | -0.0022 | 1691 |
| 500 | ssl_imnet | arm C - arm A | balanced_accuracy | 0.0800 | 0.0491 | 0.1178 | 15 | 5 | partition_means_t | 0.0800 | 0.0190 | 0.1409 | 0.0219 | 0.0197 | 0.1375 | 1691 |
| 500 | ssl_colon | arm C - arm A | balanced_accuracy | 0.0180 | 0.0617 | 0.1186 | 15 | 5 | partition_means_t | 0.0180 | -0.0586 | 0.0946 | 0.5503 | -0.0727 | 0.1001 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm A | balanced_accuracy | -0.0085 | 0.1098 | 0.1248 | 15 | 5 | partition_means_t | -0.0085 | -0.1448 | 0.1278 | 0.8710 | -0.0973 | 0.0742 | 1691 |
| 500 | mean of regimes | arm C - arm A | balanced_accuracy | 0.0035 | 0.0458 | 0.0893 | 15 | 5 | partition_means_t | 0.0035 | -0.0534 | 0.0604 | 0.8715 | -0.0704 | 0.0722 | 1691 |
| 500 | sup_imnet | arm D - arm A | auroc | 0.0488 | 0.0381 | 0.0558 | 15 | 5 | partition_means_t | 0.0488 | 0.0015 | 0.0960 | 0.0457 | -0.0300 | 0.1148 | 1691 |
| 500 | ssl_imnet | arm D - arm A | auroc | 0.1008 | 0.0679 | 0.0785 | 15 | 5 | partition_means_t | 0.1008 | 0.0165 | 0.1851 | 0.0294 | 0.0231 | 0.1754 | 1691 |
| 500 | ssl_colon | arm D - arm A | auroc | 0.1472 | 0.0965 | 0.1214 | 15 | 5 | partition_means_t | 0.1472 | 0.0273 | 0.2670 | 0.0271 | -0.0219 | 0.2791 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm A | auroc | 0.1872 | 0.1190 | 0.1498 | 15 | 5 | partition_means_t | 0.1872 | 0.0394 | 0.3349 | 0.0245 | 0.0160 | 0.3289 | 1691 |
| 500 | mean of regimes | arm D - arm A | auroc | 0.1210 | 0.0755 | 0.0809 | 15 | 5 | partition_means_t | 0.1210 | 0.0273 | 0.2147 | 0.0231 | 1.58e-04 | 0.2198 | 1691 |
| 500 | sup_imnet | arm D - arm A | auprc | 0.0541 | 0.0400 | 0.0779 | 15 | 5 | partition_means_t | 0.0541 | 0.0045 | 0.1038 | 0.0388 | -0.0151 | 0.1122 | 1691 |
| 500 | ssl_imnet | arm D - arm A | auprc | 0.0797 | 0.0666 | 0.0993 | 15 | 5 | partition_means_t | 0.0797 | -0.0029 | 0.1624 | 0.0553 | 0.0068 | 0.1433 | 1691 |
| 500 | ssl_colon | arm D - arm A | auprc | 0.1470 | 0.0903 | 0.1513 | 15 | 5 | partition_means_t | 0.1470 | 0.0349 | 0.2591 | 0.0219 | -0.0153 | 0.2525 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm A | auprc | 0.1870 | 0.1104 | 0.1733 | 15 | 5 | partition_means_t | 0.1870 | 0.0499 | 0.3242 | 0.0193 | 0.0039 | 0.3004 | 1691 |
| 500 | mean of regimes | arm D - arm A | auprc | 0.1170 | 0.0727 | 0.1027 | 15 | 5 | partition_means_t | 0.1170 | 0.0267 | 0.2072 | 0.0228 | -0.0013 | 0.1989 | 1691 |
| 500 | sup_imnet | arm D - arm A | recall | -0.0819 | 0.0483 | 0.0972 | 15 | 5 | partition_means_t | -0.0819 | -0.1419 | -0.0218 | 0.0193 | -0.1307 | -0.0458 | 2000 |
| 500 | ssl_imnet | arm D - arm A | recall | -0.0801 | 0.0416 | 0.0740 | 15 | 5 | partition_means_t | -0.0801 | -0.1317 | -0.0285 | 0.0126 | -0.1394 | -0.0503 | 2000 |
| 500 | ssl_colon | arm D - arm A | recall | -0.1047 | 0.0681 | 0.1034 | 15 | 5 | partition_means_t | -0.1047 | -0.1893 | -0.0201 | 0.0264 | -0.2236 | -0.0053 | 2000 |
| 500 | ssl_imnet_matched | arm D - arm A | recall | -0.0043 | 0.1348 | 0.0892 | 15 | 5 | partition_means_t | -0.0043 | -0.1716 | 0.1630 | 0.9463 | -0.0902 | 0.0732 | 2000 |
| 500 | mean of regimes | arm D - arm A | recall | -0.0677 | 0.0463 | 0.0340 | 15 | 5 | partition_means_t | -0.0677 | -0.1252 | -0.0103 | 0.0307 | -0.1433 | -0.0108 | 2000 |
| 500 | sup_imnet | arm D - arm A | specificity | 0.1843 | 0.1135 | 0.2026 | 15 | 5 | partition_means_t | 0.1843 | 0.0434 | 0.3251 | 0.0221 | 0.0448 | 0.2976 | 1691 |
| 500 | ssl_imnet | arm D - arm A | specificity | 0.3416 | 0.0795 | 0.2217 | 15 | 5 | partition_means_t | 0.3416 | 0.2429 | 0.4403 | 6.55e-04 | 0.2594 | 0.4319 | 1691 |
| 500 | ssl_colon | arm D - arm A | specificity | 0.2694 | 0.1581 | 0.1591 | 15 | 5 | partition_means_t | 0.2694 | 0.0730 | 0.4657 | 0.0190 | 0.1261 | 0.3896 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm A | specificity | 0.1721 | 0.2019 | 0.2068 | 15 | 5 | partition_means_t | 0.1721 | -0.0787 | 0.4228 | 0.1295 | 0.0388 | 0.2736 | 1691 |
| 500 | mean of regimes | arm D - arm A | specificity | 0.2418 | 0.1260 | 0.1343 | 15 | 5 | partition_means_t | 0.2418 | 0.0854 | 0.3982 | 0.0127 | 0.1265 | 0.3425 | 1691 |
| 500 | sup_imnet | arm D - arm A | balanced_accuracy | 0.0512 | 0.0517 | 0.0662 | 15 | 5 | partition_means_t | 0.0512 | -0.0130 | 0.1154 | 0.0911 | -0.0242 | 0.1109 | 1691 |
| 500 | ssl_imnet | arm D - arm A | balanced_accuracy | 0.1308 | 0.0362 | 0.0952 | 15 | 5 | partition_means_t | 0.1308 | 0.0858 | 0.1757 | 0.0013 | 0.0786 | 0.1799 | 1691 |
| 500 | ssl_colon | arm D - arm A | balanced_accuracy | 0.0823 | 0.0490 | 0.0512 | 15 | 5 | partition_means_t | 0.0823 | 0.0215 | 0.1432 | 0.0198 | -0.0142 | 0.1613 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm A | balanced_accuracy | 0.0839 | 0.0764 | 0.0866 | 15 | 5 | partition_means_t | 0.0839 | -0.0110 | 0.1788 | 0.0702 | 0.0060 | 0.1498 | 1691 |
| 500 | mean of regimes | arm D - arm A | balanced_accuracy | 0.0870 | 0.0508 | 0.0551 | 15 | 5 | partition_means_t | 0.0870 | 0.0240 | 0.1501 | 0.0186 | 0.0182 | 0.1433 | 1691 |
| 500 | sup_imnet | arm D - arm C | recall | 0.3510 | 0.1647 | 0.3167 | 15 | 5 | partition_means_t | 0.3510 | 0.1465 | 0.5555 | 0.0089 | 0.3365 | 0.3592 | 2000 |
| 500 | ssl_imnet | arm D - arm C | recall | 0.1135 | 0.1015 | 0.2249 | 15 | 5 | partition_means_t | 0.1135 | -0.0126 | 0.2396 | 0.0669 | 0.1013 | 0.1231 | 2000 |
| 500 | ssl_colon | arm D - arm C | recall | 0.1729 | 0.2113 | 0.2919 | 15 | 5 | partition_means_t | 0.1729 | -0.0894 | 0.4352 | 0.1412 | 0.1442 | 0.1869 | 2000 |
| 500 | ssl_imnet_matched | arm D - arm C | recall | 0.3812 | 0.2640 | 0.2755 | 15 | 5 | partition_means_t | 0.3812 | 0.0534 | 0.7090 | 0.0320 | 0.3499 | 0.4089 | 2000 |
| 500 | mean of regimes | arm D - arm C | recall | 0.2546 | 0.0399 | 0.1575 | 15 | 5 | partition_means_t | 0.2546 | 0.2050 | 0.3043 | 1.41e-04 | 0.2383 | 0.2664 | 2000 |
| 500 | sup_imnet | arm D - arm C | specificity | -0.0980 | 0.0871 | 0.2386 | 15 | 5 | partition_means_t | -0.0980 | -0.2061 | 0.0102 | 0.0657 | -0.1550 | -0.0539 | 1691 |
| 500 | ssl_imnet | arm D - arm C | specificity | -0.0119 | 0.1224 | 0.1958 | 15 | 5 | partition_means_t | -0.0119 | -0.1638 | 0.1401 | 0.8389 | -0.0417 | 0.0300 | 1691 |
| 500 | ssl_colon | arm D - arm C | specificity | -0.0442 | 0.1488 | 0.1598 | 15 | 5 | partition_means_t | -0.0442 | -0.2290 | 0.1406 | 0.5429 | -0.0789 | -0.0055 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm C | specificity | -0.1965 | 0.1489 | 0.1640 | 15 | 5 | partition_means_t | -0.1965 | -0.3814 | -0.0115 | 0.0420 | -0.2609 | -0.1430 | 1691 |
| 500 | mean of regimes | arm D - arm C | specificity | -0.0876 | 0.0561 | 0.1011 | 15 | 5 | partition_means_t | -0.0876 | -0.1573 | -0.0179 | 0.0251 | -0.1209 | -0.0587 | 1691 |
| 500 | sup_imnet | arm D - arm C | balanced_accuracy | 0.1265 | 0.0910 | 0.0685 | 15 | 5 | partition_means_t | 0.1265 | 0.0135 | 0.2395 | 0.0359 | 0.0974 | 0.1487 | 1691 |
| 500 | ssl_imnet | arm D - arm C | balanced_accuracy | 0.0508 | 0.0209 | 0.0674 | 15 | 5 | partition_means_t | 0.0508 | 0.0248 | 0.0768 | 0.0056 | 0.0342 | 0.0719 | 1691 |
| 500 | ssl_colon | arm D - arm C | balanced_accuracy | 0.0643 | 0.0544 | 0.1102 | 15 | 5 | partition_means_t | 0.0643 | -0.0032 | 0.1319 | 0.0573 | 0.0425 | 0.0854 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm C | balanced_accuracy | 0.0924 | 0.0638 | 0.0850 | 15 | 5 | partition_means_t | 0.0924 | 0.0131 | 0.1716 | 0.0318 | 0.0539 | 0.1211 | 1691 |
| 500 | mean of regimes | arm D - arm C | balanced_accuracy | 0.0835 | 0.0179 | 0.0520 | 15 | 5 | partition_means_t | 0.0835 | 0.0613 | 0.1058 | 4.79e-04 | 0.0640 | 0.0990 | 1691 |
| 500 | sup_imnet | arm D - arm B | auroc | 0.0488 | 0.0381 | 0.0558 | 15 | 5 | partition_means_t | 0.0488 | 0.0015 | 0.0960 | 0.0457 | -0.0300 | 0.1148 | 1691 |
| 500 | ssl_imnet | arm D - arm B | auroc | 0.1008 | 0.0679 | 0.0785 | 15 | 5 | partition_means_t | 0.1008 | 0.0165 | 0.1851 | 0.0294 | 0.0231 | 0.1754 | 1691 |
| 500 | ssl_colon | arm D - arm B | auroc | 0.1472 | 0.0965 | 0.1214 | 15 | 5 | partition_means_t | 0.1472 | 0.0273 | 0.2670 | 0.0271 | -0.0219 | 0.2791 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm B | auroc | 0.1872 | 0.1190 | 0.1498 | 15 | 5 | partition_means_t | 0.1872 | 0.0394 | 0.3349 | 0.0245 | 0.0160 | 0.3289 | 1691 |
| 500 | mean of regimes | arm D - arm B | auroc | 0.1210 | 0.0755 | 0.0809 | 15 | 5 | partition_means_t | 0.1210 | 0.0273 | 0.2147 | 0.0231 | 1.58e-04 | 0.2198 | 1691 |
| 500 | sup_imnet | arm D - arm B | auprc | 0.0541 | 0.0400 | 0.0779 | 15 | 5 | partition_means_t | 0.0541 | 0.0045 | 0.1038 | 0.0388 | -0.0151 | 0.1122 | 1691 |
| 500 | ssl_imnet | arm D - arm B | auprc | 0.0797 | 0.0666 | 0.0993 | 15 | 5 | partition_means_t | 0.0797 | -0.0029 | 0.1624 | 0.0553 | 0.0068 | 0.1433 | 1691 |
| 500 | ssl_colon | arm D - arm B | auprc | 0.1470 | 0.0903 | 0.1513 | 15 | 5 | partition_means_t | 0.1470 | 0.0349 | 0.2591 | 0.0219 | -0.0153 | 0.2525 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm B | auprc | 0.1870 | 0.1104 | 0.1733 | 15 | 5 | partition_means_t | 0.1870 | 0.0499 | 0.3242 | 0.0193 | 0.0039 | 0.3004 | 1691 |
| 500 | mean of regimes | arm D - arm B | auprc | 0.1170 | 0.0727 | 0.1027 | 15 | 5 | partition_means_t | 0.1170 | 0.0267 | 0.2072 | 0.0228 | -0.0013 | 0.1989 | 1691 |
| 500 | sup_imnet | arm D - arm B | recall | -0.0351 | 0.0494 | 0.1075 | 15 | 5 | partition_means_t | -0.0351 | -0.0964 | 0.0263 | 0.1880 | -0.0741 | 0.0051 | 2000 |
| 500 | ssl_imnet | arm D - arm B | recall | -0.0282 | 0.0483 | 0.0806 | 15 | 5 | partition_means_t | -0.0282 | -0.0882 | 0.0318 | 0.2616 | -0.0765 | 0.0054 | 2000 |
| 500 | ssl_colon | arm D - arm B | recall | -0.1163 | 0.0815 | 0.0819 | 15 | 5 | partition_means_t | -0.1163 | -0.2175 | -0.0151 | 0.0332 | -0.2310 | -0.0196 | 2000 |
| 500 | ssl_imnet_matched | arm D - arm B | recall | -0.0714 | 0.0496 | 0.0745 | 15 | 5 | partition_means_t | -0.0714 | -0.1329 | -0.0098 | 0.0323 | -0.1509 | -0.0051 | 2000 |
| 500 | mean of regimes | arm D - arm B | recall | -0.0627 | 0.0294 | 0.0374 | 15 | 5 | partition_means_t | -0.0627 | -0.0993 | -0.0262 | 0.0089 | -0.1310 | -0.0071 | 2000 |
| 500 | sup_imnet | arm D - arm B | specificity | 0.1061 | 0.1160 | 0.2102 | 15 | 5 | partition_means_t | 0.1061 | -0.0379 | 0.2501 | 0.1101 | -0.0459 | 0.2357 | 1691 |
| 500 | ssl_imnet | arm D - arm B | specificity | 0.2458 | 0.1307 | 0.2063 | 15 | 5 | partition_means_t | 0.2458 | 0.0836 | 0.4080 | 0.0136 | 0.1476 | 0.3424 | 1691 |
| 500 | ssl_colon | arm D - arm B | specificity | 0.2972 | 0.1796 | 0.1642 | 15 | 5 | partition_means_t | 0.2972 | 0.0742 | 0.5202 | 0.0208 | 0.1556 | 0.4215 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm B | specificity | 0.2980 | 0.1391 | 0.2172 | 15 | 5 | partition_means_t | 0.2980 | 0.1253 | 0.4708 | 0.0087 | 0.1897 | 0.3626 | 1691 |
| 500 | mean of regimes | arm D - arm B | specificity | 0.2368 | 0.1165 | 0.1055 | 15 | 5 | partition_means_t | 0.2368 | 0.0921 | 0.3815 | 0.0105 | 0.1224 | 0.3293 | 1691 |
| 500 | sup_imnet | arm D - arm B | balanced_accuracy | 0.0355 | 0.0430 | 0.0638 | 15 | 5 | partition_means_t | 0.0355 | -0.0178 | 0.0889 | 0.1381 | -0.0436 | 0.0998 | 1691 |
| 500 | ssl_imnet | arm D - arm B | balanced_accuracy | 0.1088 | 0.0461 | 0.0848 | 15 | 5 | partition_means_t | 0.1088 | 0.0515 | 0.1661 | 0.0062 | 0.0534 | 0.1590 | 1691 |
| 500 | ssl_colon | arm D - arm B | balanced_accuracy | 0.0905 | 0.0514 | 0.0641 | 15 | 5 | partition_means_t | 0.0905 | 0.0266 | 0.1543 | 0.0171 | -0.0033 | 0.1712 | 1691 |
| 500 | ssl_imnet_matched | arm D - arm B | balanced_accuracy | 0.1133 | 0.0627 | 0.0946 | 15 | 5 | partition_means_t | 0.1133 | 0.0355 | 0.1912 | 0.0156 | 0.0437 | 0.1598 | 1691 |
| 500 | mean of regimes | arm D - arm B | balanced_accuracy | 0.0870 | 0.0446 | 0.0428 | 15 | 5 | partition_means_t | 0.0870 | 0.0317 | 0.1424 | 0.0120 | 0.0185 | 0.1414 | 1691 |
| 500 | sup_imnet | arm C - arm B | auroc | 0.0488 | 0.0381 | 0.0558 | 15 | 5 | partition_means_t | 0.0488 | 0.0015 | 0.0960 | 0.0457 | -0.0300 | 0.1148 | 1691 |
| 500 | ssl_imnet | arm C - arm B | auroc | 0.1008 | 0.0679 | 0.0785 | 15 | 5 | partition_means_t | 0.1008 | 0.0165 | 0.1851 | 0.0294 | 0.0231 | 0.1754 | 1691 |
| 500 | ssl_colon | arm C - arm B | auroc | 0.1472 | 0.0965 | 0.1214 | 15 | 5 | partition_means_t | 0.1472 | 0.0273 | 0.2670 | 0.0271 | -0.0219 | 0.2791 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm B | auroc | 0.1872 | 0.1190 | 0.1498 | 15 | 5 | partition_means_t | 0.1872 | 0.0394 | 0.3349 | 0.0245 | 0.0160 | 0.3289 | 1691 |
| 500 | mean of regimes | arm C - arm B | auroc | 0.1210 | 0.0755 | 0.0809 | 15 | 5 | partition_means_t | 0.1210 | 0.0273 | 0.2147 | 0.0231 | 1.58e-04 | 0.2198 | 1691 |
| 500 | sup_imnet | arm C - arm B | auprc | 0.0541 | 0.0400 | 0.0779 | 15 | 5 | partition_means_t | 0.0541 | 0.0045 | 0.1038 | 0.0388 | -0.0151 | 0.1122 | 1691 |
| 500 | ssl_imnet | arm C - arm B | auprc | 0.0797 | 0.0666 | 0.0993 | 15 | 5 | partition_means_t | 0.0797 | -0.0029 | 0.1624 | 0.0553 | 0.0068 | 0.1433 | 1691 |
| 500 | ssl_colon | arm C - arm B | auprc | 0.1470 | 0.0903 | 0.1513 | 15 | 5 | partition_means_t | 0.1470 | 0.0349 | 0.2591 | 0.0219 | -0.0153 | 0.2525 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm B | auprc | 0.1870 | 0.1104 | 0.1733 | 15 | 5 | partition_means_t | 0.1870 | 0.0499 | 0.3242 | 0.0193 | 0.0039 | 0.3004 | 1691 |
| 500 | mean of regimes | arm C - arm B | auprc | 0.1170 | 0.0727 | 0.1027 | 15 | 5 | partition_means_t | 0.1170 | 0.0267 | 0.2072 | 0.0228 | -0.0013 | 0.1989 | 1691 |
| 500 | sup_imnet | arm C - arm B | recall | -0.3861 | 0.2101 | 0.2835 | 15 | 5 | partition_means_t | -0.3861 | -0.6470 | -0.1251 | 0.0148 | -0.4291 | -0.3400 | 2000 |
| 500 | ssl_imnet | arm C - arm B | recall | -0.1417 | 0.1044 | 0.2296 | 15 | 5 | partition_means_t | -0.1417 | -0.2713 | -0.0121 | 0.0386 | -0.1966 | -0.1048 | 2000 |
| 500 | ssl_colon | arm C - arm B | recall | -0.2892 | 0.1932 | 0.3063 | 15 | 5 | partition_means_t | -0.2892 | -0.5291 | -0.0493 | 0.0287 | -0.3828 | -0.1963 | 2000 |
| 500 | ssl_imnet_matched | arm C - arm B | recall | -0.4526 | 0.2521 | 0.2463 | 15 | 5 | partition_means_t | -0.4526 | -0.7656 | -0.1395 | 0.0159 | -0.5403 | -0.3588 | 2000 |
| 500 | mean of regimes | arm C - arm B | recall | -0.3174 | 0.0451 | 0.1584 | 15 | 5 | partition_means_t | -0.3174 | -0.3734 | -0.2614 | 9.52e-05 | -0.3836 | -0.2501 | 2000 |
| 500 | sup_imnet | arm C - arm B | specificity | 0.2041 | 0.0952 | 0.1799 | 15 | 5 | partition_means_t | 0.2041 | 0.0860 | 0.3223 | 0.0087 | 0.0421 | 0.3527 | 1691 |
| 500 | ssl_imnet | arm C - arm B | specificity | 0.2577 | 0.2227 | 0.2714 | 15 | 5 | partition_means_t | 0.2577 | -0.0189 | 0.5342 | 0.0609 | 0.1352 | 0.3675 | 1691 |
| 500 | ssl_colon | arm C - arm B | specificity | 0.3414 | 0.1524 | 0.2453 | 15 | 5 | partition_means_t | 0.3414 | 0.1522 | 0.5306 | 0.0074 | 0.1958 | 0.4730 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm B | specificity | 0.4945 | 0.1433 | 0.2585 | 15 | 5 | partition_means_t | 0.4945 | 0.3166 | 0.6724 | 0.0015 | 0.3640 | 0.5895 | 1691 |
| 500 | mean of regimes | arm C - arm B | specificity | 0.3244 | 0.1049 | 0.1351 | 15 | 5 | partition_means_t | 0.3244 | 0.1942 | 0.4547 | 0.0023 | 0.1986 | 0.4352 | 1691 |
| 500 | sup_imnet | arm C - arm B | balanced_accuracy | -0.0910 | 0.0841 | 0.1004 | 15 | 5 | partition_means_t | -0.0910 | -0.1955 | 0.0135 | 0.0730 | -0.1755 | -0.0135 | 1691 |
| 500 | ssl_imnet | arm C - arm B | balanced_accuracy | 0.0580 | 0.0636 | 0.0972 | 15 | 5 | partition_means_t | 0.0580 | -0.0210 | 0.1370 | 0.1113 | -0.0109 | 0.1155 | 1691 |
| 500 | ssl_colon | arm C - arm B | balanced_accuracy | 0.0261 | 0.0764 | 0.1209 | 15 | 5 | partition_means_t | 0.0261 | -0.0687 | 0.1210 | 0.4872 | -0.0626 | 0.1050 | 1691 |
| 500 | ssl_imnet_matched | arm C - arm B | balanced_accuracy | 0.0210 | 0.1028 | 0.1323 | 15 | 5 | partition_means_t | 0.0210 | -0.1067 | 0.1487 | 0.6722 | -0.0597 | 0.0906 | 1691 |
| 500 | mean of regimes | arm C - arm B | balanced_accuracy | 0.0035 | 0.0454 | 0.0801 | 15 | 5 | partition_means_t | 0.0035 | -0.0528 | 0.0599 | 0.8705 | -0.0689 | 0.0668 | 1691 |
