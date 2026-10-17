### Sensitivity: negative-source decomposition leave-one-centre-out (no bootstrap)

| excluded | slot | regime | comparison | metric | hypothesis | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | 0.1481 | 0.0289 | 0.0212 | 15 | 5 | partition_means_t | 0.1481 | 0.1123 | 0.1840 | 3.30e-04 |  |  | 0 |
| C1 |  | sup_imnet | sequence - isolated | auroc |  | 0.1930 | 0.0192 | 0.0430 | 15 | 5 | partition_means_t | 0.1930 | 0.1691 | 0.2168 | 2.31e-05 |  |  | 0 |
| C1 |  | ssl_imnet | sequence - isolated | auroc |  | 0.1250 | 0.0387 | 0.0327 | 15 | 5 | partition_means_t | 0.1250 | 0.0770 | 0.1730 | 0.0019 |  |  | 0 |
| C1 |  | ssl_colon | sequence - isolated | auroc |  | 0.1414 | 0.0356 | 0.0206 | 15 | 5 | partition_means_t | 0.1414 | 0.0973 | 0.1856 | 8.84e-04 |  |  | 0 |
| C1 |  | ssl_imnet_matched | sequence - isolated | auroc |  | 0.1330 | 0.0391 | 0.0248 | 15 | 5 | partition_means_t | 0.1330 | 0.0844 | 0.1816 | 0.0016 |  |  | 0 |
| C1 | isolated |  | ssl_colon - sup_imnet | auroc |  | -0.0439 | 0.0163 | 0.0321 | 15 | 5 | partition_means_t | -0.0439 | -0.0642 | -0.0236 | 0.0039 |  |  | 0 |
| C1 | isolated |  | ssl_colon - sup_imnet | auprc |  | -0.0095 | 0.0058 | 0.0083 | 15 | 5 | partition_means_t | -0.0095 | -0.0167 | -0.0023 | 0.0218 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet | auroc |  | -0.0574 | 0.0175 | 0.0251 | 15 | 5 | partition_means_t | -0.0574 | -0.0791 | -0.0356 | 0.0019 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet | auprc |  | -0.0173 | 0.0056 | 0.0079 | 15 | 5 | partition_means_t | -0.0173 | -0.0243 | -0.0103 | 0.0024 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0216 | 0.0181 | 0.0177 | 15 | 5 | partition_means_t | 0.0216 | -8.73e-04 | 0.0440 | 0.0559 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0066 | 0.0074 | 0.0057 | 15 | 5 | partition_means_t | 0.0066 | -0.0026 | 0.0157 | 0.1174 |  |  | 0 |
| C1 | sequence |  | ssl_colon - sup_imnet | auroc |  | -0.0954 | 0.0185 | 0.0450 | 15 | 5 | partition_means_t | -0.0954 | -0.1184 | -0.0724 | 3.25e-04 |  |  | 0 |
| C1 | sequence |  | ssl_colon - sup_imnet | auprc |  | -0.2168 | 0.0397 | 0.1173 | 15 | 5 | partition_means_t | -0.2168 | -0.2660 | -0.1675 | 2.58e-04 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet | auroc |  | -0.0409 | 0.0287 | 0.0274 | 15 | 5 | partition_means_t | -0.0409 | -0.0766 | -0.0053 | 0.0334 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet | auprc |  | -0.1465 | 0.0811 | 0.0575 | 15 | 5 | partition_means_t | -0.1465 | -0.2472 | -0.0458 | 0.0156 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0300 | 0.0104 | 0.0284 | 15 | 5 | partition_means_t | 0.0300 | 0.0171 | 0.0429 | 0.0030 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0617 | 0.0168 | 0.0677 | 15 | 5 | partition_means_t | 0.0617 | 0.0408 | 0.0826 | 0.0012 |  |  | 0 |
| C2 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | 0.1531 | 0.0322 | 0.0242 | 15 | 5 | partition_means_t | 0.1531 | 0.1131 | 0.1932 | 4.45e-04 |  |  | 0 |
| C2 |  | sup_imnet | sequence - isolated | auroc |  | 0.2127 | 0.0208 | 0.0468 | 15 | 5 | partition_means_t | 0.2127 | 0.1869 | 0.2385 | 2.16e-05 |  |  | 0 |
| C2 |  | ssl_imnet | sequence - isolated | auroc |  | 0.1371 | 0.0412 | 0.0429 | 15 | 5 | partition_means_t | 0.1371 | 0.0859 | 0.1883 | 0.0017 |  |  | 0 |
| C2 |  | ssl_colon | sequence - isolated | auroc |  | 0.1294 | 0.0446 | 0.0291 | 15 | 5 | partition_means_t | 0.1294 | 0.0740 | 0.1847 | 0.0029 |  |  | 0 |
| C2 |  | ssl_imnet_matched | sequence - isolated | auroc |  | 0.1333 | 0.0350 | 0.0290 | 15 | 5 | partition_means_t | 0.1333 | 0.0898 | 0.1767 | 0.0010 |  |  | 0 |
| C2 | isolated |  | ssl_colon - sup_imnet | auroc |  | -0.0325 | 0.0208 | 0.0460 | 15 | 5 | partition_means_t | -0.0325 | -0.0583 | -0.0067 | 0.0250 |  |  | 0 |
| C2 | isolated |  | ssl_colon - sup_imnet | auprc |  | -9.06e-05 | 0.0059 | 0.0110 | 15 | 5 | partition_means_t | -9.06e-05 | -0.0074 | 0.0072 | 0.9741 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet | auroc |  | -0.0430 | 0.0097 | 0.0386 | 15 | 5 | partition_means_t | -0.0430 | -0.0551 | -0.0309 | 5.93e-04 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet | auprc |  | -0.0069 | 0.0034 | 0.0093 | 15 | 5 | partition_means_t | -0.0069 | -0.0112 | -0.0027 | 0.0104 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0082 | 0.0164 | 0.0230 | 15 | 5 | partition_means_t | 0.0082 | -0.0122 | 0.0285 | 0.3274 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0047 | 0.0064 | 0.0065 | 15 | 5 | partition_means_t | 0.0047 | -0.0032 | 0.0127 | 0.1735 |  |  | 0 |
| C2 | sequence |  | ssl_colon - sup_imnet | auroc |  | -0.1158 | 0.0228 | 0.0563 | 15 | 5 | partition_means_t | -0.1158 | -0.1441 | -0.0875 | 3.41e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - sup_imnet | auprc |  | -0.2372 | 0.0414 | 0.1254 | 15 | 5 | partition_means_t | -0.2372 | -0.2885 | -0.1858 | 2.14e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet | auroc |  | -0.0507 | 0.0326 | 0.0352 | 15 | 5 | partition_means_t | -0.0507 | -0.0912 | -0.0102 | 0.0255 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet | auprc |  | -0.1500 | 0.0760 | 0.0545 | 15 | 5 | partition_means_t | -0.1500 | -0.2444 | -0.0556 | 0.0116 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0043 | 0.0151 | 0.0321 | 15 | 5 | partition_means_t | 0.0043 | -0.0145 | 0.0230 | 0.5594 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0183 | 0.0192 | 0.0576 | 15 | 5 | partition_means_t | 0.0183 | -0.0056 | 0.0422 | 0.1012 |  |  | 0 |
| C3 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | 0.1070 | 0.0213 | 0.0206 | 15 | 5 | partition_means_t | 0.1070 | 0.0806 | 0.1334 | 3.55e-04 |  |  | 0 |
| C3 |  | sup_imnet | sequence - isolated | auroc |  | 0.0979 | 0.0039 | 0.0321 | 15 | 5 | partition_means_t | 0.0979 | 0.0930 | 0.1027 | 6.05e-07 |  |  | 0 |
| C3 |  | ssl_imnet | sequence - isolated | auroc |  | 0.0598 | 0.0329 | 0.0286 | 15 | 5 | partition_means_t | 0.0598 | 0.0190 | 0.1007 | 0.0153 |  |  | 0 |
| C3 |  | ssl_colon | sequence - isolated | auroc |  | 0.1550 | 0.0341 | 0.0371 | 15 | 5 | partition_means_t | 0.1550 | 0.1127 | 0.1973 | 5.27e-04 |  |  | 0 |
| C3 |  | ssl_imnet_matched | sequence - isolated | auroc |  | 0.1154 | 0.0343 | 0.0348 | 15 | 5 | partition_means_t | 0.1154 | 0.0728 | 0.1581 | 0.0017 |  |  | 0 |
| C3 | isolated |  | ssl_colon - sup_imnet | auroc |  | -0.1629 | 0.0364 | 0.0664 | 15 | 5 | partition_means_t | -0.1629 | -0.2080 | -0.1177 | 5.59e-04 |  |  | 0 |
| C3 | isolated |  | ssl_colon - sup_imnet | auprc |  | -0.0425 | 0.0077 | 0.0131 | 15 | 5 | partition_means_t | -0.0425 | -0.0521 | -0.0329 | 2.53e-04 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet | auroc |  | -0.1457 | 0.0213 | 0.0512 | 15 | 5 | partition_means_t | -0.1457 | -0.1721 | -0.1192 | 1.07e-04 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet | auprc |  | -0.0393 | 0.0075 | 0.0115 | 15 | 5 | partition_means_t | -0.0393 | -0.0487 | -0.0300 | 3.07e-04 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | -0.0147 | 0.0232 | 0.0335 | 15 | 5 | partition_means_t | -0.0147 | -0.0435 | 0.0142 | 0.2306 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | -0.0064 | 0.0062 | 0.0064 | 15 | 5 | partition_means_t | -0.0064 | -0.0140 | 0.0013 | 0.0833 |  |  | 0 |
| C3 | sequence |  | ssl_colon - sup_imnet | auroc |  | -0.1058 | 0.0230 | 0.0491 | 15 | 5 | partition_means_t | -0.1058 | -0.1343 | -0.0773 | 5.00e-04 |  |  | 0 |
| C3 | sequence |  | ssl_colon - sup_imnet | auprc |  | -0.2225 | 0.0467 | 0.1208 | 15 | 5 | partition_means_t | -0.2225 | -0.2805 | -0.1645 | 4.39e-04 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet | auroc |  | -0.0505 | 0.0309 | 0.0295 | 15 | 5 | partition_means_t | -0.0505 | -0.0889 | -0.0121 | 0.0217 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet | auprc |  | -0.1438 | 0.0816 | 0.0617 | 15 | 5 | partition_means_t | -0.1438 | -0.2450 | -0.0425 | 0.0169 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0249 | 0.0124 | 0.0356 | 15 | 5 | partition_means_t | 0.0249 | 0.0095 | 0.0403 | 0.0109 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0542 | 0.0180 | 0.0605 | 15 | 5 | partition_means_t | 0.0542 | 0.0319 | 0.0766 | 0.0025 |  |  | 0 |
| C4 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | 0.2122 | 0.0315 | 0.0245 | 15 | 5 | partition_means_t | 0.2122 | 0.1732 | 0.2513 | 1.13e-04 |  |  | 0 |
| C4 |  | sup_imnet | sequence - isolated | auroc |  | 0.2727 | 0.0274 | 0.0551 | 15 | 5 | partition_means_t | 0.2727 | 0.2387 | 0.3068 | 2.43e-05 |  |  | 0 |
| C4 |  | ssl_imnet | sequence - isolated | auroc |  | 0.1949 | 0.0361 | 0.0299 | 15 | 5 | partition_means_t | 0.1949 | 0.1500 | 0.2398 | 2.71e-04 |  |  | 0 |
| C4 |  | ssl_colon | sequence - isolated | auroc |  | 0.2093 | 0.0379 | 0.0267 | 15 | 5 | partition_means_t | 0.2093 | 0.1623 | 0.2564 | 2.46e-04 |  |  | 0 |
| C4 |  | ssl_imnet_matched | sequence - isolated | auroc |  | 0.1719 | 0.0513 | 0.0228 | 15 | 5 | partition_means_t | 0.1719 | 0.1082 | 0.2356 | 0.0017 |  |  | 0 |
| C4 | isolated |  | ssl_colon - sup_imnet | auroc |  | -0.0400 | 0.0185 | 0.0231 | 15 | 5 | partition_means_t | -0.0400 | -0.0630 | -0.0170 | 0.0085 |  |  | 0 |
| C4 | isolated |  | ssl_colon - sup_imnet | auprc |  | -0.0054 | 0.0063 | 0.0065 | 15 | 5 | partition_means_t | -0.0054 | -0.0132 | 0.0024 | 0.1258 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet | auroc |  | -0.0630 | 0.0299 | 0.0183 | 15 | 5 | partition_means_t | -0.0630 | -0.1002 | -0.0258 | 0.0093 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet | auprc |  | -0.0117 | 0.0053 | 0.0062 | 15 | 5 | partition_means_t | -0.0117 | -0.0183 | -0.0051 | 0.0080 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | -0.0144 | 0.0331 | 0.0215 | 15 | 5 | partition_means_t | -0.0144 | -0.0555 | 0.0268 | 0.3875 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | -0.0020 | 0.0076 | 0.0053 | 15 | 5 | partition_means_t | -0.0020 | -0.0114 | 0.0074 | 0.5901 |  |  | 0 |
| C4 | sequence |  | ssl_colon - sup_imnet | auroc |  | -0.1034 | 0.0196 | 0.0525 | 15 | 5 | partition_means_t | -0.1034 | -0.1277 | -0.0790 | 2.96e-04 |  |  | 0 |
| C4 | sequence |  | ssl_colon - sup_imnet | auprc |  | -0.2251 | 0.0414 | 0.1237 | 15 | 5 | partition_means_t | -0.2251 | -0.2765 | -0.1736 | 2.63e-04 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet | auroc |  | -0.0486 | 0.0265 | 0.0305 | 15 | 5 | partition_means_t | -0.0486 | -0.0814 | -0.0157 | 0.0148 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet | auprc |  | -0.1539 | 0.0794 | 0.0626 | 15 | 5 | partition_means_t | -0.1539 | -0.2526 | -0.0553 | 0.0123 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0231 | 0.0120 | 0.0313 | 15 | 5 | partition_means_t | 0.0231 | 0.0082 | 0.0379 | 0.0125 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0500 | 0.0175 | 0.0690 | 15 | 5 | partition_means_t | 0.0500 | 0.0283 | 0.0718 | 0.0031 |  |  | 0 |
| C5 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | 0.1562 | 0.0281 | 0.0204 | 15 | 5 | partition_means_t | 0.1562 | 0.1213 | 0.1911 | 2.41e-04 |  |  | 0 |
| C5 |  | sup_imnet | sequence - isolated | auroc |  | 0.1908 | 0.0152 | 0.0372 | 15 | 5 | partition_means_t | 0.1908 | 0.1720 | 0.2096 | 9.46e-06 |  |  | 0 |
| C5 |  | ssl_imnet | sequence - isolated | auroc |  | 0.1289 | 0.0365 | 0.0314 | 15 | 5 | partition_means_t | 0.1289 | 0.0837 | 0.1742 | 0.0014 |  |  | 0 |
| C5 |  | ssl_colon | sequence - isolated | auroc |  | 0.1646 | 0.0398 | 0.0223 | 15 | 5 | partition_means_t | 0.1646 | 0.1152 | 0.2140 | 7.59e-04 |  |  | 0 |
| C5 |  | ssl_imnet_matched | sequence - isolated | auroc |  | 0.1403 | 0.0388 | 0.0251 | 15 | 5 | partition_means_t | 0.1403 | 0.0921 | 0.1885 | 0.0013 |  |  | 0 |
| C5 | isolated |  | ssl_colon - sup_imnet | auroc |  | -0.0953 | 0.0233 | 0.0415 | 15 | 5 | partition_means_t | -0.0953 | -0.1242 | -0.0664 | 7.89e-04 |  |  | 0 |
| C5 | isolated |  | ssl_colon - sup_imnet | auprc |  | -0.0249 | 0.0066 | 0.0102 | 15 | 5 | partition_means_t | -0.0249 | -0.0330 | -0.0167 | 0.0011 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet | auroc |  | -0.0998 | 0.0130 | 0.0294 | 15 | 5 | partition_means_t | -0.0998 | -0.1159 | -0.0837 | 6.66e-05 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet | auprc |  | -0.0296 | 0.0043 | 0.0092 | 15 | 5 | partition_means_t | -0.0296 | -0.0349 | -0.0243 | 1.02e-04 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | -0.0156 | 0.0150 | 0.0221 | 15 | 5 | partition_means_t | -0.0156 | -0.0343 | 0.0030 | 0.0807 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | -0.0063 | 0.0060 | 0.0060 | 15 | 5 | partition_means_t | -0.0063 | -0.0137 | 0.0012 | 0.0804 |  |  | 0 |
| C5 | sequence |  | ssl_colon - sup_imnet | auroc |  | -0.1215 | 0.0226 | 0.0561 | 15 | 5 | partition_means_t | -0.1215 | -0.1495 | -0.0935 | 2.73e-04 |  |  | 0 |
| C5 | sequence |  | ssl_colon - sup_imnet | auprc |  | -0.2553 | 0.0414 | 0.1280 | 15 | 5 | partition_means_t | -0.2553 | -0.3068 | -0.2039 | 1.60e-04 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet | auroc |  | -0.0641 | 0.0303 | 0.0317 | 15 | 5 | partition_means_t | -0.0641 | -0.1018 | -0.0265 | 0.0091 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet | auprc |  | -0.1763 | 0.0789 | 0.0530 | 15 | 5 | partition_means_t | -0.1763 | -0.2742 | -0.0784 | 0.0075 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0087 | 0.0154 | 0.0346 | 15 | 5 | partition_means_t | 0.0087 | -0.0105 | 0.0278 | 0.2761 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0247 | 0.0193 | 0.0589 | 15 | 5 | partition_means_t | 0.0247 | 7.98e-04 | 0.0487 | 0.0455 |  |  | 0 |
| C6 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | 0.1593 | 0.0277 | 0.0200 | 15 | 5 | partition_means_t | 0.1593 | 0.1248 | 0.1937 | 2.12e-04 |  |  | 0 |
| C6 |  | sup_imnet | sequence - isolated | auroc |  | 0.1990 | 0.0157 | 0.0394 | 15 | 5 | partition_means_t | 0.1990 | 0.1795 | 0.2185 | 9.24e-06 |  |  | 0 |
| C6 |  | ssl_imnet | sequence - isolated | auroc |  | 0.1364 | 0.0379 | 0.0322 | 15 | 5 | partition_means_t | 0.1364 | 0.0893 | 0.1834 | 0.0013 |  |  | 0 |
| C6 |  | ssl_colon | sequence - isolated | auroc |  | 0.1599 | 0.0384 | 0.0230 | 15 | 5 | partition_means_t | 0.1599 | 0.1122 | 0.2077 | 7.43e-04 |  |  | 0 |
| C6 |  | ssl_imnet_matched | sequence - isolated | auroc |  | 0.1417 | 0.0354 | 0.0239 | 15 | 5 | partition_means_t | 0.1417 | 0.0978 | 0.1856 | 8.58e-04 |  |  | 0 |
| C6 | isolated |  | ssl_colon - sup_imnet | auroc |  | -0.0635 | 0.0205 | 0.0385 | 15 | 5 | partition_means_t | -0.0635 | -0.0890 | -0.0380 | 0.0023 |  |  | 0 |
| C6 | isolated |  | ssl_colon - sup_imnet | auprc |  | -0.0135 | 0.0055 | 0.0083 | 15 | 5 | partition_means_t | -0.0135 | -0.0204 | -0.0066 | 0.0055 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet | auroc |  | -0.0707 | 0.0146 | 0.0293 | 15 | 5 | partition_means_t | -0.0707 | -0.0888 | -0.0527 | 4.07e-04 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet | auprc |  | -0.0187 | 0.0047 | 0.0080 | 15 | 5 | partition_means_t | -0.0187 | -0.0245 | -0.0130 | 8.43e-04 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | -1.00e-06 | 0.0162 | 0.0186 | 15 | 5 | partition_means_t | -1.00e-06 | -0.0201 | 0.0201 | 0.9999 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | -2.98e-04 | 0.0064 | 0.0051 | 15 | 5 | partition_means_t | -2.98e-04 | -0.0083 | 0.0077 | 0.9226 |  |  | 0 |
| C6 | sequence |  | ssl_colon - sup_imnet | auroc |  | -0.1026 | 0.0213 | 0.0508 | 15 | 5 | partition_means_t | -0.1026 | -0.1290 | -0.0762 | 4.18e-04 |  |  | 0 |
| C6 | sequence |  | ssl_colon - sup_imnet | auprc |  | -0.2207 | 0.0393 | 0.1180 | 15 | 5 | partition_means_t | -0.2207 | -0.2696 | -0.1719 | 2.32e-04 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet | auroc |  | -0.0472 | 0.0293 | 0.0293 | 15 | 5 | partition_means_t | -0.0472 | -0.0836 | -0.0107 | 0.0229 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet | auprc |  | -0.1476 | 0.0762 | 0.0551 | 15 | 5 | partition_means_t | -0.1476 | -0.2422 | -0.0529 | 0.0124 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0182 | 0.0125 | 0.0304 | 15 | 5 | partition_means_t | 0.0182 | 0.0027 | 0.0337 | 0.0309 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0425 | 0.0171 | 0.0637 | 15 | 5 | partition_means_t | 0.0425 | 0.0213 | 0.0637 | 0.0051 |  |  | 0 |
