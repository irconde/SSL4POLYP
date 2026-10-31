# E4 arm contrasts: registered vs (a) degenerate pairs excluded vs (b) seed medians

Post hoc sensitivity analysis, not registered. Balanced accuracy, mean of the four regimes.

### Pairs excluded in (a)

| id | comparison | budget | pairs | pairs_excluded | pairs_excluded_treat_degenerate | pairs_excluded_base_degenerate | runs_dropped | excluded_sup_imnet | excluded_ssl_imnet | excluded_ssl_colon | excluded_ssl_imnet_matched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H6 | B - A | mean of budgets | 240 | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| H7 | C - A | mean of budgets | 240 | 45 | 44 | 1 | 0 | 9 | 5 | 8 | 23 |
| arm/50/pooled/B-A/balanced_accuracy | B - A | 50 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arm/50/pooled/C-A/balanced_accuracy | C - A | 50 | 60 | 12 | 12 | 0 | 0 | 1 | 3 | 1 | 7 |
| arm/50/pooled/D-C/balanced_accuracy | D - C | 50 | 60 | 12 | 0 | 12 | 0 | 1 | 3 | 1 | 7 |
| arm/50/pooled/D-A/balanced_accuracy | D - A | 50 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arm/100/pooled/B-A/balanced_accuracy | B - A | 100 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arm/100/pooled/C-A/balanced_accuracy | C - A | 100 | 60 | 15 | 15 | 0 | 0 | 4 | 2 | 2 | 7 |
| arm/100/pooled/D-C/balanced_accuracy | D - C | 100 | 60 | 17 | 2 | 15 | 0 | 4 | 2 | 3 | 8 |
| arm/100/pooled/D-A/balanced_accuracy | D - A | 100 | 60 | 2 | 2 | 0 | 0 | 0 | 0 | 1 | 1 |
| arm/200/pooled/B-A/balanced_accuracy | B - A | 200 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arm/200/pooled/C-A/balanced_accuracy | C - A | 200 | 60 | 13 | 13 | 0 | 0 | 3 | 0 | 3 | 7 |
| arm/200/pooled/D-C/balanced_accuracy | D - C | 200 | 60 | 14 | 1 | 13 | 0 | 3 | 0 | 3 | 8 |
| arm/200/pooled/D-A/balanced_accuracy | D - A | 200 | 60 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| arm/500/pooled/B-A/balanced_accuracy | B - A | 500 | 60 | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| arm/500/pooled/C-A/balanced_accuracy | C - A | 500 | 60 | 5 | 4 | 1 | 0 | 1 | 0 | 2 | 2 |
| arm/500/pooled/D-C/balanced_accuracy | D - C | 500 | 60 | 4 | 0 | 4 | 0 | 1 | 0 | 1 | 2 |
| arm/500/pooled/D-A/balanced_accuracy | D - A | 500 | 60 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |

### Contrasts

| id | variant | n_runs | estimate | t_ci_lo | t_ci_hi | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|
| H6 | registered | 15 | -0.0064 | -0.0241 | 0.0114 | -0.0184 | 0.0058 | 1691 |
| H6 | a_exclusion | 15 | -0.0062 | -0.0243 | 0.0119 | -0.0182 | 0.0060 | 1691 |
| H6 | b_seed_median | 15 | -0.0044 | -0.0183 | 0.0096 | -0.0143 | 0.0087 | 1691 |
| H7 | registered | 15 | -0.0748 | -0.1012 | -0.0485 | -0.1353 | -0.0150 | 1691 |
| H7 | a_exclusion | 15 | -0.0423 | -0.0693 | -0.0154 | -0.1000 | 0.0142 | 1691 |
| H7 | b_seed_median | 15 | -0.0817 | -0.1209 | -0.0425 | -0.1416 | -0.0176 | 1691 |
| arm/50/pooled/B-A/balanced_accuracy | registered | 15 | -0.0049 | -0.0279 | 0.0182 | -0.0217 | 0.0101 | 2000 |
| arm/50/pooled/B-A/balanced_accuracy | a_exclusion | 15 | -0.0049 | -0.0279 | 0.0182 | -0.0217 | 0.0101 | 2000 |
| arm/50/pooled/B-A/balanced_accuracy | b_seed_median | 15 | 9.77e-04 | -0.0193 | 0.0212 | -0.0178 | 0.0153 | 2000 |
| arm/50/pooled/C-A/balanced_accuracy | registered | 15 | -0.0962 | -0.1234 | -0.0690 | -0.1447 | -0.0440 | 2000 |
| arm/50/pooled/C-A/balanced_accuracy | a_exclusion | 15 | -0.0627 | -0.1133 | -0.0121 | -0.1053 | -0.0192 | 2000 |
| arm/50/pooled/C-A/balanced_accuracy | b_seed_median | 15 | -0.0904 | -0.1153 | -0.0655 | -0.1398 | -0.0398 | 2000 |
| arm/50/pooled/D-C/balanced_accuracy | registered | 15 | 0.0355 | 0.0181 | 0.0529 | 0.0213 | 0.0457 | 2000 |
| arm/50/pooled/D-C/balanced_accuracy | a_exclusion | 15 | 0.0157 | -0.0135 | 0.0449 | 0.0042 | 0.0243 | 2000 |
| arm/50/pooled/D-C/balanced_accuracy | b_seed_median | 15 | 0.0453 | 0.0266 | 0.0640 | 0.0269 | 0.0573 | 2000 |
| arm/50/pooled/D-A/balanced_accuracy | registered | 15 | -0.0607 | -0.0947 | -0.0267 | -0.1101 | -0.0111 | 2000 |
| arm/50/pooled/D-A/balanced_accuracy | a_exclusion | 15 | -0.0607 | -0.0947 | -0.0267 | -0.1101 | -0.0111 | 2000 |
| arm/50/pooled/D-A/balanced_accuracy | b_seed_median | 15 | -0.0561 | -0.0821 | -0.0302 | -0.1025 | -0.0101 | 2000 |
| arm/100/pooled/B-A/balanced_accuracy | registered | 15 | -0.0140 | -0.0390 | 0.0110 | -0.0295 | 0.0020 | 2000 |
| arm/100/pooled/B-A/balanced_accuracy | a_exclusion | 15 | -0.0140 | -0.0390 | 0.0110 | -0.0295 | 0.0020 | 2000 |
| arm/100/pooled/B-A/balanced_accuracy | b_seed_median | 15 | -0.0079 | -0.0299 | 0.0141 | -0.0243 | 0.0070 | 2000 |
| arm/100/pooled/C-A/balanced_accuracy | registered | 15 | -0.1042 | -0.1255 | -0.0829 | -0.1683 | -0.0336 | 2000 |
| arm/100/pooled/C-A/balanced_accuracy | a_exclusion | 15 | -0.0603 | -0.0844 | -0.0362 | -0.1197 | 0.0047 | 2000 |
| arm/100/pooled/C-A/balanced_accuracy | b_seed_median | 15 | -0.1014 | -0.1196 | -0.0832 | -0.1512 | -0.0437 | 2000 |
| arm/100/pooled/D-C/balanced_accuracy | registered | 15 | 0.0527 | 0.0136 | 0.0917 | 0.0383 | 0.0640 | 2000 |
| arm/100/pooled/D-C/balanced_accuracy | a_exclusion | 15 | 0.0210 | 1.24e-04 | 0.0418 | 0.0136 | 0.0272 | 2000 |
| arm/100/pooled/D-C/balanced_accuracy | b_seed_median | 15 | 0.0706 | 0.0260 | 0.1152 | 0.0481 | 0.0827 | 2000 |
| arm/100/pooled/D-A/balanced_accuracy | registered | 15 | -0.0515 | -0.0766 | -0.0264 | -0.1103 | 0.0115 | 2000 |
| arm/100/pooled/D-A/balanced_accuracy | a_exclusion | 15 | -0.0469 | -0.0650 | -0.0288 | -0.1055 | 0.0165 | 2000 |
| arm/100/pooled/D-A/balanced_accuracy | b_seed_median | 15 | -0.0410 | -0.0913 | 0.0094 | -0.0941 | 0.0190 | 2000 |
| arm/200/pooled/B-A/balanced_accuracy | registered | 15 | -0.0065 | -0.0143 | 0.0012 | -0.0191 | 0.0067 | 2000 |
| arm/200/pooled/B-A/balanced_accuracy | a_exclusion | 15 | -0.0065 | -0.0143 | 0.0012 | -0.0191 | 0.0067 | 2000 |
| arm/200/pooled/B-A/balanced_accuracy | b_seed_median | 15 | -0.0062 | -0.0154 | 0.0029 | -0.0223 | 0.0090 | 2000 |
| arm/200/pooled/C-A/balanced_accuracy | registered | 15 | -0.1025 | -0.1352 | -0.0698 | -0.1730 | -0.0232 | 2000 |
| arm/200/pooled/C-A/balanced_accuracy | a_exclusion | 15 | -0.0690 | -0.0992 | -0.0388 | -0.1391 | 0.0064 | 2000 |
| arm/200/pooled/C-A/balanced_accuracy | b_seed_median | 15 | -0.0870 | -0.1274 | -0.0466 | -0.1584 | -0.0142 | 2000 |
| arm/200/pooled/D-C/balanced_accuracy | registered | 15 | 0.0426 | 0.0021 | 0.0830 | 0.0254 | 0.0544 | 2000 |
| arm/200/pooled/D-C/balanced_accuracy | a_exclusion | 15 | 0.0245 | -0.0225 | 0.0714 | 0.0114 | 0.0345 | 2000 |
| arm/200/pooled/D-C/balanced_accuracy | b_seed_median | 15 | 0.0547 | 0.0170 | 0.0923 | 0.0332 | 0.0654 | 2000 |
| arm/200/pooled/D-A/balanced_accuracy | registered | 15 | -0.0599 | -0.0924 | -0.0275 | -0.1323 | 0.0177 | 2000 |
| arm/200/pooled/D-A/balanced_accuracy | a_exclusion | 15 | -0.0579 | -0.0878 | -0.0279 | -0.1297 | 0.0188 | 2000 |
| arm/200/pooled/D-A/balanced_accuracy | b_seed_median | 15 | -0.0548 | -0.0976 | -0.0119 | -0.1238 | 0.0211 | 2000 |
| arm/500/pooled/B-A/balanced_accuracy | registered | 15 | 7.31e-06 | -0.0209 | 0.0209 | -0.0089 | 0.0101 | 1691 |
| arm/500/pooled/B-A/balanced_accuracy | a_exclusion | 15 | 0.0030 | -0.0254 | 0.0314 | -0.0063 | 0.0132 | 1691 |
| arm/500/pooled/B-A/balanced_accuracy | b_seed_median | 15 | -0.0031 | -0.0251 | 0.0189 | -0.0125 | 0.0123 | 1691 |
| arm/500/pooled/C-A/balanced_accuracy | registered | 15 | 0.0035 | -0.0534 | 0.0604 | -0.0704 | 0.0722 | 1691 |
| arm/500/pooled/C-A/balanced_accuracy | a_exclusion | 15 | 0.0188 | -0.0381 | 0.0757 | -0.0538 | 0.0862 | 1691 |
| arm/500/pooled/C-A/balanced_accuracy | b_seed_median | 15 | 0.0122 | -0.0754 | 0.0997 | -0.0722 | 0.0975 | 1691 |
| arm/500/pooled/D-C/balanced_accuracy | registered | 15 | 0.0835 | 0.0613 | 0.1058 | 0.0640 | 0.0990 | 1691 |
| arm/500/pooled/D-C/balanced_accuracy | a_exclusion | 15 | 0.0710 | 0.0547 | 0.0873 | 0.0521 | 0.0872 | 1691 |
| arm/500/pooled/D-C/balanced_accuracy | b_seed_median | 15 | 0.0677 | 0.0415 | 0.0939 | 0.0480 | 0.0979 | 1691 |
| arm/500/pooled/D-A/balanced_accuracy | registered | 15 | 0.0870 | 0.0240 | 0.1501 | 0.0182 | 0.1433 | 1691 |
| arm/500/pooled/D-A/balanced_accuracy | a_exclusion | 15 | 0.0877 | 0.0237 | 0.1517 | 0.0199 | 0.1432 | 1691 |
| arm/500/pooled/D-A/balanced_accuracy | b_seed_median | 15 | 0.0829 | 0.0044 | 0.1613 | 0.0129 | 0.1484 | 1691 |

Registered aggregation recomputed: 108/108 quantities equal to contrast_summary.csv.
