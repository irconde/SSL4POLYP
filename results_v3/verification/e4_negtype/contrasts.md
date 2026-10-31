# E4: support size vs type of negatives (common evaluation set)

Post hoc sensitivity analysis, not registered. Mean of the four regimes (SUP-imnet, SSL-imnet, SSL-colon, SSL-imnet-matched). Estimate = mean of the paired per-run values (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, the registered draws).

### Arm means (15 runs × 4 regimes)

| condition | arm | balanced_accuracy | recall | specificity | auroc |
|---|---|---|---|---|---|
| 200iso | A | 0.7163 | 0.9123 | 0.5203 | 0.7776 |
| 200iso | B | 0.7074 | 0.8843 | 0.5304 | 0.7776 |
| 200iso | C | 0.6215 | 0.5179 | 0.7251 | 0.7444 |
| 200iso | D | 0.6610 | 0.7996 | 0.5224 | 0.7444 |
| 200seq | A | 0.7163 | 0.9123 | 0.5203 | 0.7776 |
| 200seq | B | 0.7286 | 0.8865 | 0.5708 | 0.7776 |
| 200seq | C | 0.8133 | 0.8347 | 0.7918 | 0.9163 |
| 200seq | D | 0.8148 | 0.8578 | 0.7717 | 0.9163 |
| 500 | A | 0.7163 | 0.9123 | 0.5203 | 0.7776 |
| 500 | B | 0.7163 | 0.9073 | 0.5253 | 0.7776 |
| 500 | C | 0.7198 | 0.5899 | 0.8498 | 0.8986 |
| 500 | D | 0.8033 | 0.8446 | 0.7621 | 0.8986 |

### Contrasts

| kind | condition | comparison | metric | estimate | t_ci_lo | t_ci_hi | boot_lo | boot_hi |
|---|---|---|---|---|---|---|---|---|
| arm | 200iso | B - A | balanced_accuracy | -0.0089 | -0.0228 | 0.0050 | -0.0213 | 0.0046 |
| arm | 200iso | B - A | recall | -0.0280 | -0.0839 | 0.0279 | -0.0426 | -0.0183 |
| arm | 200iso | B - A | specificity | 0.0101 | -0.0639 | 0.0842 | -0.0116 | 0.0341 |
| arm | 200iso | C - A | balanced_accuracy | -0.0948 | -0.1612 | -0.0284 | -0.1750 | -0.0228 |
| arm | 200iso | C - A | recall | -0.3944 | -0.5047 | -0.2841 | -0.4501 | -0.3322 |
| arm | 200iso | C - A | specificity | 0.2048 | 0.0609 | 0.3486 | 0.0583 | 0.3344 |
| arm | 200iso | C - A | auroc | -0.0332 | -0.1018 | 0.0353 | -0.1769 | 0.0866 |
| arm | 200iso | D - C | balanced_accuracy | 0.0395 | -0.0066 | 0.0856 | 0.0202 | 0.0533 |
| arm | 200iso | D - C | recall | 0.2816 | 0.1969 | 0.3663 | 0.2543 | 0.2959 |
| arm | 200iso | D - C | specificity | -0.2026 | -0.3303 | -0.0749 | -0.2300 | -0.1802 |
| arm | 200iso | D - A | balanced_accuracy | -0.0553 | -0.0922 | -0.0184 | -0.1403 | 0.0129 |
| arm | 200iso | D - A | recall | -0.1127 | -0.1802 | -0.0453 | -0.1868 | -0.0481 |
| arm | 200iso | D - A | specificity | 0.0021 | -0.0179 | 0.0222 | -0.1470 | 0.1254 |
| arm | 200iso | D - A | auroc | -0.0332 | -0.1018 | 0.0353 | -0.1769 | 0.0866 |
| arm | 200seq | B - A | balanced_accuracy | 0.0123 | -0.0015 | 0.0262 | 0.0015 | 0.0232 |
| arm | 200seq | B - A | recall | -0.0258 | -0.0795 | 0.0278 | -0.0380 | -0.0181 |
| arm | 200seq | B - A | specificity | 0.0505 | -0.0267 | 0.1276 | 0.0328 | 0.0708 |
| arm | 200seq | C - A | balanced_accuracy | 0.0970 | 0.0655 | 0.1284 | 0.0526 | 0.1333 |
| arm | 200seq | C - A | recall | -0.0776 | -0.1634 | 0.0082 | -0.1281 | -0.0394 |
| arm | 200seq | C - A | specificity | 0.2715 | 0.1874 | 0.3557 | 0.1991 | 0.3310 |
| arm | 200seq | C - A | auroc | 0.1387 | 0.0659 | 0.2116 | 0.0496 | 0.2140 |
| arm | 200seq | D - C | balanced_accuracy | 0.0015 | -0.0083 | 0.0113 | -0.0057 | 0.0124 |
| arm | 200seq | D - C | recall | 0.0231 | -0.0217 | 0.0679 | 0.0194 | 0.0283 |
| arm | 200seq | D - C | specificity | -0.0201 | -0.0489 | 0.0087 | -0.0344 | 4.71e-04 |
| arm | 200seq | D - A | balanced_accuracy | 0.0985 | 0.0676 | 0.1293 | 0.0611 | 0.1324 |
| arm | 200seq | D - A | recall | -0.0545 | -0.1036 | -0.0054 | -0.1060 | -0.0168 |
| arm | 200seq | D - A | specificity | 0.2514 | 0.1907 | 0.3122 | 0.1943 | 0.3072 |
| arm | 200seq | D - A | auroc | 0.1387 | 0.0659 | 0.2116 | 0.0496 | 0.2140 |
| arm | 500 | B - A | balanced_accuracy | 7.31e-06 | -0.0209 | 0.0209 | -0.0089 | 0.0101 |
| arm | 500 | B - A | recall | -0.0050 | -0.0594 | 0.0494 | -0.0148 | 0.0013 |
| arm | 500 | B - A | specificity | 0.0050 | -0.0835 | 0.0936 | -0.0105 | 0.0240 |
| arm | 500 | C - A | balanced_accuracy | 0.0035 | -0.0534 | 0.0604 | -0.0704 | 0.0722 |
| arm | 500 | C - A | recall | -0.3224 | -0.4069 | -0.2379 | -0.3949 | -0.2564 |
| arm | 500 | C - A | specificity | 0.3295 | 0.2019 | 0.4570 | 0.2023 | 0.4487 |
| arm | 500 | C - A | auroc | 0.1210 | 0.0273 | 0.2147 | 1.58e-04 | 0.2198 |
| arm | 500 | D - C | balanced_accuracy | 0.0835 | 0.0613 | 0.1058 | 0.0640 | 0.0990 |
| arm | 500 | D - C | recall | 0.2546 | 0.2050 | 0.3043 | 0.2383 | 0.2664 |
| arm | 500 | D - C | specificity | -0.0876 | -0.1573 | -0.0179 | -0.1209 | -0.0587 |
| arm | 500 | D - A | balanced_accuracy | 0.0870 | 0.0240 | 0.1501 | 0.0182 | 0.1433 |
| arm | 500 | D - A | recall | -0.0677 | -0.1252 | -0.0103 | -0.1433 | -0.0108 |
| arm | 500 | D - A | specificity | 0.2418 | 0.0854 | 0.3982 | 0.1265 | 0.3425 |
| arm | 500 | D - A | auroc | 0.1210 | 0.0273 | 0.2147 | 1.58e-04 | 0.2198 |
| type | 200seq - 200iso | [D - A](200seq) - [D - A](200iso) | balanced_accuracy | 0.1538 | 0.1363 | 0.1713 | 0.0955 | 0.2148 |
| type | 200seq - 200iso | [D - A](200seq) - [D - A](200iso) | recall | 0.0583 | 0.0236 | 0.0929 | -0.0048 | 0.1069 |
| type | 200seq - 200iso | [D - A](200seq) - [D - A](200iso) | specificity | 0.2493 | 0.1992 | 0.2994 | 0.1524 | 0.3613 |
| type | 200seq - 200iso | [D - A](200seq) - [D - A](200iso) | auroc | 0.1720 | 0.1565 | 0.1875 | 0.1031 | 0.2436 |
| type | 200seq - 200iso | [C - A](200seq) - [C - A](200iso) | balanced_accuracy | 0.1918 | 0.1546 | 0.2289 | 0.1329 | 0.2424 |
| type | 200seq - 200iso | [C - A](200seq) - [C - A](200iso) | recall | 0.3168 | 0.2597 | 0.3739 | 0.2510 | 0.3616 |
| type | 200seq - 200iso | [C - A](200seq) - [C - A](200iso) | specificity | 0.0668 | -0.0094 | 0.1429 | -0.0308 | 0.1598 |
| type | 200seq - 200iso | [C - A](200seq) - [C - A](200iso) | auroc | 0.1720 | 0.1565 | 0.1875 | 0.1031 | 0.2436 |
| size | 500 - 200seq | [D - A](500) - [D - A](200seq) | balanced_accuracy | -0.0114 | -0.0513 | 0.0285 | -0.0594 | 0.0337 |
| size | 500 - 200seq | [D - A](500) - [D - A](200seq) | recall | -0.0133 | -0.0486 | 0.0221 | -0.0571 | 0.0369 |
| size | 500 - 200seq | [D - A](500) - [D - A](200seq) | specificity | -0.0096 | -0.1237 | 0.1045 | -0.0903 | 0.0739 |
| size | 500 - 200seq | [D - A](500) - [D - A](200seq) | auroc | -0.0178 | -0.0519 | 0.0164 | -0.0644 | 0.0233 |
| size | 500 - 200seq | [C - A](500) - [C - A](200seq) | balanced_accuracy | -0.0934 | -0.1193 | -0.0676 | -0.1394 | -0.0442 |
| size | 500 - 200seq | [C - A](500) - [C - A](200seq) | recall | -0.2448 | -0.2773 | -0.2123 | -0.2925 | -0.1834 |
| size | 500 - 200seq | [C - A](500) - [C - A](200seq) | specificity | 0.0579 | -0.0097 | 0.1256 | -0.0193 | 0.1436 |
| size | 500 - 200seq | [C - A](500) - [C - A](200seq) | auroc | -0.0178 | -0.0519 | 0.0164 | -0.0644 | 0.0233 |

200iso and 500 recomputed: 168/168 quantities equal to e4_common_eval (max |diff| 4.97e-11).
