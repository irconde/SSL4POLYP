# Exploratory experiment: common learning rate 1e-3 vs campaign v3

Exploratory (not registered in the SAP): no Holm adjustment, no confirmation verdicts. Estimate = mean of the paired per-run differences (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, samples shared by both campaigns). diff = common − v3 per paired run. v3 recomputed on three regimes (H5 and H7 average three regimes, not four).

### Contrasts of H1, H3, H4, H5 and H7

| base_id | metric | comparison | common_estimate | common_t_ci_lo | common_t_ci_hi | v3_estimate | v3_t_ci_lo | v3_t_ci_hi | diff_estimate | diff_t_ci_lo | diff_t_ci_hi | diff_boot_lo | diff_boot_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | AULC(auprc) | ssl_colon - sup_imnet | 0.1047 | 0.0495 | 0.1598 | 0.0475 | 0.0286 | 0.0663 | 0.0572 | 0.0125 | 0.1019 | 0.0394 | 0.0743 |
| H3 | auroc | ssl_colon - sup_imnet | 0.0721 | 0.0072 | 0.1370 | 0.0380 | 0.0122 | 0.0639 | 0.0340 | -0.0516 | 0.1197 | -2.43e-04 | 0.0692 |
| H4 | auroc | ssl_colon - sup_imnet | 0.0792 | 0.0284 | 0.1299 | -0.0720 | -0.0985 | -0.0455 | 0.1512 | 0.1083 | 0.1941 | -0.0563 | 0.3969 |
| H5 | auroc | sequence - isolated | 0.1659 | 0.1201 | 0.2117 | 0.1603 | 0.1254 | 0.1952 | 0.0056 | -0.0389 | 0.0501 | -0.1177 | 0.1007 |
| H7 | balanced_accuracy | arm C - arm A | -0.0918 | -0.1171 | -0.0665 | -0.0601 | -0.0991 | -0.0212 | -0.0317 | -0.0712 | 0.0079 | -0.0617 | -6.43e-04 |

### Secondary contrasts (subset of metrics; all in contrasts_common_vs_v3.csv)

| base_id | metric | comparison | common_estimate | common_t_ci_lo | common_t_ci_hi | v3_estimate | v3_t_ci_lo | v3_t_ci_hi | diff_estimate | diff_t_ci_lo | diff_t_ci_hi | diff_boot_lo | diff_boot_hi | experiment | variant | slot | regime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmp/e1/5/all/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1862 | 0.0856 | 0.2868 | 0.0797 | 0.0115 | 0.1478 | 0.1065 | -0.0132 | 0.2263 | 0.0701 | 0.1419 | e1 | 5 | all |  |
| cmp/e1/5/all/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.2333 | 0.1395 | 0.3271 | 0.1197 | 0.0216 | 0.2178 | 0.1135 | 0.0106 | 0.2165 | 0.0697 | 0.1515 | e1 | 5 | all |  |
| cmp/e1/5/all/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1081 | 0.0326 | 0.1835 | 0.0399 | -0.0161 | 0.0958 | 0.0682 | -0.0176 | 0.1541 | 0.0415 | 0.0962 | e1 | 5 | all |  |
| cmp/e1/5/all/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1453 | 0.0945 | 0.1960 | 0.1417 | 0.0880 | 0.1954 | 0.0036 | -0.0757 | 0.0829 | -0.0470 | 0.0535 | e1 | 5 | all |  |
| cmp/e1/5/all/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1800 | 0.0939 | 0.2660 | 0.1660 | 0.1203 | 0.2118 | 0.0140 | -0.0849 | 0.1129 | -0.0393 | 0.0658 | e1 | 5 | all |  |
| cmp/e1/5/all/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1027 | 0.0558 | 0.1496 | 0.1164 | 0.0324 | 0.2003 | -0.0137 | -0.0722 | 0.0449 | -0.0393 | 0.0138 | e1 | 5 | all |  |
| cmp/e1/10/all/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1675 | 0.0679 | 0.2672 | 0.0954 | 0.0403 | 0.1504 | 0.0722 | 0.0255 | 0.1189 | 0.0325 | 0.1127 | e1 | 10 | all |  |
| cmp/e1/10/all/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.2114 | 0.1113 | 0.3115 | 0.1379 | 0.0635 | 0.2122 | 0.0735 | 0.0253 | 0.1217 | 0.0268 | 0.1175 | e1 | 10 | all |  |
| cmp/e1/10/all/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1184 | 0.0459 | 0.1908 | 0.0685 | 5.89e-04 | 0.1364 | 0.0499 | 0.0086 | 0.0912 | 0.0178 | 0.0822 | e1 | 10 | all |  |
| cmp/e1/10/all/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1702 | 0.0904 | 0.2499 | 0.0869 | 0.0359 | 0.1380 | 0.0832 | -0.0226 | 0.1891 | 0.0446 | 0.1226 | e1 | 10 | all |  |
| cmp/e1/10/all/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.2112 | 0.1201 | 0.3024 | 0.1002 | 0.0416 | 0.1589 | 0.1110 | -0.0078 | 0.2298 | 0.0650 | 0.1513 | e1 | 10 | all |  |
| cmp/e1/10/all/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1144 | 0.0592 | 0.1696 | 0.0750 | 0.0232 | 0.1269 | 0.0394 | -0.0504 | 0.1292 | 0.0098 | 0.0710 | e1 | 10 | all |  |
| cmp/e1/25/all/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0550 | -0.0261 | 0.1362 | 0.0376 | 0.0043 | 0.0709 | 0.0174 | -0.0467 | 0.0816 | -0.0096 | 0.0422 | e1 | 25 | all |  |
| cmp/e1/25/all/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0835 | -0.0160 | 0.1830 | 0.0453 | 0.0086 | 0.0820 | 0.0382 | -0.0533 | 0.1298 | 0.0057 | 0.0658 | e1 | 25 | all |  |
| cmp/e1/25/all/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0354 | -0.0357 | 0.1064 | 0.0406 | 0.0080 | 0.0733 | -0.0053 | -0.0679 | 0.0574 | -0.0303 | 0.0188 | e1 | 25 | all |  |
| cmp/e1/25/all/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0634 | -0.0154 | 0.1422 | -0.0070 | -0.0344 | 0.0204 | 0.0704 | 0.0098 | 0.1310 | 0.0339 | 0.1071 | e1 | 25 | all |  |
| cmp/e1/25/all/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0953 | 0.0022 | 0.1885 | -0.0095 | -0.0375 | 0.0186 | 0.1048 | 0.0289 | 0.1806 | 0.0649 | 0.1420 | e1 | 25 | all |  |
| cmp/e1/25/all/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0521 | 0.0056 | 0.0985 | -0.0168 | -0.0512 | 0.0176 | 0.0688 | 0.0246 | 0.1131 | 0.0329 | 0.1039 | e1 | 25 | all |  |
| cmp/e1/50/all/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0857 | 0.0213 | 0.1501 | 0.0447 | 0.0206 | 0.0689 | 0.0410 | -0.0187 | 0.1007 | 0.0180 | 0.0647 | e1 | 50 | all |  |
| cmp/e1/50/all/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1011 | 0.0307 | 0.1715 | 0.0384 | 0.0138 | 0.0630 | 0.0627 | -0.0042 | 0.1297 | 0.0363 | 0.0872 | e1 | 50 | all |  |
| cmp/e1/50/all/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1053 | 0.0215 | 0.1891 | 0.0706 | 0.0225 | 0.1186 | 0.0347 | -0.0330 | 0.1025 | 0.0091 | 0.0616 | e1 | 50 | all |  |
| cmp/e1/50/all/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1225 | 0.0298 | 0.2152 | -0.0046 | -0.0249 | 0.0156 | 0.1272 | 0.0290 | 0.2253 | 0.0956 | 0.1607 | e1 | 50 | all |  |
| cmp/e1/50/all/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1511 | 0.0512 | 0.2510 | -0.0052 | -0.0234 | 0.0131 | 0.1563 | 0.0531 | 0.2594 | 0.1205 | 0.1892 | e1 | 50 | all |  |
| cmp/e1/50/all/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1354 | 0.0404 | 0.2303 | -0.0146 | -0.0586 | 0.0294 | 0.1500 | 0.0562 | 0.2437 | 0.1143 | 0.1862 | e1 | 50 | all |  |
| cmp/e1/100/all/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0677 | 0.0230 | 0.1123 | 0.0220 | 0.0074 | 0.0365 | 0.0457 | -0.0076 | 0.0989 | 0.0250 | 0.0696 | e1 | 100 | all |  |
| cmp/e1/100/all/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0713 | 0.0170 | 0.1257 | 0.0195 | 0.0057 | 0.0332 | 0.0519 | -0.0092 | 0.1129 | 0.0288 | 0.0786 | e1 | 100 | all |  |
| cmp/e1/100/all/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1040 | 0.0474 | 0.1605 | 0.0509 | 0.0044 | 0.0974 | 0.0531 | -0.0286 | 0.1347 | 0.0221 | 0.0849 | e1 | 100 | all |  |
| cmp/e1/100/all/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0698 | 0.0187 | 0.1209 | -0.0025 | -0.0091 | 0.0041 | 0.0723 | 0.0195 | 0.1250 | 0.0530 | 0.0928 | e1 | 100 | all |  |
| cmp/e1/100/all/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0815 | 0.0182 | 0.1447 | -0.0020 | -0.0073 | 0.0033 | 0.0835 | 0.0195 | 0.1475 | 0.0581 | 0.1081 | e1 | 100 | all |  |
| cmp/e1/100/all/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1018 | 0.0329 | 0.1707 | 6.73e-05 | -0.0252 | 0.0253 | 0.1018 | 0.0257 | 0.1778 | 0.0743 | 0.1292 | e1 | 100 | all |  |
| cmp/e2/None/overall/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0677 | 0.0230 | 0.1123 | 0.0220 | 0.0074 | 0.0365 | 0.0457 | -0.0076 | 0.0989 | 0.0255 | 0.0701 | e2 |  | overall |  |
| cmp/e2/None/overall/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0713 | 0.0170 | 0.1257 | 0.0195 | 0.0057 | 0.0332 | 0.0519 | -0.0092 | 0.1129 | 0.0290 | 0.0781 | e2 |  | overall |  |
| cmp/e2/None/overall/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1040 | 0.0474 | 0.1605 | 0.0509 | 0.0044 | 0.0974 | 0.0531 | -0.0286 | 0.1347 | 0.0213 | 0.0841 | e2 |  | overall |  |
| cmp/e2/None/overall/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0698 | 0.0187 | 0.1209 | -0.0025 | -0.0091 | 0.0041 | 0.0723 | 0.0195 | 0.1250 | 0.0535 | 0.0940 | e2 |  | overall |  |
| cmp/e2/None/overall/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0815 | 0.0182 | 0.1447 | -0.0020 | -0.0073 | 0.0033 | 0.0835 | 0.0195 | 0.1475 | 0.0587 | 0.1088 | e2 |  | overall |  |
| cmp/e2/None/overall/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1018 | 0.0329 | 0.1707 | 6.73e-05 | -0.0252 | 0.0253 | 0.1018 | 0.0257 | 0.1778 | 0.0754 | 0.1268 | e2 |  | overall |  |
| cmp/e2/None/flat/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0721 | 0.0072 | 0.1370 | 0.0380 | 0.0122 | 0.0639 | 0.0340 | -0.0516 | 0.1197 | -2.43e-04 | 0.0692 | e2 |  | flat |  |
| cmp/e2/None/flat/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1631 | 0.0013 | 0.3249 | 0.0781 | 0.0186 | 0.1375 | 0.0850 | -0.1187 | 0.2888 | 0.0067 | 0.1483 | e2 |  | flat |  |
| cmp/e2/None/flat/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0974 | 0.0297 | 0.1651 | 0.0729 | 0.0109 | 0.1349 | 0.0245 | -0.0828 | 0.1318 | -0.0148 | 0.0636 | e2 |  | flat |  |
| cmp/e2/None/flat/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0655 | 0.0111 | 0.1200 | -0.0068 | -0.0255 | 0.0119 | 0.0723 | 0.0207 | 0.1240 | 0.0469 | 0.1008 | e2 |  | flat |  |
| cmp/e2/None/flat/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1596 | 0.0288 | 0.2905 | -0.0121 | -0.0441 | 0.0198 | 0.1718 | 0.0421 | 0.3014 | 0.1068 | 0.2228 | e2 |  | flat |  |
| cmp/e2/None/flat/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0799 | 0.0140 | 0.1457 | -0.0034 | -0.0420 | 0.0352 | 0.0832 | 0.0154 | 0.1510 | 0.0525 | 0.1125 | e2 |  | flat |  |
| cmp/e2/None/polypoid/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0655 | 0.0286 | 0.1023 | 0.0140 | 0.0014 | 0.0265 | 0.0515 | 0.0100 | 0.0930 | 0.0324 | 0.0748 | e2 |  | polypoid |  |
| cmp/e2/None/polypoid/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0898 | 0.0289 | 0.1506 | 0.0172 | 9.23e-04 | 0.0335 | 0.0725 | 0.0073 | 0.1378 | 0.0443 | 0.1052 | e2 |  | polypoid |  |
| cmp/e2/None/polypoid/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1073 | 0.0512 | 0.1633 | 0.0399 | -0.0023 | 0.0822 | 0.0673 | -0.0079 | 0.1426 | 0.0351 | 0.1000 | e2 |  | polypoid |  |
| cmp/e2/None/polypoid/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0719 | 0.0185 | 0.1253 | -3.55e-04 | -0.0017 | 9.98e-04 | 0.0722 | 0.0176 | 0.1268 | 0.0532 | 0.0942 | e2 |  | polypoid |  |
| cmp/e2/None/polypoid/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1080 | 0.0233 | 0.1927 | -4.18e-04 | -0.0022 | 0.0014 | 0.1085 | 0.0223 | 0.1946 | 0.0769 | 0.1386 | e2 |  | polypoid |  |
| cmp/e2/None/polypoid/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1128 | 0.0315 | 0.1941 | 0.0018 | -0.0172 | 0.0208 | 0.1110 | 0.0244 | 0.1976 | 0.0844 | 0.1369 | e2 |  | polypoid |  |
| cmp/e3/None/all/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0792 | 0.0284 | 0.1299 | -0.0720 | -0.0985 | -0.0455 | 0.1512 | 0.1083 | 0.1941 | -0.0563 | 0.3969 | e3 |  | all |  |
| cmp/e3/None/all/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0240 | 0.0112 | 0.0368 | -0.0170 | -0.0258 | -0.0083 | 0.0411 | 0.0350 | 0.0472 | -0.0234 | 0.1210 | e3 |  | all |  |
| cmp/e3/None/all/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0419 | 0.0032 | 0.0806 | -0.0314 | -0.0454 | -0.0174 | 0.0733 | 0.0232 | 0.1234 | -0.0110 | 0.1457 | e3 |  | all |  |
| cmp/e3/None/all/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0728 | -0.0013 | 0.1469 | -0.0786 | -0.0989 | -0.0583 | 0.1514 | 0.0905 | 0.2124 | 0.0213 | 0.2872 | e3 |  | all |  |
| cmp/e3/None/all/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0220 | 0.0026 | 0.0414 | -0.0223 | -0.0287 | -0.0158 | 0.0443 | 0.0287 | 0.0598 | 0.0047 | 0.1013 | e3 |  | all |  |
| cmp/e3/None/all/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0514 | 0.0045 | 0.0983 | -0.0412 | -0.0650 | -0.0175 | 0.0927 | 0.0366 | 0.1488 | 0.0144 | 0.1717 | e3 |  | all |  |
| cmp/e4/50/A/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0278 | -0.0068 | 0.0625 | -0.1056 | -0.1308 | -0.0805 | 0.1335 | 0.0908 | 0.1761 | 0.0739 | 0.2090 | e4 | 50 | A |  |
| cmp/e4/50/A/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0135 | -0.0118 | 0.0387 | -0.2101 | -0.2540 | -0.1662 | 0.2236 | 0.1664 | 0.2807 | 0.0655 | 0.3243 | e4 | 50 | A |  |
| cmp/e4/50/A/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0438 | 0.0072 | 0.0804 | -0.0704 | -0.0836 | -0.0572 | 0.1142 | 0.0711 | 0.1573 | 0.0752 | 0.1624 | e4 | 50 | A |  |
| cmp/e4/50/A/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0319 | -0.0385 | 0.1023 | -0.0516 | -0.0869 | -0.0164 | 0.0835 | 0.0136 | 0.1534 | 0.0302 | 0.1449 | e4 | 50 | A |  |
| cmp/e4/50/A/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 6.49e-04 | -0.0842 | 0.0855 | -0.1459 | -0.2334 | -0.0583 | 0.1465 | 0.0353 | 0.2577 | 0.0310 | 0.2208 | e4 | 50 | A |  |
| cmp/e4/50/A/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0440 | -0.0192 | 0.1072 | -0.0011 | -0.0382 | 0.0360 | 0.0451 | -0.0212 | 0.1114 | 0.0022 | 0.0948 | e4 | 50 | A |  |
| cmp/e4/50/B/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0278 | -0.0068 | 0.0625 | -0.1056 | -0.1308 | -0.0805 | 0.1335 | 0.0908 | 0.1761 | 0.0739 | 0.2090 | e4 | 50 | B |  |
| cmp/e4/50/B/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0135 | -0.0118 | 0.0387 | -0.2101 | -0.2540 | -0.1662 | 0.2236 | 0.1664 | 0.2807 | 0.0655 | 0.3243 | e4 | 50 | B |  |
| cmp/e4/50/B/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0150 | -0.0107 | 0.0408 | -0.0666 | -0.1137 | -0.0195 | 0.0817 | 0.0096 | 0.1537 | 0.0423 | 0.1239 | e4 | 50 | B |  |
| cmp/e4/50/B/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0319 | -0.0385 | 0.1023 | -0.0516 | -0.0869 | -0.0164 | 0.0835 | 0.0136 | 0.1534 | 0.0302 | 0.1449 | e4 | 50 | B |  |
| cmp/e4/50/B/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 6.49e-04 | -0.0842 | 0.0855 | -0.1459 | -0.2334 | -0.0583 | 0.1465 | 0.0353 | 0.2577 | 0.0310 | 0.2208 | e4 | 50 | B |  |
| cmp/e4/50/B/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0347 | -0.0550 | 0.1244 | -0.0320 | -0.0854 | 0.0213 | 0.0668 | -0.0446 | 0.1781 | 0.0239 | 0.1095 | e4 | 50 | B |  |
| cmp/e4/50/C/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1032 | -0.0145 | 0.2209 | -0.1013 | -0.1651 | -0.0375 | 0.2045 | 0.0489 | 0.3601 | 0.1554 | 0.2516 | e4 | 50 | C |  |
| cmp/e4/50/C/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1010 | 0.0070 | 0.1951 | -0.1658 | -0.2630 | -0.0686 | 0.2668 | 0.1261 | 0.4075 | 0.1668 | 0.3528 | e4 | 50 | C |  |
| cmp/e4/50/C/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0092 | -0.0410 | 0.0593 | -0.0363 | -0.1183 | 0.0457 | 0.0455 | -0.0608 | 0.1518 | 0.0072 | 0.0855 | e4 | 50 | C |  |
| cmp/e4/50/C/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1195 | -0.0642 | 0.3032 | -0.0037 | -0.0404 | 0.0331 | 0.1231 | -0.0506 | 0.2969 | 0.0717 | 0.1762 | e4 | 50 | C |  |
| cmp/e4/50/C/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1002 | -0.0671 | 0.2675 | -0.0465 | -0.1659 | 0.0728 | 0.1467 | -0.0677 | 0.3612 | 0.0677 | 0.2320 | e4 | 50 | C |  |
| cmp/e4/50/C/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0433 | -0.0658 | 0.1525 | 0.0039 | -0.0705 | 0.0784 | 0.0394 | -0.0582 | 0.1370 | 0.0069 | 0.0733 | e4 | 50 | C |  |
| cmp/e4/50/D/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1032 | -0.0145 | 0.2209 | -0.1013 | -0.1651 | -0.0375 | 0.2045 | 0.0489 | 0.3601 | 0.1554 | 0.2516 | e4 | 50 | D |  |
| cmp/e4/50/D/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1010 | 0.0070 | 0.1951 | -0.1658 | -0.2630 | -0.0686 | 0.2668 | 0.1261 | 0.4075 | 0.1668 | 0.3528 | e4 | 50 | D |  |
| cmp/e4/50/D/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0347 | -0.0552 | 0.1245 | -0.0727 | -0.1290 | -0.0165 | 0.1074 | -0.0372 | 0.2519 | 0.0709 | 0.1459 | e4 | 50 | D |  |
| cmp/e4/50/D/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1195 | -0.0642 | 0.3032 | -0.0037 | -0.0404 | 0.0331 | 0.1231 | -0.0506 | 0.2969 | 0.0717 | 0.1762 | e4 | 50 | D |  |
| cmp/e4/50/D/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1002 | -0.0671 | 0.2675 | -0.0465 | -0.1659 | 0.0728 | 0.1467 | -0.0677 | 0.3612 | 0.0677 | 0.2320 | e4 | 50 | D |  |
| cmp/e4/50/D/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0522 | -0.0132 | 0.1175 | -0.0132 | -0.0394 | 0.0131 | 0.0653 | -0.0121 | 0.1427 | 0.0313 | 0.1038 | e4 | 50 | D |  |
| cmp/e4/100/A/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0275 | -0.0074 | 0.0623 | -0.1058 | -0.1312 | -0.0804 | 0.1333 | 0.0900 | 0.1766 | 0.0732 | 0.2088 | e4 | 100 | A |  |
| cmp/e4/100/A/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0123 | -0.0123 | 0.0369 | -0.2128 | -0.2579 | -0.1676 | 0.2251 | 0.1668 | 0.2833 | 0.0664 | 0.3256 | e4 | 100 | A |  |
| cmp/e4/100/A/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0438 | 0.0077 | 0.0800 | -0.0708 | -0.0842 | -0.0574 | 0.1146 | 0.0720 | 0.1573 | 0.0755 | 0.1635 | e4 | 100 | A |  |
| cmp/e4/100/A/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0316 | -0.0388 | 0.1020 | -0.0515 | -0.0869 | -0.0162 | 0.0832 | 0.0133 | 0.1531 | 0.0299 | 0.1448 | e4 | 100 | A |  |
| cmp/e4/100/A/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | -3.59e-04 | -0.0863 | 0.0856 | -0.1470 | -0.2361 | -0.0580 | 0.1467 | 0.0348 | 0.2586 | 0.0284 | 0.2212 | e4 | 100 | A |  |
| cmp/e4/100/A/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0439 | -0.0194 | 0.1073 | -0.0010 | -0.0389 | 0.0368 | 0.0449 | -0.0220 | 0.1119 | 0.0021 | 0.0950 | e4 | 100 | A |  |
| cmp/e4/100/B/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0275 | -0.0074 | 0.0623 | -0.1058 | -0.1312 | -0.0804 | 0.1333 | 0.0900 | 0.1766 | 0.0732 | 0.2088 | e4 | 100 | B |  |
| cmp/e4/100/B/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0123 | -0.0123 | 0.0369 | -0.2128 | -0.2579 | -0.1676 | 0.2251 | 0.1668 | 0.2833 | 0.0664 | 0.3256 | e4 | 100 | B |  |
| cmp/e4/100/B/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0540 | -0.0016 | 0.1096 | -0.0960 | -0.1629 | -0.0290 | 0.1499 | 0.0465 | 0.2533 | 0.1036 | 0.2023 | e4 | 100 | B |  |
| cmp/e4/100/B/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0316 | -0.0388 | 0.1020 | -0.0515 | -0.0869 | -0.0162 | 0.0832 | 0.0133 | 0.1531 | 0.0299 | 0.1448 | e4 | 100 | B |  |
| cmp/e4/100/B/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | -3.59e-04 | -0.0863 | 0.0856 | -0.1470 | -0.2361 | -0.0580 | 0.1467 | 0.0348 | 0.2586 | 0.0284 | 0.2212 | e4 | 100 | B |  |
| cmp/e4/100/B/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0573 | -0.0403 | 0.1548 | -0.0257 | -0.0970 | 0.0455 | 0.0830 | -0.0301 | 0.1960 | 0.0436 | 0.1294 | e4 | 100 | B |  |
| cmp/e4/100/C/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1155 | -0.0012 | 0.2322 | -0.1562 | -0.2819 | -0.0305 | 0.2717 | 0.0724 | 0.4710 | 0.1897 | 0.3422 | e4 | 100 | C |  |
| cmp/e4/100/C/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1685 | 0.0686 | 0.2684 | -0.2108 | -0.3530 | -0.0687 | 0.3793 | 0.1599 | 0.5988 | 0.2617 | 0.4635 | e4 | 100 | C |  |
| cmp/e4/100/C/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0190 | -0.0111 | 0.0491 | -0.0177 | -0.1529 | 0.1175 | 0.0367 | -0.1100 | 0.1833 | -0.0080 | 0.0792 | e4 | 100 | C |  |
| cmp/e4/100/C/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1533 | -0.0446 | 0.3511 | -0.1122 | -0.2029 | -0.0215 | 0.2655 | 0.0636 | 0.4673 | 0.1720 | 0.3456 | e4 | 100 | C |  |
| cmp/e4/100/C/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1806 | -0.0019 | 0.3631 | -0.1655 | -0.2776 | -0.0535 | 0.3461 | 0.1486 | 0.5436 | 0.2452 | 0.4264 | e4 | 100 | C |  |
| cmp/e4/100/C/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0564 | 0.0037 | 0.1091 | -0.0561 | -0.1413 | 0.0292 | 0.1125 | 0.0257 | 0.1992 | 0.0695 | 0.1546 | e4 | 100 | C |  |
| cmp/e4/100/D/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1155 | -0.0012 | 0.2322 | -0.1562 | -0.2819 | -0.0305 | 0.2717 | 0.0724 | 0.4710 | 0.1897 | 0.3422 | e4 | 100 | D |  |
| cmp/e4/100/D/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1685 | 0.0686 | 0.2684 | -0.2108 | -0.3530 | -0.0687 | 0.3793 | 0.1599 | 0.5988 | 0.2617 | 0.4635 | e4 | 100 | D |  |
| cmp/e4/100/D/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0637 | 0.0115 | 0.1158 | -0.1199 | -0.1944 | -0.0454 | 0.1836 | 0.0648 | 0.3023 | 0.1430 | 0.2200 | e4 | 100 | D |  |
| cmp/e4/100/D/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1533 | -0.0446 | 0.3511 | -0.1122 | -0.2029 | -0.0215 | 0.2655 | 0.0636 | 0.4673 | 0.1720 | 0.3456 | e4 | 100 | D |  |
| cmp/e4/100/D/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1806 | -0.0019 | 0.3631 | -0.1655 | -0.2776 | -0.0535 | 0.3461 | 0.1486 | 0.5436 | 0.2452 | 0.4264 | e4 | 100 | D |  |
| cmp/e4/100/D/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0868 | -0.0168 | 0.1904 | -0.0683 | -0.1136 | -0.0231 | 0.1552 | 0.0660 | 0.2443 | 0.1064 | 0.1990 | e4 | 100 | D |  |
| cmp/e4/200/A/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0265 | -0.0091 | 0.0622 | -0.1068 | -0.1328 | -0.0808 | 0.1333 | 0.0888 | 0.1779 | 0.0742 | 0.2092 | e4 | 200 | A |  |
| cmp/e4/200/A/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0108 | -0.0134 | 0.0351 | -0.2177 | -0.2647 | -0.1706 | 0.2285 | 0.1676 | 0.2894 | 0.0696 | 0.3294 | e4 | 200 | A |  |
| cmp/e4/200/A/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0441 | 0.0068 | 0.0813 | -0.0718 | -0.0856 | -0.0580 | 0.1159 | 0.0725 | 0.1593 | 0.0762 | 0.1657 | e4 | 200 | A |  |
| cmp/e4/200/A/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0313 | -0.0401 | 0.1027 | -0.0510 | -0.0873 | -0.0148 | 0.0823 | 0.0106 | 0.1541 | 0.0277 | 0.1447 | e4 | 200 | A |  |
| cmp/e4/200/A/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | -0.0014 | -0.0887 | 0.0859 | -0.1486 | -0.2399 | -0.0573 | 0.1472 | 0.0328 | 0.2617 | 0.0256 | 0.2227 | e4 | 200 | A |  |
| cmp/e4/200/A/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0440 | -0.0207 | 0.1088 | -3.11e-04 | -0.0394 | 0.0388 | 0.0443 | -0.0243 | 0.1129 | 0.0013 | 0.0959 | e4 | 200 | A |  |
| cmp/e4/200/B/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0265 | -0.0091 | 0.0622 | -0.1068 | -0.1328 | -0.0808 | 0.1333 | 0.0888 | 0.1779 | 0.0742 | 0.2092 | e4 | 200 | B |  |
| cmp/e4/200/B/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0108 | -0.0134 | 0.0351 | -0.2177 | -0.2647 | -0.1706 | 0.2285 | 0.1676 | 0.2894 | 0.0696 | 0.3294 | e4 | 200 | B |  |
| cmp/e4/200/B/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0571 | 0.0069 | 0.1073 | -0.0867 | -0.1456 | -0.0278 | 0.1438 | 0.0583 | 0.2292 | 0.0911 | 0.2037 | e4 | 200 | B |  |
| cmp/e4/200/B/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0313 | -0.0401 | 0.1027 | -0.0510 | -0.0873 | -0.0148 | 0.0823 | 0.0106 | 0.1541 | 0.0277 | 0.1447 | e4 | 200 | B |  |
| cmp/e4/200/B/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | -0.0014 | -0.0887 | 0.0859 | -0.1486 | -0.2399 | -0.0573 | 0.1472 | 0.0328 | 0.2617 | 0.0256 | 0.2227 | e4 | 200 | B |  |
| cmp/e4/200/B/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0592 | -0.0475 | 0.1659 | -0.0316 | -0.0844 | 0.0212 | 0.0908 | -0.0215 | 0.2030 | 0.0482 | 0.1447 | e4 | 200 | B |  |
| cmp/e4/200/C/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1155 | -3.34e-04 | 0.2314 | -0.1529 | -0.2670 | -0.0388 | 0.2684 | 0.0673 | 0.4695 | 0.1749 | 0.3574 | e4 | 200 | C |  |
| cmp/e4/200/C/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1371 | 0.0106 | 0.2636 | -0.2384 | -0.3316 | -0.1452 | 0.3755 | 0.1620 | 0.5889 | 0.2797 | 0.4423 | e4 | 200 | C |  |
| cmp/e4/200/C/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0469 | 0.0206 | 0.0732 | -0.0339 | -0.1003 | 0.0324 | 0.0809 | 0.0174 | 0.1443 | 0.0531 | 0.1141 | e4 | 200 | C |  |
| cmp/e4/200/C/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1187 | -0.0381 | 0.2755 | -0.0764 | -0.1805 | 0.0277 | 0.1951 | 0.0465 | 0.3437 | 0.1087 | 0.2825 | e4 | 200 | C |  |
| cmp/e4/200/C/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1155 | 0.0198 | 0.2112 | -0.1542 | -0.2533 | -0.0552 | 0.2697 | 0.1257 | 0.4138 | 0.1893 | 0.3244 | e4 | 200 | C |  |
| cmp/e4/200/C/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0691 | 0.0128 | 0.1254 | -0.0642 | -0.1435 | 0.0151 | 0.1333 | 0.0674 | 0.1992 | 0.0932 | 0.1775 | e4 | 200 | C |  |
| cmp/e4/200/D/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1155 | -3.34e-04 | 0.2314 | -0.1529 | -0.2670 | -0.0388 | 0.2684 | 0.0673 | 0.4695 | 0.1749 | 0.3574 | e4 | 200 | D |  |
| cmp/e4/200/D/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1371 | 0.0106 | 0.2636 | -0.2384 | -0.3316 | -0.1452 | 0.3755 | 0.1620 | 0.5889 | 0.2797 | 0.4423 | e4 | 200 | D |  |
| cmp/e4/200/D/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0554 | 0.0037 | 0.1071 | -0.0779 | -0.1544 | -0.0015 | 0.1334 | 0.0419 | 0.2249 | 0.0852 | 0.1840 | e4 | 200 | D |  |
| cmp/e4/200/D/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1187 | -0.0381 | 0.2755 | -0.0764 | -0.1805 | 0.0277 | 0.1951 | 0.0465 | 0.3437 | 0.1087 | 0.2825 | e4 | 200 | D |  |
| cmp/e4/200/D/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.1155 | 0.0198 | 0.2112 | -0.1542 | -0.2533 | -0.0552 | 0.2697 | 0.1257 | 0.4138 | 0.1893 | 0.3244 | e4 | 200 | D |  |
| cmp/e4/200/D/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0475 | -0.0427 | 0.1377 | -0.0282 | -0.0671 | 0.0107 | 0.0757 | -0.0121 | 0.1635 | 0.0144 | 0.1347 | e4 | 200 | D |  |
| cmp/e4/500/A/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0256 | -0.0312 | 0.0824 | -0.1219 | -0.1858 | -0.0579 | 0.1475 | 0.0838 | 0.2111 | 0.0727 | 0.2092 | e4 | 500 | A |  |
| cmp/e4/500/A/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0151 | -0.0244 | 0.0546 | -0.1170 | -0.1934 | -0.0407 | 0.1321 | 0.0544 | 0.2098 | 0.0351 | 0.1830 | e4 | 500 | A |  |
| cmp/e4/500/A/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0499 | -0.0044 | 0.1041 | -0.0693 | -0.0847 | -0.0539 | 0.1191 | 0.0718 | 0.1665 | 0.0729 | 0.1723 | e4 | 500 | A |  |
| cmp/e4/500/A/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0180 | -0.1032 | 0.1392 | -0.0633 | -0.1484 | 0.0218 | 0.0813 | -0.0435 | 0.2061 | 0.0234 | 0.1373 | e4 | 500 | A |  |
| cmp/e4/500/A/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0071 | -0.0866 | 0.1008 | -0.0698 | -0.1581 | 0.0185 | 0.0769 | -0.0232 | 0.1771 | 0.0138 | 0.1194 | e4 | 500 | A |  |
| cmp/e4/500/A/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0291 | -0.0846 | 0.1429 | 0.0020 | -0.0505 | 0.0545 | 0.0271 | -0.1058 | 0.1600 | -0.0084 | 0.0605 | e4 | 500 | A |  |
| cmp/e4/500/B/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.0256 | -0.0312 | 0.0824 | -0.1219 | -0.1858 | -0.0579 | 0.1475 | 0.0838 | 0.2111 | 0.0727 | 0.2092 | e4 | 500 | B |  |
| cmp/e4/500/B/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.0151 | -0.0244 | 0.0546 | -0.1170 | -0.1934 | -0.0407 | 0.1321 | 0.0544 | 0.2098 | 0.0351 | 0.1830 | e4 | 500 | B |  |
| cmp/e4/500/B/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.0588 | -0.0394 | 0.1571 | -0.0931 | -0.1330 | -0.0532 | 0.1519 | 0.0740 | 0.2298 | 0.0855 | 0.2301 | e4 | 500 | B |  |
| cmp/e4/500/B/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.0180 | -0.1032 | 0.1392 | -0.0633 | -0.1484 | 0.0218 | 0.0813 | -0.0435 | 0.2061 | 0.0234 | 0.1373 | e4 | 500 | B |  |
| cmp/e4/500/B/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0071 | -0.0866 | 0.1008 | -0.0698 | -0.1581 | 0.0185 | 0.0769 | -0.0232 | 0.1771 | 0.0138 | 0.1194 | e4 | 500 | B |  |
| cmp/e4/500/B/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0441 | -0.0921 | 0.1803 | -0.0281 | -0.1029 | 0.0467 | 0.0722 | -0.0906 | 0.2349 | 0.0265 | 0.1180 | e4 | 500 | B |  |
| cmp/e4/500/C/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1639 | 0.0723 | 0.2556 | -0.0235 | -0.0628 | 0.0159 | 0.1874 | 0.1258 | 0.2491 | 0.1313 | 0.2419 | e4 | 500 | C |  |
| cmp/e4/500/C/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1238 | 0.0575 | 0.1902 | -0.0241 | -0.0621 | 0.0139 | 0.1480 | 0.0997 | 0.1962 | 0.0791 | 0.1989 | e4 | 500 | C |  |
| cmp/e4/500/C/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1406 | 0.0308 | 0.2505 | 0.0240 | -0.1017 | 0.1497 | 0.1166 | 0.0054 | 0.2278 | 0.0730 | 0.1554 | e4 | 500 | C |  |
| cmp/e4/500/C/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1209 | 0.0265 | 0.2153 | -0.0169 | -0.0385 | 0.0046 | 0.1378 | 0.0321 | 0.2435 | 0.0652 | 0.2048 | e4 | 500 | C |  |
| cmp/e4/500/C/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0968 | 0.0253 | 0.1683 | -0.0025 | -0.0353 | 0.0303 | 0.0994 | 0.0266 | 0.1721 | 0.0466 | 0.1483 | e4 | 500 | C |  |
| cmp/e4/500/C/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.0730 | -0.0887 | 0.2347 | -0.0600 | -0.1518 | 0.0319 | 0.1330 | -0.0222 | 0.2882 | 0.0675 | 0.1874 | e4 | 500 | C |  |
| cmp/e4/500/D/ssl_colon-sup_imnet/auroc | auroc | ssl_colon - sup_imnet | 0.1639 | 0.0723 | 0.2556 | -0.0235 | -0.0628 | 0.0159 | 0.1874 | 0.1258 | 0.2491 | 0.1313 | 0.2419 | e4 | 500 | D |  |
| cmp/e4/500/D/ssl_colon-sup_imnet/auprc | auprc | ssl_colon - sup_imnet | 0.1238 | 0.0575 | 0.1902 | -0.0241 | -0.0621 | 0.0139 | 0.1480 | 0.0997 | 0.1962 | 0.0791 | 0.1989 | e4 | 500 | D |  |
| cmp/e4/500/D/ssl_colon-sup_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - sup_imnet | 0.1549 | 0.0747 | 0.2351 | -0.0382 | -0.0783 | 0.0019 | 0.1930 | 0.1282 | 0.2579 | 0.1539 | 0.2321 | e4 | 500 | D |  |
| cmp/e4/500/D/ssl_colon-ssl_imnet/auroc | auroc | ssl_colon - ssl_imnet | 0.1209 | 0.0265 | 0.2153 | -0.0169 | -0.0385 | 0.0046 | 0.1378 | 0.0321 | 0.2435 | 0.0652 | 0.2048 | e4 | 500 | D |  |
| cmp/e4/500/D/ssl_colon-ssl_imnet/auprc | auprc | ssl_colon - ssl_imnet | 0.0968 | 0.0253 | 0.1683 | -0.0025 | -0.0353 | 0.0303 | 0.0994 | 0.0266 | 0.1721 | 0.0466 | 0.1483 | e4 | 500 | D |  |
| cmp/e4/500/D/ssl_colon-ssl_imnet/balanced_accuracy | balanced_accuracy | ssl_colon - ssl_imnet | 0.1040 | 0.0496 | 0.1583 | -0.0464 | -0.0939 | 0.0010 | 0.1504 | 0.0787 | 0.2221 | 0.0944 | 0.2078 | e4 | 500 | D |  |
| aulc/ssl_colon-sup_imnet/auroc | AULC(auroc) | ssl_colon - sup_imnet | 0.0858 | 0.0382 | 0.1334 | 0.0435 | 0.0277 | 0.0593 | 0.0423 | 0.0019 | 0.0827 | 0.0256 | 0.0591 | e1 |  |  |  |
| aulc/ssl_colon-ssl_imnet/auroc | AULC(auroc) | ssl_colon - ssl_imnet | 0.1018 | 0.0490 | 0.1546 | 0.0089 | -0.0031 | 0.0209 | 0.0929 | 0.0397 | 0.1461 | 0.0705 | 0.1167 | e1 |  |  |  |
| aulc_level/sup_imnet/auroc | AULC(auroc) |  | 0.8363 | 0.7890 | 0.8836 | 0.9095 | 0.8885 | 0.9305 | -0.0732 | -0.1208 | -0.0256 | -0.0916 | -0.0566 | e1 |  |  | sup_imnet |
| aulc_level/ssl_imnet/auroc | AULC(auroc) |  | 0.8203 | 0.7586 | 0.8820 | 0.9441 | 0.9187 | 0.9695 | -0.1238 | -0.1891 | -0.0585 | -0.1481 | -0.1008 | e1 |  |  | ssl_imnet |
| aulc_level/ssl_colon/auroc | AULC(auroc) |  | 0.9221 | 0.9044 | 0.9398 | 0.9530 | 0.9370 | 0.9690 | -0.0309 | -0.0436 | -0.0183 | -0.0411 | -0.0214 | e1 |  |  | ssl_colon |
| aulc/ssl_colon-sup_imnet/auprc | AULC(auprc) | ssl_colon - sup_imnet | 0.1047 | 0.0495 | 0.1598 | 0.0475 | 0.0286 | 0.0663 | 0.0572 | 0.0125 | 0.1019 | 0.0394 | 0.0743 | e1 |  |  |  |
| aulc/ssl_colon-ssl_imnet/auprc | AULC(auprc) | ssl_colon - ssl_imnet | 0.1281 | 0.0672 | 0.1891 | 0.0104 | -0.0038 | 0.0246 | 0.1178 | 0.0580 | 0.1776 | 0.0920 | 0.1403 | e1 |  |  |  |
| aulc_level/sup_imnet/auprc | AULC(auprc) |  | 0.8194 | 0.7583 | 0.8805 | 0.9093 | 0.8913 | 0.9274 | -0.0899 | -0.1468 | -0.0330 | -0.1086 | -0.0707 | e1 |  |  | sup_imnet |
| aulc_level/ssl_imnet/auprc | AULC(auprc) |  | 0.7960 | 0.7241 | 0.8678 | 0.9464 | 0.9244 | 0.9685 | -0.1505 | -0.2252 | -0.0758 | -0.1753 | -0.1224 | e1 |  |  | ssl_imnet |
| aulc_level/ssl_colon/auprc | AULC(auprc) |  | 0.9241 | 0.9097 | 0.9384 | 0.9568 | 0.9435 | 0.9701 | -0.0327 | -0.0483 | -0.0172 | -0.0426 | -0.0231 | e1 |  |  | ssl_colon |
| arm/50/sup_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0148 | -0.0299 | 3.08e-04 | -0.0110 | -0.0472 | 0.0251 | -0.0037 | -0.0457 | 0.0382 | -0.0365 | 0.0281 | e4 | 50 |  | sup_imnet |
| arm/50/ssl_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0343 | -0.0604 | -0.0082 | 0.0237 | 0.0052 | 0.0421 | -0.0579 | -0.0952 | -0.0207 | -0.0983 | -0.0174 | e4 | 50 |  | ssl_imnet |
| arm/50/ssl_colon/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0436 | -0.0659 | -0.0212 | -0.0073 | -0.0333 | 0.0187 | -0.0363 | -0.0639 | -0.0086 | -0.0703 | -0.0018 | e4 | 50 |  | ssl_colon |
| arm/50/pooled/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0309 | -0.0352 | -0.0266 | 0.0018 | -0.0200 | 0.0235 | -0.0327 | -0.0539 | -0.0115 | -0.0650 | 4.87e-05 | e4 | 50 |  | mean of regimes |
| arm/50/sup_imnet/C-A/auroc | auroc | arm C - arm A | -0.0982 | -0.2145 | 0.0180 | -0.0050 | -0.0265 | 0.0164 | -0.0932 | -0.2244 | 0.0381 | -0.1756 | 0.0034 | e4 | 50 |  | sup_imnet |
| arm/50/ssl_imnet/C-A/auroc | auroc | arm C - arm A | -0.1105 | -0.2544 | 0.0334 | -0.0487 | -0.0833 | -0.0140 | -0.0618 | -0.2242 | 0.1006 | -0.1323 | 0.0226 | e4 | 50 |  | ssl_imnet |
| arm/50/ssl_colon/C-A/auroc | auroc | arm C - arm A | -0.0229 | -0.1020 | 0.0562 | -7.01e-04 | -0.0484 | 0.0470 | -0.0222 | -0.1257 | 0.0814 | -0.0496 | 0.0076 | e4 | 50 |  | ssl_colon |
| arm/50/pooled/C-A/auroc | auroc | arm C - arm A | -0.0772 | -0.1432 | -0.0112 | -0.0181 | -0.0458 | 0.0095 | -0.0590 | -0.1380 | 0.0199 | -0.1164 | 0.0079 | e4 | 50 |  | mean of regimes |
| arm/50/sup_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.0729 | -0.1015 | -0.0444 | -0.1001 | -0.1856 | -0.0146 | 0.0272 | -0.0687 | 0.1231 | -0.0083 | 0.0685 | e4 | 50 |  | sup_imnet |
| arm/50/ssl_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1069 | -0.1860 | -0.0278 | -0.0711 | -0.1144 | -0.0277 | -0.0358 | -0.0904 | 0.0187 | -0.0767 | 0.0030 | e4 | 50 |  | ssl_imnet |
| arm/50/ssl_colon/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1076 | -0.1609 | -0.0543 | -0.0661 | -0.1339 | 0.0018 | -0.0415 | -0.1615 | 0.0785 | -0.0754 | -0.0100 | e4 | 50 |  | ssl_colon |
| arm/50/pooled/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.0958 | -0.1365 | -0.0551 | -0.0791 | -0.1196 | -0.0386 | -0.0167 | -0.0865 | 0.0530 | -0.0450 | 0.0124 | e4 | 50 |  | mean of regimes |
| arm/100/sup_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0356 | -0.0562 | -0.0151 | 0.0025 | -0.0385 | 0.0434 | -0.0381 | -0.0884 | 0.0122 | -0.0767 | 0.0010 | e4 | 100 |  | sup_imnet |
| arm/100/ssl_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0388 | -0.0664 | -0.0113 | 0.0020 | -0.0500 | 0.0540 | -0.0408 | -0.1084 | 0.0268 | -0.0752 | -0.0075 | e4 | 100 |  | ssl_imnet |
| arm/100/ssl_colon/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0255 | -0.0470 | -0.0040 | -0.0227 | -0.0475 | 0.0020 | -0.0028 | -0.0180 | 0.0124 | -0.0274 | 0.0178 | e4 | 100 |  | ssl_colon |
| arm/100/pooled/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0333 | -0.0403 | -0.0263 | -0.0061 | -0.0326 | 0.0204 | -0.0272 | -0.0525 | -0.0019 | -0.0584 | 0.0023 | e4 | 100 |  | mean of regimes |
| arm/100/sup_imnet/C-A/auroc | auroc | arm C - arm A | -0.1245 | -0.2356 | -0.0134 | -0.0177 | -0.0502 | 0.0148 | -0.1068 | -0.2192 | 0.0056 | -0.2066 | 0.0111 | e4 | 100 |  | sup_imnet |
| arm/100/ssl_imnet/C-A/auroc | auroc | arm C - arm A | -0.1581 | -0.2855 | -0.0307 | -0.0074 | -0.0667 | 0.0519 | -0.1507 | -0.3309 | 0.0295 | -0.2590 | -0.0203 | e4 | 100 |  | ssl_imnet |
| arm/100/ssl_colon/C-A/auroc | auroc | arm C - arm A | -0.0364 | -0.1300 | 0.0571 | -0.0680 | -0.1915 | 0.0554 | 0.0316 | -0.1323 | 0.1955 | -6.82e-04 | 0.0638 | e4 | 100 |  | ssl_colon |
| arm/100/pooled/C-A/auroc | auroc | arm C - arm A | -0.1063 | -0.1610 | -0.0517 | -0.0310 | -0.0887 | 0.0267 | -0.0753 | -0.1581 | 0.0075 | -0.1496 | 0.0095 | e4 | 100 |  | mean of regimes |
| arm/100/sup_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1012 | -0.1282 | -0.0742 | -0.1385 | -0.2530 | -0.0240 | 0.0373 | -0.1001 | 0.1748 | -0.0023 | 0.0806 | e4 | 100 |  | sup_imnet |
| arm/100/ssl_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1385 | -0.1986 | -0.0785 | -0.0304 | -0.0846 | 0.0238 | -0.1081 | -0.2066 | -0.0096 | -0.1666 | -0.0506 | e4 | 100 |  | ssl_imnet |
| arm/100/ssl_colon/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1260 | -0.1726 | -0.0795 | -0.0854 | -0.1379 | -0.0329 | -0.0406 | -0.1177 | 0.0365 | -0.0705 | -0.0152 | e4 | 100 |  | ssl_colon |
| arm/100/pooled/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1219 | -0.1484 | -0.0954 | -0.0848 | -0.1203 | -0.0492 | -0.0371 | -0.0871 | 0.0128 | -0.0721 | -0.0012 | e4 | 100 |  | mean of regimes |
| arm/200/sup_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0408 | -0.0700 | -0.0117 | -0.0011 | -0.0363 | 0.0341 | -0.0397 | -0.0872 | 0.0077 | -0.0836 | 0.0038 | e4 | 200 |  | sup_imnet |
| arm/200/ssl_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0430 | -0.0776 | -0.0085 | 0.0152 | -0.0041 | 0.0346 | -0.0583 | -0.0979 | -0.0186 | -0.0985 | -0.0183 | e4 | 200 |  | ssl_imnet |
| arm/200/ssl_colon/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0279 | -0.0572 | 0.0014 | -0.0160 | -0.0380 | 0.0059 | -0.0118 | -0.0490 | 0.0253 | -0.0328 | 0.0057 | e4 | 200 |  | ssl_colon |
| arm/200/pooled/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0373 | -0.0593 | -0.0153 | -6.53e-04 | -0.0144 | 0.0131 | -0.0366 | -0.0640 | -0.0092 | -0.0695 | -0.0039 | e4 | 200 |  | mean of regimes |
| arm/200/sup_imnet/C-A/auroc | auroc | arm C - arm A | -0.1676 | -0.2539 | -0.0814 | -0.0188 | -0.0474 | 0.0098 | -0.1488 | -0.2412 | -0.0565 | -0.2752 | 0.0056 | e4 | 200 |  | sup_imnet |
| arm/200/ssl_imnet/C-A/auroc | auroc | arm C - arm A | -0.1661 | -0.3072 | -0.0249 | -0.0395 | -0.0893 | 0.0103 | -0.1266 | -0.3012 | 0.0480 | -0.2461 | 0.0146 | e4 | 200 |  | ssl_imnet |
| arm/200/ssl_colon/C-A/auroc | auroc | arm C - arm A | -0.0787 | -0.1702 | 0.0129 | -0.0649 | -0.1677 | 0.0379 | -0.0138 | -0.1520 | 0.1245 | -0.0614 | 0.0350 | e4 | 200 |  | ssl_colon |
| arm/200/pooled/C-A/auroc | auroc | arm C - arm A | -0.1375 | -0.2215 | -0.0534 | -0.0411 | -0.0783 | -0.0038 | -0.0964 | -0.1924 | -3.81e-04 | -0.1870 | 0.0108 | e4 | 200 |  | mean of regimes |
| arm/200/sup_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1177 | -0.1493 | -0.0862 | -0.1307 | -0.2289 | -0.0325 | 0.0130 | -0.0654 | 0.0914 | -0.0304 | 0.0572 | e4 | 200 |  | sup_imnet |
| arm/200/ssl_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1400 | -0.1707 | -0.1093 | -0.0290 | -0.0921 | 0.0341 | -0.1110 | -0.1914 | -0.0306 | -0.1689 | -0.0526 | e4 | 200 |  | ssl_imnet |
| arm/200/ssl_colon/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1149 | -0.1479 | -0.0819 | -0.0929 | -0.1382 | -0.0476 | -0.0220 | -0.0676 | 0.0235 | -0.0587 | 0.0130 | e4 | 200 |  | ssl_colon |
| arm/200/pooled/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.1242 | -0.1448 | -0.1036 | -0.0842 | -0.1189 | -0.0495 | -0.0400 | -0.0713 | -0.0087 | -0.0808 | 0.0033 | e4 | 200 |  | mean of regimes |
| arm/500/sup_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0209 | -0.0528 | 0.0109 | 0.0157 | -0.0085 | 0.0398 | -0.0366 | -0.0542 | -0.0190 | -0.1016 | 0.0178 | e4 | 500 |  | sup_imnet |
| arm/500/ssl_imnet/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0269 | -0.0519 | -0.0020 | 0.0220 | -0.0171 | 0.0611 | -0.0489 | -0.0925 | -0.0053 | -0.1125 | 0.0053 | e4 | 500 |  | ssl_imnet |
| arm/500/ssl_colon/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0120 | -0.0281 | 0.0042 | -0.0081 | -0.0374 | 0.0211 | -0.0038 | -0.0418 | 0.0342 | -0.0369 | 0.0211 | e4 | 500 |  | ssl_colon |
| arm/500/pooled/B-A/balanced_accuracy | balanced_accuracy | arm B - arm A | -0.0199 | -0.0331 | -0.0068 | 0.0098 | -0.0152 | 0.0349 | -0.0298 | -0.0548 | -0.0048 | -0.0813 | 0.0116 | e4 | 500 |  | mean of regimes |
| arm/500/sup_imnet/C-A/auroc | auroc | arm C - arm A | 0.0272 | -0.0549 | 0.1093 | 0.0488 | 0.0015 | 0.0960 | -0.0216 | -0.0629 | 0.0198 | -0.0813 | 0.0307 | e4 | 500 |  | sup_imnet |
| arm/500/ssl_imnet/C-A/auroc | auroc | arm C - arm A | 0.0627 | -0.0881 | 0.2135 | 0.1008 | 0.0165 | 0.1851 | -0.0381 | -0.1692 | 0.0929 | -0.1246 | 0.0511 | e4 | 500 |  | ssl_imnet |
| arm/500/ssl_colon/C-A/auroc | auroc | arm C - arm A | 0.1656 | 0.0322 | 0.2989 | 0.1472 | 0.0273 | 0.2670 | 0.0184 | 0.0021 | 0.0346 | -0.0306 | 0.0622 | e4 | 500 |  | ssl_colon |
| arm/500/pooled/C-A/auroc | auroc | arm C - arm A | 0.0852 | -0.0292 | 0.1995 | 0.0989 | 0.0211 | 0.1767 | -0.0138 | -0.0708 | 0.0433 | -0.0635 | 0.0343 | e4 | 500 |  | mean of regimes |
| arm/500/sup_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.0711 | -0.1484 | 0.0063 | -0.0753 | -0.1899 | 0.0393 | 0.0042 | -0.0619 | 0.0704 | -0.0426 | 0.0595 | e4 | 500 |  | sup_imnet |
| arm/500/ssl_imnet/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.0242 | -0.0946 | 0.0463 | 0.0800 | 0.0190 | 0.1409 | -0.1041 | -0.1607 | -0.0476 | -0.1567 | -0.0573 | e4 | 500 |  | ssl_imnet |
| arm/500/ssl_colon/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | 0.0197 | -0.0331 | 0.0725 | 0.0180 | -0.0586 | 0.0946 | 0.0017 | -0.0806 | 0.0840 | -0.0504 | 0.0437 | e4 | 500 |  | ssl_colon |
| arm/500/pooled/C-A/balanced_accuracy | balanced_accuracy | arm C - arm A | -0.0252 | -0.0850 | 0.0346 | 0.0075 | -0.0492 | 0.0643 | -0.0327 | -0.0749 | 0.0094 | -0.0610 | -0.0050 | e4 | 500 |  | mean of regimes |
| gap/sup_imnet/auroc | auroc | flat - polypoid | -0.0333 | -0.0513 | -0.0153 | -0.0328 | -0.0706 | 0.0051 | -4.91e-04 | -0.0469 | 0.0459 | -0.0340 | 0.0314 | e2 |  |  | sup_imnet |
| gap/ssl_imnet/auroc | auroc | flat - polypoid | -0.0203 | -0.0342 | -0.0064 | -0.0023 | -0.0051 | 5.73e-04 | -0.0180 | -0.0310 | -0.0051 | -0.0436 | 0.0069 | e2 |  |  | ssl_imnet |
| gap/ssl_colon/auroc | auroc | flat - polypoid | -0.0266 | -0.0634 | 0.0101 | -0.0087 | -0.0292 | 0.0118 | -0.0179 | -0.0352 | -6.66e-04 | -0.0313 | -0.0058 | e2 |  |  | ssl_colon |
| gapdiff/ssl_colon-sup_imnet/auroc | auroc | gap ssl_colon - gap sup_imnet | 0.0066 | -0.0311 | 0.0444 | 0.0241 | 0.0012 | 0.0469 | -0.0174 | -0.0759 | 0.0410 | -0.0489 | 0.0128 | e2 |  |  |  |
| gapdiff/ssl_colon-ssl_imnet/auroc | auroc | gap ssl_colon - gap ssl_imnet | -0.0063 | -0.0418 | 0.0291 | -0.0064 | -0.0247 | 0.0118 | 1.04e-04 | -0.0203 | 0.0205 | -0.0193 | 0.0198 | e2 |  |  |  |
