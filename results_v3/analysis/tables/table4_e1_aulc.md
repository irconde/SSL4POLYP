### Table 4: E1 AULC (trapezoid over 5–100 %, normalised by the axis width)

| regime | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B | comparison |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sup_imnet | AULC(auroc) | 0.9095 | 0.0169 | 0.0238 | 15 | 5 | partition_means_t | 0.9095 | 0.8885 | 0.9305 | 2.87e-08 | 0.8901 | 0.9274 | 2000 |  |
| ssl_imnet | AULC(auroc) | 0.9441 | 0.0204 | 0.0105 | 15 | 5 | partition_means_t | 0.9441 | 0.9187 | 0.9695 | 5.27e-08 | 0.9290 | 0.9568 | 2000 |  |
| ssl_colon | AULC(auroc) | 0.9530 | 0.0129 | 0.0073 | 15 | 5 | partition_means_t | 0.9530 | 0.9370 | 0.9690 | 8.04e-09 | 0.9379 | 0.9655 | 2000 |  |
| ssl_imnet_matched | AULC(auroc) | 0.9681 | 0.0169 | 0.0082 | 15 | 5 | partition_means_t | 0.9681 | 0.9471 | 0.9890 | 2.22e-08 | 0.9556 | 0.9784 | 2000 |  |
| sup_imnet | AULC(auprc) | 0.9093 | 0.0145 | 0.0232 | 15 | 5 | partition_means_t | 0.9093 | 0.8913 | 0.9274 | 1.57e-08 | 0.8917 | 0.9269 | 2000 |  |
| ssl_imnet | AULC(auprc) | 0.9464 | 0.0178 | 0.0103 | 15 | 5 | partition_means_t | 0.9464 | 0.9244 | 0.9685 | 2.97e-08 | 0.9336 | 0.9583 | 2000 |  |
| ssl_colon | AULC(auprc) | 0.9568 | 0.0107 | 0.0067 | 15 | 5 | partition_means_t | 0.9568 | 0.9435 | 0.9701 | 3.78e-09 | 0.9434 | 0.9689 | 2000 |  |
| ssl_imnet_matched | AULC(auprc) | 0.9706 | 0.0145 | 0.0085 | 15 | 5 | partition_means_t | 0.9706 | 0.9526 | 0.9887 | 1.20e-08 | 0.9597 | 0.9798 | 2000 |  |
| sup_imnet | AULC(f1) | 0.8334 | 0.0192 | 0.0303 | 15 | 5 | partition_means_t | 0.8334 | 0.8096 | 0.8572 | 6.73e-08 | 0.8104 | 0.8553 | 2000 |  |
| ssl_imnet | AULC(f1) | 0.8892 | 0.0233 | 0.0146 | 15 | 5 | partition_means_t | 0.8892 | 0.8602 | 0.9182 | 1.14e-07 | 0.8699 | 0.9075 | 2000 |  |
| ssl_colon | AULC(f1) | 0.8877 | 0.0225 | 0.0167 | 15 | 5 | partition_means_t | 0.8877 | 0.8598 | 0.9156 | 9.85e-08 | 0.8671 | 0.9066 | 2000 |  |
| ssl_imnet_matched | AULC(f1) | 0.9138 | 0.0354 | 0.0156 | 15 | 5 | partition_means_t | 0.9138 | 0.8699 | 0.9578 | 5.41e-07 | 0.8941 | 0.9308 | 2000 |  |
| sup_imnet | AULC(precision) | 0.8296 | 0.0442 | 0.0401 | 15 | 5 | partition_means_t | 0.8296 | 0.7748 | 0.8844 | 1.92e-06 | 0.8062 | 0.8570 | 2000 |  |
| ssl_imnet | AULC(precision) | 0.8780 | 0.0260 | 0.0235 | 15 | 5 | partition_means_t | 0.8780 | 0.8458 | 0.9102 | 1.83e-07 | 0.8557 | 0.9029 | 2000 |  |
| ssl_colon | AULC(precision) | 0.8798 | 0.0176 | 0.0136 | 15 | 5 | partition_means_t | 0.8798 | 0.8580 | 0.9017 | 3.84e-08 | 0.8562 | 0.9062 | 2000 |  |
| ssl_imnet_matched | AULC(precision) | 0.9137 | 0.0345 | 0.0238 | 15 | 5 | partition_means_t | 0.9137 | 0.8708 | 0.9566 | 4.88e-07 | 0.8921 | 0.9366 | 2000 |  |
| sup_imnet | AULC(recall) | 0.8524 | 0.0529 | 0.0390 | 15 | 5 | partition_means_t | 0.8524 | 0.7867 | 0.9181 | 3.55e-06 | 0.8230 | 0.8810 | 2000 |  |
| ssl_imnet | AULC(recall) | 0.9102 | 0.0444 | 0.0261 | 15 | 5 | partition_means_t | 0.9102 | 0.8551 | 0.9653 | 1.35e-06 | 0.8835 | 0.9344 | 2000 |  |
| ssl_colon | AULC(recall) | 0.9032 | 0.0494 | 0.0309 | 15 | 5 | partition_means_t | 0.9032 | 0.8419 | 0.9645 | 2.14e-06 | 0.8746 | 0.9271 | 2000 |  |
| ssl_imnet_matched | AULC(recall) | 0.9211 | 0.0557 | 0.0227 | 15 | 5 | partition_means_t | 0.9211 | 0.8519 | 0.9902 | 3.19e-06 | 0.8952 | 0.9421 | 2000 |  |
| sup_imnet | AULC(specificity) | 0.7983 | 0.0717 | 0.0597 | 15 | 5 | partition_means_t | 0.7983 | 0.7093 | 0.8873 | 1.55e-05 | 0.7637 | 0.8331 | 2000 |  |
| ssl_imnet | AULC(specificity) | 0.8533 | 0.0351 | 0.0320 | 15 | 5 | partition_means_t | 0.8533 | 0.8097 | 0.8969 | 6.86e-07 | 0.8219 | 0.8825 | 2000 |  |
| ssl_colon | AULC(specificity) | 0.8636 | 0.0317 | 0.0198 | 15 | 5 | partition_means_t | 0.8636 | 0.8243 | 0.9030 | 4.33e-07 | 0.8323 | 0.8938 | 2000 |  |
| ssl_imnet_matched | AULC(specificity) | 0.9040 | 0.0430 | 0.0314 | 15 | 5 | partition_means_t | 0.9040 | 0.8506 | 0.9574 | 1.22e-06 | 0.8755 | 0.9296 | 2000 |  |
| sup_imnet | AULC(balanced_accuracy) | 0.8254 | 0.0218 | 0.0339 | 15 | 5 | partition_means_t | 0.8254 | 0.7983 | 0.8524 | 1.17e-07 | 0.8031 | 0.8475 | 2000 |  |
| ssl_imnet | AULC(balanced_accuracy) | 0.8817 | 0.0227 | 0.0146 | 15 | 5 | partition_means_t | 0.8817 | 0.8536 | 0.9099 | 1.05e-07 | 0.8614 | 0.9007 | 2000 |  |
| ssl_colon | AULC(balanced_accuracy) | 0.8834 | 0.0189 | 0.0138 | 15 | 5 | partition_means_t | 0.8834 | 0.8600 | 0.9069 | 5.01e-08 | 0.8621 | 0.9032 | 2000 |  |
| ssl_imnet_matched | AULC(balanced_accuracy) | 0.9125 | 0.0332 | 0.0161 | 15 | 5 | partition_means_t | 0.9125 | 0.8713 | 0.9537 | 4.19e-07 | 0.8940 | 0.9297 | 2000 |  |
| sup_imnet | AULC(brier) | 0.1258 | 0.0133 | 0.0234 | 15 | 5 | partition_means_t | 0.1258 | 0.1093 | 0.1423 | 2.95e-05 | 0.1101 | 0.1414 | 2000 |  |
| ssl_imnet | AULC(brier) | 0.0864 | 0.0177 | 0.0122 | 15 | 5 | partition_means_t | 0.0864 | 0.0644 | 0.1084 | 4.01e-04 | 0.0725 | 0.1013 | 2000 |  |
| ssl_colon | AULC(brier) | 0.0863 | 0.0173 | 0.0058 | 15 | 5 | partition_means_t | 0.0863 | 0.0648 | 0.1078 | 3.70e-04 | 0.0718 | 0.1022 | 2000 |  |
| ssl_imnet_matched | AULC(brier) | 0.0642 | 0.0241 | 0.0110 | 15 | 5 | partition_means_t | 0.0642 | 0.0343 | 0.0942 | 0.0040 | 0.0513 | 0.0785 | 2000 |  |
| sup_imnet | AULC(ece) | 0.0850 | 0.0124 | 0.0244 | 15 | 5 | partition_means_t | 0.0850 | 0.0696 | 0.1003 | 1.05e-04 | 0.0805 | 0.1094 | 2000 |  |
| ssl_imnet | AULC(ece) | 0.0653 | 0.0171 | 0.0150 | 15 | 5 | partition_means_t | 0.0653 | 0.0441 | 0.0865 | 0.0010 | 0.0596 | 0.0861 | 2000 |  |
| ssl_colon | AULC(ece) | 0.0674 | 0.0180 | 0.0081 | 15 | 5 | partition_means_t | 0.0674 | 0.0450 | 0.0898 | 0.0011 | 0.0619 | 0.0886 | 2000 |  |
| ssl_imnet_matched | AULC(ece) | 0.0517 | 0.0176 | 0.0130 | 15 | 5 | partition_means_t | 0.0517 | 0.0299 | 0.0735 | 0.0028 | 0.0464 | 0.0710 | 2000 |  |
| sup_imnet | AULC(per_polyp_sensitivity_ge1) | 0.9991 | 0.0021 | 0.0036 | 15 | 5 | partition_means_t | 0.9991 | 0.9965 | 1.0017 | 4.62e-12 | 0.9972 | 1 | 2000 |  |
| ssl_imnet | AULC(per_polyp_sensitivity_ge1) | 0.9946 | 0.0096 | 0.0127 | 15 | 5 | partition_means_t | 0.9946 | 0.9827 | 1.0066 | 2.11e-09 | 0.9875 | 1 | 2000 |  |
| ssl_colon | AULC(per_polyp_sensitivity_ge1) | 0.9982 | 0.0039 | 0.0068 | 15 | 5 | partition_means_t | 0.9982 | 0.9934 | 1.0031 | 5.72e-11 | 0.9947 | 1 | 2000 |  |
| ssl_imnet_matched | AULC(per_polyp_sensitivity_ge1) | 0.9892 | 0.0215 | 0.0072 | 15 | 5 | partition_means_t | 0.9892 | 0.9625 | 1.0160 | 5.39e-08 | 0.9785 | 0.9991 | 2000 |  |
| sup_imnet | AULC(per_polyp_sensitivity_ge5) | 0.9863 | 0.0148 | 0.0155 | 15 | 5 | partition_means_t | 0.9863 | 0.9680 | 1.0046 | 1.20e-08 | 0.9740 | 0.9963 | 2000 |  |
| ssl_imnet | AULC(per_polyp_sensitivity_ge5) | 0.9836 | 0.0228 | 0.0180 | 15 | 5 | partition_means_t | 0.9836 | 0.9553 | 1.0120 | 6.96e-08 | 0.9669 | 0.9967 | 2000 |  |
| ssl_colon | AULC(per_polyp_sensitivity_ge5) | 0.9909 | 0.0085 | 0.0189 | 15 | 5 | partition_means_t | 0.9909 | 0.9803 | 1.0015 | 1.31e-09 | 0.9793 | 0.9999 | 2000 |  |
| ssl_imnet_matched | AULC(per_polyp_sensitivity_ge5) | 0.9847 | 0.0258 | 0.0095 | 15 | 5 | partition_means_t | 0.9847 | 0.9527 | 1.0167 | 1.13e-07 | 0.9711 | 0.9958 | 2000 |  |
|  | AULC(auroc) | 0.0435 | 0.0127 | 0.0248 | 15 | 5 | partition_means_t | 0.0435 | 0.0277 | 0.0593 | 0.0016 | 0.0315 | 0.0569 | 2000 | ssl_colon - sup_imnet |
|  | AULC(auroc) | 0.0089 | 0.0097 | 0.0113 | 15 | 5 | partition_means_t | 0.0089 | -0.0031 | 0.0209 | 0.1088 | 5.54e-05 | 0.0186 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(auroc) | -0.0150 | 0.0137 | 0.0074 | 15 | 5 | partition_means_t | -0.0150 | -0.0321 | 0.0020 | 0.0708 | -0.0234 | -0.0071 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(auprc) | 0.0475 | 0.0152 | 0.0239 | 15 | 5 | partition_means_t | 0.0475 | 0.0286 | 0.0663 | 0.0022 | 0.0355 | 0.0592 | 2000 | ssl_colon - sup_imnet |
|  | AULC(auprc) | 0.0104 | 0.0114 | 0.0113 | 15 | 5 | partition_means_t | 0.0104 | -0.0038 | 0.0246 | 0.1126 | 0.0013 | 0.0197 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(auprc) | -0.0138 | 0.0118 | 0.0084 | 15 | 5 | partition_means_t | -0.0138 | -0.0285 | 8.67e-04 | 0.0592 | -0.0214 | -0.0064 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(f1) | 0.0544 | 0.0148 | 0.0290 | 15 | 5 | partition_means_t | 0.0544 | 0.0360 | 0.0728 | 0.0012 | 0.0382 | 0.0698 | 2000 | ssl_colon - sup_imnet |
|  | AULC(f1) | -0.0015 | 0.0131 | 0.0243 | 15 | 5 | partition_means_t | -0.0015 | -0.0177 | 0.0148 | 0.8133 | -0.0154 | 0.0127 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(f1) | -0.0261 | 0.0263 | 0.0157 | 15 | 5 | partition_means_t | -0.0261 | -0.0587 | 0.0065 | 0.0905 | -0.0373 | -0.0139 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(precision) | 0.0502 | 0.0322 | 0.0413 | 15 | 5 | partition_means_t | 0.0502 | 0.0103 | 0.0902 | 0.0251 | 0.0277 | 0.0710 | 2000 | ssl_colon - sup_imnet |
|  | AULC(precision) | 0.0018 | 0.0212 | 0.0231 | 15 | 5 | partition_means_t | 0.0018 | -0.0245 | 0.0281 | 0.8595 | -0.0195 | 0.0233 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(precision) | -0.0339 | 0.0223 | 0.0187 | 15 | 5 | partition_means_t | -0.0339 | -0.0616 | -0.0062 | 0.0275 | -0.0486 | -0.0191 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(recall) | 0.0508 | 0.0280 | 0.0392 | 15 | 5 | partition_means_t | 0.0508 | 0.0160 | 0.0856 | 0.0154 | 0.0309 | 0.0702 | 2000 | ssl_colon - sup_imnet |
|  | AULC(recall) | -0.0070 | 0.0250 | 0.0480 | 15 | 5 | partition_means_t | -0.0070 | -0.0380 | 0.0240 | 0.5654 | -0.0208 | 0.0081 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(recall) | -0.0179 | 0.0382 | 0.0228 | 15 | 5 | partition_means_t | -0.0179 | -0.0653 | 0.0296 | 0.3555 | -0.0326 | -0.0035 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(specificity) | 0.0653 | 0.0427 | 0.0647 | 15 | 5 | partition_means_t | 0.0653 | 0.0124 | 0.1183 | 0.0267 | 0.0376 | 0.0917 | 2000 | ssl_colon - sup_imnet |
|  | AULC(specificity) | 0.0103 | 0.0262 | 0.0350 | 15 | 5 | partition_means_t | 0.0103 | -0.0222 | 0.0429 | 0.4280 | -0.0163 | 0.0361 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(specificity) | -0.0404 | 0.0256 | 0.0201 | 15 | 5 | partition_means_t | -0.0404 | -0.0722 | -0.0086 | 0.0244 | -0.0596 | -0.0238 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(balanced_accuracy) | 0.0581 | 0.0170 | 0.0341 | 15 | 5 | partition_means_t | 0.0581 | 0.0369 | 0.0792 | 0.0016 | 0.0415 | 0.0747 | 2000 | ssl_colon - sup_imnet |
|  | AULC(balanced_accuracy) | 0.0017 | 0.0133 | 0.0204 | 15 | 5 | partition_means_t | 0.0017 | -0.0149 | 0.0182 | 0.7922 | -0.0132 | 0.0165 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(balanced_accuracy) | -0.0291 | 0.0242 | 0.0148 | 15 | 5 | partition_means_t | -0.0291 | -0.0592 | 9.57e-04 | 0.0548 | -0.0404 | -0.0177 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(brier) | -0.0395 | 0.0143 | 0.0237 | 15 | 5 | partition_means_t | -0.0395 | -0.0572 | -0.0218 | 0.0034 | -0.0518 | -0.0270 | 2000 | ssl_colon - sup_imnet |
|  | AULC(brier) | -1.44e-04 | 0.0145 | 0.0126 | 15 | 5 | partition_means_t | -1.44e-04 | -0.0182 | 0.0179 | 0.9834 | -0.0116 | 0.0115 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(brier) | 0.0220 | 0.0166 | 0.0106 | 15 | 5 | partition_means_t | 0.0220 | 0.0015 | 0.0426 | 0.0409 | 0.0129 | 0.0310 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(ece) | -0.0176 | 0.0161 | 0.0280 | 15 | 5 | partition_means_t | -0.0176 | -0.0376 | 0.0024 | 0.0710 | -0.0312 | -0.0083 | 2000 | ssl_colon - sup_imnet |
|  | AULC(ece) | 0.0021 | 0.0285 | 0.0187 | 15 | 5 | partition_means_t | 0.0021 | -0.0333 | 0.0374 | 0.8785 | -0.0099 | 0.0153 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(ece) | 0.0157 | 0.0083 | 0.0146 | 15 | 5 | partition_means_t | 0.0157 | 0.0054 | 0.0260 | 0.0135 | 0.0073 | 0.0268 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(per_polyp_sensitivity_ge1) | -8.19e-04 | 0.0049 | 0.0077 | 15 | 5 | partition_means_t | -8.19e-04 | -0.0069 | 0.0052 | 0.7269 | -0.0053 | 0.0020 | 2000 | ssl_colon - sup_imnet |
|  | AULC(per_polyp_sensitivity_ge1) | 0.0036 | 0.0115 | 0.0144 | 15 | 5 | partition_means_t | 0.0036 | -0.0106 | 0.0179 | 0.5189 | -0.0026 | 0.0109 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(per_polyp_sensitivity_ge1) | 0.0090 | 0.0176 | 0.0050 | 15 | 5 | partition_means_t | 0.0090 | -0.0129 | 0.0309 | 0.3173 | 9.36e-04 | 0.0177 | 2000 | ssl_colon - ssl_imnet_matched |
|  | AULC(per_polyp_sensitivity_ge5) | 0.0046 | 0.0077 | 0.0230 | 15 | 5 | partition_means_t | 0.0046 | -0.0050 | 0.0141 | 0.2569 | -0.0050 | 0.0159 | 2000 | ssl_colon - sup_imnet |
|  | AULC(per_polyp_sensitivity_ge5) | 0.0073 | 0.0170 | 0.0307 | 15 | 5 | partition_means_t | 0.0073 | -0.0138 | 0.0283 | 0.3931 | -0.0035 | 0.0219 | 2000 | ssl_colon - ssl_imnet |
|  | AULC(per_polyp_sensitivity_ge5) | 0.0062 | 0.0239 | 0.0199 | 15 | 5 | partition_means_t | 0.0062 | -0.0234 | 0.0358 | 0.5924 | -0.0048 | 0.0175 | 2000 | ssl_colon - ssl_imnet_matched |
