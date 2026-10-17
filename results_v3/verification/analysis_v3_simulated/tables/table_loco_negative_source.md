### Sensitivity: negative-source decomposition leave-one-centre-out (no bootstrap)

| excluded | slot | regime | comparison | metric | hypothesis | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | -0.0709 | 0.0111 | 0.0081 | 15 | 5 | partition_means_t | -0.0709 | -0.0848 | -0.0571 | 1.41e-04 |  |  | 0 |
| C1 |  | sup_imnet | sequence - isolated | auroc |  | -0.0692 | 0.0088 | 0.0170 | 15 | 5 | partition_means_t | -0.0692 | -0.0802 | -0.0583 | 6.21e-05 |  |  | 0 |
| C1 |  | ssl_imnet | sequence - isolated | auroc |  | -0.0795 | 0.0202 | 0.0101 | 15 | 5 | partition_means_t | -0.0795 | -0.1046 | -0.0544 | 9.18e-04 |  |  | 0 |
| C1 |  | ssl_colon | sequence - isolated | auroc |  | -0.0680 | 0.0109 | 0.0144 | 15 | 5 | partition_means_t | -0.0680 | -0.0816 | -0.0545 | 1.52e-04 |  |  | 0 |
| C1 |  | ssl_imnet_matched | sequence - isolated | auroc |  | -0.0669 | 0.0134 | 0.0190 | 15 | 5 | partition_means_t | -0.0669 | -0.0836 | -0.0502 | 3.70e-04 |  |  | 0 |
| C1 | isolated |  | ssl_colon - sup_imnet | auroc |  | 0.0506 | 0.0175 | 0.0256 | 15 | 5 | partition_means_t | 0.0506 | 0.0288 | 0.0723 | 0.0030 |  |  | 0 |
| C1 | isolated |  | ssl_colon - sup_imnet | auprc |  | 0.0100 | 0.0036 | 0.0056 | 15 | 5 | partition_means_t | 0.0100 | 0.0055 | 0.0144 | 0.0034 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet | auroc |  | 0.0043 | 0.0167 | 0.0191 | 15 | 5 | partition_means_t | 0.0043 | -0.0165 | 0.0251 | 0.5947 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet | auprc |  | 5.15e-04 | 0.0029 | 0.0040 | 15 | 5 | partition_means_t | 5.15e-04 | -0.0031 | 0.0041 | 0.7135 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0311 | 0.0117 | 0.0218 | 15 | 5 | partition_means_t | 0.0311 | 0.0166 | 0.0457 | 0.0040 |  |  | 0 |
| C1 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0053 | 0.0019 | 0.0046 | 15 | 5 | partition_means_t | 0.0053 | 0.0029 | 0.0077 | 0.0036 |  |  | 0 |
| C1 | sequence |  | ssl_colon - sup_imnet | auroc |  | 0.0517 | 0.0129 | 0.0133 | 15 | 5 | partition_means_t | 0.0517 | 0.0357 | 0.0677 | 8.49e-04 |  |  | 0 |
| C1 | sequence |  | ssl_colon - sup_imnet | auprc |  | 0.0924 | 0.0261 | 0.0172 | 15 | 5 | partition_means_t | 0.0924 | 0.0600 | 0.1248 | 0.0014 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet | auroc |  | 0.0158 | 0.0029 | 0.0112 | 15 | 5 | partition_means_t | 0.0158 | 0.0121 | 0.0194 | 2.74e-04 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet | auprc |  | 0.0283 | 0.0130 | 0.0217 | 15 | 5 | partition_means_t | 0.0283 | 0.0122 | 0.0445 | 0.0081 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0300 | 0.0073 | 0.0075 | 15 | 5 | partition_means_t | 0.0300 | 0.0210 | 0.0391 | 7.78e-04 |  |  | 0 |
| C1 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0573 | 0.0141 | 0.0197 | 15 | 5 | partition_means_t | 0.0573 | 0.0398 | 0.0748 | 8.11e-04 |  |  | 0 |
| C2 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | -0.0733 | 0.0116 | 0.0066 | 15 | 5 | partition_means_t | -0.0733 | -0.0877 | -0.0590 | 1.43e-04 |  |  | 0 |
| C2 |  | sup_imnet | sequence - isolated | auroc |  | -0.0692 | 0.0076 | 0.0095 | 15 | 5 | partition_means_t | -0.0692 | -0.0786 | -0.0597 | 3.48e-05 |  |  | 0 |
| C2 |  | ssl_imnet | sequence - isolated | auroc |  | -0.0791 | 0.0180 | 0.0110 | 15 | 5 | partition_means_t | -0.0791 | -0.1014 | -0.0567 | 6.04e-04 |  |  | 0 |
| C2 |  | ssl_colon | sequence - isolated | auroc |  | -0.0719 | 0.0128 | 0.0144 | 15 | 5 | partition_means_t | -0.0719 | -0.0878 | -0.0561 | 2.28e-04 |  |  | 0 |
| C2 |  | ssl_imnet_matched | sequence - isolated | auroc |  | -0.0732 | 0.0146 | 0.0152 | 15 | 5 | partition_means_t | -0.0732 | -0.0913 | -0.0550 | 3.65e-04 |  |  | 0 |
| C2 | isolated |  | ssl_colon - sup_imnet | auroc |  | 0.0556 | 0.0187 | 0.0243 | 15 | 5 | partition_means_t | 0.0556 | 0.0324 | 0.0788 | 0.0027 |  |  | 0 |
| C2 | isolated |  | ssl_colon - sup_imnet | auprc |  | 0.0119 | 0.0046 | 0.0071 | 15 | 5 | partition_means_t | 0.0119 | 0.0062 | 0.0177 | 0.0045 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet | auroc |  | 0.0117 | 0.0127 | 0.0210 | 15 | 5 | partition_means_t | 0.0117 | -0.0041 | 0.0274 | 0.1088 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet | auprc |  | 0.0024 | 0.0025 | 0.0050 | 15 | 5 | partition_means_t | 0.0024 | -7.58e-04 | 0.0055 | 0.1033 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0316 | 0.0052 | 0.0220 | 15 | 5 | partition_means_t | 0.0316 | 0.0251 | 0.0381 | 1.76e-04 |  |  | 0 |
| C2 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0059 | 9.24e-04 | 0.0047 | 15 | 5 | partition_means_t | 0.0059 | 0.0047 | 0.0070 | 1.42e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - sup_imnet | auroc |  | 0.0528 | 0.0094 | 0.0126 | 15 | 5 | partition_means_t | 0.0528 | 0.0411 | 0.0645 | 2.30e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - sup_imnet | auprc |  | 0.0946 | 0.0207 | 0.0189 | 15 | 5 | partition_means_t | 0.0946 | 0.0689 | 0.1202 | 5.13e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet | auroc |  | 0.0188 | 0.0031 | 0.0141 | 15 | 5 | partition_means_t | 0.0188 | 0.0149 | 0.0227 | 1.78e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet | auprc |  | 0.0344 | 0.0108 | 0.0270 | 15 | 5 | partition_means_t | 0.0344 | 0.0210 | 0.0478 | 0.0020 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0328 | 0.0053 | 0.0096 | 15 | 5 | partition_means_t | 0.0328 | 0.0262 | 0.0394 | 1.60e-04 |  |  | 0 |
| C2 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0617 | 0.0095 | 0.0193 | 15 | 5 | partition_means_t | 0.0617 | 0.0499 | 0.0735 | 1.31e-04 |  |  | 0 |
| C3 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | -0.0702 | 0.0106 | 0.0072 | 15 | 5 | partition_means_t | -0.0702 | -0.0834 | -0.0571 | 1.21e-04 |  |  | 0 |
| C3 |  | sup_imnet | sequence - isolated | auroc |  | -0.0663 | 0.0118 | 0.0146 | 15 | 5 | partition_means_t | -0.0663 | -0.0810 | -0.0517 | 2.32e-04 |  |  | 0 |
| C3 |  | ssl_imnet | sequence - isolated | auroc |  | -0.0773 | 0.0154 | 0.0103 | 15 | 5 | partition_means_t | -0.0773 | -0.0965 | -0.0581 | 3.63e-04 |  |  | 0 |
| C3 |  | ssl_colon | sequence - isolated | auroc |  | -0.0698 | 0.0095 | 0.0126 | 15 | 5 | partition_means_t | -0.0698 | -0.0816 | -0.0580 | 8.03e-05 |  |  | 0 |
| C3 |  | ssl_imnet_matched | sequence - isolated | auroc |  | -0.0674 | 0.0126 | 0.0179 | 15 | 5 | partition_means_t | -0.0674 | -0.0831 | -0.0518 | 2.79e-04 |  |  | 0 |
| C3 | isolated |  | ssl_colon - sup_imnet | auroc |  | 0.0530 | 0.0158 | 0.0231 | 15 | 5 | partition_means_t | 0.0530 | 0.0335 | 0.0726 | 0.0017 |  |  | 0 |
| C3 | isolated |  | ssl_colon - sup_imnet | auprc |  | 0.0107 | 0.0032 | 0.0056 | 15 | 5 | partition_means_t | 0.0107 | 0.0067 | 0.0146 | 0.0018 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet | auroc |  | 0.0056 | 0.0109 | 0.0152 | 15 | 5 | partition_means_t | 0.0056 | -0.0079 | 0.0191 | 0.3109 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet | auprc |  | 0.0010 | 0.0019 | 0.0031 | 15 | 5 | partition_means_t | 0.0010 | -0.0013 | 0.0034 | 0.2927 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0319 | 0.0047 | 0.0248 | 15 | 5 | partition_means_t | 0.0319 | 0.0261 | 0.0377 | 1.10e-04 |  |  | 0 |
| C3 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0055 | 0.0011 | 0.0050 | 15 | 5 | partition_means_t | 0.0055 | 0.0041 | 0.0069 | 3.78e-04 |  |  | 0 |
| C3 | sequence |  | ssl_colon - sup_imnet | auroc |  | 0.0496 | 0.0116 | 0.0120 | 15 | 5 | partition_means_t | 0.0496 | 0.0352 | 0.0640 | 6.70e-04 |  |  | 0 |
| C3 | sequence |  | ssl_colon - sup_imnet | auprc |  | 0.0887 | 0.0251 | 0.0156 | 15 | 5 | partition_means_t | 0.0887 | 0.0575 | 0.1199 | 0.0014 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet | auroc |  | 0.0131 | 0.0030 | 0.0136 | 15 | 5 | partition_means_t | 0.0131 | 0.0094 | 0.0168 | 5.78e-04 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet | auprc |  | 0.0246 | 0.0111 | 0.0251 | 15 | 5 | partition_means_t | 0.0246 | 0.0109 | 0.0384 | 0.0076 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0295 | 0.0069 | 0.0100 | 15 | 5 | partition_means_t | 0.0295 | 0.0209 | 0.0381 | 6.83e-04 |  |  | 0 |
| C3 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0547 | 0.0155 | 0.0218 | 15 | 5 | partition_means_t | 0.0547 | 0.0355 | 0.0739 | 0.0014 |  |  | 0 |
| C4 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | -0.0702 | 0.0111 | 0.0068 | 15 | 5 | partition_means_t | -0.0702 | -0.0841 | -0.0564 | 1.47e-04 |  |  | 0 |
| C4 |  | sup_imnet | sequence - isolated | auroc |  | -0.0643 | 0.0125 | 0.0113 | 15 | 5 | partition_means_t | -0.0643 | -0.0798 | -0.0488 | 3.24e-04 |  |  | 0 |
| C4 |  | ssl_imnet | sequence - isolated | auroc |  | -0.0795 | 0.0147 | 0.0081 | 15 | 5 | partition_means_t | -0.0795 | -0.0977 | -0.0613 | 2.66e-04 |  |  | 0 |
| C4 |  | ssl_colon | sequence - isolated | auroc |  | -0.0693 | 0.0123 | 0.0147 | 15 | 5 | partition_means_t | -0.0693 | -0.0845 | -0.0540 | 2.26e-04 |  |  | 0 |
| C4 |  | ssl_imnet_matched | sequence - isolated | auroc |  | -0.0678 | 0.0163 | 0.0205 | 15 | 5 | partition_means_t | -0.0678 | -0.0880 | -0.0476 | 7.38e-04 |  |  | 0 |
| C4 | isolated |  | ssl_colon - sup_imnet | auroc |  | 0.0575 | 0.0228 | 0.0222 | 15 | 5 | partition_means_t | 0.0575 | 0.0291 | 0.0858 | 0.0049 |  |  | 0 |
| C4 | isolated |  | ssl_colon - sup_imnet | auprc |  | 0.0115 | 0.0046 | 0.0056 | 15 | 5 | partition_means_t | 0.0115 | 0.0058 | 0.0171 | 0.0049 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet | auroc |  | 0.0060 | 0.0099 | 0.0187 | 15 | 5 | partition_means_t | 0.0060 | -0.0063 | 0.0182 | 0.2471 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet | auprc |  | 0.0011 | 0.0015 | 0.0036 | 15 | 5 | partition_means_t | 0.0011 | -7.30e-04 | 0.0029 | 0.1736 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0330 | 0.0090 | 0.0279 | 15 | 5 | partition_means_t | 0.0330 | 0.0218 | 0.0442 | 0.0012 |  |  | 0 |
| C4 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0057 | 0.0016 | 0.0055 | 15 | 5 | partition_means_t | 0.0057 | 0.0037 | 0.0077 | 0.0014 |  |  | 0 |
| C4 | sequence |  | ssl_colon - sup_imnet | auroc |  | 0.0525 | 0.0120 | 0.0155 | 15 | 5 | partition_means_t | 0.0525 | 0.0376 | 0.0675 | 6.21e-04 |  |  | 0 |
| C4 | sequence |  | ssl_colon - sup_imnet | auprc |  | 0.0920 | 0.0252 | 0.0213 | 15 | 5 | partition_means_t | 0.0920 | 0.0607 | 0.1233 | 0.0012 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet | auroc |  | 0.0162 | 0.0042 | 0.0136 | 15 | 5 | partition_means_t | 0.0162 | 0.0110 | 0.0214 | 9.86e-04 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet | auprc |  | 0.0287 | 0.0098 | 0.0272 | 15 | 5 | partition_means_t | 0.0287 | 0.0166 | 0.0409 | 0.0028 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0316 | 0.0043 | 0.0062 | 15 | 5 | partition_means_t | 0.0316 | 0.0262 | 0.0369 | 8.11e-05 |  |  | 0 |
| C4 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0573 | 0.0095 | 0.0156 | 15 | 5 | partition_means_t | 0.0573 | 0.0455 | 0.0690 | 1.75e-04 |  |  | 0 |
| C5 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | -0.0726 | 0.0107 | 0.0061 | 15 | 5 | partition_means_t | -0.0726 | -0.0859 | -0.0593 | 1.11e-04 |  |  | 0 |
| C5 |  | sup_imnet | sequence - isolated | auroc |  | -0.0691 | 0.0082 | 0.0145 | 15 | 5 | partition_means_t | -0.0691 | -0.0792 | -0.0589 | 4.64e-05 |  |  | 0 |
| C5 |  | ssl_imnet | sequence - isolated | auroc |  | -0.0802 | 0.0156 | 0.0062 | 15 | 5 | partition_means_t | -0.0802 | -0.0996 | -0.0608 | 3.29e-04 |  |  | 0 |
| C5 |  | ssl_colon | sequence - isolated | auroc |  | -0.0694 | 0.0097 | 0.0117 | 15 | 5 | partition_means_t | -0.0694 | -0.0815 | -0.0573 | 9.08e-05 |  |  | 0 |
| C5 |  | ssl_imnet_matched | sequence - isolated | auroc |  | -0.0717 | 0.0142 | 0.0182 | 15 | 5 | partition_means_t | -0.0717 | -0.0893 | -0.0541 | 3.49e-04 |  |  | 0 |
| C5 | isolated |  | ssl_colon - sup_imnet | auroc |  | 0.0533 | 0.0178 | 0.0217 | 15 | 5 | partition_means_t | 0.0533 | 0.0312 | 0.0755 | 0.0026 |  |  | 0 |
| C5 | isolated |  | ssl_colon - sup_imnet | auprc |  | 0.0108 | 0.0036 | 0.0051 | 15 | 5 | partition_means_t | 0.0108 | 0.0063 | 0.0153 | 0.0026 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet | auroc |  | 0.0061 | 0.0108 | 0.0189 | 15 | 5 | partition_means_t | 0.0061 | -0.0073 | 0.0195 | 0.2747 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet | auprc |  | 9.82e-04 | 0.0019 | 0.0038 | 15 | 5 | partition_means_t | 9.82e-04 | -0.0014 | 0.0034 | 0.3154 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0320 | 0.0072 | 0.0206 | 15 | 5 | partition_means_t | 0.0320 | 0.0230 | 0.0410 | 5.94e-04 |  |  | 0 |
| C5 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0056 | 0.0013 | 0.0043 | 15 | 5 | partition_means_t | 0.0056 | 0.0040 | 0.0072 | 6.73e-04 |  |  | 0 |
| C5 | sequence |  | ssl_colon - sup_imnet | auroc |  | 0.0530 | 0.0120 | 0.0153 | 15 | 5 | partition_means_t | 0.0530 | 0.0380 | 0.0679 | 5.97e-04 |  |  | 0 |
| C5 | sequence |  | ssl_colon - sup_imnet | auprc |  | 0.0938 | 0.0257 | 0.0184 | 15 | 5 | partition_means_t | 0.0938 | 0.0619 | 0.1256 | 0.0012 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet | auroc |  | 0.0168 | 0.0033 | 0.0146 | 15 | 5 | partition_means_t | 0.0168 | 0.0128 | 0.0209 | 3.30e-04 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet | auprc |  | 0.0317 | 0.0089 | 0.0276 | 15 | 5 | partition_means_t | 0.0317 | 0.0206 | 0.0427 | 0.0014 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0342 | 0.0045 | 0.0076 | 15 | 5 | partition_means_t | 0.0342 | 0.0286 | 0.0398 | 6.93e-05 |  |  | 0 |
| C5 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0618 | 0.0103 | 0.0197 | 15 | 5 | partition_means_t | 0.0618 | 0.0490 | 0.0746 | 1.79e-04 |  |  | 0 |
| C6 |  |  | sequence - isolated (mean of regimes) | auroc | H5 | -0.0777 | 0.0103 | 0.0072 | 15 | 5 | partition_means_t | -0.0777 | -0.0905 | -0.0650 | 7.12e-05 |  |  | 0 |
| C6 |  | sup_imnet | sequence - isolated | auroc |  | -0.0756 | 0.0109 | 0.0144 | 15 | 5 | partition_means_t | -0.0756 | -0.0891 | -0.0621 | 9.93e-05 |  |  | 0 |
| C6 |  | ssl_imnet | sequence - isolated | auroc |  | -0.0846 | 0.0144 | 0.0095 | 15 | 5 | partition_means_t | -0.0846 | -0.1025 | -0.0667 | 1.95e-04 |  |  | 0 |
| C6 |  | ssl_colon | sequence - isolated | auroc |  | -0.0750 | 0.0088 | 0.0148 | 15 | 5 | partition_means_t | -0.0750 | -0.0860 | -0.0641 | 4.46e-05 |  |  | 0 |
| C6 |  | ssl_imnet_matched | sequence - isolated | auroc |  | -0.0757 | 0.0147 | 0.0131 | 15 | 5 | partition_means_t | -0.0757 | -0.0940 | -0.0574 | 3.29e-04 |  |  | 0 |
| C6 | isolated |  | ssl_colon - sup_imnet | auroc |  | 0.0527 | 0.0164 | 0.0208 | 15 | 5 | partition_means_t | 0.0527 | 0.0323 | 0.0731 | 0.0020 |  |  | 0 |
| C6 | isolated |  | ssl_colon - sup_imnet | auprc |  | 0.0102 | 0.0032 | 0.0052 | 15 | 5 | partition_means_t | 0.0102 | 0.0062 | 0.0142 | 0.0021 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet | auroc |  | 0.0074 | 0.0103 | 0.0186 | 15 | 5 | partition_means_t | 0.0074 | -0.0054 | 0.0201 | 0.1845 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet | auprc |  | 0.0013 | 0.0018 | 0.0036 | 15 | 5 | partition_means_t | 0.0013 | -8.84e-04 | 0.0035 | 0.1707 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0328 | 0.0092 | 0.0198 | 15 | 5 | partition_means_t | 0.0328 | 0.0214 | 0.0442 | 0.0013 |  |  | 0 |
| C6 | isolated |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0056 | 0.0017 | 0.0042 | 15 | 5 | partition_means_t | 0.0056 | 0.0034 | 0.0077 | 0.0019 |  |  | 0 |
| C6 | sequence |  | ssl_colon - sup_imnet | auroc |  | 0.0533 | 0.0123 | 0.0160 | 15 | 5 | partition_means_t | 0.0533 | 0.0380 | 0.0685 | 6.35e-04 |  |  | 0 |
| C6 | sequence |  | ssl_colon - sup_imnet | auprc |  | 0.0924 | 0.0262 | 0.0213 | 15 | 5 | partition_means_t | 0.0924 | 0.0599 | 0.1250 | 0.0014 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet | auroc |  | 0.0169 | 0.0045 | 0.0136 | 15 | 5 | partition_means_t | 0.0169 | 0.0113 | 0.0226 | 0.0011 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet | auprc |  | 0.0298 | 0.0124 | 0.0257 | 15 | 5 | partition_means_t | 0.0298 | 0.0145 | 0.0452 | 0.0057 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet_matched | auroc |  | 0.0334 | 0.0040 | 0.0084 | 15 | 5 | partition_means_t | 0.0334 | 0.0284 | 0.0383 | 4.79e-05 |  |  | 0 |
| C6 | sequence |  | ssl_colon - ssl_imnet_matched | auprc |  | 0.0608 | 0.0110 | 0.0185 | 15 | 5 | partition_means_t | 0.0608 | 0.0472 | 0.0745 | 2.45e-04 |  |  | 0 |
