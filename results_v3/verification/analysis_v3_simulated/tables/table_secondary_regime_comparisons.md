### Secondary: SSL-colon vs each comparator by experiment, budget, stratum, arm and metric

Descriptive (no multiplicity correction). Markdown shows a subset of metrics; the CSV has all.

| experiment | variant | slot | comparison | metric | mean | sd_partition | sd_seed | n_runs | n_partitions | model_method | model_estimate | model_ci_lo | model_ci_hi | model_p | boot_lo | boot_hi | boot_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sun_full |  | all | ssl_colon - sup_imnet | auroc | 0.0374 | 0.0028 | 0.0150 | 15 | 5 | fallback_t | 0.0374 | 0.0338 | 0.0409 | 7.90e-06 | 0.0275 | 0.0479 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | auprc | 0.0331 | 0.0061 | 0.0161 | 15 | 5 | fallback_t | 0.0331 | 0.0255 | 0.0406 | 2.64e-04 | 0.0233 | 0.0443 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0377 | 0.0092 | 0.0262 | 15 | 5 | fallback_t | 0.0377 | 0.0262 | 0.0492 | 7.98e-04 | 0.0287 | 0.0477 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | brier | -0.0189 | 0.0037 | 0.0099 | 15 | 5 | fallback_t | -0.0189 | -0.0235 | -0.0144 | 3.18e-04 | -0.0237 | -0.0144 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | ece | 0.0270 | 0.0130 | 0.0212 | 15 | 5 | lmer | 0.0270 | 0.0109 | 0.0432 | 0.0096 | 0.0137 | 0.0305 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | auroc | 0.0088 | 0.0073 | 0.0104 | 15 | 5 | lmer | 0.0088 | -2.44e-04 | 0.0178 | 0.0540 | 0.0039 | 0.0141 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | auprc | 0.0083 | 0.0085 | 0.0128 | 15 | 5 | lmer | 0.0083 | -0.0022 | 0.0189 | 0.0938 | 0.0035 | 0.0141 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0059 | 0.0091 | 0.0159 | 15 | 5 | fallback_t | 0.0059 | -0.0055 | 0.0172 | 0.2253 | -0.0021 | 0.0139 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | brier | -0.0108 | 0.0065 | 0.0151 | 15 | 5 | fallback_t | -0.0108 | -0.0189 | -0.0027 | 0.0211 | -0.0143 | -0.0073 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | ece | -0.0097 | 0.0126 | 0.0270 | 15 | 5 | fallback_t | -0.0097 | -0.0253 | 0.0059 | 0.1580 | -0.0176 | -0.0023 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | auroc | 0.0212 | 0.0044 | 0.0126 | 15 | 5 | fallback_t | 0.0212 | 0.0157 | 0.0267 | 4.19e-04 | 0.0149 | 0.0277 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | auprc | 0.0204 | 0.0044 | 0.0143 | 15 | 5 | fallback_t | 0.0204 | 0.0149 | 0.0259 | 4.95e-04 | 0.0135 | 0.0275 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0237 | 0.0110 | 0.0226 | 15 | 5 | fallback_t | 0.0237 | 0.0100 | 0.0374 | 0.0086 | 0.0154 | 0.0321 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | brier | -0.0114 | 0.0032 | 0.0096 | 15 | 5 | fallback_t | -0.0114 | -0.0153 | -0.0075 | 0.0013 | -0.0147 | -0.0080 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | ece | 0.0158 | 0.0080 | 0.0255 | 15 | 5 | fallback_t | 0.0158 | 0.0058 | 0.0258 | 0.0117 | 0.0057 | 0.0190 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| sun_full |  | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | auroc | 0.0357 | 0.0149 | 0.0183 | 15 | 5 | lmer | 0.0357 | 0.0172 | 0.0541 | 0.0058 | 0.0267 | 0.0451 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | auprc | 0.0412 | 0.0161 | 0.0158 | 15 | 5 | lmer | 0.0412 | 0.0211 | 0.0612 | 0.0047 | 0.0299 | 0.0522 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0349 | 0.0142 | 0.0298 | 15 | 5 | fallback_t | 0.0349 | 0.0173 | 0.0526 | 0.0053 | 0.0256 | 0.0444 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | brier | -0.0148 | 0.0082 | 0.0096 | 15 | 5 | lmer | -0.0148 | -0.0250 | -0.0047 | 0.0155 | -0.0186 | -0.0108 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | ece | 0.0133 | 0.0078 | 0.0359 | 15 | 5 | fallback_t | 0.0133 | 0.0037 | 0.0230 | 0.0185 | -7.38e-04 | 0.0177 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | auroc | 0.0072 | 0.0112 | 0.0164 | 15 | 5 | lmer | 0.0072 | -0.0068 | 0.0211 | 0.2265 | -5.73e-04 | 0.0146 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | auprc | 0.0062 | 0.0094 | 0.0192 | 15 | 5 | fallback_t | 0.0062 | -0.0054 | 0.0179 | 0.2109 | -0.0032 | 0.0152 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0098 | 0.0181 | 0.0215 | 15 | 5 | lmer | 0.0098 | -0.0127 | 0.0322 | 0.2939 | 8.74e-04 | 0.0189 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | brier | -0.0073 | 0.0055 | 0.0126 | 15 | 5 | fallback_t | -0.0073 | -0.0141 | -4.66e-04 | 0.0413 | -0.0110 | -0.0036 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | ece | -0.0085 | 0.0198 | 0.0354 | 15 | 5 | fallback_t | -0.0085 | -0.0331 | 0.0161 | 0.3910 | -0.0172 | -4.62e-04 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | auroc | 0.0315 | 0.0164 | 0.0191 | 15 | 5 | lmer | 0.0315 | 0.0111 | 0.0519 | 0.0128 | 0.0228 | 0.0405 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | auprc | 0.0301 | 0.0166 | 0.0200 | 15 | 5 | lmer | 0.0301 | 0.0095 | 0.0507 | 0.0155 | 0.0204 | 0.0400 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0351 | 0.0180 | 0.0249 | 15 | 5 | lmer | 0.0351 | 0.0128 | 0.0574 | 0.0120 | 0.0256 | 0.0453 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | brier | -0.0137 | 0.0081 | 0.0130 | 15 | 5 | lmer | -0.0137 | -0.0239 | -0.0036 | 0.0196 | -0.0175 | -0.0100 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | ece | 0.0126 | 0.0049 | 0.0445 | 15 | 5 | fallback_t | 0.0126 | 0.0066 | 0.0187 | 0.0044 | 0.0011 | 0.0162 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 5 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | auroc | 0.0413 | 0.0105 | 0.0158 | 15 | 5 | lmer | 0.0413 | 0.0283 | 0.0543 | 9.14e-04 | 0.0330 | 0.0508 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | auprc | 0.0428 | 0.0104 | 0.0222 | 15 | 5 | fallback_t | 0.0428 | 0.0299 | 0.0558 | 7.73e-04 | 0.0339 | 0.0524 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0380 | 0.0135 | 0.0154 | 15 | 5 | lmer | 0.0380 | 0.0213 | 0.0548 | 0.0033 | 0.0289 | 0.0477 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | brier | -0.0172 | 0.0060 | 0.0078 | 15 | 5 | lmer | -0.0172 | -0.0247 | -0.0098 | 0.0030 | -0.0220 | -0.0126 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | ece | 0.0216 | 0.0236 | 0.0165 | 15 | 5 | lmer | 0.0216 | -0.0078 | 0.0509 | 0.1106 | 0.0080 | 0.0288 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | auroc | 0.0100 | 0.0071 | 0.0140 | 15 | 5 | fallback_t | 0.0100 | 0.0012 | 0.0188 | 0.0345 | 0.0018 | 0.0183 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | auprc | 0.0095 | 0.0050 | 0.0183 | 15 | 5 | fallback_t | 0.0095 | 0.0033 | 0.0158 | 0.0135 | 0.0010 | 0.0178 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0108 | 0.0173 | 0.0216 | 15 | 5 | lmer | 0.0108 | -0.0107 | 0.0322 | 0.2354 | 0.0013 | 0.0205 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | brier | -0.0071 | 0.0106 | 0.0132 | 15 | 5 | lmer | -0.0071 | -0.0203 | 0.0061 | 0.2091 | -0.0113 | -0.0029 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | ece | -9.15e-04 | 0.0183 | 0.0405 | 15 | 5 | fallback_t | -9.15e-04 | -0.0236 | 0.0218 | 0.9162 | -0.0088 | 0.0081 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | auroc | 0.0263 | 0.0098 | 0.0100 | 15 | 5 | lmer | 0.0263 | 0.0142 | 0.0385 | 0.0039 | 0.0180 | 0.0349 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | auprc | 0.0253 | 0.0079 | 0.0139 | 15 | 5 | fallback_t | 0.0253 | 0.0155 | 0.0351 | 0.0020 | 0.0170 | 0.0338 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0373 | 0.0106 | 0.0312 | 15 | 5 | fallback_t | 0.0373 | 0.0242 | 0.0505 | 0.0014 | 0.0282 | 0.0466 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | brier | -0.0132 | 0.0099 | 0.0108 | 15 | 5 | lmer | -0.0132 | -0.0255 | -8.87e-04 | 0.0409 | -0.0174 | -0.0088 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | ece | 0.0085 | 0.0297 | 0.0453 | 15 | 5 | lmer | 0.0085 | -0.0283 | 0.0453 | 0.5560 | -0.0030 | 0.0159 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 10 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | auroc | 0.0347 | 0.0075 | 0.0171 | 15 | 5 | fallback_t | 0.0347 | 0.0254 | 0.0440 | 4.93e-04 | 0.0267 | 0.0435 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | auprc | 0.0358 | 0.0085 | 0.0168 | 15 | 5 | fallback_t | 0.0358 | 0.0253 | 0.0464 | 7.10e-04 | 0.0265 | 0.0453 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0376 | 0.0095 | 0.0211 | 15 | 5 | fallback_t | 0.0376 | 0.0258 | 0.0493 | 8.90e-04 | 0.0286 | 0.0465 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | brier | -0.0169 | 0.0061 | 0.0126 | 15 | 5 | fallback_t | -0.0169 | -0.0245 | -0.0093 | 0.0035 | -0.0214 | -0.0126 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | ece | 0.0206 | 0.0149 | 0.0269 | 15 | 5 | fallback_t | 0.0206 | 0.0022 | 0.0391 | 0.0361 | 0.0065 | 0.0240 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 25 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | auroc | 0.0056 | 0.0083 | 0.0189 | 15 | 5 | fallback_t | 0.0056 | -0.0047 | 0.0158 | 0.2057 | -7.89e-04 | 0.0114 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | auprc | 0.0076 | 0.0059 | 0.0195 | 15 | 5 | fallback_t | 0.0076 | 1.90e-04 | 0.0149 | 0.0465 | 9.87e-04 | 0.0139 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0024 | 0.0137 | 0.0239 | 15 | 5 | fallback_t | 0.0024 | -0.0146 | 0.0194 | 0.7199 | -0.0067 | 0.0106 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | brier | -0.0026 | 0.0068 | 0.0090 | 15 | 5 | lmer | -0.0026 | -0.0110 | 0.0058 | 0.4420 | -0.0060 | 9.08e-04 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | ece | 0.0038 | 0.0198 | 0.0241 | 15 | 5 | lmer | 0.0038 | -0.0207 | 0.0284 | 0.6863 | -0.0051 | 0.0085 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | auroc | 0.0361 | 0.0110 | 0.0177 | 15 | 5 | lmer | 0.0361 | 0.0224 | 0.0498 | 0.0019 | 0.0280 | 0.0450 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | auprc | 0.0383 | 0.0109 | 0.0167 | 15 | 5 | lmer | 0.0383 | 0.0247 | 0.0519 | 0.0014 | 0.0293 | 0.0479 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0381 | 0.0199 | 0.0228 | 15 | 5 | lmer | 0.0381 | 0.0134 | 0.0628 | 0.0128 | 0.0291 | 0.0474 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | brier | -0.0181 | 0.0085 | 0.0110 | 15 | 5 | lmer | -0.0181 | -0.0287 | -0.0076 | 0.0088 | -0.0227 | -0.0142 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | ece | 0.0162 | 0.0246 | 0.0280 | 15 | 5 | lmer | 0.0162 | -0.0143 | 0.0467 | 0.2142 | 0.0044 | 0.0211 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 25 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | auroc | 0.0357 | 0.0119 | 0.0174 | 15 | 5 | lmer | 0.0357 | 0.0208 | 0.0505 | 0.0026 | 0.0278 | 0.0437 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | auprc | 0.0347 | 0.0111 | 0.0182 | 15 | 5 | lmer | 0.0347 | 0.0210 | 0.0484 | 0.0022 | 0.0267 | 0.0434 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0327 | 0.0180 | 0.0163 | 15 | 5 | lmer | 0.0327 | 0.0103 | 0.0550 | 0.0154 | 0.0242 | 0.0411 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | brier | -0.0180 | 0.0055 | 0.0182 | 15 | 5 | fallback_t | -0.0180 | -0.0248 | -0.0111 | 0.0019 | -0.0223 | -0.0136 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | ece | 0.0184 | 0.0241 | 0.0333 | 15 | 5 | lmer | 0.0184 | -0.0115 | 0.0483 | 0.1627 | 0.0072 | 0.0247 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | auroc | 0.0072 | 0.0032 | 0.0136 | 15 | 5 | fallback_t | 0.0072 | 0.0032 | 0.0112 | 0.0073 | 0.0017 | 0.0131 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | auprc | 0.0064 | 0.0075 | 0.0167 | 15 | 5 | fallback_t | 0.0064 | -0.0029 | 0.0157 | 0.1282 | -8.04e-04 | 0.0134 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0064 | 0.0114 | 0.0180 | 15 | 5 | lmer | 0.0064 | -0.0078 | 0.0206 | 0.2785 | -0.0015 | 0.0147 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | brier | -0.0041 | 0.0106 | 0.0141 | 15 | 5 | lmer | -0.0041 | -0.0173 | 0.0090 | 0.4321 | -0.0076 | -9.40e-04 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | ece | 9.39e-04 | 0.0254 | 0.0345 | 15 | 5 | lmer | 9.39e-04 | -0.0307 | 0.0325 | 0.9382 | -0.0071 | 0.0094 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | auroc | 0.0336 | 0.0108 | 0.0076 | 15 | 5 | lmer | 0.0336 | 0.0202 | 0.0470 | 0.0022 | 0.0263 | 0.0413 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | auprc | 0.0355 | 0.0112 | 0.0070 | 15 | 5 | lmer | 0.0355 | 0.0216 | 0.0495 | 0.0021 | 0.0266 | 0.0447 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0366 | 0.0166 | 0.0108 | 15 | 5 | lmer | 0.0366 | 0.0159 | 0.0572 | 0.0080 | 0.0283 | 0.0450 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | brier | -0.0177 | 0.0062 | 0.0106 | 15 | 5 | lmer | -0.0177 | -0.0254 | -0.0099 | 0.0032 | -0.0210 | -0.0139 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | ece | 0.0161 | 0.0139 | 0.0244 | 15 | 5 | fallback_t | 0.0161 | -0.0012 | 0.0334 | 0.0607 | 0.0074 | 0.0224 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 50 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | auroc | 0.0394 | 0.0144 | 0.0178 | 15 | 5 | lmer | 0.0394 | 0.0215 | 0.0573 | 0.0036 | 0.0312 | 0.0475 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | auprc | 0.0404 | 0.0170 | 0.0168 | 15 | 5 | lmer | 0.0404 | 0.0192 | 0.0615 | 0.0061 | 0.0310 | 0.0496 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | balanced_accuracy | 0.0442 | 0.0180 | 0.0175 | 15 | 5 | lmer | 0.0442 | 0.0219 | 0.0665 | 0.0053 | 0.0360 | 0.0520 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | brier | -0.0171 | 0.0098 | 0.0199 | 15 | 5 | fallback_t | -0.0171 | -0.0292 | -0.0049 | 0.0174 | -0.0206 | -0.0135 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | ece | 0.0331 | 0.0240 | 0.0299 | 15 | 5 | lmer | 0.0331 | 0.0033 | 0.0628 | 0.0367 | 0.0196 | 0.0370 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | auroc | 0.0091 | 0.0047 | 0.0145 | 15 | 5 | fallback_t | 0.0091 | 0.0032 | 0.0149 | 0.0126 | 0.0042 | 0.0139 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | auprc | 0.0085 | 0.0045 | 0.0149 | 15 | 5 | fallback_t | 0.0085 | 0.0028 | 0.0141 | 0.0140 | 0.0031 | 0.0141 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | balanced_accuracy | 0.0142 | 0.0111 | 0.0209 | 15 | 5 | fallback_t | 0.0142 | 4.46e-04 | 0.0280 | 0.0456 | 0.0067 | 0.0212 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | brier | -0.0023 | 0.0018 | 0.0175 | 15 | 5 | fallback_t | -0.0023 | -0.0046 | 4.22e-06 | 0.0503 | -0.0050 | 5.22e-04 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | ece | 0.0127 | 0.0137 | 0.0277 | 15 | 5 | fallback_t | 0.0127 | -0.0043 | 0.0298 | 0.1069 | 0.0070 | 0.0191 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | auroc | 0.0356 | 0.0114 | 0.0142 | 15 | 5 | lmer | 0.0356 | 0.0214 | 0.0498 | 0.0022 | 0.0282 | 0.0430 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | auprc | 0.0345 | 0.0116 | 0.0175 | 15 | 5 | lmer | 0.0345 | 0.0201 | 0.0490 | 0.0027 | 0.0271 | 0.0424 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0409 | 0.0063 | 0.0175 | 15 | 5 | fallback_t | 0.0409 | 0.0332 | 0.0487 | 1.27e-04 | 0.0330 | 0.0490 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | brier | -0.0169 | 0.0061 | 0.0163 | 15 | 5 | fallback_t | -0.0169 | -0.0244 | -0.0094 | 0.0034 | -0.0205 | -0.0133 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | ece | 0.0217 | 0.0131 | 0.0341 | 15 | 5 | fallback_t | 0.0217 | 0.0055 | 0.0379 | 0.0206 | 0.0136 | 0.0293 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e1 | 100 | all | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | auroc | 0.0374 | 0.0028 | 0.0150 | 15 | 5 | fallback_t | 0.0374 | 0.0338 | 0.0409 | 7.90e-06 | 0.0298 | 0.0456 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | auprc | 0.0331 | 0.0061 | 0.0161 | 15 | 5 | fallback_t | 0.0331 | 0.0255 | 0.0406 | 2.64e-04 | 0.0246 | 0.0417 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | balanced_accuracy | 0.0377 | 0.0092 | 0.0262 | 15 | 5 | fallback_t | 0.0377 | 0.0262 | 0.0492 | 7.98e-04 | 0.0288 | 0.0467 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | brier | -0.0189 | 0.0037 | 0.0099 | 15 | 5 | fallback_t | -0.0189 | -0.0235 | -0.0144 | 3.18e-04 | -0.0225 | -0.0155 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | ece | 0.0270 | 0.0130 | 0.0212 | 15 | 5 | lmer | 0.0270 | 0.0109 | 0.0432 | 0.0096 | 0.0159 | 0.0305 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | auroc | 0.0088 | 0.0073 | 0.0104 | 15 | 5 | lmer | 0.0088 | -2.44e-04 | 0.0178 | 0.0540 | 0.0042 | 0.0138 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | auprc | 0.0083 | 0.0085 | 0.0128 | 15 | 5 | lmer | 0.0083 | -0.0022 | 0.0189 | 0.0938 | 0.0034 | 0.0138 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | balanced_accuracy | 0.0059 | 0.0091 | 0.0159 | 15 | 5 | fallback_t | 0.0059 | -0.0055 | 0.0172 | 0.2253 | -0.0016 | 0.0137 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | brier | -0.0108 | 0.0065 | 0.0151 | 15 | 5 | fallback_t | -0.0108 | -0.0189 | -0.0027 | 0.0211 | -0.0140 | -0.0076 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | ece | -0.0097 | 0.0126 | 0.0270 | 15 | 5 | fallback_t | -0.0097 | -0.0253 | 0.0059 | 0.1580 | -0.0176 | -0.0025 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | auroc | 0.0212 | 0.0044 | 0.0126 | 15 | 5 | fallback_t | 0.0212 | 0.0157 | 0.0267 | 4.19e-04 | 0.0156 | 0.0273 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | auprc | 0.0204 | 0.0044 | 0.0143 | 15 | 5 | fallback_t | 0.0204 | 0.0149 | 0.0259 | 4.95e-04 | 0.0137 | 0.0272 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0237 | 0.0110 | 0.0226 | 15 | 5 | fallback_t | 0.0237 | 0.0100 | 0.0374 | 0.0086 | 0.0160 | 0.0321 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | brier | -0.0114 | 0.0032 | 0.0096 | 15 | 5 | fallback_t | -0.0114 | -0.0153 | -0.0075 | 0.0013 | -0.0144 | -0.0084 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | ece | 0.0158 | 0.0080 | 0.0255 | 15 | 5 | fallback_t | 0.0158 | 0.0058 | 0.0258 | 0.0117 | 0.0062 | 0.0188 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | overall | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | auroc | 0.0784 | 0.0119 | 0.0302 | 15 | 5 | fallback_t | 0.0784 | 0.0637 | 0.0932 | 1.23e-04 | 0.0642 | 0.0939 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | auprc | 0.1425 | 0.0097 | 0.0573 | 15 | 5 | fallback_t | 0.1425 | 0.1305 | 0.1545 | 5.08e-06 | 0.1133 | 0.1669 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | balanced_accuracy | 0.0617 | 0.0070 | 0.0401 | 15 | 5 | fallback_t | 0.0617 | 0.0530 | 0.0704 | 3.98e-05 | 0.0486 | 0.0745 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | brier | -0.0253 | 0.0140 | 0.0250 | 15 | 5 | fallback_t | -0.0253 | -0.0426 | -0.0080 | 0.0155 | -0.0300 | -0.0207 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | ece | 0.0060 | 0.0469 | 0.0603 | 15 | 5 | lmer | 0.0060 | -0.0523 | 0.0643 | 0.7892 | 2.27e-04 | 0.0143 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | auroc | 0.0183 | 0.0151 | 0.0172 | 15 | 5 | lmer | 0.0183 | -3.82e-04 | 0.0371 | 0.0530 | 0.0100 | 0.0275 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | auprc | 0.0342 | 0.0398 | 0.0435 | 15 | 5 | lmer | 0.0342 | -0.0152 | 0.0836 | 0.1268 | 0.0182 | 0.0516 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | balanced_accuracy | 0.0100 | 0.0115 | 0.0197 | 15 | 5 | lmer | 0.0100 | -0.0044 | 0.0243 | 0.1257 | -0.0022 | 0.0218 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | brier | -0.0173 | 0.0198 | 0.0343 | 15 | 5 | lmer | -0.0173 | -0.0419 | 0.0074 | 0.1235 | -0.0214 | -0.0131 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | ece | -0.0176 | 0.0591 | 0.0726 | 15 | 5 | lmer | -0.0176 | -0.0910 | 0.0557 | 0.5414 | -0.0256 | -0.0123 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | auroc | 0.0344 | 0.0138 | 0.0255 | 15 | 5 | fallback_t | 0.0344 | 0.0172 | 0.0515 | 0.0051 | 0.0248 | 0.0446 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | auprc | 0.0591 | 0.0161 | 0.0598 | 15 | 5 | fallback_t | 0.0591 | 0.0391 | 0.0790 | 0.0012 | 0.0387 | 0.0773 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0222 | 0.0084 | 0.0340 | 15 | 5 | fallback_t | 0.0222 | 0.0118 | 0.0326 | 0.0041 | 0.0108 | 0.0341 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | brier | -0.0139 | 0.0053 | 0.0230 | 15 | 5 | fallback_t | -0.0139 | -0.0205 | -0.0073 | 0.0043 | -0.0180 | -0.0097 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | ece | -0.0030 | 0.0307 | 0.0489 | 15 | 5 | lmer | -0.0030 | -0.0411 | 0.0351 | 0.8398 | -0.0087 | 0.0035 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | flat | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | auroc | 0.0168 | 0.0076 | 0.0115 | 15 | 5 | lmer | 0.0168 | 0.0073 | 0.0263 | 0.0079 | 0.0104 | 0.0236 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | auprc | 0.0209 | 0.0125 | 0.0185 | 15 | 5 | lmer | 0.0209 | 0.0053 | 0.0364 | 0.0203 | 0.0113 | 0.0312 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | balanced_accuracy | 0.0257 | 0.0113 | 0.0279 | 15 | 5 | fallback_t | 0.0257 | 0.0117 | 0.0398 | 0.0070 | 0.0168 | 0.0346 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | brier | -0.0111 | 0.0140 | 0.0163 | 15 | 5 | lmer | -0.0111 | -0.0284 | 0.0063 | 0.1514 | -0.0148 | -0.0074 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | ece | 0.0048 | 0.0242 | 0.0387 | 15 | 5 | lmer | 0.0048 | -0.0252 | 0.0349 | 0.6773 | -0.0019 | 0.0110 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - sup_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | auroc | 0.0040 | 0.0051 | 0.0117 | 15 | 5 | fallback_t | 0.0040 | -0.0024 | 0.0103 | 0.1555 | -5.08e-04 | 0.0085 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | auprc | 0.0058 | 0.0077 | 0.0175 | 15 | 5 | fallback_t | 0.0058 | -0.0038 | 0.0154 | 0.1676 | -3.67e-04 | 0.0127 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | balanced_accuracy | 0.0038 | 0.0152 | 0.0216 | 15 | 5 | lmer | 0.0038 | -0.0151 | 0.0227 | 0.6059 | -0.0039 | 0.0120 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | brier | -0.0117 | 0.0162 | 0.0262 | 15 | 5 | lmer | -0.0117 | -0.0318 | 0.0084 | 0.1807 | -0.0152 | -0.0080 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | ece | -0.0178 | 0.0251 | 0.0389 | 15 | 5 | lmer | -0.0178 | -0.0489 | 0.0133 | 0.1879 | -0.0235 | -0.0094 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | auroc | 0.0146 | 0.0086 | 0.0099 | 15 | 5 | lmer | 0.0146 | 0.0039 | 0.0253 | 0.0191 | 0.0093 | 0.0203 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | auprc | 0.0206 | 0.0113 | 0.0157 | 15 | 5 | lmer | 0.0206 | 0.0066 | 0.0346 | 0.0150 | 0.0122 | 0.0297 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0244 | 0.0175 | 0.0246 | 15 | 5 | lmer | 0.0244 | 0.0027 | 0.0462 | 0.0355 | 0.0163 | 0.0329 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | brier | -0.0096 | 0.0096 | 0.0163 | 15 | 5 | lmer | -0.0096 | -0.0215 | 0.0023 | 0.0884 | -0.0127 | -0.0065 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | ece | 0.0055 | 0.0137 | 0.0292 | 15 | 5 | fallback_t | 0.0055 | -0.0115 | 0.0225 | 0.4198 | -0.0017 | 0.0106 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge1 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e2 |  | polypoid | ssl_colon - ssl_imnet_matched | per_polyp_sensitivity_ge5 | 0.0000 | 0.0000 | 0.0000 | 15 | 5 | constant | 0.0000 |  |  |  | 0.0000 | 0.0000 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | auroc | 0.0542 | 0.0160 | 0.0187 | 15 | 5 | lmer | 0.0542 | 0.0343 | 0.0740 | 0.0016 | 0.0438 | 0.0643 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | auprc | 0.0120 | 0.0042 | 0.0050 | 15 | 5 | lmer | 0.0120 | 0.0069 | 0.0172 | 0.0029 | 0.0086 | 0.0163 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | balanced_accuracy | -0.0019 | 0.0041 | 0.0115 | 15 | 5 | fallback_t | -0.0019 | -0.0070 | 0.0032 | 0.3597 | -0.0044 | 9.97e-04 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | brier | -0.0161 | 0.0718 | 0.0813 | 15 | 5 | lmer | -0.0161 | -0.1053 | 0.0730 | 0.6419 | -0.0197 | -0.0126 | 2000 |
| e3 |  | all | ssl_colon - sup_imnet | ece | -0.0100 | 0.0581 | 0.0676 | 15 | 5 | lmer | -0.0100 | -0.0822 | 0.0622 | 0.7200 | -0.0127 | -0.0071 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | auroc | 0.0087 | 0.0068 | 0.0164 | 15 | 5 | fallback_t | 0.0087 | 1.99e-04 | 0.0172 | 0.0468 | 0.0025 | 0.0152 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | auprc | 0.0017 | 0.0013 | 0.0039 | 15 | 5 | fallback_t | 0.0017 | 1.22e-04 | 0.0033 | 0.0403 | 6.00e-05 | 0.0039 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | balanced_accuracy | -0.0010 | 0.0061 | 0.0110 | 15 | 5 | fallback_t | -0.0010 | -0.0086 | 0.0066 | 0.7296 | -0.0054 | 0.0036 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | brier | 0.0171 | 0.0889 | 0.1013 | 15 | 5 | lmer | 0.0171 | -0.0933 | 0.1274 | 0.6896 | 0.0106 | 0.0227 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet | ece | 0.0159 | 0.0734 | 0.0818 | 15 | 5 | lmer | 0.0159 | -0.0752 | 0.1071 | 0.6532 | 0.0105 | 0.0208 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | auroc | 0.0353 | 0.0071 | 0.0181 | 15 | 5 | fallback_t | 0.0353 | 0.0265 | 0.0441 | 3.71e-04 | 0.0276 | 0.0426 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | auprc | 0.0071 | 0.0013 | 0.0044 | 15 | 5 | fallback_t | 0.0071 | 0.0055 | 0.0087 | 2.74e-04 | 0.0051 | 0.0092 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | balanced_accuracy | -7.28e-05 | 0.0034 | 0.0062 | 15 | 5 | fallback_t | -7.28e-05 | -0.0043 | 0.0041 | 0.9639 | -0.0042 | 0.0043 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | brier | -0.0295 | 0.0349 | 0.0611 | 15 | 5 | fallback_t | -0.0295 | -0.0728 | 0.0139 | 0.1320 | -0.0355 | -0.0227 | 2000 |
| e3 |  | all | ssl_colon - ssl_imnet_matched | ece | -0.0230 | 0.0301 | 0.0508 | 15 | 5 | lmer | -0.0230 | -0.0603 | 0.0143 | 0.1624 | -0.0272 | -0.0182 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | auroc | 0.0522 | 0.0119 | 0.0133 | 15 | 5 | lmer | 0.0522 | 0.0375 | 0.0670 | 6.02e-04 | 0.0457 | 0.0594 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | auprc | 0.0901 | 0.0243 | 0.0172 | 15 | 5 | lmer | 0.0901 | 0.0598 | 0.1203 | 0.0012 | 0.0766 | 0.1005 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | balanced_accuracy | -6.95e-04 | 0.0046 | 0.0111 | 15 | 5 | fallback_t | -6.95e-04 | -0.0064 | 0.0050 | 0.7517 | -0.0034 | 0.0019 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | brier | -0.0086 | 0.0147 | 0.0148 | 15 | 5 | lmer | -0.0086 | -0.0268 | 0.0096 | 0.2597 | -0.0104 | -0.0070 | 2000 |
| e4 | 50 | A | ssl_colon - sup_imnet | ece | 0.0119 | 0.0369 | 0.0422 | 15 | 5 | lmer | 0.0119 | -0.0339 | 0.0577 | 0.5109 | 0.0074 | 0.0168 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | auroc | 0.0158 | 0.0033 | 0.0130 | 15 | 5 | fallback_t | 0.0158 | 0.0118 | 0.0199 | 4.13e-04 | 0.0096 | 0.0234 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | auprc | 0.0278 | 0.0097 | 0.0241 | 15 | 5 | fallback_t | 0.0278 | 0.0158 | 0.0399 | 0.0030 | 0.0157 | 0.0409 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | balanced_accuracy | -3.44e-04 | 0.0059 | 0.0114 | 15 | 5 | fallback_t | -3.44e-04 | -0.0077 | 0.0070 | 0.9025 | -0.0047 | 0.0045 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | brier | 0.0010 | 0.0172 | 0.0208 | 15 | 5 | lmer | 0.0010 | -0.0204 | 0.0224 | 0.9008 | -0.0014 | 0.0041 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet | ece | 0.0136 | 0.0449 | 0.0502 | 15 | 5 | lmer | 0.0136 | -0.0422 | 0.0693 | 0.5367 | 0.0102 | 0.0156 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0319 | 0.0048 | 0.0078 | 15 | 5 | lmer | 0.0319 | 0.0260 | 0.0379 | 1.16e-04 | 0.0239 | 0.0400 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0565 | 0.0105 | 0.0178 | 15 | 5 | lmer | 0.0565 | 0.0434 | 0.0696 | 2.77e-04 | 0.0420 | 0.0697 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 8.68e-04 | 0.0033 | 0.0060 | 15 | 5 | fallback_t | 8.68e-04 | -0.0033 | 0.0050 | 0.5911 | -0.0032 | 0.0042 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | brier | -0.0084 | 0.0065 | 0.0110 | 15 | 5 | lmer | -0.0084 | -0.0165 | -3.15e-04 | 0.0448 | -0.0114 | -0.0056 | 2000 |
| e4 | 50 | A | ssl_colon - ssl_imnet_matched | ece | -0.0023 | 0.0187 | 0.0329 | 15 | 5 | fallback_t | -0.0023 | -0.0256 | 0.0210 | 0.7970 | -0.0070 | 0.0040 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | auroc | 0.0522 | 0.0119 | 0.0133 | 15 | 5 | lmer | 0.0522 | 0.0375 | 0.0670 | 6.02e-04 | 0.0457 | 0.0594 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | auprc | 0.0901 | 0.0243 | 0.0172 | 15 | 5 | lmer | 0.0901 | 0.0598 | 0.1203 | 0.0012 | 0.0766 | 0.1005 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | balanced_accuracy | 0.0014 | 0.0062 | 0.0124 | 15 | 5 | fallback_t | 0.0014 | -0.0063 | 0.0091 | 0.6435 | -0.0034 | 0.0061 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | brier | -0.0086 | 0.0147 | 0.0148 | 15 | 5 | lmer | -0.0086 | -0.0268 | 0.0096 | 0.2597 | -0.0104 | -0.0070 | 2000 |
| e4 | 50 | B | ssl_colon - sup_imnet | ece | 0.0119 | 0.0369 | 0.0422 | 15 | 5 | lmer | 0.0119 | -0.0339 | 0.0577 | 0.5109 | 0.0074 | 0.0168 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | auroc | 0.0158 | 0.0033 | 0.0130 | 15 | 5 | fallback_t | 0.0158 | 0.0118 | 0.0199 | 4.13e-04 | 0.0096 | 0.0234 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | auprc | 0.0278 | 0.0097 | 0.0241 | 15 | 5 | fallback_t | 0.0278 | 0.0158 | 0.0399 | 0.0030 | 0.0157 | 0.0409 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | balanced_accuracy | -4.96e-04 | 0.0086 | 0.0131 | 15 | 5 | lmer | -4.96e-04 | -0.0112 | 0.0102 | 0.9035 | -0.0065 | 0.0053 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | brier | 0.0010 | 0.0172 | 0.0208 | 15 | 5 | lmer | 0.0010 | -0.0204 | 0.0224 | 0.9008 | -0.0014 | 0.0041 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet | ece | 0.0136 | 0.0449 | 0.0502 | 15 | 5 | lmer | 0.0136 | -0.0422 | 0.0693 | 0.5367 | 0.0102 | 0.0156 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0319 | 0.0048 | 0.0078 | 15 | 5 | lmer | 0.0319 | 0.0260 | 0.0379 | 1.16e-04 | 0.0239 | 0.0400 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0565 | 0.0105 | 0.0178 | 15 | 5 | lmer | 0.0565 | 0.0434 | 0.0696 | 2.77e-04 | 0.0420 | 0.0697 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0028 | 0.0069 | 0.0099 | 15 | 5 | lmer | 0.0028 | -0.0058 | 0.0113 | 0.4207 | -0.0028 | 0.0081 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | brier | -0.0084 | 0.0065 | 0.0110 | 15 | 5 | lmer | -0.0084 | -0.0165 | -3.15e-04 | 0.0448 | -0.0114 | -0.0056 | 2000 |
| e4 | 50 | B | ssl_colon - ssl_imnet_matched | ece | -0.0023 | 0.0187 | 0.0329 | 15 | 5 | fallback_t | -0.0023 | -0.0256 | 0.0210 | 0.7970 | -0.0070 | 0.0040 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | auroc | 0.0329 | 0.0054 | 0.0083 | 15 | 5 | lmer | 0.0329 | 0.0262 | 0.0396 | 1.71e-04 | 0.0275 | 0.0386 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | auprc | 0.0643 | 0.0111 | 0.0173 | 15 | 5 | lmer | 0.0643 | 0.0506 | 0.0781 | 2.02e-04 | 0.0503 | 0.0757 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | balanced_accuracy | -7.45e-04 | 0.0038 | 0.0125 | 15 | 5 | fallback_t | -7.45e-04 | -0.0055 | 0.0040 | 0.6850 | -0.0062 | 0.0053 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | brier | -0.0091 | 0.0136 | 0.0142 | 15 | 5 | lmer | -0.0091 | -0.0260 | 0.0078 | 0.2096 | -0.0111 | -0.0075 | 2000 |
| e4 | 50 | C | ssl_colon - sup_imnet | ece | 0.0163 | 0.0213 | 0.0242 | 15 | 5 | lmer | 0.0163 | -0.0101 | 0.0427 | 0.1618 | 0.0106 | 0.0192 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | auroc | 0.0115 | 0.0022 | 0.0078 | 15 | 5 | fallback_t | 0.0115 | 0.0088 | 0.0143 | 3.29e-04 | 0.0081 | 0.0147 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | auprc | 0.0236 | 0.0043 | 0.0147 | 15 | 5 | fallback_t | 0.0236 | 0.0183 | 0.0290 | 2.56e-04 | 0.0165 | 0.0302 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | balanced_accuracy | -2.67e-04 | 0.0050 | 0.0105 | 15 | 5 | fallback_t | -2.67e-04 | -0.0065 | 0.0059 | 0.9106 | -0.0028 | 0.0018 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | brier | 3.11e-04 | 0.0160 | 0.0198 | 15 | 5 | lmer | 3.11e-04 | -0.0195 | 0.0201 | 0.9673 | -0.0011 | 0.0021 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet | ece | 0.0087 | 0.0227 | 0.0309 | 15 | 5 | lmer | 0.0087 | -0.0194 | 0.0368 | 0.4388 | 0.0053 | 0.0123 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0176 | 0.0032 | 0.0072 | 15 | 5 | fallback_t | 0.0176 | 0.0136 | 0.0216 | 2.65e-04 | 0.0138 | 0.0215 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | auprc | 0.0365 | 0.0058 | 0.0195 | 15 | 5 | fallback_t | 0.0365 | 0.0293 | 0.0437 | 1.49e-04 | 0.0262 | 0.0459 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0059 | 0.0043 | 0.0095 | 15 | 5 | fallback_t | 0.0059 | 5.76e-04 | 0.0112 | 0.0370 | 0.0021 | 0.0095 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | brier | -0.0084 | 0.0058 | 0.0118 | 15 | 5 | fallback_t | -0.0084 | -0.0155 | -0.0012 | 0.0315 | -0.0112 | -0.0063 | 2000 |
| e4 | 50 | C | ssl_colon - ssl_imnet_matched | ece | 0.0030 | 0.0111 | 0.0186 | 15 | 5 | lmer | 0.0030 | -0.0108 | 0.0168 | 0.5809 | -0.0030 | 0.0077 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | auroc | 0.0329 | 0.0054 | 0.0083 | 15 | 5 | lmer | 0.0329 | 0.0262 | 0.0396 | 1.71e-04 | 0.0275 | 0.0386 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | auprc | 0.0643 | 0.0111 | 0.0173 | 15 | 5 | lmer | 0.0643 | 0.0506 | 0.0781 | 2.02e-04 | 0.0503 | 0.0757 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | balanced_accuracy | 3.29e-04 | 0.0041 | 0.0127 | 15 | 5 | fallback_t | 3.29e-04 | -0.0048 | 0.0055 | 0.8673 | -0.0040 | 0.0045 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | brier | -0.0091 | 0.0136 | 0.0142 | 15 | 5 | lmer | -0.0091 | -0.0260 | 0.0078 | 0.2096 | -0.0111 | -0.0075 | 2000 |
| e4 | 50 | D | ssl_colon - sup_imnet | ece | 0.0163 | 0.0213 | 0.0242 | 15 | 5 | lmer | 0.0163 | -0.0101 | 0.0427 | 0.1618 | 0.0106 | 0.0192 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | auroc | 0.0115 | 0.0022 | 0.0078 | 15 | 5 | fallback_t | 0.0115 | 0.0088 | 0.0143 | 3.29e-04 | 0.0081 | 0.0147 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | auprc | 0.0236 | 0.0043 | 0.0147 | 15 | 5 | fallback_t | 0.0236 | 0.0183 | 0.0290 | 2.56e-04 | 0.0165 | 0.0302 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | balanced_accuracy | -3.12e-04 | 0.0050 | 0.0108 | 15 | 5 | fallback_t | -3.12e-04 | -0.0065 | 0.0058 | 0.8951 | -0.0031 | 0.0033 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | brier | 3.11e-04 | 0.0160 | 0.0198 | 15 | 5 | lmer | 3.11e-04 | -0.0195 | 0.0201 | 0.9673 | -0.0011 | 0.0021 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet | ece | 0.0087 | 0.0227 | 0.0309 | 15 | 5 | lmer | 0.0087 | -0.0194 | 0.0368 | 0.4388 | 0.0053 | 0.0123 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0176 | 0.0032 | 0.0072 | 15 | 5 | fallback_t | 0.0176 | 0.0136 | 0.0216 | 2.65e-04 | 0.0138 | 0.0215 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | auprc | 0.0365 | 0.0058 | 0.0195 | 15 | 5 | fallback_t | 0.0365 | 0.0293 | 0.0437 | 1.49e-04 | 0.0262 | 0.0459 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0049 | 0.0038 | 0.0089 | 15 | 5 | fallback_t | 0.0049 | 2.13e-04 | 0.0097 | 0.0440 | 0.0014 | 0.0081 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | brier | -0.0084 | 0.0058 | 0.0118 | 15 | 5 | fallback_t | -0.0084 | -0.0155 | -0.0012 | 0.0315 | -0.0112 | -0.0063 | 2000 |
| e4 | 50 | D | ssl_colon - ssl_imnet_matched | ece | 0.0030 | 0.0111 | 0.0186 | 15 | 5 | lmer | 0.0030 | -0.0108 | 0.0168 | 0.5809 | -0.0030 | 0.0077 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | auroc | 0.0514 | 0.0105 | 0.0130 | 15 | 5 | lmer | 0.0514 | 0.0384 | 0.0644 | 3.92e-04 | 0.0450 | 0.0586 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | auprc | 0.0903 | 0.0234 | 0.0167 | 15 | 5 | lmer | 0.0903 | 0.0613 | 0.1193 | 9.88e-04 | 0.0762 | 0.1012 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | balanced_accuracy | 1.19e-04 | 0.0038 | 0.0114 | 15 | 5 | fallback_t | 1.19e-04 | -0.0046 | 0.0049 | 0.9481 | -0.0024 | 0.0026 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | brier | -0.0086 | 0.0144 | 0.0146 | 15 | 5 | lmer | -0.0086 | -0.0265 | 0.0093 | 0.2532 | -0.0104 | -0.0070 | 2000 |
| e4 | 100 | A | ssl_colon - sup_imnet | ece | 0.0119 | 0.0367 | 0.0415 | 15 | 5 | lmer | 0.0119 | -0.0337 | 0.0574 | 0.5096 | 0.0071 | 0.0170 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | auroc | 0.0155 | 0.0031 | 0.0126 | 15 | 5 | fallback_t | 0.0155 | 0.0117 | 0.0193 | 3.56e-04 | 0.0095 | 0.0226 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | auprc | 0.0280 | 0.0103 | 0.0231 | 15 | 5 | fallback_t | 0.0280 | 0.0153 | 0.0408 | 0.0037 | 0.0164 | 0.0404 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | balanced_accuracy | 9.68e-05 | 0.0057 | 0.0111 | 15 | 5 | fallback_t | 9.68e-05 | -0.0069 | 0.0071 | 0.9713 | -0.0036 | 0.0042 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | brier | 9.39e-04 | 0.0169 | 0.0206 | 15 | 5 | lmer | 9.39e-04 | -0.0200 | 0.0219 | 0.9070 | -0.0014 | 0.0038 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet | ece | 0.0133 | 0.0445 | 0.0504 | 15 | 5 | lmer | 0.0133 | -0.0420 | 0.0685 | 0.5417 | 0.0101 | 0.0154 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0316 | 0.0049 | 0.0076 | 15 | 5 | lmer | 0.0316 | 0.0255 | 0.0377 | 1.33e-04 | 0.0237 | 0.0394 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0569 | 0.0099 | 0.0179 | 15 | 5 | fallback_t | 0.0569 | 0.0445 | 0.0692 | 2.14e-04 | 0.0430 | 0.0693 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0014 | 0.0035 | 0.0057 | 15 | 5 | lmer | 0.0014 | -0.0029 | 0.0058 | 0.4134 | -0.0022 | 0.0044 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | brier | -0.0083 | 0.0063 | 0.0110 | 15 | 5 | fallback_t | -0.0083 | -0.0161 | -5.45e-04 | 0.0411 | -0.0114 | -0.0055 | 2000 |
| e4 | 100 | A | ssl_colon - ssl_imnet_matched | ece | -0.0020 | 0.0184 | 0.0331 | 15 | 5 | fallback_t | -0.0020 | -0.0249 | 0.0209 | 0.8185 | -0.0071 | 0.0049 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | auroc | 0.0514 | 0.0105 | 0.0130 | 15 | 5 | lmer | 0.0514 | 0.0384 | 0.0644 | 3.92e-04 | 0.0450 | 0.0586 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | auprc | 0.0903 | 0.0234 | 0.0167 | 15 | 5 | lmer | 0.0903 | 0.0613 | 0.1193 | 9.88e-04 | 0.0762 | 0.1012 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | balanced_accuracy | -0.0020 | 0.0087 | 0.0146 | 15 | 5 | lmer | -0.0020 | -0.0128 | 0.0089 | 0.6410 | -0.0060 | 0.0023 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | brier | -0.0086 | 0.0144 | 0.0146 | 15 | 5 | lmer | -0.0086 | -0.0265 | 0.0093 | 0.2532 | -0.0104 | -0.0070 | 2000 |
| e4 | 100 | B | ssl_colon - sup_imnet | ece | 0.0119 | 0.0367 | 0.0415 | 15 | 5 | lmer | 0.0119 | -0.0337 | 0.0574 | 0.5096 | 0.0071 | 0.0170 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | auroc | 0.0155 | 0.0031 | 0.0126 | 15 | 5 | fallback_t | 0.0155 | 0.0117 | 0.0193 | 3.56e-04 | 0.0095 | 0.0226 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | auprc | 0.0280 | 0.0103 | 0.0231 | 15 | 5 | fallback_t | 0.0280 | 0.0153 | 0.0408 | 0.0037 | 0.0164 | 0.0404 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | balanced_accuracy | -0.0027 | 0.0077 | 0.0128 | 15 | 5 | lmer | -0.0027 | -0.0123 | 0.0069 | 0.4763 | -0.0078 | 0.0025 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | brier | 9.39e-04 | 0.0169 | 0.0206 | 15 | 5 | lmer | 9.39e-04 | -0.0200 | 0.0219 | 0.9070 | -0.0014 | 0.0038 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet | ece | 0.0133 | 0.0445 | 0.0504 | 15 | 5 | lmer | 0.0133 | -0.0420 | 0.0685 | 0.5417 | 0.0101 | 0.0154 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0316 | 0.0049 | 0.0076 | 15 | 5 | lmer | 0.0316 | 0.0255 | 0.0377 | 1.33e-04 | 0.0237 | 0.0394 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0569 | 0.0099 | 0.0179 | 15 | 5 | fallback_t | 0.0569 | 0.0445 | 0.0692 | 2.14e-04 | 0.0430 | 0.0693 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 8.28e-04 | 0.0065 | 0.0113 | 15 | 5 | fallback_t | 8.28e-04 | -0.0073 | 0.0089 | 0.7902 | -0.0047 | 0.0058 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | brier | -0.0083 | 0.0063 | 0.0110 | 15 | 5 | fallback_t | -0.0083 | -0.0161 | -5.45e-04 | 0.0411 | -0.0114 | -0.0055 | 2000 |
| e4 | 100 | B | ssl_colon - ssl_imnet_matched | ece | -0.0020 | 0.0184 | 0.0331 | 15 | 5 | fallback_t | -0.0020 | -0.0249 | 0.0209 | 0.8185 | -0.0071 | 0.0049 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | auroc | 0.0334 | 0.0037 | 0.0092 | 15 | 5 | fallback_t | 0.0334 | 0.0288 | 0.0380 | 3.55e-05 | 0.0276 | 0.0396 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | auprc | 0.0638 | 0.0052 | 0.0220 | 15 | 5 | fallback_t | 0.0638 | 0.0573 | 0.0703 | 1.07e-05 | 0.0503 | 0.0775 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | balanced_accuracy | 9.46e-04 | 0.0091 | 0.0130 | 15 | 5 | lmer | 9.46e-04 | -0.0104 | 0.0123 | 0.8279 | -0.0065 | 0.0075 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | brier | -0.0091 | 0.0138 | 0.0151 | 15 | 5 | lmer | -0.0091 | -0.0262 | 0.0080 | 0.2150 | -0.0118 | -0.0067 | 2000 |
| e4 | 100 | C | ssl_colon - sup_imnet | ece | 0.0165 | 0.0238 | 0.0257 | 15 | 5 | lmer | 0.0165 | -0.0131 | 0.0461 | 0.1962 | 0.0105 | 0.0203 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | auroc | 0.0111 | 0.0038 | 0.0053 | 15 | 5 | lmer | 0.0111 | 0.0063 | 0.0158 | 0.0029 | 0.0086 | 0.0139 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | auprc | 0.0244 | 0.0050 | 0.0147 | 15 | 5 | fallback_t | 0.0244 | 0.0182 | 0.0306 | 4.02e-04 | 0.0163 | 0.0319 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | balanced_accuracy | 0.0027 | 0.0076 | 0.0094 | 15 | 5 | lmer | 0.0027 | -0.0067 | 0.0121 | 0.4678 | -0.0024 | 0.0064 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | brier | -1.86e-04 | 0.0154 | 0.0201 | 15 | 5 | lmer | -1.86e-04 | -0.0193 | 0.0189 | 0.9797 | -0.0017 | 0.0015 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet | ece | 0.0096 | 0.0259 | 0.0296 | 15 | 5 | lmer | 0.0096 | -0.0225 | 0.0418 | 0.4523 | 0.0059 | 0.0123 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0182 | 0.0046 | 0.0074 | 15 | 5 | lmer | 0.0182 | 0.0125 | 0.0239 | 8.82e-04 | 0.0148 | 0.0221 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | auprc | 0.0348 | 0.0055 | 0.0174 | 15 | 5 | fallback_t | 0.0348 | 0.0279 | 0.0416 | 1.49e-04 | 0.0255 | 0.0428 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0016 | 0.0083 | 0.0134 | 15 | 5 | lmer | 0.0016 | -0.0087 | 0.0119 | 0.6890 | -0.0017 | 0.0041 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | brier | -0.0072 | 0.0069 | 0.0110 | 15 | 5 | lmer | -0.0072 | -0.0158 | 0.0014 | 0.0796 | -0.0099 | -0.0053 | 2000 |
| e4 | 100 | C | ssl_colon - ssl_imnet_matched | ece | 0.0059 | 0.0103 | 0.0157 | 15 | 5 | lmer | 0.0059 | -0.0068 | 0.0187 | 0.2654 | -0.0024 | 0.0126 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | auroc | 0.0334 | 0.0037 | 0.0092 | 15 | 5 | fallback_t | 0.0334 | 0.0288 | 0.0380 | 3.55e-05 | 0.0276 | 0.0396 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | auprc | 0.0638 | 0.0052 | 0.0220 | 15 | 5 | fallback_t | 0.0638 | 0.0573 | 0.0703 | 1.07e-05 | 0.0503 | 0.0775 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | balanced_accuracy | 0.0011 | 0.0098 | 0.0123 | 15 | 5 | lmer | 0.0011 | -0.0111 | 0.0134 | 0.8073 | -0.0060 | 0.0089 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | brier | -0.0091 | 0.0138 | 0.0151 | 15 | 5 | lmer | -0.0091 | -0.0262 | 0.0080 | 0.2150 | -0.0118 | -0.0067 | 2000 |
| e4 | 100 | D | ssl_colon - sup_imnet | ece | 0.0165 | 0.0238 | 0.0257 | 15 | 5 | lmer | 0.0165 | -0.0131 | 0.0461 | 0.1962 | 0.0105 | 0.0203 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | auroc | 0.0111 | 0.0038 | 0.0053 | 15 | 5 | lmer | 0.0111 | 0.0063 | 0.0158 | 0.0029 | 0.0086 | 0.0139 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | auprc | 0.0244 | 0.0050 | 0.0147 | 15 | 5 | fallback_t | 0.0244 | 0.0182 | 0.0306 | 4.02e-04 | 0.0163 | 0.0319 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | balanced_accuracy | 0.0024 | 0.0072 | 0.0091 | 15 | 5 | lmer | 0.0024 | -0.0066 | 0.0114 | 0.4993 | -0.0023 | 0.0067 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | brier | -1.86e-04 | 0.0154 | 0.0201 | 15 | 5 | lmer | -1.86e-04 | -0.0193 | 0.0189 | 0.9797 | -0.0017 | 0.0015 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet | ece | 0.0096 | 0.0259 | 0.0296 | 15 | 5 | lmer | 0.0096 | -0.0225 | 0.0418 | 0.4523 | 0.0059 | 0.0123 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0182 | 0.0046 | 0.0074 | 15 | 5 | lmer | 0.0182 | 0.0125 | 0.0239 | 8.82e-04 | 0.0148 | 0.0221 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | auprc | 0.0348 | 0.0055 | 0.0174 | 15 | 5 | fallback_t | 0.0348 | 0.0279 | 0.0416 | 1.49e-04 | 0.0255 | 0.0428 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 6.53e-04 | 0.0086 | 0.0140 | 15 | 5 | lmer | 6.53e-04 | -0.0100 | 0.0113 | 0.8729 | -0.0021 | 0.0031 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | brier | -0.0072 | 0.0069 | 0.0110 | 15 | 5 | lmer | -0.0072 | -0.0158 | 0.0014 | 0.0796 | -0.0099 | -0.0053 | 2000 |
| e4 | 100 | D | ssl_colon - ssl_imnet_matched | ece | 0.0059 | 0.0103 | 0.0157 | 15 | 5 | lmer | 0.0059 | -0.0068 | 0.0187 | 0.2654 | -0.0024 | 0.0126 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | auroc | 0.0521 | 0.0114 | 0.0136 | 15 | 5 | lmer | 0.0521 | 0.0379 | 0.0663 | 5.20e-04 | 0.0454 | 0.0594 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | auprc | 0.0909 | 0.0234 | 0.0177 | 15 | 5 | lmer | 0.0909 | 0.0617 | 0.1200 | 9.76e-04 | 0.0756 | 0.1026 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | balanced_accuracy | -4.25e-04 | 0.0043 | 0.0119 | 15 | 5 | fallback_t | -4.25e-04 | -0.0058 | 0.0049 | 0.8359 | -0.0028 | 0.0024 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | brier | -0.0085 | 0.0136 | 0.0139 | 15 | 5 | lmer | -0.0085 | -0.0253 | 0.0083 | 0.2332 | -0.0101 | -0.0071 | 2000 |
| e4 | 200 | A | ssl_colon - sup_imnet | ece | 0.0125 | 0.0361 | 0.0409 | 15 | 5 | lmer | 0.0125 | -0.0323 | 0.0574 | 0.4806 | 0.0079 | 0.0173 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | auroc | 0.0166 | 0.0031 | 0.0134 | 15 | 5 | fallback_t | 0.0166 | 0.0128 | 0.0204 | 2.72e-04 | 0.0101 | 0.0243 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | auprc | 0.0298 | 0.0105 | 0.0250 | 15 | 5 | fallback_t | 0.0298 | 0.0168 | 0.0428 | 0.0031 | 0.0167 | 0.0437 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | balanced_accuracy | 5.75e-04 | 0.0066 | 0.0106 | 15 | 5 | lmer | 5.75e-04 | -0.0076 | 0.0088 | 0.8554 | -0.0041 | 0.0055 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | brier | 5.81e-04 | 0.0159 | 0.0196 | 15 | 5 | lmer | 5.81e-04 | -0.0191 | 0.0203 | 0.9387 | -0.0020 | 0.0038 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet | ece | 0.0135 | 0.0434 | 0.0497 | 15 | 5 | lmer | 0.0135 | -0.0404 | 0.0675 | 0.5240 | 0.0099 | 0.0157 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0317 | 0.0050 | 0.0083 | 15 | 5 | lmer | 0.0317 | 0.0255 | 0.0380 | 1.47e-04 | 0.0229 | 0.0403 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0572 | 0.0099 | 0.0176 | 15 | 5 | fallback_t | 0.0572 | 0.0449 | 0.0695 | 2.09e-04 | 0.0413 | 0.0715 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0013 | 0.0040 | 0.0060 | 15 | 5 | lmer | 0.0013 | -0.0036 | 0.0063 | 0.4967 | -0.0034 | 0.0054 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | brier | -0.0080 | 0.0057 | 0.0101 | 15 | 5 | fallback_t | -0.0080 | -0.0151 | -0.0010 | 0.0338 | -0.0113 | -0.0049 | 2000 |
| e4 | 200 | A | ssl_colon - ssl_imnet_matched | ece | -0.0015 | 0.0180 | 0.0326 | 15 | 5 | fallback_t | -0.0015 | -0.0238 | 0.0208 | 0.8643 | -0.0065 | 0.0054 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | auroc | 0.0521 | 0.0114 | 0.0136 | 15 | 5 | lmer | 0.0521 | 0.0379 | 0.0663 | 5.20e-04 | 0.0454 | 0.0594 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | auprc | 0.0909 | 0.0234 | 0.0177 | 15 | 5 | lmer | 0.0909 | 0.0617 | 0.1200 | 9.76e-04 | 0.0756 | 0.1026 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | balanced_accuracy | -0.0021 | 0.0104 | 0.0145 | 15 | 5 | lmer | -0.0021 | -0.0150 | 0.0107 | 0.6718 | -0.0072 | 0.0033 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | brier | -0.0085 | 0.0136 | 0.0139 | 15 | 5 | lmer | -0.0085 | -0.0253 | 0.0083 | 0.2332 | -0.0101 | -0.0071 | 2000 |
| e4 | 200 | B | ssl_colon - sup_imnet | ece | 0.0125 | 0.0361 | 0.0409 | 15 | 5 | lmer | 0.0125 | -0.0323 | 0.0574 | 0.4806 | 0.0079 | 0.0173 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | auroc | 0.0166 | 0.0031 | 0.0134 | 15 | 5 | fallback_t | 0.0166 | 0.0128 | 0.0204 | 2.72e-04 | 0.0101 | 0.0243 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | auprc | 0.0298 | 0.0105 | 0.0250 | 15 | 5 | fallback_t | 0.0298 | 0.0168 | 0.0428 | 0.0031 | 0.0167 | 0.0437 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | balanced_accuracy | -1.45e-05 | 0.0067 | 0.0125 | 15 | 5 | fallback_t | -1.45e-05 | -0.0084 | 0.0084 | 0.9964 | -0.0069 | 0.0069 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | brier | 5.81e-04 | 0.0159 | 0.0196 | 15 | 5 | lmer | 5.81e-04 | -0.0191 | 0.0203 | 0.9387 | -0.0020 | 0.0038 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet | ece | 0.0135 | 0.0434 | 0.0497 | 15 | 5 | lmer | 0.0135 | -0.0404 | 0.0675 | 0.5240 | 0.0099 | 0.0157 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0317 | 0.0050 | 0.0083 | 15 | 5 | lmer | 0.0317 | 0.0255 | 0.0380 | 1.47e-04 | 0.0229 | 0.0403 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0572 | 0.0099 | 0.0176 | 15 | 5 | fallback_t | 0.0572 | 0.0449 | 0.0695 | 2.09e-04 | 0.0413 | 0.0715 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0017 | 0.0063 | 0.0108 | 15 | 5 | lmer | 0.0017 | -0.0061 | 0.0095 | 0.5811 | -0.0055 | 0.0087 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | brier | -0.0080 | 0.0057 | 0.0101 | 15 | 5 | fallback_t | -0.0080 | -0.0151 | -0.0010 | 0.0338 | -0.0113 | -0.0049 | 2000 |
| e4 | 200 | B | ssl_colon - ssl_imnet_matched | ece | -0.0015 | 0.0180 | 0.0326 | 15 | 5 | fallback_t | -0.0015 | -0.0238 | 0.0208 | 0.8643 | -0.0065 | 0.0054 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | auroc | 0.0331 | 0.0068 | 0.0099 | 15 | 5 | lmer | 0.0331 | 0.0247 | 0.0415 | 3.95e-04 | 0.0278 | 0.0386 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | auprc | 0.0665 | 0.0113 | 0.0196 | 15 | 5 | lmer | 0.0665 | 0.0525 | 0.0805 | 1.83e-04 | 0.0522 | 0.0779 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | balanced_accuracy | 0.0010 | 0.0076 | 0.0081 | 15 | 5 | lmer | 0.0010 | -0.0084 | 0.0104 | 0.7786 | -0.0047 | 0.0063 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | brier | -0.0092 | 0.0132 | 0.0145 | 15 | 5 | lmer | -0.0092 | -0.0256 | 0.0072 | 0.1944 | -0.0118 | -0.0072 | 2000 |
| e4 | 200 | C | ssl_colon - sup_imnet | ece | 0.0154 | 0.0161 | 0.0254 | 15 | 5 | lmer | 0.0154 | -0.0045 | 0.0354 | 0.0983 | 0.0114 | 0.0185 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | auroc | 0.0082 | 0.0043 | 0.0062 | 15 | 5 | lmer | 0.0082 | 0.0029 | 0.0135 | 0.0125 | 0.0047 | 0.0123 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | auprc | 0.0165 | 0.0074 | 0.0135 | 15 | 5 | fallback_t | 0.0165 | 0.0073 | 0.0256 | 0.0075 | 0.0088 | 0.0245 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | balanced_accuracy | -7.74e-05 | 0.0075 | 0.0102 | 15 | 5 | lmer | -7.74e-05 | -0.0094 | 0.0092 | 0.9827 | -0.0054 | 0.0049 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | brier | 8.40e-04 | 0.0151 | 0.0184 | 15 | 5 | lmer | 8.40e-04 | -0.0179 | 0.0196 | 0.9068 | -7.93e-04 | 0.0029 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet | ece | 0.0070 | 0.0178 | 0.0302 | 15 | 5 | lmer | 0.0070 | -0.0151 | 0.0291 | 0.4264 | 0.0026 | 0.0117 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0171 | 0.0050 | 0.0086 | 15 | 5 | fallback_t | 0.0171 | 0.0109 | 0.0233 | 0.0016 | 0.0132 | 0.0211 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | auprc | 0.0348 | 0.0094 | 0.0229 | 15 | 5 | fallback_t | 0.0348 | 0.0232 | 0.0465 | 0.0012 | 0.0243 | 0.0446 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0040 | 0.0030 | 0.0092 | 15 | 5 | fallback_t | 0.0040 | 2.61e-04 | 0.0077 | 0.0411 | -4.01e-04 | 0.0079 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | brier | -0.0076 | 0.0070 | 0.0132 | 15 | 5 | fallback_t | -0.0076 | -0.0163 | 0.0011 | 0.0721 | -0.0101 | -0.0057 | 2000 |
| e4 | 200 | C | ssl_colon - ssl_imnet_matched | ece | 0.0042 | 0.0059 | 0.0209 | 15 | 5 | fallback_t | 0.0042 | -0.0031 | 0.0115 | 0.1851 | -0.0016 | 0.0099 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | auroc | 0.0331 | 0.0068 | 0.0099 | 15 | 5 | lmer | 0.0331 | 0.0247 | 0.0415 | 3.95e-04 | 0.0278 | 0.0386 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | auprc | 0.0665 | 0.0113 | 0.0196 | 15 | 5 | lmer | 0.0665 | 0.0525 | 0.0805 | 1.83e-04 | 0.0522 | 0.0779 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | balanced_accuracy | -7.09e-04 | 0.0061 | 0.0068 | 15 | 5 | lmer | -7.09e-04 | -0.0082 | 0.0068 | 0.8063 | -0.0062 | 0.0042 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | brier | -0.0092 | 0.0132 | 0.0145 | 15 | 5 | lmer | -0.0092 | -0.0256 | 0.0072 | 0.1944 | -0.0118 | -0.0072 | 2000 |
| e4 | 200 | D | ssl_colon - sup_imnet | ece | 0.0154 | 0.0161 | 0.0254 | 15 | 5 | lmer | 0.0154 | -0.0045 | 0.0354 | 0.0983 | 0.0114 | 0.0185 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | auroc | 0.0082 | 0.0043 | 0.0062 | 15 | 5 | lmer | 0.0082 | 0.0029 | 0.0135 | 0.0125 | 0.0047 | 0.0123 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | auprc | 0.0165 | 0.0074 | 0.0135 | 15 | 5 | fallback_t | 0.0165 | 0.0073 | 0.0256 | 0.0075 | 0.0088 | 0.0245 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | balanced_accuracy | -8.76e-04 | 0.0090 | 0.0122 | 15 | 5 | lmer | -8.76e-04 | -0.0121 | 0.0103 | 0.8384 | -0.0074 | 0.0050 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | brier | 8.40e-04 | 0.0151 | 0.0184 | 15 | 5 | lmer | 8.40e-04 | -0.0179 | 0.0196 | 0.9068 | -7.93e-04 | 0.0029 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet | ece | 0.0070 | 0.0178 | 0.0302 | 15 | 5 | lmer | 0.0070 | -0.0151 | 0.0291 | 0.4264 | 0.0026 | 0.0117 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0171 | 0.0050 | 0.0086 | 15 | 5 | fallback_t | 0.0171 | 0.0109 | 0.0233 | 0.0016 | 0.0132 | 0.0211 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | auprc | 0.0348 | 0.0094 | 0.0229 | 15 | 5 | fallback_t | 0.0348 | 0.0232 | 0.0465 | 0.0012 | 0.0243 | 0.0446 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0027 | 0.0044 | 0.0103 | 15 | 5 | fallback_t | 0.0027 | -0.0027 | 0.0081 | 0.2407 | -0.0019 | 0.0066 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | brier | -0.0076 | 0.0070 | 0.0132 | 15 | 5 | fallback_t | -0.0076 | -0.0163 | 0.0011 | 0.0721 | -0.0101 | -0.0057 | 2000 |
| e4 | 200 | D | ssl_colon - ssl_imnet_matched | ece | 0.0042 | 0.0059 | 0.0209 | 15 | 5 | fallback_t | 0.0042 | -0.0031 | 0.0115 | 0.1851 | -0.0016 | 0.0099 | 2000 |
| e4 | 500 | A | ssl_colon - sup_imnet | auroc | 0.0527 | 0.0147 | 0.0185 | 15 | 5 | lmer | 0.0527 | 0.0344 | 0.0709 | 0.0013 | 0.0441 | 0.0615 | 1274 |
| e4 | 500 | A | ssl_colon - sup_imnet | auprc | 0.0443 | 0.0164 | 0.0214 | 15 | 5 | lmer | 0.0443 | 0.0240 | 0.0647 | 0.0038 | 0.0305 | 0.0569 | 1274 |
| e4 | 500 | A | ssl_colon - sup_imnet | balanced_accuracy | -7.58e-04 | 0.0057 | 0.0138 | 15 | 5 | fallback_t | -7.58e-04 | -0.0079 | 0.0064 | 0.7822 | -0.0033 | 0.0020 | 1274 |
| e4 | 500 | A | ssl_colon - sup_imnet | brier | -0.0150 | 0.0413 | 0.0471 | 15 | 5 | lmer | -0.0150 | -0.0663 | 0.0363 | 0.4620 | -0.0239 | -0.0077 | 2000 |
| e4 | 500 | A | ssl_colon - sup_imnet | ece | -0.0029 | 0.0499 | 0.0576 | 15 | 5 | lmer | -0.0029 | -0.0649 | 0.0592 | 0.9046 | -0.0088 | 0.0015 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet | auroc | 0.0165 | 0.0081 | 0.0139 | 15 | 5 | lmer | 0.0165 | 0.0065 | 0.0265 | 0.0103 | 0.0096 | 0.0253 | 1274 |
| e4 | 500 | A | ssl_colon - ssl_imnet | auprc | 0.0144 | 0.0120 | 0.0172 | 15 | 5 | lmer | 0.0144 | -4.70e-04 | 0.0293 | 0.0547 | 0.0073 | 0.0233 | 1274 |
| e4 | 500 | A | ssl_colon - ssl_imnet | balanced_accuracy | -0.0010 | 0.0047 | 0.0135 | 15 | 5 | fallback_t | -0.0010 | -0.0068 | 0.0047 | 0.6421 | -0.0054 | 0.0028 | 1274 |
| e4 | 500 | A | ssl_colon - ssl_imnet | brier | 0.0047 | 0.0508 | 0.0620 | 15 | 5 | lmer | 0.0047 | -0.0584 | 0.0677 | 0.8473 | -0.0045 | 0.0158 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet | ece | 0.0143 | 0.0608 | 0.0707 | 15 | 5 | lmer | 0.0143 | -0.0612 | 0.0898 | 0.6270 | 0.0100 | 0.0183 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | auroc | 0.0321 | 0.0066 | 0.0190 | 15 | 5 | fallback_t | 0.0321 | 0.0239 | 0.0402 | 3.98e-04 | 0.0238 | 0.0409 | 1274 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | auprc | 0.0273 | 0.0075 | 0.0177 | 15 | 5 | fallback_t | 0.0273 | 0.0180 | 0.0366 | 0.0012 | 0.0173 | 0.0371 | 1274 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | balanced_accuracy | 0.0021 | 0.0051 | 0.0054 | 15 | 5 | lmer | 0.0021 | -0.0043 | 0.0084 | 0.4199 | -0.0017 | 0.0059 | 1274 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | brier | -0.0209 | 0.0227 | 0.0355 | 15 | 5 | lmer | -0.0209 | -0.0491 | 0.0074 | 0.1094 | -0.0304 | -0.0139 | 2000 |
| e4 | 500 | A | ssl_colon - ssl_imnet_matched | ece | -0.0165 | 0.0262 | 0.0443 | 15 | 5 | lmer | -0.0165 | -0.0490 | 0.0160 | 0.2315 | -0.0220 | -0.0119 | 2000 |
| e4 | 500 | B | ssl_colon - sup_imnet | auroc | 0.0527 | 0.0147 | 0.0185 | 15 | 5 | lmer | 0.0527 | 0.0344 | 0.0709 | 0.0013 | 0.0441 | 0.0615 | 1274 |
| e4 | 500 | B | ssl_colon - sup_imnet | auprc | 0.0443 | 0.0164 | 0.0214 | 15 | 5 | lmer | 0.0443 | 0.0240 | 0.0647 | 0.0038 | 0.0305 | 0.0569 | 1274 |
| e4 | 500 | B | ssl_colon - sup_imnet | balanced_accuracy | -3.98e-04 | 0.0111 | 0.0164 | 15 | 5 | lmer | -3.98e-04 | -0.0141 | 0.0133 | 0.9397 | -0.0053 | 0.0052 | 1274 |
| e4 | 500 | B | ssl_colon - sup_imnet | brier | -0.0150 | 0.0413 | 0.0471 | 15 | 5 | lmer | -0.0150 | -0.0663 | 0.0363 | 0.4620 | -0.0239 | -0.0077 | 2000 |
| e4 | 500 | B | ssl_colon - sup_imnet | ece | -0.0029 | 0.0499 | 0.0576 | 15 | 5 | lmer | -0.0029 | -0.0649 | 0.0592 | 0.9046 | -0.0088 | 0.0015 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet | auroc | 0.0165 | 0.0081 | 0.0139 | 15 | 5 | lmer | 0.0165 | 0.0065 | 0.0265 | 0.0103 | 0.0096 | 0.0253 | 1274 |
| e4 | 500 | B | ssl_colon - ssl_imnet | auprc | 0.0144 | 0.0120 | 0.0172 | 15 | 5 | lmer | 0.0144 | -4.70e-04 | 0.0293 | 0.0547 | 0.0073 | 0.0233 | 1274 |
| e4 | 500 | B | ssl_colon - ssl_imnet | balanced_accuracy | -2.87e-04 | 0.0108 | 0.0124 | 15 | 5 | lmer | -2.87e-04 | -0.0137 | 0.0132 | 0.9556 | -0.0053 | 0.0049 | 1274 |
| e4 | 500 | B | ssl_colon - ssl_imnet | brier | 0.0047 | 0.0508 | 0.0620 | 15 | 5 | lmer | 0.0047 | -0.0584 | 0.0677 | 0.8473 | -0.0045 | 0.0158 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet | ece | 0.0143 | 0.0608 | 0.0707 | 15 | 5 | lmer | 0.0143 | -0.0612 | 0.0898 | 0.6270 | 0.0100 | 0.0183 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | auroc | 0.0321 | 0.0066 | 0.0190 | 15 | 5 | fallback_t | 0.0321 | 0.0239 | 0.0402 | 3.98e-04 | 0.0238 | 0.0409 | 1274 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | auprc | 0.0273 | 0.0075 | 0.0177 | 15 | 5 | fallback_t | 0.0273 | 0.0180 | 0.0366 | 0.0012 | 0.0173 | 0.0371 | 1274 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | balanced_accuracy | 8.45e-04 | 0.0075 | 0.0090 | 15 | 5 | lmer | 8.45e-04 | -0.0084 | 0.0101 | 0.8127 | -0.0050 | 0.0072 | 1274 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | brier | -0.0209 | 0.0227 | 0.0355 | 15 | 5 | lmer | -0.0209 | -0.0491 | 0.0074 | 0.1094 | -0.0304 | -0.0139 | 2000 |
| e4 | 500 | B | ssl_colon - ssl_imnet_matched | ece | -0.0165 | 0.0262 | 0.0443 | 15 | 5 | lmer | -0.0165 | -0.0490 | 0.0160 | 0.2315 | -0.0220 | -0.0119 | 2000 |
| e4 | 500 | C | ssl_colon - sup_imnet | auroc | 0.0318 | 0.0077 | 0.0185 | 15 | 5 | fallback_t | 0.0318 | 0.0223 | 0.0413 | 7.55e-04 | 0.0249 | 0.0381 | 1274 |
| e4 | 500 | C | ssl_colon - sup_imnet | auprc | 0.0267 | 0.0091 | 0.0170 | 15 | 5 | fallback_t | 0.0267 | 0.0155 | 0.0380 | 0.0028 | 0.0170 | 0.0361 | 1274 |
| e4 | 500 | C | ssl_colon - sup_imnet | balanced_accuracy | -6.97e-04 | 0.0045 | 0.0127 | 15 | 5 | fallback_t | -6.97e-04 | -0.0063 | 0.0049 | 0.7470 | -0.0053 | 0.0042 | 1274 |
| e4 | 500 | C | ssl_colon - sup_imnet | brier | -0.0165 | 0.0393 | 0.0525 | 15 | 5 | lmer | -0.0165 | -0.0653 | 0.0323 | 0.4002 | -0.0258 | -0.0084 | 2000 |
| e4 | 500 | C | ssl_colon - sup_imnet | ece | -0.0063 | 0.0503 | 0.0670 | 15 | 5 | lmer | -0.0063 | -0.0687 | 0.0562 | 0.7947 | -0.0134 | 0.0029 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet | auroc | 0.0098 | 0.0048 | 0.0054 | 15 | 5 | lmer | 0.0098 | 0.0039 | 0.0157 | 0.0099 | 0.0047 | 0.0153 | 1274 |
| e4 | 500 | C | ssl_colon - ssl_imnet | auprc | 0.0082 | 0.0052 | 0.0063 | 15 | 5 | lmer | 0.0082 | 0.0018 | 0.0146 | 0.0239 | 0.0035 | 0.0122 | 1274 |
| e4 | 500 | C | ssl_colon - ssl_imnet | balanced_accuracy | -0.0029 | 0.0125 | 0.0154 | 15 | 5 | lmer | -0.0029 | -0.0184 | 0.0125 | 0.6250 | -0.0092 | 0.0019 | 1274 |
| e4 | 500 | C | ssl_colon - ssl_imnet | brier | 0.0033 | 0.0492 | 0.0624 | 15 | 5 | lmer | 0.0033 | -0.0578 | 0.0644 | 0.8881 | -0.0050 | 0.0134 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet | ece | 0.0125 | 0.0647 | 0.0729 | 15 | 5 | lmer | 0.0125 | -0.0679 | 0.0928 | 0.6891 | 0.0059 | 0.0195 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | auroc | 0.0195 | 0.0041 | 0.0112 | 15 | 5 | fallback_t | 0.0195 | 0.0144 | 0.0246 | 4.53e-04 | 0.0141 | 0.0251 | 1274 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | auprc | 0.0163 | 0.0053 | 0.0096 | 15 | 5 | fallback_t | 0.0163 | 0.0097 | 0.0228 | 0.0023 | 0.0108 | 0.0213 | 1274 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0049 | 0.0105 | 0.0148 | 15 | 5 | lmer | -0.0049 | -0.0179 | 0.0081 | 0.3577 | -0.0108 | -0.0012 | 1274 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | brier | -0.0189 | 0.0194 | 0.0379 | 15 | 5 | fallback_t | -0.0189 | -0.0429 | 0.0052 | 0.0947 | -0.0289 | -0.0108 | 2000 |
| e4 | 500 | C | ssl_colon - ssl_imnet_matched | ece | -0.0163 | 0.0262 | 0.0505 | 15 | 5 | fallback_t | -0.0163 | -0.0487 | 0.0162 | 0.2369 | -0.0235 | -0.0066 | 2000 |
| e4 | 500 | D | ssl_colon - sup_imnet | auroc | 0.0318 | 0.0077 | 0.0185 | 15 | 5 | fallback_t | 0.0318 | 0.0223 | 0.0413 | 7.55e-04 | 0.0249 | 0.0381 | 1274 |
| e4 | 500 | D | ssl_colon - sup_imnet | auprc | 0.0267 | 0.0091 | 0.0170 | 15 | 5 | fallback_t | 0.0267 | 0.0155 | 0.0380 | 0.0028 | 0.0170 | 0.0361 | 1274 |
| e4 | 500 | D | ssl_colon - sup_imnet | balanced_accuracy | -0.0014 | 0.0062 | 0.0139 | 15 | 5 | fallback_t | -0.0014 | -0.0091 | 0.0063 | 0.6400 | -0.0071 | 0.0040 | 1274 |
| e4 | 500 | D | ssl_colon - sup_imnet | brier | -0.0165 | 0.0393 | 0.0525 | 15 | 5 | lmer | -0.0165 | -0.0653 | 0.0323 | 0.4002 | -0.0258 | -0.0084 | 2000 |
| e4 | 500 | D | ssl_colon - sup_imnet | ece | -0.0063 | 0.0503 | 0.0670 | 15 | 5 | lmer | -0.0063 | -0.0687 | 0.0562 | 0.7947 | -0.0134 | 0.0029 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet | auroc | 0.0098 | 0.0048 | 0.0054 | 15 | 5 | lmer | 0.0098 | 0.0039 | 0.0157 | 0.0099 | 0.0047 | 0.0153 | 1274 |
| e4 | 500 | D | ssl_colon - ssl_imnet | auprc | 0.0082 | 0.0052 | 0.0063 | 15 | 5 | lmer | 0.0082 | 0.0018 | 0.0146 | 0.0239 | 0.0035 | 0.0122 | 1274 |
| e4 | 500 | D | ssl_colon - ssl_imnet | balanced_accuracy | -0.0018 | 0.0123 | 0.0185 | 15 | 5 | lmer | -0.0018 | -0.0170 | 0.0134 | 0.7619 | -0.0090 | 0.0037 | 1274 |
| e4 | 500 | D | ssl_colon - ssl_imnet | brier | 0.0033 | 0.0492 | 0.0624 | 15 | 5 | lmer | 0.0033 | -0.0578 | 0.0644 | 0.8881 | -0.0050 | 0.0134 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet | ece | 0.0125 | 0.0647 | 0.0729 | 15 | 5 | lmer | 0.0125 | -0.0679 | 0.0928 | 0.6891 | 0.0059 | 0.0195 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | auroc | 0.0195 | 0.0041 | 0.0112 | 15 | 5 | fallback_t | 0.0195 | 0.0144 | 0.0246 | 4.53e-04 | 0.0141 | 0.0251 | 1274 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | auprc | 0.0163 | 0.0053 | 0.0096 | 15 | 5 | fallback_t | 0.0163 | 0.0097 | 0.0228 | 0.0023 | 0.0108 | 0.0213 | 1274 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | balanced_accuracy | -0.0048 | 0.0102 | 0.0145 | 15 | 5 | lmer | -0.0048 | -0.0174 | 0.0079 | 0.3572 | -0.0099 | -0.0013 | 1274 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | brier | -0.0189 | 0.0194 | 0.0379 | 15 | 5 | fallback_t | -0.0189 | -0.0429 | 0.0052 | 0.0947 | -0.0289 | -0.0108 | 2000 |
| e4 | 500 | D | ssl_colon - ssl_imnet_matched | ece | -0.0163 | 0.0262 | 0.0505 | 15 | 5 | fallback_t | -0.0163 | -0.0487 | 0.0162 | 0.2369 | -0.0235 | -0.0066 | 2000 |
| neg | 50 | isolated | ssl_colon - sup_imnet | auroc | 0.0538 | 0.0178 | 0.0209 | 15 | 5 | lmer | 0.0538 | 0.0317 | 0.0759 | 0.0025 | 0.0429 | 0.0641 | 1997 |
| neg | 50 | isolated | ssl_colon - sup_imnet | auprc | 0.0109 | 0.0037 | 0.0052 | 15 | 5 | lmer | 0.0109 | 0.0062 | 0.0155 | 0.0029 | 0.0036 | 0.0191 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet | auroc | 0.0068 | 0.0113 | 0.0171 | 15 | 5 | lmer | 0.0068 | -0.0072 | 0.0209 | 0.2481 | -0.0016 | 0.0150 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet | auprc | 0.0012 | 0.0020 | 0.0035 | 15 | 5 | fallback_t | 0.0012 | -0.0012 | 0.0036 | 0.2419 | -5.22e-04 | 0.0035 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet_matched | auroc | 0.0321 | 0.0070 | 0.0217 | 15 | 5 | fallback_t | 0.0321 | 0.0234 | 0.0407 | 5.06e-04 | 0.0240 | 0.0400 | 1997 |
| neg | 50 | isolated | ssl_colon - ssl_imnet_matched | auprc | 0.0056 | 0.0012 | 0.0045 | 15 | 5 | fallback_t | 0.0056 | 0.0041 | 0.0071 | 5.47e-04 | 0.0019 | 0.0099 | 1997 |
| neg | 50 | sequence | ssl_colon - sup_imnet | auroc | 0.0521 | 0.0116 | 0.0136 | 15 | 5 | lmer | 0.0521 | 0.0377 | 0.0666 | 5.59e-04 | 0.0455 | 0.0595 | 2000 |
| neg | 50 | sequence | ssl_colon - sup_imnet | auprc | 0.0875 | 0.0234 | 0.0169 | 15 | 5 | lmer | 0.0875 | 0.0584 | 0.1166 | 0.0011 | 0.0724 | 0.0988 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet | auroc | 0.0162 | 0.0032 | 0.0130 | 15 | 5 | fallback_t | 0.0162 | 0.0123 | 0.0202 | 3.37e-04 | 0.0099 | 0.0239 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet | auprc | 0.0277 | 0.0097 | 0.0234 | 15 | 5 | fallback_t | 0.0277 | 0.0157 | 0.0397 | 0.0030 | 0.0157 | 0.0407 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet_matched | auroc | 0.0319 | 0.0050 | 0.0075 | 15 | 5 | lmer | 0.0319 | 0.0257 | 0.0382 | 1.40e-04 | 0.0236 | 0.0401 | 2000 |
| neg | 50 | sequence | ssl_colon - ssl_imnet_matched | auprc | 0.0554 | 0.0106 | 0.0169 | 15 | 5 | lmer | 0.0554 | 0.0423 | 0.0686 | 3.04e-04 | 0.0404 | 0.0687 | 2000 |
