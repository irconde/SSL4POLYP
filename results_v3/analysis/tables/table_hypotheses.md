### Primary hypotheses H1–H7 (SAP sections 3–5)

Estimate = mean of the paired per-run differences; model CI and p from the one-sample t-test on the K partition means (SAP v3.3, df = K − 1); Holm over the hypotheses listed in holm_family; confirmed = Holm p < 0.05 and the cluster-bootstrap 95 % CI excludes 0.

| hypothesis | definition | estimate | sd_partition | sd_seed | model_method | model_ci_lo | model_ci_hi | p | p_holm | boot_lo | boot_hi | confirmed | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | E1, SSL-colon − SUP-imnet, AULC of AUPRC (5–100 %) | 0.0475 | 0.0152 | 0.0239 | partition_means_t | 0.0286 | 0.0663 | 0.0022 | 0.0088 | 0.0355 | 0.0592 | yes | final |  |
| H2 | E1, SSL-colon − SSL-imnet-matched, AULC of AUPRC (5–100 %) | -0.0138 | 0.0118 | 0.0084 | partition_means_t | -0.0285 | 8.67e-04 | 0.0592 | 0.1185 | -0.0214 | -0.0064 | no | final |  |
| H3 | E2, SSL-colon − SUP-imnet, AUROC, flat stratum | 0.0380 | 0.0208 | 0.0303 | partition_means_t | 0.0122 | 0.0639 | 0.0151 | 0.0452 | 0.0202 | 0.0593 | yes | final |  |
| H4 | E3, SSL-colon − SUP-imnet, AUROC, polypgen_clean_test | -0.0720 | 0.0214 | 0.0361 | partition_means_t | -0.0985 | -0.0455 | 0.0017 | 0.0084 | -0.3447 | 0.0944 | no | final |  |
| H5 | E3 decomposition, sequence − isolated negatives, AUROC, mean of regimes, 50-frame split, common positives | 0.1548 | 0.0279 | 0.0205 | partition_means_t | 0.1202 | 0.1895 | 2.42e-04 | 0.0017 | -0.0310 | 0.3093 | no | final |  |
| H6 | E4, arm B − arm A, balanced accuracy, mean of budgets and regimes | -0.0064 | 0.0143 | 0.0183 | partition_means_t | -0.0241 | 0.0114 | 0.3757 | 0.3757 | -0.0184 | 0.0058 | no | final |  |
| H7 | E4, arm C − arm A, balanced accuracy, mean of budgets and regimes | -0.0748 | 0.0212 | 0.0464 | partition_means_t | -0.1012 | -0.0485 | 0.0014 | 0.0084 | -0.1353 | -0.0150 | yes | final |  |
