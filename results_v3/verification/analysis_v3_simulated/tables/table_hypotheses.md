### Primary hypotheses H1–H7 (SAP sections 3–5)

Estimate = mean of the paired per-run differences; model CI from lme4/lmerTest (Satterthwaite) or the fallback t-test; Holm over the hypotheses listed in holm_family; confirmed = Holm p < 0.05 and the cluster-bootstrap 95 % CI excludes 0.

| hypothesis | definition | estimate | sd_partition | sd_seed | model_method | model_ci_lo | model_ci_hi | p | p_holm | boot_lo | boot_hi | confirmed | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | E1, SSL-colon − SUP-imnet, AULC of AUPRC (5–100 %) | 0.0375 | 0.0109 | 0.0112 | lmer | 0.0239 | 0.0510 | 0.0016 | 0.0047 | 0.0309 | 0.0443 | yes | final |  |
| H2 | E1, SSL-colon − SSL-imnet-matched, AULC of AUPRC (5–100 %) | 0.0346 | 0.0089 | 0.0065 | lmer | 0.0236 | 0.0457 | 9.67e-04 | 0.0039 | 0.0283 | 0.0413 | yes | final |  |
| H3 | E2, SSL-colon − SUP-imnet, AUROC, flat stratum | 0.0784 | 0.0119 | 0.0302 | fallback_t | 0.0637 | 0.0932 | 1.23e-04 | 6.82e-04 | 0.0642 | 0.0939 | yes | final |  |
| H4 | E3, SSL-colon − SUP-imnet, AUROC, polypgen_clean_test | 0.0542 | 0.0160 | 0.0187 | lmer | 0.0343 | 0.0740 | 0.0016 | 0.0047 | 0.0438 | 0.0643 | yes | final |  |
| H5 | E3 decomposition, sequence − isolated negatives, AUROC, mean of regimes, 50-frame split, common positives | -0.0725 | 0.0108 | 0.0064 | lmer | -0.0859 | -0.0591 | 1.14e-04 | 6.82e-04 | -0.0991 | -0.0457 | yes | final |  |
| H6 | E4, arm B − arm A, balanced accuracy, mean of budgets and regimes | 0.0467 | 0.0137 | 0.0069 | lmer | 0.0297 | 0.0636 | 0.0016 | 0.0047 | 0.0437 | 0.0497 | yes | final |  |
| H7 | E4, arm C − arm A, balanced accuracy, mean of budgets and regimes | 0.0815 | 0.0091 | 0.0093 | lmer | 0.0702 | 0.0929 | 3.74e-05 | 2.62e-04 | 0.0786 | 0.0857 | yes | final |  |
