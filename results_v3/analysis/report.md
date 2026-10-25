# Campaign v3 analysis (SAP v3.3)

- SAP implemented: v3.3, SHA-256 `fd6b0342d529f77897ae6fd8c08e3d5cb101ab090fc09fe7540f40ea70f7ff56`; file matches: True; last SAP registered in RUNLOG: SAP v3.3 (`fd6b0342d529…`), matches: True.
- Roots: `<REPO>/checkpoints_v3/campaign/classification`, `<REPO>/checkpoints_v3/campaign_matched/classification`
- Runs: 720; regimes: sup_imnet, ssl_imnet, ssl_colon, ssl_imnet_matched; partitions: [1, 2, 3, 4, 5]; seeds: [13, 29, 47].
- Views evaluated: 3060; contrasts: 2313; bootstrap B = 2000, seed 12345.
- Integrity checks: 840 run, 0 failed.
- Elapsed: 858.1 s.

## Completeness

- All expected runs present for the regimes found.

## Primary hypotheses

### H1–H7

| hypothesis | estimate | model_method | model_ci_lo | model_ci_hi | p | p_holm | boot_lo | boot_hi | confirmed | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 0.0475 | partition_means_t | 0.0286 | 0.0663 | 0.0022 | 0.0088 | 0.0355 | 0.0592 | yes | final |  |
| H2 | -0.0138 | partition_means_t | -0.0285 | 8.67e-04 | 0.0592 | 0.1185 | -0.0214 | -0.0064 | no | final |  |
| H3 | 0.0380 | partition_means_t | 0.0122 | 0.0639 | 0.0151 | 0.0452 | 0.0202 | 0.0593 | yes | final |  |
| H4 | -0.0720 | partition_means_t | -0.0985 | -0.0455 | 0.0017 | 0.0084 | -0.3447 | 0.0944 | no | final |  |
| H5 | 0.1548 | partition_means_t | 0.1202 | 0.1895 | 2.42e-04 | 0.0017 | -0.0310 | 0.3093 | no | final |  |
| H6 | -0.0064 | partition_means_t | -0.0241 | 0.0114 | 0.3757 | 0.3757 | -0.0184 | 0.0058 | no | final |  |
| H7 | -0.0748 | partition_means_t | -0.1012 | -0.0485 | 0.0014 | 0.0084 | -0.1353 | -0.0150 | yes | final |  |

Confirmed (Holm p < 0.05 and bootstrap CI excluding 0): H1, H3, H7.

## Outputs

Tables:
- `tables/table_hypotheses.csv`
- `tables/table_sun_full.csv`
- `tables/table3_e1_performance.csv`
- `tables/table4_e1_aulc.csv`
- `tables/table5_e2_morphology.csv`
- `tables/table5b_e2_gap.csv`
- `tables/table6_e3_zeroshot.csv`
- `tables/table7_negative_source.csv`
- `tables/table7_negative_source_contrasts.csv`
- `tables/table7_count_checks.csv`
- `tables/table8_e4_arms.csv`
- `tables/table8b_e4_ap_within_pack.csv`
- `tables/table9_e4_arm_contrasts.csv`
- `tables/table_secondary_regime_comparisons.csv`
- `tables/table_degenerate.csv`
- `tables/table_calibration.csv`
- `tables/table_reliability.csv`
- `tables/table_per_polyp.csv`
- `tables/table_sensitivity_k1_no_s500.csv`
- `tables/table_loco_e3.csv`
- `tables/table_loco_negative_source.csv`
- `tables/table_locase_e2_flat.csv`

Figures:
- `figures/fig_e1_learning_curves.png`
- `figures/fig_hypotheses_forest.png`
- `figures/fig_e2_strata.png`
- `figures/fig_e4_arms_balanced_accuracy.png`
- `figures/fig_reliability.png`
- `figures/fig_e3_loco.png`
