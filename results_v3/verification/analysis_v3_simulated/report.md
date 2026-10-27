# Campaign v3 analysis (SAP v3.3) — SIMULATED DATA (effects scenario, sim seed 2026)

- SAP implemented: v3.3, SHA-256 `fd6b0342d529f77897ae6fd8c08e3d5cb101ab090fc09fe7540f40ea70f7ff56`; file matches: True; last SAP registered in RUNLOG: SAP v3.3 (`fd6b0342d529…`), matches: True.
- Roots: `<SCRATCH>/sim_final/campaign/classification`, `<SCRATCH>/sim_final/campaign_matched/classification`
- Runs: 720; regimes: sup_imnet, ssl_imnet, ssl_colon, ssl_imnet_matched; partitions: [1, 2, 3, 4, 5]; seeds: [13, 29, 47].
- Views evaluated: 3060; contrasts: 2313; bootstrap B = 2000, seed 12345.
- Integrity checks: 840 run, 0 failed.
- Elapsed: 1126.9 s.

## Completeness

- All expected runs present for the regimes found.

## Primary hypotheses

### H1–H7

| hypothesis | estimate | model_method | model_ci_lo | model_ci_hi | p | p_holm | boot_lo | boot_hi | confirmed | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 0.0375 | partition_means_t | 0.0239 | 0.0510 | 0.0016 | 0.0047 | 0.0309 | 0.0443 | yes | final |  |
| H2 | 0.0346 | partition_means_t | 0.0236 | 0.0457 | 9.67e-04 | 0.0039 | 0.0283 | 0.0413 | yes | final |  |
| H3 | 0.0784 | partition_means_t | 0.0637 | 0.0932 | 1.23e-04 | 6.82e-04 | 0.0642 | 0.0939 | yes | final |  |
| H4 | 0.0542 | partition_means_t | 0.0343 | 0.0740 | 0.0016 | 0.0047 | 0.0438 | 0.0643 | yes | final |  |
| H5 | -0.0725 | partition_means_t | -0.0859 | -0.0591 | 1.14e-04 | 6.82e-04 | -0.0991 | -0.0457 | yes | final |  |
| H6 | 0.0467 | partition_means_t | 0.0297 | 0.0636 | 0.0016 | 0.0047 | 0.0437 | 0.0497 | yes | final |  |
| H7 | 0.0815 | partition_means_t | 0.0702 | 0.0929 | 3.74e-05 | 2.62e-04 | 0.0786 | 0.0857 | yes | final |  |

Confirmed (Holm p < 0.05 and bootstrap CI excluding 0): H1, H2, H3, H4, H5, H6, H7.

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
