# Monte Carlo validation of the SAP v3.2c analysis (simulated data only)

Settings: bootstrap B = 200 per repetition (the analysis itself uses 2000); null 20, effects 6, zero partition variance 4 repetitions. Each repetition simulates a full campaign (4 regimes × 5 partitions × 3 seeds; `analysis/v3/simulate.py`) and runs H1–H7 (model, fallback, Holm over 7, bootstrap, double rule). Partition and seed SD of every difference: 0.008 (0 between partitions in the third scenario). No simulated target was clipped.

Produced by `pytest analysis/v3/tests` (log: `pytest_log.txt`, 23 passed); per-repetition rows in `validation_reps.csv`.

## effects

- repetitions: 6; repetitions with ≥ 1 confirmed hypothesis: 1.00; confirmations per test: 1.000

| H | effect | mean estimate | bias | model CI coverage | bootstrap CI coverage | confirmed | fallback used |
|---|---|---|---|---|---|---|---|
| H1 | +0.040 | +0.0403 | +0.0003 | 1.00 | 1.00 | 1.00 | 0.00 |
| H2 | +0.030 | +0.0272 | -0.0028 | 1.00 | 0.67 | 1.00 | 0.33 |
| H3 | +0.070 | +0.0702 | +0.0002 | 1.00 | 1.00 | 1.00 | 0.67 |
| H4 | +0.050 | +0.0520 | +0.0020 | 1.00 | 0.67 | 1.00 | 0.33 |
| H5 | -0.080 | -0.0772 | +0.0028 | 1.00 | 1.00 | 1.00 | 0.00 |
| H6 | +0.040 | +0.0401 | +0.0001 | 1.00 | 0.83 | 1.00 | 0.17 |
| H7 | +0.080 | +0.0806 | +0.0006 | 1.00 | 1.00 | 1.00 | 0.00 |

## null

- repetitions: 20; repetitions with ≥ 1 confirmed hypothesis: 0.15; confirmations per test: 0.021

- family-wise false confirmation: 3/20 = 0.15 (exact 95 % CI 0.03–0.38); per-test: 3/140

| H | effect | mean estimate | bias | model CI coverage | bootstrap CI coverage | confirmed | fallback used |
|---|---|---|---|---|---|---|---|
| H1 | +0.000 | +0.0004 | +0.0004 | 0.85 | 0.45 | 0.10 | 0.25 |
| H2 | +0.000 | +0.0007 | +0.0007 | 1.00 | 0.50 | 0.00 | 0.15 |
| H3 | +0.000 | +0.0015 | +0.0015 | 1.00 | 0.90 | 0.00 | 0.40 |
| H4 | +0.000 | +0.0008 | +0.0008 | 1.00 | 0.60 | 0.00 | 0.25 |
| H5 | +0.000 | +0.0002 | +0.0002 | 0.95 | 1.00 | 0.00 | 0.20 |
| H6 | +0.000 | -0.0005 | -0.0005 | 0.95 | 0.05 | 0.00 | 0.10 |
| H7 | +0.000 | +0.0016 | +0.0016 | 0.85 | 0.20 | 0.05 | 0.15 |

## no_partition_variance

- repetitions: 4; repetitions with ≥ 1 confirmed hypothesis: 1.00; confirmations per test: 1.000

| H | effect | mean estimate | bias | model CI coverage | bootstrap CI coverage | confirmed | fallback used |
|---|---|---|---|---|---|---|---|
| H1 | +0.040 | +0.0386 | -0.0014 | 1.00 | 1.00 | 1.00 | 0.50 |
| H2 | +0.030 | +0.0275 | -0.0025 | 0.75 | 0.75 | 1.00 | 0.50 |
| H3 | +0.070 | +0.0677 | -0.0023 | 1.00 | 1.00 | 1.00 | 0.00 |
| H4 | +0.050 | +0.0477 | -0.0023 | 1.00 | 1.00 | 1.00 | 0.25 |
| H5 | -0.080 | -0.0798 | +0.0002 | 1.00 | 1.00 | 1.00 | 0.50 |
| H6 | +0.040 | +0.0409 | +0.0009 | 1.00 | 1.00 | 1.00 | 0.50 |
| H7 | +0.080 | +0.0795 | -0.0005 | 1.00 | 1.00 | 1.00 | 0.50 |

## Reading

- (a) Effects: every effect is recovered (|bias| ≤ 0.003) and covered by the model CI in all repetitions; all seven are confirmed in every repetition.
- (b) Null: per-test false confirmation 3/140; family-wise 3/20 (H1 twice, H7 once). The model CI of H1 and H7 covered 0 in 17/20 repetitions (nominal 19/20). With 20 repetitions the family-wise rate is compatible with values from about 0.03 to 0.38, so it does not establish whether the procedure is above the nominal 0.05; a larger run (`python -m analysis.v3.validate --null-reps 200`) would.
- The bootstrap CI covers the true μ much less often than the model CI (e.g. H6 0.05 under the null): it measures test-sampling uncertainty only, conditional on the trained models, as SAP section 5 states. In the double rule it adds protection against test-set noise but not against partition/seed variability, which rests on the model.
- (c) Holm recomputed from the stored p-values equals the reported adjusted p-values in every repetition, and `confirmed` equals (Holm p < 0.05 and bootstrap CI excluding 0) in every row.
- (d) Every singular fit used the fallback. With K = 5 the REML estimate of σ²_partition is often exactly 0 even when the true value is not (fallback 0–67 % per hypothesis in the effects scenario); with true σ²_partition = 0 the fallback was used in 0–50 % of fits (REML does not always estimate 0). A deterministic test (`test_fallback_when_partition_variance_is_zero`) checks the switch and the t-test on the K partition means.
