# Variant validation: results and decision

Plan and rule: `PLAN.md` (commit `0b6dd67`, pushed before the run). 200 null and 50 effects repetitions of a simulated campaign (B = 200 per repetition), CPU only, 12 processes at nice 19, 2026-10-17. Per-repetition rows: `variant_reps.csv`; summary: `variant_summary.json`; log: `run_log.txt`.

## Per variant

| Variant | Family-wise false confirmations (null) | Rate | Exact 95 % CI | Min. coverage (null and effects) | Mean power | Meets both conditions |
|---|---|---|---|---|---|---|
| V1 lmer + Satterthwaite + fallback | 7/200 | 0.035 | 0.014–0.071 | 0.920 | 1.000 | no |
| V2 lmer + Kenward–Roger + fallback | 7/200 | 0.035 | 0.014–0.071 | 0.920 | 1.000 | no |
| V3 t-test on K partition means | 7/200 | 0.035 | 0.014–0.071 | 0.920 | 1.000 | no |

## Per hypothesis (identical for V1, V2 and V3)

| H | False confirmation (null) | Model-CI coverage, null | Model-CI coverage, effects | Power (effects) |
|---|---|---|---|---|
| H1 | 0.005 | 0.960 | 0.940 | 1.000 |
| H2 | 0.005 | 0.975 | 0.940 | 1.000 |
| H3 | 0.000 | 0.950 | 0.980 | 1.000 |
| H4 | 0.005 | 0.950 | 0.920 | 1.000 |
| H5 | 0.000 | 0.935 | 0.920 | 1.000 |
| H6 | 0.015 | 0.950 | 0.960 | 1.000 |
| H7 | 0.005 | 0.955 | 0.940 | 1.000 |

Agreement between variants: max |Δp| V1–V2 2.8e-04, V1–V3 2.8e-04, V2–V3 1.7e-05; confirmation decisions differ in 0 of 1,750 hypothesis tests per pair. As noted in `PLAN.md` before the run, in a complete balanced design the three variants are the same test (the residual differences come from the numerical optimum of non-singular REML fits).

## Decision (rule applied as written)

1. V1: family-wise upper limit 0.071 ≤ 0.08 (met); minimum coverage 0.92 < 0.93 (not met: H4 and H5 with effects, 46/50) → V1 is not kept.
2. V2 and V3 give the same results, so neither meets the coverage condition.
3. Neither meets both conditions → **V3** (always the one-sample t-test on the K partition means, df = K − 1). SAP v3.3.

## Notes

- Coverage 46/50 is compatible with the nominal 0.95 (P(X ≤ 46 | n = 50, p = 0.95) = 0.24); the coverage condition is decided by the 50 effect repetitions. Null coverage is ≥ 0.935 in every hypothesis.
- 13 simulated targets were clipped in 7 repetitions (6 null, 1 effects). Without those repetitions: 7/194 family-wise (exact CI 0.015–0.073), minimum coverage 0.918; same decision. The rule is applied to all repetitions, as registered.
- Because the variants coincide when every run is present, the change to V3 does not alter any result of a complete campaign; it removes the dependence on the singularity check and fixes the method if runs are missing (V3 then averages the available seeds of each partition with equal weight per partition).
- The first automatic summary of this run read the literal scenario name "null" as a missing value (pandas default) and reported no null repetitions; fixed in `read_reps` and re-summarised from the same `variant_reps.csv` (no repetition was rerun).
