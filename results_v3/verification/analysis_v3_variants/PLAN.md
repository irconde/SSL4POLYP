# Variant validation of the primary analysis: plan and decision rule (fixed before the run)

Written and committed on 2026-10-17, before running any repetition of this validation and before any analysis
of the real campaign results. Code: `analysis/v3/validate_variants.py`.

## Design

- Simulated campaigns only (`analysis/v3/simulate.py`, minimal files): 200 repetitions of the null scenario
  (simulation seeds 1000–1199) and 50 of the effects scenario (seeds 2000–2049). Same scenarios as
  `analysis_v3_validation/`: 4 regimes × 5 partitions × 3 seeds, partition and seed SD 0.008 on every difference;
  effects H1 +0.04, H2 +0.03, H3 +0.07, H4 +0.05, H5 −0.08, H6 +0.04, H7 +0.08.
- In each repetition the per-run differences of H1–H7 and their cluster bootstrap (B = 200 per repetition, seed
  12345; the analysis itself uses B = 2000) are computed once, and three variants of the model are applied to the
  same data:
  - **V1** (SAP v3.2c): `lmer`, Satterthwaite df, fallback t-test on the K partition means when singular or not converged.
  - **V2**: `lmer`, Kenward–Roger df (pbkrtest), same fallback rule (only the df method changes).
  - **V3**: always the one-sample t-test on the K partition means (df = K − 1).
- For each variant: Holm over H1–H7 and the double rule (Holm p < 0.05 and bootstrap CI excluding 0).
- CPU only, one process per repetition, at most 12 processes at `nice 19`.

## Reported per variant

Per hypothesis: false confirmation rate (null), coverage of the model 95 % CI of the true μ (null and effects),
confirmation rate with effects (power). Per variant: family-wise false confirmation rate (≥ 1 confirmed among
H1–H7 in a null repetition) with its exact (Clopper–Pearson) 95 % CI; mean power over H1–H7.

## Decision rule (as given, 2026-10-17)

1. If V1 has a family-wise rate whose exact 95 % CI upper limit is ≤ 0.08 and a coverage of at least 0.93 in all
   hypotheses, V1 is kept and the SAP does not change.
2. Otherwise, among V2 and V3, the variant with the highest mean power that meets those two conditions is chosen.
   If none meets them, V3 is chosen.
3. If the variant changes: SAP v3.3 with the change and the validation as justification, hash in RUNLOG and
   SAP_deviations.md, stating that the change was made before any analysis of the real data; tag `v3-cmpb-sap33`.

Operational reading of the conditions (applied as written): "coverage of at least 0.93 in all hypotheses" =
the point estimate of the model-CI coverage is ≥ 0.93 for each of H1–H7 in both the null and the effects
scenario.

## Note recorded before the run

In a complete, balanced design (every partition with all three seeds, as in the campaign and in these simulations)
the REML fit of `d ~ 1 + (1 | k)`, when not singular, gives for μ exactly the one-sample t-test on the K partition
means (same estimate, standard error and df = K − 1, with Satterthwaite and with Kenward–Roger); when singular, V1
and V2 use that same t-test. The three variants are therefore expected to give the same p-values and decisions
here (a smoke test of 3 repetitions agreed to |Δp| < 2e-8). They differ only when runs are missing.
