# SAP deviations, campaign v3

## 2026-10-15: learning-rate selection (SAP v3.1)

- **What changed:** section 10 "Selección de learning rate" added (`docs/v3/plan_analisis_estadistico_v3_1.md`,
  SHA-256 `a2fc1b8325885216a15b410792f689acf9805267f43342e8fb9b28968818e7bd`). Each regime's fine-tuning
  learning rate is chosen by a validation-only sweep (1e-4, 3e-4, 1e-3; seeds 13, 29, 47; sun_full,
  partition k = 1) by mean validation AP; 1e-3 for all regimes is kept only if it wins in all three.
- **When:** after the phase 0 smoke test and before any campaign result; the rule was written and its
  hash registered before the sweep was run. The test split is never evaluated in the sweep.
- **Why:** check that the common learning rate does not favour any regime.
- **Effect:** no change to hypotheses, metrics, statistical model or confirmation rules; may change the
  fine-tuning learning rate (sun_full, E1, E2) of one or more regimes. E4's adaptation schedule is unchanged.
