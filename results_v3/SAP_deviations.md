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

## 2026-10-15: partitions, E2 and SSL-imnet-matched learning rate (SAP v3.2)

- **SAP:** `docs/v3/plan_analisis_estadistico_v3_2.md`, SHA-256 `38f8ab0535ab024c4f292ea17a95d1288f0e7602e2e74e2ae742d2fdd2375c70` (deviation 9.2).
- **When:** after the v3.1 learning-rate sweep, before generating any new partition and before any campaign result.
- **What changed:** SSL-imnet-matched goes through the same learning-rate sweep and rule (no longer inherits
  SSL-imnet's rate); K = 5 independent morphology-stratified partitions (k = 1 = current `sun_morphology`,
  seed 47; k = 2–5 = first 4 seeds of 101–120 passing the assertions; test sets may overlap); E2 evaluated per
  stratum on each partition's sun_full models without its own training (H3 unchanged); nested `sun_subsets`
  per partition (`subsets-canonical`, seed 13); few-shot pack seed 1000·k + s.
- **Deviation:** the v3.1 learning-rate sweep used the validation split of the v2 `sun_full`, which shares
  cases with the new k = 1. No test split was used in the sweep.
- **Effect:** no change to H1–H7, metrics, statistical model or confirmation rules.
