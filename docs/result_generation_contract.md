# Result Generation Contract

This document defines the uniform, deterministic recipe for materialising tables and
figures across all SSL4Polyp experiments (Exp1–Exp5). The goal is to guarantee that
identical inputs yield identical aggregates regardless of when or where the reporting
scripts run.

---

## 1. Required Per-Run Artifacts

Each training/evaluation run **must** emit the following payloads:

### 1.1 `*.metrics.json`

```
{
  "seed": 13,
  "epoch": 17,
  "val": { ... },
  "test_primary": { ... },
  "test_sensitivity": { ... },
  "thresholds": {
    "policy": "f1_opt_on_val",
    "values": { ... },
    "sources": { ... },
    "primary": {
      "policy": "f1_opt_on_val",
      "tau": 0.4237,
      "split": "sun_full/val",
      "n_candidates": 187,
      "tiebreakers": ["higher_recall", "lower_tau"],
      "epoch": 17,
      "degenerate_val": false,
      "notes": {}
    },
    "sensitivity": { ... }
  },
  "run": { ... },
  "provenance": { ... }
}
```

The following content requirements apply:

- **Validation block (`val`)** — present for training runs. Must include at least
  `loss`, `auprc`, `auroc`.
- **Primary test block (`test_primary`)** — always required. Must expose:
  `tp`, `fp`, `tn`, `fn`, `loss`, `auprc`, `auroc`, `recall`, `precision`, `f1`,
  `balanced_accuracy`, `mcc`, `n_pos`, `n_neg`, `prevalence`, `tau`, `tau_info`.
- **Sensitivity block (`test_sensitivity`)** — optional per experiment but, when
  present, must contain the same fields as `test_primary` (except `tau`, which is
  optional if sensitivity uses an alternate threshold).
- **Held-out naming** — even if a dataset pack labels the held-out split as `eval`,
  all exported metrics **must** use the `test_*` keys (e.g., `test_primary`,
  `test_sensitivity`) so downstream tooling remains consistent.
- **Threshold provenance** — `thresholds` section populated for every run. The
  `primary` record **must** exist and match the policy mandated by the experiment.
  When a secondary threshold is required (e.g., sensitivity), its record is stored at
  `thresholds.sensitivity`.
- **Run metadata** — `run` section describing model key, architecture, pretraining,
  experiment name, selection tag, seed, mode, world size, etc.
- **Data provenance** — `provenance` must capture dataset pack names, percent/seed
  details, canonical roots, and SHA256 digests for any referenced CSVs.

### 1.2 Optional Artifacts

| Artifact | Purpose | Requirements |
| --- | --- | --- |
| `*_test_outputs.csv` | Per-frame predictions. | Columns: `frame_id`, `prob`, `label`, `pred`. When available, also include `center_id`, `origin`, `sequence_id`. SHA256 digests must be recorded in the metrics payload. |
| `*_roc_curve.csv`, `*_pr_curve.csv` | Deterministic curves (~200 points). | Stored under `curves/`, referenced via `thresholds` or `curve_exports` metadata. |


## 2. Deterministic Aggregation Rules

All experiment-level tables and figures must obey the following rules.

### 2.1 Seed Axis Normalisation

- Valid training seeds are exactly `{13, 29, 47}`.
- A model/condition is only aggregated if **all** required seeds are present.
- Results for missing seeds cause a hard failure (see Guardrails).

### 2.2 Statistics

- `mean` = arithmetic mean over seeds.
- `sd` = sample standard deviation over seeds.
- `paired_delta` between two conditions uses per-seed differences first, then mean ± sd
  and 95% CI from bootstrap (below).

### 2.3 Bootstrap Configuration

- Number of resamples `B = 2000`.
- Confidence intervals = percentile 95% (`[2.5%, 97.5%]`).
- Thresholds **must not** be recomputed inside the bootstrap; re-use τ stored in
  the metrics payload.

#### Clustering Strategy

| Dataset | Preferred cluster ID | Fallback |
| --- | --- | --- |
| SUN splits | `case_id` | `frame_id` (singleton) |
| PolypGen | `center_id` | For negatives: `sequence_id`; for positives: singleton by `frame_id`. |

- Domain shift deltas resample matched indices jointly across domains per seed.
  (e.g., SUN vs PolypGen: sample cluster IDs, apply identical selections to both
  domains before recomputing metrics.)

### 2.4 Formatting

- Numeric display: 3 decimal places.
- Confidence intervals: `0.742 [0.713, 0.771]`.
- Paired deltas: same format, optionally add `Δ` prefix.

### 2.5 Checkpoint Tie-breaking

- When multiple checkpoints achieve the same monitor value (within `1e-12`), prefer
  the **earliest epoch**. The selection tag must reflect this decision so downstream
  tooling can confirm deterministic behaviour.


## 3. Guardrails (Fail-Fast Assertions)

Reporting scripts must abort with a clear error message if any of the following is
violated:

1. **Confusion Consistency** — `tp + fp + tn + fn == n_pos + n_neg` for every block.
2. **CSV Integrity** — The SHA256 digest of fixed splits (e.g., SUN test) must match
   across all runs/conditions. Mixed digests indicate a data leak or misaligned
   evaluation set.
3. **Seed Completeness** — Every model/condition must include the full seed set.
   Missing or extra seeds are disallowed.
4. **Threshold Provenance** — `thresholds.primary` exists and aligns with the expected
   experiment policy (`f1_opt_on_val`, `sun_val_frozen`, etc.). If a sensitivity
   threshold is mandated, ensure `thresholds.sensitivity` is present and matches.
5. **Curve Availability** — When a report references ROC/PR curves, verify the files
   exist and match the recorded digest.
6. **Metadata Sanity** — `run` and `provenance` sections contain enough information to
   reconstruct pack paths, seeds, and parent checkpoints. Missing fields are fatal.


## 4. Suggested Implementation Hooks

To retrofit existing reporting scripts (`scripts/exp*_report.py`):

1. Introduce a `ResultLoader` utility that encapsulates:
   - Guardrail validation.
   - Extraction of primary/sensitivity metrics.
   - Normalisation of curve metadata.
2. Add a shared `Bootstrapper` that accepts:
   - Metrics matrix arrayed by seed.
   - Cluster assignments per seed/condition.
   - Delta pairing semantics.
   Use a cryptographically neutral RNG seed (e.g., `rng = np.random.default_rng(1337)`)
   so runs are reproducible.
3. Provide display helpers (formatting to 3 decimals, embedding CI strings) shared by
   all experiment scripts.
4. Extend CLI arguments to allow `--runs-root`, `--output`, and optional
   `--strict/--no-strict` toggles for guardrails, defaulting to strict enforcement.
5. Produce a final manifest (e.g., `report_manifest.json`) that records:
   - Timestamp, git commit, RNG seed.
   - List of runs included (model, seed, path).
   - Hashes of output tables/figures.


## 5. Experiment-specific Notes

| Experiment | Primary Policy | Sensitivity Policy | Notes |
| --- | --- | --- | --- |
| Exp1 | `f1_opt_on_val` | `youden_on_val` | Requires SUN ROC/PR curves. |
| Exp2 | `f1_opt_on_val` | `youden_on_val` | Domain vs. generic delta: resample paired SUN test clusters. |
| Exp3 | `f1_opt_on_val` | `youden_on_val` | Morphology strata must be preserved per cluster. |
| Exp4 | `f1_opt_on_val` | `youden_on_val` | Ensure subset percentages appear in provenance to filter runs. |
| Exp5A | `sun_val_frozen` | — | Sensitivity not computed; guardrail should confirm absence. |
| Exp5B | `sun_val_frozen` | — | Perturbation metadata must match manifest. |
| Exp5C | `sun_val_frozen` | `val_opt_youden` | Few-shot budget recorded via provenance `fewshot_budget`. |


## 6. Next Steps

1. Refactor `scripts/exp*_report.py` to consume the shared loader once implemented.
2. Port existing tests (e.g., `tests/test_exp*_report.py`) to assert the new guardrails
   and formatting rules.
3. Add integration tests that run the full pipeline on fixture runs to ensure
   deterministic output hashes.
4. Document the CLI usage in `README.md` after the refactor, referencing this contract.

---

Version: 2025-10-08
