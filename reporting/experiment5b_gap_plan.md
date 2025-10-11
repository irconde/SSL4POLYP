# Experiment 5B Gap Remediation Plan

## Objective
Ensure the Experiment 5B robustness reporting pipeline fully complies with the specification covering primary policy validation, perturbation coverage, and cluster-aware AUSC confidence intervals.

## Current Gaps
1. **Primary τ enforcement** – The pipeline records τ values but does not guarantee that every run used `sun_val_frozen`.
2. **Perturbation coverage validation** – Severity families are inferred from available tags without checking that all mandated severities are present.
3. **Case-clustered ΔAUSC bootstrapping** – The bootstrap routine lacks support for cluster identifiers, so confidence intervals are not case-clustered.

## Remediation Tasks
### 1. Enforce primary τ consistency
- [ ] Extend the provenance loading logic to assert that all Experiment 5B runs report `tau_policy == "sun_val_frozen"`.
- [ ] Add a configuration flag or constant listing the expected primary τ and surface clear errors when mismatches occur.
- [ ] Update unit tests (or add new ones) to cover both compliant and non-compliant τ scenarios.

### 2. Validate perturbation severity coverage
- [ ] Introduce canonical definitions for Experiment 5B perturbation families (blur, JPEG, brightness, contrast, occlusion) and their required severities.
- [ ] During metrics ingestion, verify that each family/severity pair is present across the aggregated runs; raise descriptive exceptions if any are missing.
- [ ] Add tests ensuring the validator passes with complete coverage and fails when a severity is absent.

### 3. Support cluster-aware ΔAUSC bootstrapping
- [ ] Update `_bootstrap_family_delta` (or create a new helper) to accept per-case cluster identifiers.
- [ ] Modify the bootstrap sampling procedure to resample clusters rather than individual cases when computing ΔAUSC confidence intervals.
- [ ] Thread cluster identifiers through the reporting pipeline, ensuring they originate from the underlying metrics artifacts.
- [ ] Expand existing bootstrap tests to include scenarios with shared cluster IDs and verify confidence intervals remain deterministic.

## Validation & Documentation
- [ ] Document the new validation checks and bootstrap behaviour in the Experiment 5B reporting README or relevant developer notes.
- [ ] Re-run the Experiment 5B reporting notebook/script to confirm compliance once code changes are implemented.

## Risks & Considerations
- Enforcing τ and severity coverage may surface historical runs that need to be re-generated; plan for reruns if validation fails.
- Cluster metadata must be available in the underlying metrics; coordinate with data generation if additional fields are required.
- Determinism should be preserved when adding cluster-aware bootstrapping by seeding the resampling process consistently.

