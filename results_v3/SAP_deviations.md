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

## 2026-10-16: SSL-imnet-matched pretraining environment and code

- Commit `32ca324` (authorised 2026-10-16), four changes to the MAE pretraining code:
  1. `util/misc.py`: `from math import inf` replaces `from torch._six import inf` (same `float("inf")`,
     used only in the gradient-norm helper).
  2. `models_mae.py`: `pos_embed` imported relative to the package, falling back to `util.pos_embed`
     when `main_pretrain.py` runs as a script.
  3. `engine_pretrain.py`: `getattr(log_writer, "log_dir", None)` in a log message.
  4. `main_pretrain.py`: TF32 set explicitly to matmul = True, cuDNN = True, as in fine-tuning.
  None of the code changes alters the training computation. Changes 1–3 only make the script run; change 4
  sets TF32 to the torch 1.9 default (it acts only on fp32 operations; AMP fp16 operations are unaffected),
  i.e. it aligns the numerical setting with the SSL-colon pretraining rather than changing the algorithm.
- PyTorch version: the `torch._six` import shows that SSL-colon (HyperKvasir) was pretrained with an earlier
  PyTorch, probably 1.9 (RunPod documentation; the v2 environment record is torch 1.9.0+cu111). The
  SSL-imnet-matched regime is pretrained with torch 2.4.1 (`ssl4polyp-pretrain`, timm 0.3.2) because torch
  1.9 does not support the current GPUs (RTX 6000 Ada).
- Iterations per epoch and learning-rate schedule: with 99,417 images, batch size 128 and `drop_last=True`
  there are 776 iterations per epoch (observed in the probe), the same as for the 99,417 HyperKvasir
  unlabelled images; the per-iteration half-cycle cosine schedule (`lr_sched.adjust_learning_rate` at
  `epoch + iter / 776`, lr 1e-3, min_lr 0, 40 warmup epochs, 400 epochs) is therefore the same. The
  HyperKvasir training logs are not available, so this follows from the stored arguments and dataset size.
- **Effect on the SAP:** none on hypotheses, metrics or rules; the matched regime differs from SSL-colon in
  the PyTorch version used for pretraining (2.4.1 vs, probably, 1.9), declared as a limitation.

## 2026-10-17: inference method of section 4 (SAP v3.3)

- **SAP:** `docs/v3/plan_analisis_estadistico_v3_3.md`, SHA-256 `fd6b0342d529f77897ae6fd8c08e3d5cb101ab090fc09fe7540f40ea70f7ff56` (deviation 9.3), tag `v3-cmpb-sap33`.
- **When:** after the four-regime campaign finished and **before any analysis of the real data**; no metric of the
  real runs had been computed or seen.
- **What changed:** the inference on μ is always the one-sample t-test on the K partition means (df = K − 1),
  instead of `lmer` with Satterthwaite df and the fallback rule for singular or non-converged fits.
- **Why:** simulation validation with a decision rule fixed and committed before the run (`0b6dd67`): 200 null and
  50 effects repetitions, variants V1 (v3.2c), V2 (Kenward–Roger) and V3 (t-test on partition means). The three
  gave identical decisions; family-wise false confirmations 7/200 (exact 95 % CI 0.014–0.071); minimum coverage
  0.92 < 0.93 → no variant met both conditions → V3 by the rule
  (`results_v3/verification/analysis_v3_variants/RESULTS.md`).
- **Effect:** none on H1–H7, metrics, Holm, bootstrap or the double confirmation rule. With all runs present the
  results are identical to v3.2c; the change only matters if runs are missing and removes the singularity check.
