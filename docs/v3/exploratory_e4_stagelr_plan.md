# Exploratory plan: E4 adaptation with the declared stage learning rates (E4-stagelr)

## Status

**Exploratory analysis, not registered in the SAP, decided on 2026-11-04 after seeing the results** of the registered
campaign v3 analysis (SAP v3.3) and of the earlier exploratory and sensitivity analyses. It answers a finding of the
code review of section 2.5: the E4 adaptation did not follow the schedule its configuration declares.

**Its results will not be used to confirm or refute hypotheses**: no Holm correction and no confirmation verdicts;
estimates and intervals are reported as observed. The registered analysis and its results do not change, and it
remains the confirmatory one. No registered run, result, manifest, `RUN_COMPLETE.json`, `events.jsonl` or registered
plan is modified.

## Question

Do H7 (arm C − arm A) and the E4 arm contrasts hold when the adaptation follows the schedule as declared in
`config/exp/v3/e4_budgets/s*.yaml`?

## Background: what the registered E4 did

The budget configurations declare a two-stage schedule (head only with the encoder frozen, then the head plus the last
one or two blocks with a much lower backbone rate) and a global linear warm-up plus cosine schedule
(`scheduler.warmup_epochs` per budget; `cosine` from `config/base.yaml`).

| Budget | Epochs | Stage 1 (head only) | Stage 2 | Warm-up |
|---|---|---|---|---|
| 50 | 30 | 10 epochs, head 5e-4 | 20 epochs, `head+1`, head 5e-4, backbone 5e-6 | 6 |
| 100 | 40 | 4 epochs, head 1e-3 | 36 epochs, `head+2`, head 5e-4, backbone 1.5e-5 | 4 |
| 200 | 50 | 3 epochs, head 1e-3 | 47 epochs, `head+2`, head 5e-4, backbone 2e-5 | 3 |
| 500 | 50 | 5 epochs, head 1e-3 | 45 epochs, `head+2`, head 5e-4, backbone 2e-5 | 5 |

Two implementation facts, found in the registered code and logs (code at `f203f16`…`9548c8c`, unchanged for these
paths since the campaign):

1. **Stage rates held for one epoch only (all 240 runs).** The two optimizer groups are created with the base rate
   (`train_classification.py:6032-6047`, at 9548c8c), the epoch `LambdaLR` is created from it (`:6255`), the stage
   rates are assigned at the start of each stage (`FinetuneScheduleRuntime.apply_if_needed`, `:1203-1209`), and every
   `scheduler.step()` (`:7148`) resets both groups to `base_lr × λ(epoch)`. From the second epoch of each stage, both
   groups ran at the base rate (5e-4 at 50 frames, 1e-3 otherwise) times the global factor: the unfrozen blocks were
   trained 50–100 times above the declared backbone rate, and the head at about twice its stage-2 rate. In the 2,658
   stage-2 epochs after the first, no epoch used the declared rates.
2. **SUP-imnet adapted the head only (58 runs).** `configure_finetune_parameters` unfreezes `model.blocks` only when it
   is an `nn.ModuleList`, `list` or `tuple` (`src/ssl4polyp/classification/finetune.py:77`). In timm 0.4.12 the ViT
   stores its blocks as `nn.Sequential` (`timm/models/vision_transformer.py:269`), so `head+1`/`head+2` unfroze no
   block in SUP-imnet; the MAE regimes (`nn.ModuleList`, `models_mae.py:42`) did unfreeze them. All 58 SUP-imnet runs
   that reached stage 2 had 0 trainable backbone parameters in it (the other 2, at 50 frames, stopped earlier). The
   final LayerNorm stays frozen in every regime.

The documentation (configs, README v2, `tests/test_finetune_schedule.py`, SAP v3.1–v3.3 section 10) states the stage
rates but not how they combine with the warm-up and cosine factor. This plan uses interpretation **I2** (decided):
the stage rate of each group is its base rate, scaled by the registered global factor,
lr_g(e) = r_g(stage(e)) · λ(e − 1), with λ(i) = (i + 1)/W for i < W and ½(1 + cos(π (i − W)/(T − W))) after. I2 honours
every declared parameter; a constant rate per stage (I1) would leave the declared warm-up and cosine without effect and
is not run.

## Design

### Variants

| Variant | Regimes | Flags | Runs | Purpose |
|---|---|---|---|---|
| **declared** (primary) | all four | `--e4-stage-lr-persist`; SUP-imnet also `--e4-unfreeze-sequential-blocks` | 240 | every regime follows the declared schedule |
| **persist-only** (secondary) | SUP-imnet | `--e4-stage-lr-persist` | 60 | separates, for SUP-imnet, the effect of the rates (head-only adaptation, as registered) from that of the unfreezing |

In SUP-imnet the declared variant changes two things at once (rates and unfreezing); C' − C mixes them. With the
persist-only variant, C'p − C isolates the rates and C' − C'p the unfreezing. For the MAE regimes the unfreezing flag
would be a no-op, so they are run once.

### Runs

- **Matrix:** `campaign_plan.stagelr_matrix(regimes, variant)`, phase `e4stagelr` of `run_campaign.py`
  (`--stagelr-variant declared|persistonly`). Each run is the registered E4 command line (budget config, partition,
  seed, regime, the registered `sun_full` parent of `checkpoints_v3/campaign/classification`) plus the flags.
- **Same as the registered E4:** parents (SHA-256 equal to the value recorded by the registered E4 runs, 60/60), packs
  (`data_packs/v3`, seed 1000k + s), seeds, configurations, batch sizes (4/8/16/32), early stopping, τ policies and
  evaluation. Only the flags change.
- **Code:** commit `61a3099` adds the two opt-in flags (`argparse.SUPPRESS`; without them the namespace, the config
  dump, the logs and every output are unchanged) and the matrix and phase. With `--e4-stage-lr-persist` the run block
  records `e4_stage_lr_persist: true` and the log a `[stage-lr] epoch=… head=… backbone=…` line per epoch; with
  `--e4-unfreeze-sequential-blocks`, `e4_unfreeze_sequential_blocks: true`.
- **Execution** (after the revision-1 tag; not started):
  - `python scripts/v3/run_campaign.py e4stagelr --stagelr-variant declared` then `--stagelr-variant persistonly`,
    in `ssl4polyp-ft`; no commits while a phase runs.
  - **GPUs (r1):** the two RTX 6000 Ada (GPUs 0 and 1), one run per GPU, from a shared queue as in the registered
    campaign; the GPU of every run is recorded (`RUN_COMPLETE.json`, `events.jsonl`).
  - **Failures (r1):** a failed run is retried once with the same command line and seed, and both attempts are
    recorded. `run_campaign.py` retries transient failures itself; any other failure stops the phase, which is then
    relaunched once (it resumes, and the failed run starts again from scratch). A second failure of the same run stops
    the experiment and is reported before anything else is done.
  - Outputs: `checkpoints_v3/exp_e4_stagelr/classification/` and `checkpoints_v3/exp_e4_stagelr_persistonly/classification/`
    (not in git); records `results_v3/exp_e4_stagelr/e4stagelr_{declared,persistonly}/`.
- **Verification of each phase:**
  - every run complete, with `[stage-lr]` lines satisfying |lr − r_g · λ(e − 1)| / max(lr, 1e-30) < 1e-9 in every
    epoch and group, backbone 0 in stage 1, and trainable backbone parameters 7,087,872 (one block, 50 frames) or
    14,175,744 (two blocks) in stage 2, in all regimes of the declared variant; 0 in stage 2 for persist-only;
  - the zero-shot outputs (arms A and B) byte-identical to those of the registered run (all runs);
  - the parent's SHA-256 equal to the value recorded by the registered E4 run;
  - the three flag-on control runs (k = 1, seed 13: SUP-imnet s100 and s50, SSL-colon s50) repeated in the declared
    phase byte-identical to the controls (every output CSV; weights compared through the SHA-256 of
    `results_v3/verification/exp_e4_stagelr_controls/files_manifest.csv`);
  - the integrity checks of `verify_phase.py e4stagelr --stagelr-variant …` (section 6 checks of the campaign plus the
    flags in the run block), and a results manifest with the path, bytes and SHA-256 of every file and checkpoint
    (`scripts/v3/batch_records.py`).
  - The rate, stage, arm A/B, parent and control checks are made by `scripts/v3/check_e4_stagelr_runs.py --variant …`
    (r1), which prints no metric; the weights are deleted only after all checks and the manifest.
- **Weights (r1):** the adapted weights are deleted once the manifest records their SHA-256. Parents and every
  registered file are untouched.
- **Expected duration:** the registered 240 adaptations took 2.2 GPU-hours (66 min on two GPUs). With the declared
  rates, early stopping may come later (at most 30–50 epochs of about 1 s): ≤ 4.5 GPU-hours for the 240 runs and
  ≤ 1 GPU-hour for the 60 persist-only runs. Disk: ~94 GB of weights at most before deletion; 277 GB free at planning.

## Estimands

Balanced accuracy at each arm's τ. Arms:
- **A, B** (unadapted): reused from the registered runs (zero-shot outputs and `e4_arms.A/B`);
- **C, D** (adapted, registered schedule): the registered runs, for C' − C and D' − D;
- **C', D'** (declared variant) and **C'p, D'p** (persist-only, SUP-imnet): recomputed, from the new runs' test
  outputs and `e4_arms.C/D` (C keeps the inherited τ; D re-estimates τ on the adapted support-validation scores and
  keeps the inherited τ when that split is degenerate, as in the campaign).

Per run (partition k, seed s), each contrast is the mean over its (regime, budget) pairs of the paired arm differences,
the registered definition of H7 (`analysis/v3/contrasts.py`). Reported:

1. **Declared variant:** H7' = C' − A, D' − A, D' − C', C' − C and D' − D:
   - global (4 regimes × 4 budgets);
   - per budget (mean of regimes);
   - per regime (mean of budgets) and per regime × budget, descriptive.
2. **Registered references**, computed by the same code on the registered runs: C − A (= H7), D − A, D − C, with the
   same breakdown.
3. **SUP-imnet:** C'p − A, D'p − A, D'p − C'p, C'p − C, D'p − D, C' − C'p and D' − D'p, per budget and mean of budgets,
   descriptive.
4. **Degenerate predictors** (specificity or sensitivity < 0.10 at τ, the registered rule) in C', D', C'p and D'p
   against the registered C (44 views) and D (3), per budget and regime.

Inference, with the registered functions:
- t-test on the 5 partition means (`mixed.fit_many`, variant `ttest`, SAP v3.3);
- cluster bootstrap of the E4 units (`design.bootstrap`, B = 2000, seed 12345). The bootstrap space is checked to be
  the registered e4 space, so the draws are those of the registered analysis; a replicate with an undefined run is
  discarded, as registered (1,691 valid replicates for contrasts involving the 500-frame budget).

Analysis code: `analysis/v3/exploratory_e4_stagelr.py` (`ssl4polyp-analysis`), written and checked before any run of
the experiment:

    PYTHONPATH=. python -m analysis.v3.exploratory_e4_stagelr --out results_v3/analysis_exploratory_e4_stagelr

The code is not changed after this plan, except for a bug fix that is reported.

## Interpretation fixed in advance

Applied only to the three global contrasts H7' (C' − A), D' − A and D' − C', each against the sign of its registered
counterpart (H7 = C − A, D − A, D − C, global). "The two intervals" are the t-test CI and the bootstrap CI; an
interval "excludes zero" on a side when it lies entirely on that side of zero. The categories are exclusive and
exhaustive:

- **Robust:** the estimate keeps the registered sign and both intervals exclude zero on that side.
- **Attenuated:** the estimate keeps the registered sign and at least one interval includes zero.
- **Not supported:** the sign changes; **reversed** when, in addition, both intervals exclude zero on the opposite
  side.

**Reference category (r1).** The same rule, applied to each registered contrast against its own sign, gives its
reference category: H7 = C − A **robust**, D − A **attenuated** (its bootstrap CI includes 0), D − C **robust**
(values in Controls, item 5). For each of the three new contrasts the report states whether its category is
**maintained**, **weakened** or **strengthened** with respect to the reference, with the order robust > attenuated >
not supported > reversed. No other claim is derived from these categories.

Per budget, per regime, the SUP-imnet variant contrasts and the degenerate counts are reported without a category.

## What will not be claimed

- That this replaces the registered H7, changes its verdict, or invalidates the registered E4 results: the registered
  analysis remains the confirmatory one.
- That the declared stage rates are good or optimal: they are not tuned here, and no schedule other than I2 is run.
- That the declared variant separates rates from unfreezing in SUP-imnet; only the persist-only variant does, and only
  for SUP-imnet.
- Anything beyond these packs, budgets, regimes and seeds.

## Controls (run before this plan was tagged)

Records under `results_v3/verification/` (`exp_e4_stagelr_controls/`, `campaign_plan_*exp_e4_stagelr*`,
`exp_e4_stagelr_analysis_check/`).

1. **Code, default path:**
   - unit check with the real models: the existing `tests/test_finetune_schedule.py` logic passes; without the flag the
     rates follow the registered pattern; with it they equal r_g · λ(e − 1) exactly (s500 schedule, 50 epochs);
     SUP-imnet with the unfreezing flag has 7,087,872 / 14,175,744 trainable backbone parameters in `head+1` /
     `head+2`, the same as the MAE regimes, with the final LayerNorm frozen; `bind_scheduler` is called before the first
     `apply_if_needed` (checked at run time) and the scheduler has one `lr_lambda` per optimizer group;
   - **dry run:** the 720 registered runs resolve byte-identically with the changed code and with HEAD `9548c8c`
     (namespace and prepared configuration, after normalising the root of the extracted HEAD tree); the 240 + 60 new
     runs resolve with no failure and differ from the registered E4 run only in the flags and in the output-derived
     paths (`output_dir`, `thresholds_root`); the 60 parents match the registered SHA-256.
2. **Bit-for-bit reproduction, flag off** (same GPU as the registered run): `E4_k1_sup_imnet_seed13_s100`,
   `E4_k1_ssl_colon_seed13_s50`, `E4_k1_ssl_imnet_seed13_s200`, `E4_k1_ssl_imnet_matched_seed13_s500`. All 36 CSVs of
   the four runs (test, validation, train and zero-shot outputs, pack copies) are byte-identical to the registered ones,
   and `e4_arms` (τ and test metrics of A–D) is equal. No `[stage-lr]` line and no new run-block key.
3. **Rates, flag on** (declared variant): SUP-imnet s100 (two blocks), SUP-imnet s50 (one block) and SSL-colon s50,
   k = 1, seed 13. In every epoch of the three runs the `[stage-lr]` rates satisfy the I2 formula (largest relative
   error 3.0e-11, the 10-digit print), the backbone is 0 in stage 1, and stage 2 has 14,175,744 / 7,087,872 / 7,087,872
   trainable backbone parameters; the run block records the flags.
4. **Arms A and B unchanged:** in those three runs, the zero-shot outputs (test, train, validation) are byte-identical
   to the registered ones and `e4_arms.A/B` is equal.
5. **Analysis identity check:** with the registered root in place of both new roots, the analysis reproduces H7 and
   the per-budget arm contrasts C − A, D − A, D − C (pooled and per regime) of
   `results_v3/analysis/contrast_summary.csv` (estimate, t-test CI, bootstrap CI and B), and every new − registered
   contrast (C' − C, D' − D, C'p − C, D'p − D, C' − C'p, D' − D'p) is exactly 0 in the estimate and in every replicate.
   Result (2026-11-04): 366/366 quantities equal (61 contrasts × 6; max |diff| 5.0e-11, the stored precision); the
   70 new − registered contrasts are exactly 0 in the estimate and in every finite replicate; the bootstrap space equals
   the registered e4 space (6 positive and 24 negative units); the 300 substituted runs have the registered evaluation
   frames, inherited τ and zero-shot outputs; degenerate counts reproduce `run_metrics.csv` (C 44, D 3). Registered
   references of the three ruled contrasts (global, mean of 4 regimes × 4 budgets): H7 = C − A −0.0748 (t-test CI
   −0.1012 to −0.0485; bootstrap CI −0.1353 to −0.0150), D − A −0.0213 (−0.0329 to −0.0097; −0.0801 to 0.0337),
   D − C 0.0536 (0.0334 to 0.0738; 0.0407 to 0.0626); bootstrap B = 1,691 for all three.
   Run again with the revision-1 code (2026-11-04): the same 366/366 quantities and zero contrasts; reference
   categories H7 robust, D − A attenuated, D − C robust, and "maintained" for the three contrasts, as expected when the
   new runs are the registered ones.

The control runs at k = 1, seed 13 are the same runs as the corresponding runs of the experiment; their outputs were
inspected only for the checks above (the balanced accuracy of C' and D' of the three flag-on runs was printed while
checking `e4_arms`). Their weights were deleted after recording their SHA-256 (`files_manifest.csv`).

## Outputs

- **Runs:** `checkpoints_v3/exp_e4_stagelr/classification`, `checkpoints_v3/exp_e4_stagelr_persistonly/classification`
  (not in git); records `results_v3/exp_e4_stagelr/`; verification and manifests `results_v3/verification/exp_e4_stagelr_*`.
- **Analysis:** `results_v3/analysis_exploratory_e4_stagelr/`.
- **RUNLOG and tags:** a RUNLOG entry per stage; tag `v3-cmpb-exp-e4-stagelr-plan` (this plan, the code and the
  checked analysis) and, at the end, `v3-cmpb-exp-e4-stagelr-results`.

## Revision 1 (2026-11-04)

Registered before any run of the experiment (tag `v3-cmpb-exp-e4-stagelr-plan-r1`). It closes the open points of the
original plan and adds the reference categories; the design, estimands and checks are otherwise unchanged.

- **Interpretation:** the rule is also applied to the registered contrasts, giving their reference categories (H7
  robust, D − A attenuated, D − C robust), and the report states whether each new category is maintained, weakened or
  strengthened. The operational definition is written out: "excludes zero" = the interval lies entirely on the side of
  the registered sign. The analysis code writes the reference category and the change
  (`interpretation.csv`, `summary.json`); the identity check was run again with this code (Controls, item 5).
- **Control runs:** the three flag-on control runs (SUP-imnet s100 and s50, SSL-colon s50; k = 1, seed 13) are repeated
  in the main campaign and must be byte-identical to the controls. The balanced accuracy of C' and D' of these three
  runs was seen before the main experiment was launched.
- **GPUs:** both RTX 6000 Ada, shared queue as in the registered campaign, GPU recorded per run.
- **Weights:** the adapted weights are deleted after their SHA-256 is recorded in the results manifest.
- **Failures:** a failed run is retried once with the same seed and recorded; a second failure stops the experiment.
- **RUNLOG:** an entry for the preparation (phase 2) and one for the results.
- **Superseded open points (original plan, SHA-256 `a2761d07b12f8f6b18dc87720124e61e6609da054996f142077bb6caf2885236`):**
  GPU assignment, reuse or repetition of the control runs, and the RUNLOG entry, all marked TO DECIDE.
