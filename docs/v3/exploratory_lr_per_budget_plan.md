# Exploratory plan: learning-rate selection per E1 budget

## Status

**Exploratory analysis, not registered in the SAP, decided on 2026-10-28 after seeing the results** of the registered
campaign v3 analysis (SAP v3.3, `v3-cmpb-analysis-results`) and of the two earlier exploratory experiments
(`v3-cmpb-exp-lr-results`, `v3-cmpb-exp-lr2-results`). It answers a review comment: the fine-tuning learning rate of
each regime was chosen once (SAP v3.1 section 10 sweep, sun_full, all training data) and applied to every E1 budget,
so the size of H1 could depend on that choice. **Its results will not be used to confirm or refute hypotheses**: no
Holm correction and no confirmation verdicts; estimates and intervals are reported as observed.

## Question

If each regime uses its best learning rate for each E1 budget, do H1 and the between-regime E1 contrasts change?

## Design

- **Regimes:** SUP-imnet (`sup_imnet`), SSL-imnet (`ssl_imnet`), SSL-colon (`ssl_colon`), SSL-imnet-matched
  (`ssl_imnet_matched`).
- **Budgets:** E1 at 5, 10, 25, 50 and 100 % of the SUN training split (nested `sun_subsets` of packs v3).
- **Learning rates:** 1e-4, 3e-4 and 1e-3 (the grid of the SAP v3.1 section 10 sweep).
- **Partitions and seeds:** k = 1–5 × seeds 13, 29, 47 (15 runs per regime, budget and learning rate).
- **Reused runs:** the runs at each regime's registered learning rate (SUP-imnet 3e-4, SSL-imnet 3e-4, SSL-colon 1e-4,
  SSL-imnet-matched 1e-4) are the E1 runs of the registered campaigns (`checkpoints_v3/campaign/classification`,
  campaigns 3 and 4), unchanged.
- **New runs (600):** the other two learning rates of each regime: 4 regimes × 2 learning rates × 5 budgets ×
  15 runs.

  | Learning rate | Regimes | Runs |
  |---|---|---|
  | 1e-3 | all four | 300 |
  | 1e-4 | SUP-imnet, SSL-imnet | 150 |
  | 3e-4 | SSL-colon, SSL-imnet-matched | 150 |

- **Everything else identical to the registered campaign:**
  - training code and the `ssl4polyp-ft` environment;
  - packs v3 (`v3-cmpb-packs2`) and seeds;
  - classification head, augmentation and optimiser;
  - early stopping on the validation loss and best-epoch selection;
  - F1-optimal threshold on validation;
  - the section 6 integrity checks.

  The learning rate is the only change. It is set as an explicit `--override lr=<value>` and recorded in each run's
  JSON (`run.lr`, `run.lr_source` = `override`); `config/model/*.yaml` is not modified.

## Execution

- **Launcher:** `scripts/v3/run_campaign.py e1 --regimes … --lr-override <lr> --root
  checkpoints_v3/exp_lr_per_budget/lr<lr>/classification --records results_v3/exp_lr_per_budget/lr<lr>`, one
  invocation per learning rate (three in total), on GPUs 0 and 1 with one run per GPU. The registered campaign roots
  are not written to.
- **Dry run:** before launching, the campaign planner (`scripts/v3/campaign_plan.py`) resolves all 600 runs with the
  expected learning rate and source `override`, with no failures.
- **Smoke test:** one run per learning rate (k = 1, seed 13, 5 % budget) is compared with the equivalent registered run.
  The resolved configurations must differ only in the learning rate and its source.
- **Verification:** each batch must pass the section 6 checks of the registered campaign (`scripts/v3/verify_phase.py`
  e1, including the learning rate) and get a results manifest with SHA-256 hashes.
  - Failed runs are retried once and reported.
  - No commits are made while a batch is running.
- **Storage (the only difference from the registered campaign, with no effect on the computation):** the 600
  best-epoch checkpoints (about 1 GB each) do not fit on the disk (416 GB free).
  - The batches run one learning rate at a time.
  - Once a batch passes its checks and its manifest, which records each checkpoint's SHA-256, is written, its
    checkpoints are deleted.
  - E1 models have no child runs, and the analysis uses only the per-frame CSVs and metrics JSONs, which are kept.
- **Reproducibility check (declared, not used for inference):** 300 of the new runs have an earlier equivalent with the
  same code, packs and learning rate:
  - the 225 E1 runs at 1e-3 of the common-lr experiment (SUP-imnet, SSL-imnet, SSL-colon);
  - the 75 SSL-imnet E1 runs at 1e-4 of the SSL-imnet lr 1e-4 experiment.

  Their validation and test CSVs are compared and the agreement is reported. The analysis uses the new runs.

## Selection rule

For each regime and budget, the selected learning rate is the one with the highest mean validation AP of the
best-epoch weights over its 15 runs.
- The validation AP is the `val.auprc` of each run's metrics JSON, checked against its recomputation from the
  validation CSV, as in the section 10 sweep.
- The validation split is the partition's `base/val.csv`, which is the same for every budget.
- If the best mean exceeds the registered learning rate's mean by less than 1e-4, the registered learning rate is
  kept.
- The test split takes no part in the selection.

## Estimands

With the per-budget selection, for SSL-colon − SUP-imnet (H1), SSL-colon − SSL-imnet and SSL-colon −
SSL-imnet-matched:
- the AULC of AUPRC over 5–100 %;
- the AUPRC at each budget.

For each estimand, three versions on the same pairs (partition × seed):
- per-budget selection;
- registered;
- their paired difference (selection − registered).

Also reported: the table of selected learning rates, and the mean validation AP for every regime, budget and learning
rate.

## Inference

As in the registered analysis and the earlier exploratory analyses, using the `analysis/v3` code unchanged:
- the estimate is the mean of the per-run paired differences;
- the 95 % CI comes from the one-sample t-test on the 5 partition means (df = 4);
- the 95 % cluster bootstrap CI uses B = 2000 and seed 12345, with a shared case sample;
- the AULC uses the registered trapezoid weights (`aulc_weights`).

There is no Holm correction and there are no confirmation verdicts.

The registered values are recomputed by the same code and must equal those of the registered analysis
(`results_v3/analysis/`). Any difference is reported.

## Scope and output

- **Scope:** E1 only.
  - If the rule selects, at 100 %, a learning rate different from the registered one for any regime, this is
    reported, but sun_full, E2, E3 and E4 are not redone.
  - E2 and E3 use the sun_full models; E4 has its own staged schedule.
- **Analysis code:** `analysis/v3/exploratory_lr_per_budget/`, committed before it is run on the real runs.
- **Output:** `results_v3/analysis_exploratory_lr_per_budget/`:
  - the validation AP table with the selected learning rate marked;
  - the estimands in their three versions with t-test and bootstrap CIs;
  - the check of the recomputed registered values.
- **Tags:** `v3-cmpb-exp-lr-budget-plan` (this plan, before any run) and `v3-cmpb-exp-lr-budget-results`.

## Limitations

- **Hypothesis-generating:** decided after seeing results.
- **Selection and evaluation share the validation split:** the learning rate is selected on the same validation
  split used for early stopping and the threshold, and the selected configuration is not re-evaluated on independent
  validation data. The maximum over three learning rates of a mean validation AP is optimistically biased for the
  selected configuration. Test results are not affected by this bias, because the test split takes no part in the
  selection.
- **Asymmetric reuse:** the runs at each regime's registered learning rate were trained earlier, on the commits of
  campaigns 3 and 4. Since campaign 3, the fine-tuning code changed only to accept the SSL-imnet-matched
  initialisation; the other source changes affect MAE pretraining or are an import fallback. The reproducibility check
  above measures any non-determinism between launches.

## Revision 1 (2026-10-28)

**Reason:** avoid redundant runs, given that the fine-tuning is deterministic (verified below), and fix the analysis
code before any new data exist. Registered before any run of the experiment was launched. This section supersedes the
sections above where they differ.

1. **Reused runs instead of repeated ones.** The 300 existing E1 runs that the original design would have repeated are
   reused:
   - 1e-3 for SUP-imnet, SSL-imnet and SSL-colon: the 225 E1 runs of the common-lr experiment
     (`checkpoints_v3/campaign_lr_common/classification`);
   - 1e-4 for SSL-imnet: the 75 E1 runs of the SSL-imnet lr 1e-4 experiment
     (`checkpoints_v3/campaign_lr_sslimnet_1e-4/classification`).

   They count as valid only after two checks.
   - **a) Configuration and code.** For each of the 300 runs, the configuration it recorded (`config.yaml`) and its
     command line (`RUN_COMPLETE.json`) must equal those the equivalent new run would have, apart from the output path.
     The new run's configuration is resolved by the training code's own `prepare_run` (`scripts/v3/check_reused_runs.py`).
     The fine-tuning code of the commits they were trained on must not differ from the current one.
   - **b) Bit-for-bit reproduction.** Four of them are trained again with the command line of the equivalent new run
     (`scripts/v3/repro_reused_runs.py`, k = 1, seed 13):
     - SUP-imnet 1e-3 at 5 %;
     - SSL-imnet 1e-3 at 10 %;
     - SSL-colon 1e-3 at 25 %;
     - SSL-imnet 1e-4 at 50 %.

     Their validation and test per-frame output CSVs must be byte-identical to the originals. Any mismatch stops the
     experiment.

   **New runs: only the remaining 300**, in batches of one learning rate:

   | Batch | Regimes | Runs |
   |---|---|---|
   | 1e-4 | SUP-imnet | 75 |
   | 3e-4 | SSL-colon, SSL-imnet-matched | 150 |
   | 1e-3 | SSL-imnet-matched | 75 |

   The reproducibility check of the original design (comparing new runs with existing equivalents) is replaced by
   check b.
2. **Weights.** Each batch's best-epoch checkpoints are deleted once the batch passes the section 6 checks and its
   manifest records every checkpoint's SHA-256. The weights of any other experiment are not touched.
3. **Analysis code fixed before the data:** `analysis/v3/exploratory_lr_per_budget/`, committed with this revision.
   - It was tested on the registered runs and the 300 reused runs only:
     - with the selection disabled (`--selection registered`);
     - with the selection restricted to those runs.
   - In both tests it must reproduce the registered H1 and E1 contrasts of the registered analysis
     (`results_v3/analysis/contrast_summary.csv`). The test outputs are in
     `results_v3/verification/exp_lr_per_budget_analysis_check/`.
   - It is run on the new runs only after all three batches have finished and passed their checks.
   - Every contrast references only E1 views, so its cluster bootstrap draws the same case resamples as the registered
     E1 family (same seed, same resampling units).
4. **Selection data:** for each regime and budget, the candidates are the registered runs, the reused runs and the new
   runs, 15 runs per learning rate in every case. The selection rule, estimands, inference and scope are unchanged.
