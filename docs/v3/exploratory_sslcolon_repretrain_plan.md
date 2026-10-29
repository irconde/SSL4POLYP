# Exploratory plan: SSL-colon pretrained again in the SSL-imnet-matched environment (SSL-colon-r)

## Status

**Exploratory analysis, not registered in the SAP, decided on 2026-10-29 after seeing the results** of the registered
campaign v3 analysis (SAP v3.3) and of the three earlier exploratory experiments. It answers a review comment: the domain
contrast (H2) compares two pretrainings run in different conditions.

| | SSL-colon (registered) | SSL-imnet-matched |
|---|---|---|
| Data | HyperKvasir unlabelled, 99,417 images | ImageNet-1k subset, 99,417 images |
| PyTorch | earlier version, probably 1.9 (`torch._six`) | 2.4.1 (`ssl4polyp-pretrain`, timm 0.3.2) |
| GPU | another machine | RTX 6000 Ada (GPU 1) |
| Course | resumed once (`checkpoint-99.pth`, `start_epoch` 100) | continuous, epochs 0–399 |
| Records | none (training logs not kept) | `log.txt`, TensorBoard, `train.log` |

**Its results will not be used to confirm or refute hypotheses**: no Holm correction and no confirmation verdicts;
estimates and intervals are reported as observed. The registered analysis and its results do not change.

## Question

Do H2 and the other comparisons involving SSL-colon hold when SSL-colon is pretrained in the same environment as
SSL-imnet-matched?

## Declared deviation: rebuilt pretraining data

The preprocessed HyperKvasir-unlabelled set used for SSL-colon was not kept. SSL-colon-r is pretrained on a rebuild:
- **Source:** the official download (`hyper-kvasir-unlabeled-images.zip`, SHA-256 `684d0bbd…3ad21`).
- **Cleaning:** the documented cleaning, `polypdb.clean_hyperkvasir_unlabeled` with default parameters.
- **Record:** polyp-db-analysis commit `d917e41`, `docs/hyperkvasir_unlabeled_rebuild.md`.
- **Location:** `/workspace/data/hyperkvasir-unlabeled-r/`.
- **Hash list:** `docs/hyperkvasir_unlabeled/hyperkvasir_unlabeled_r_sha256.csv.gz` in that repository. SHA-256 of the
  uncompressed list: `d9ad754e868aabcc41663b41b27c64778c7fa4662a93c3dc65b59cc8a5782b25`.

The rebuild is **equivalent, not identical**: the OCR model and library versions of the original preprocessing are
unknown. That difference can contribute to any difference between SSL-colon-r and SSL-colon. It does not affect the
contrast between SSL-colon-r and SSL-imnet-matched, which are pretrained in identical conditions.

Checks done before this plan (2026-10-29, nothing trained):
- **Count:** the folder holds 99,417 files with the same names as the raw images. There are no missing or extra files.
- **Hashes:** the hash list recomputed from the files (`scripts/verify_hyperkvasir_rebuild.py` at `d917e41`) is
  byte-identical to the committed one (SHA-256 `d9ad754e…5782b25`).
- **Decoding:** all 99,417 images decode as RGB JPEG with PIL in `ssl4polyp-pretrain`.
- **Loader:** with one class folder pointing at the set (see Pretraining), `torchvision.datasets.ImageFolder` reads
  99,417 images. A `DataLoader` with batch 128 and `drop_last=True` then gives **776 iterations per epoch**, the same
  as for SSL-imnet-matched and the original SSL-colon.

## Design

### 1. Pretraining: SSL-colon-r

- **Script:** `checkpoints_v3/pretrain/mae_sslcolon_r/launch.sh`, a copy of
  `checkpoints_v3/pretrain/mae_imnet_matched/launch.sh` with only `--data_path`, `--output_dir` and `--log_dir` changed.
- **Arguments:** identical to SSL-imnet-matched. These are the arguments stored in `mae_hyperkvasir.pth`, apart from
  the paths and the resume (see the next item):
  - `mae_vit_base_patch16`, batch 128 × `accum_iter` 2, 400 epochs, 40 warmup epochs;
  - `blr` = `lr` = 1e-3, `min_lr` 0, weight decay 0.05, mask ratio 0.75, input 224, AMP, `--no_train_dir`;
  - seed 0, 8 workers, `pin_mem`;
  - checkpoints every 10 epochs and 15 min, `keep_last` 5, `keep_every_n_epochs` 50;
  - `world_size` 1.
- **No resume:** `--resume ""` and `--start_epoch 0`. The run is continuous from epoch 0 to 399. An interruption is
  reported, and the plan says what happens next.
- **Data path:** `ImageFolder` needs one level of class folders, while the rebuild is flat. `data_path` is therefore a
  folder `checkpoints_v3/pretrain/mae_sslcolon_r/data/` holding one symbolic link `images ->
  /workspace/data/hyperkvasir-unlabeled-r`. The original layout (`unlabeled-images/images/`) also has a single class
  folder. The data folder itself is not modified.
- **Environment:** `ssl4polyp-pretrain` (torch 2.4.1, timm 0.3.2), MAE code as used for SSL-imnet-matched. That code is
  unchanged since its commit (`d109d28`, formerly `32ca324`); only an unrelated demo notebook changed since.
- **Hardware:** one RTX 6000 Ada, GPU 1 (the card SSL-imnet-matched used), with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Records:**
  - `ckpts/log.txt` (per-epoch mean loss), TensorBoard and `train.log`;
  - a launch record with the commit and start time;
  - `ckpts/ckpts/checkpoint-399.pth`, copied to `checkpoints/pretrained/vit_b/mae_hyperkvasir_r.pth`, with its SHA-256;
  - checks on the final checkpoint: stored arguments, epoch 399, empty `resume`, `start_epoch` 0, all weights finite,
    and it loads into the fine-tuning backbone.
- **Expected duration:** about 18 h (SSL-imnet-matched: 17 h 52 min for the same 400 × 776 iterations on the same GPU).

### 2. Fine-tuning campaign of SSL-colon-r (180 runs)

- **Regime and learning rate:** `ssl_colon_r` (SSL-colon-r), fine-tuning learning rate **1e-4** (= SSL-colon's
  registered rate, not chosen again).
- **Runs**, the same experiments, packs (v3), partitions k = 1–5 and seeds 13, 29, 47 as SSL-colon in the registered
  campaign:

  | Phase | Runs |
  |---|---|
  | sun_full (full budget) | 15 |
  | E1 (5, 10, 25, 50, 100 %) | 75 |
  | E2 / E3 (evaluation of the sun_full models) | 15 + 15 |
  | E4 (s50, s100, s200, s500, from the sun_full parents) | 60 |
  | **Total** | **180** |

- **Code and configuration:** training code and `ssl4polyp-ft` unchanged. The preparation before launch, done after
  approval and committed before the first run, follows how SSL-imnet-matched was added (`849c096`):
  1. `config/model/ssl_colon_r.yaml`: key `ssl_colon_r`, pretraining `Hyperkvasir` (already handled as an MAE
     initialisation), checkpoint `checkpoints/pretrained/vit_b/mae_hyperkvasir_r.pth` with its SHA-256,
     `finetune_lr: 1.0e-4`.
  2. `model/ssl_colon_r` added to the model lists of `config/exp/v3/{sun_full,e1,e2,e3,e4}.yaml`.
  3. `scripts/v3/campaign_plan.py`, `run_campaign.py`, `verify_phase.py`: `ssl_colon_r` accepted by `--regimes`. It is
     not added to the default regime list, so the registered campaign's commands resolve exactly as before.
- **Before launch:**
  - **Dry run:** the planner resolves the 180 runs with the same packs, splits, seeds, fine-tuning mode, threshold
    policy and E4 parents as SSL-colon, and lr 1e-4, with no failures.
  - **Smoke config:** for one run per phase, the resolved configuration differs from SSL-colon's only in the model key,
    tag and checkpoint (path and SHA-256).
- **Execution:**
  - `run_campaign.py <phase> --regimes ssl_colon_r --root checkpoints_v3/exp_sslcolon_r/classification --records
    results_v3/exp_sslcolon_r`, phases in the order sun_full, e2e3, e1, e4;
  - GPUs 0 and 1, one run per GPU;
  - failed runs retried once and reported;
  - no commits while a phase runs.
- **Verification:**
  - each phase passes the section 6 checks of the registered campaign (`verify_phase.py`);
  - each phase gets per-run records and a results manifest with the SHA-256 of every file and checkpoint
    (`scripts/v3/batch_records.py`).
- **Weights:**
  - **E1:** the best-epoch weights are deleted once the E1 phase passes its checks and its manifest records each
    checkpoint's SHA-256 (as in the per-budget experiment).
  - **sun_full:** checkpoints are kept, because they are the parents of E2/E3 and E4.
  - **E4:** weights are kept.
  - No other experiment's weights are touched.
- **Expected duration and disk:**
  - about 7.7 GPU-hours, or ~4 h on two GPUs (SSL-colon in campaign 3: 7.68 GPU-hours for its 180 runs);
  - peak ~118 GB of fine-tuning checkpoints plus ~16 GB of pretraining checkpoints, ~75 GB after E1 deletion;
  - free at planning: 334 GB.

## Estimands

All are computed over the same pairs (partition × seed) as the registered analysis, with the registered inference:
- estimate = mean of the paired per-run values;
- t-test CI on the 5 partition means (SAP v3.3, variant `ttest`);
- cluster bootstrap CI per family (B = 2000, seed 12345).

1. **H1–H7 with SSL-colon-r in place of SSL-colon.** The definitions and code are the registered ones
   (`analysis/v3/contrasts.py`, `hypothesis_contrasts`). H5–H7 are means over the four regimes, so they change through
   SSL-colon-r. Three versions per hypothesis:
   - `r`;
   - `registered`, which must reproduce `results_v3/analysis/contrast_summary.csv`;
   - `diff` = r − registered.
2. **Between-pretraining variability**, as paired SSL-colon-r − SSL-colon differences:
   - AULC of AUPRC in E1 (= diff of H1 = diff of H2);
   - AUROC in the E2 flat stratum (= diff of H3);
   - zero-shot AUROC on `polypgen_clean_test`, E3 (= diff of H4).
3. **Pretraining loss curves:** mean train loss per epoch of SSL-colon-r against SSL-imnet-matched (`log.txt`),
   reported per epoch and at epochs 0, 50, …, 350, 399. No loss curve exists for the original SSL-colon.

Analysis code: `analysis/v3/exploratory_sslcolon_r.py` (ssl4polyp-analysis), written and checked before any
pretraining or fine-tuning:

    PYTHONPATH=. python -m analysis.v3.exploratory_sslcolon_r --out results_v3/analysis_exploratory_sslcolon_r

Check (identity substitution): run with the original SSL-colon in place of SSL-colon-r (`--r-root
checkpoints_v3/campaign/classification --r-regime ssl_colon`). Then `r` and `registered` must both equal the registered
H1–H7 (estimate, t-test CI, bootstrap CI and B, at the 10 significant digits of `contrast_summary.csv`), and `diff` must
be 0 up to floating-point rounding (|diff| < 1e-15).
- **Result (2026-10-29):** 84/84 quantities equal (max |diff| 4.7e-11, the stored precision); `diff` at most 3e-17 in
  absolute value. The loss-curve code read 400/400 epochs of each log, checked by comparing the SSL-imnet-matched log
  with itself.
- **Record:** `results_v3/verification/exp_sslcolon_r_analysis_check/`.
- **Implementation detail:** to reproduce the registered resamples, each family's registered secondary contrasts are
  bootstrapped alongside the hypotheses (1,757 contrasts, not reported). Each family's resampling spaces are then
  exactly those of the registered analysis.

The code is not changed after this plan, except for a bug fix that is reported.

## Interpretation fixed in advance

"Null" = both CIs (t-test and bootstrap) include 0. "Negative" or "advantage" = both CIs below or above 0. "Differs
much" = both CIs of the paired SSL-colon-r − SSL-colon difference in AULC of AUPRC exclude 0, or its absolute value
exceeds that of the registered H2 (0.0138).

- **(a)** H2 with SSL-colon-r is still null or negative → the environment confounder is ruled out as the explanation
  of the H2 result.
- **(b)** SSL-colon-r differs much from SSL-colon → the variability between pretrainings matters, and H2 is presented
  with that caution. That difference mixes environment, the rerun itself and the preprocessing (declared deviation), and
  this experiment cannot separate them.
- **(c)** H2 with SSL-colon-r shows a domain advantage → the conclusion about the domain is not robust.

(a) and (b) can occur together. The other hypotheses (H1, H3–H7) are reported with their `diff`, without a separate
rule.

## Outputs

- **Pretraining:** `checkpoints_v3/pretrain/mae_sslcolon_r/` (not in git, except records copied to
  `results_v3/verification/`).
- **Campaign:**
  - runs under `checkpoints_v3/exp_sslcolon_r/classification` (not in git);
  - records under `results_v3/exp_sslcolon_r/`;
  - verification, per-run records and manifests: `results_v3/verification/exp_sslcolon_r_*`.
- **Analysis:** `results_v3/analysis_exploratory_sslcolon_r/`.
- **RUNLOG and tags:** a RUNLOG entry per stage; tag `v3-cmpb-exp-sslcolon-r-plan` (this plan and the analysis code) and,
  at the end, `v3-cmpb-exp-sslcolon-r-results`.
