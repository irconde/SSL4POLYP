# RUNLOG, campaign v3 (CMPB)

## Pre-registration

| Document | Path | SHA-256 | Registered |
|---|---|---|---|
| Statistical analysis plan (SAP) | `docs/v3/plan_analisis_estadistico_v3.md` | `f2120a09687678bf04f3bb8575a4f16adb8b8613704adbb20478dd0215cc0474` | 2026-10-14, before phase 1 |
| Retraining specification v3.1 | `docs/v3/especificacion_reentrenamiento_v3_1.md` | `e614ce5385efd4e79ec5508ece43b75cf48a8ba26cd69ccd9159665576612d34` | 2026-10-14 |

Any change to the SAP after results are seen goes to `results_v3/SAP_deviations.md` (SAP section 9).

## Station and environment

- GPUs: 2 × NVIDIA RTX 6000 Ada Generation, 48 GB (driver 595.91.07). GPU 0 also drives the display.
- Conda environment `ssl4polyp`: Python 3.8.20, torch 2.4.1+cu121, torchvision 0.19.1+cu121,
  timm 0.3.2 (patched, see `env/README.md`), numpy 1.24.4, scikit-learn 1.0.2.
- Records: `env/environment.yml`, `env/pip_freeze.txt`, `results_v3/environment/`
  (`nvidia-smi.txt`, `nvidia-smi-q.txt`, `versions.json`).
- TF32 set explicitly in `config/base.yaml`: matmul = true, cuDNN = true.
- Initialisation weights (SHA-256 checked at every run):
  - SUP-imnet `checkpoints/pretrained/vit_b/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`: `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`
  - SSL-colon `checkpoints/pretrained/vit_b/mae_hyperkvasir.pth`: `faf5d0b1c67eeaf3ab333fc4dd0da23b3d294d998aa946a150ac59cf1d1d728f`
  - SSL-imnet `checkpoints/pretrained/vit_b/mae_imagenet.pth`: `aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d`
- `checkpoints/classification/clean_checkpoints.py` is disabled for this campaign (refuses `--force`).

## Phase 0

### 2026-10-14: data and pack verification

`scripts/v3/verify_data_packs.py` → `results_v3/verification/pack_hashes.csv`, `pack_verification.json`.
sun_full, sun_subsets (p05–p100), sun_morphology, polypgen_clean_test and polypgen_fewshot
(s50–s500): 39 split/hash checks, all CSV SHA-256 equal to the manifests (and to
`config/data/*.yaml` where declared), counts equal to the manifests, all images present with the
recorded size, PolypGen image SHA-1 equal to `data/polypgen/manifests/frames.csv`. The 24 distinct
CSV hashes stored in the v2 run snapshots equal the current packs.

### 2026-10-14: smoke test (partition k = 1), tag `v3-cmpb-smoke` (commit `d166ee2`)

GPU 1 (`CUDA_VISIBLE_DEVICES=1`), `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `scripts/v3/run_smoke.sh`.
Weights in `checkpoints_v3/classification/k1/` (8.7 GB), records in `results_v3/smoke/k1/`.

| Run | Start | Duration | Notes |
|---|---|---|---|
| a) sun_full, SSL-colon, seed 13 | 12:52:02 | 429 s (7.2 min) | early stop after epoch 24, best epoch 16, τ = 0.687906 |
| b) E3, polypgen_clean_test | 12:59:11 | 9 s | parent epoch 16, frozen τ |
| c) E4 s50, arms A–D | 12:59:20 | 30 s | early stop after epoch 13, best epoch 5 |

Section 6 smoke criteria: all 39 checks pass (`results_v3/smoke/k1/verification_section6.json`,
`scripts/v3/verify_run_outputs.py`): best-checkpoint reload logged, τ epoch = weights epoch,
evaluated checkpoint SHA-256 matches, val/test (and E4 support) CSVs with all spec 1.3 columns,
JSON metrics = CSV recomputation (max |diff| = 0), τ recomputed from the CSVs = stored τ
(sun_full primary, E4 arms B and D).

Incidents: none during the runs. Open points for phase 1 are listed in the smoke report
(environment differs from v2; checkpoint disk use).

### 2026-10-14: environments split

- `ssl4polyp-pretrain`: `conda create --clone` of the original environment, unchanged (timm 0.3.2); MAE
  pretraining of SSL-imnet-matched only. (`mamba create --clone` dropped the pip packages.)
- `ssl4polyp-ft`: clone of `ssl4polyp-pretrain` + `timm==0.4.12` (`--no-deps`) + matplotlib 3.7.5.
  All fine-tuning and evaluation runs. Original `ssl4polyp` environment removed.
- Networks built by the v2 code (`d678b28`) and the current code in `ssl4polyp-ft` are identical for the
  three regimes (`results_v3/verification/network_v2_vs_v3.json`).
- Code (tag `v3-cmpb-smoke2`, commit `e5f0d67`): timm .npz loader, only the best-epoch checkpoint kept,
  `negative_source = NA` for SUN.

### 2026-10-14: extended smoke test (k = 1, seed 13, three regimes), tag `v3-cmpb-smoke2`

`ssl4polyp-ft`, GPU 1, `scripts/v3/run_smoke.sh` (SMOKE_TAG=smoke2). Weights in
`checkpoints_v3/smoke2/` (4.0 GB, best-epoch checkpoints only), records in `results_v3/smoke2/k1/`.

| Model | sun_full | E3 | E4 s50 |
|---|---|---|---|
| SUP-imnet | 202 s (best epoch 3) | 8 s | 30 s |
| SSL-imnet | 358 s (best epoch 12) | 8 s | 34 s |
| SSL-colon | 428 s (best epoch 16) | 9 s | 29 s |

Section 6 criteria: 117/117 checks pass (`results_v3/verification/smoke2_section6.json`).

Comparison with v2 seed 13 (`results_v3/verification/smoke2_comparison_v2.csv`): |ΔAUROC| or |ΔAP| > 0.02
in SUP-imnet (sun_full, E3, E4 A/C), SSL-imnet (E4 A/C AUROC, +0.021/+0.022) and SSL-colon (E4 A/C).
Investigation (`results_v3/verification/smoke2_investigation_controls.sh`, runs in
`checkpoints_v3/investigation/`):
- evaluation frames, order and labels identical to v2 in every compared CSV;
- networks and initial weights identical to v2 code (above);
- v3 is deterministic: an identical repeat of SUP-imnet sun_full gives bit-identical val/test CSVs;
- TF32 off changes nothing in training (AMP fp16); for SSL-colon it only perturbs fp32 evaluation
  at ~1e-5 in the val loss, with no change in test metrics;
- epoch-1 train loss v2 vs v3: SSL-colon 0.4789/0.4793, SSL-imnet 0.6550/0.6597 (same data order and
  initialisation), then the trajectories diverge and the best epoch changes (SSL-colon 11→16,
  SSL-imnet 9→12, SUP-imnet 5→3). SUP-imnet's first epoch is unstable in v2 itself (train loss
  1.40/1.47/1.95 across seeds; v3 1.70);
- most flagged differences are within the v2 between-seed range; exceptions: SUP-imnet sun_full AUROC
  (−0.040 vs range 0.033, and v2 sun_full used last-epoch weights) and SSL-colon E4 AP (−0.127 vs
  range 0.071).
Conclusion: no code or data difference found; the gaps come from different trained weights under
torch 2.4.1 / RTX 6000 Ada versus torch 1.9.0 / A40 (fp16 kernels), amplified by training. v2 runs
cannot be reproduced seed by seed in this environment.

### 2026-10-15: power outages during the investigation

Several power cuts on 2026-10-14 interrupted the investigation control runs twice (partial outputs
deleted and the runs relaunched from scratch; one cut left `run_controls.sh` empty and it was
rewritten). The final control runs completed. Integrity check afterwards: `git fsck` clean, local
branch equal to `origin/refactoring`, `ssl4polyp-ft` loads torch/timm with CUDA, section 6 checks
re-run on all smoke2 and investigation runs (205/205 pass), the 11 kept checkpoints load, and the v2
comparison table reproduces exactly.

## Learning-rate selection (SAP v3.1)

### 2026-10-15: SAP v3.1 registered before the sweep

| Document | Path | SHA-256 |
|---|---|---|
| SAP v3.1 | `docs/v3/plan_analisis_estadistico_v3_1.md` | `a2fc1b8325885216a15b410792f689acf9805267f43342e8fb9b28968818e7bd` |

Supersedes SAP v3 (`f2120a09…cc0474`); adds section 10 (learning-rate selection) and deviation 9.1
(`results_v3/SAP_deviations.md`). Registered before any sweep run.
