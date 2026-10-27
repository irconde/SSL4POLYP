# SSL4POLYP

Reference implementation and run records of the article

> **Self-supervised pretraining for colonoscopy polyp classification: a pre-registered controlled evaluation
> of domain, scale and deployment protocol.** Vincent Yip, Ahmed Abuhalimeh, Celso Campos, Florentino Fdez-Riverola
> and Ivan Rodriguez-Conde. 2026, submitted to *Computer Methods and Programs in Biomedicine*.

This repository holds **campaign v3**, the retraining from scratch whose results are reported in the article.
The code that produced the earlier version of the study (v2) is preserved at the tags `v2-asoc-2025` and
`v2-eaai-2026`; its README is kept in [docs/legacy/README_v2.md](docs/legacy/README_v2.md).

The repository started as a fork of [SSL4GIE](https://github.com/ESandML/SSL4GIE) (Sanderson and
Matuszewski, IEEE Access 2024). See [NOTICE](NOTICE) for third-party code and licenses.

## Study summary

A ViT-B/16 frame classifier (polyp / no polyp) is fine-tuned from four initialisations:

| Regime | Initialisation | Pretraining data |
|---|---|---|
| SUP-imnet | Supervised AugReg ViT-B/16 (ImageNet-21k, fine-tuned on ImageNet-1k) | ImageNet |
| SSL-imnet | MAE | ImageNet-1k (1.28 M images) |
| SSL-colon | MAE | HyperKvasir unlabelled (99,417 images) |
| SSL-imnet-matched | MAE, same arguments as SSL-colon | Class-stratified ImageNet-1k subset of 99,417 images |

SSL-imnet-matched isolates the effect of the pretraining domain from the amount of pretraining data.

Experiments (SUN Colonoscopy Video Database for training; PolypGen for external evaluation):

| Id | Question | Design |
|---|---|---|
| sun_full | Reference models | Full SUN training split of each partition |
| E1 | Sample efficiency | Nested SUN training subsets of 5, 10, 25, 50 and 100 % |
| E2 | Morphology | sun_full models evaluated on flat and polypoid strata (no new training) |
| E3 | Zero-shot transfer | sun_full models on `polypgen_clean_test`, τ inherited from SUN |
| E4 | Few-shot adaptation | PolypGen budgets of 50, 100, 200 and 500 frames; arms A (unadapted, inherited τ), B (unadapted, recalibrated τ), C (adapted, inherited τ), D (adapted, recalibrated τ) |

Replication: K = 5 independent case-level SUN partitions (morphology-stratified) × 3 seeds (13, 29, 47)
× 4 regimes = 720 runs. The seven primary hypotheses (H1–H7), the metrics, the inference (one-sample
t-test on the K partition means, Holm correction, cluster bootstrap) and the secondary analyses are fixed
in the pre-registered statistical analysis plan (SAP, below). Every step, decision and incident of the
campaign is recorded in [results_v3/RUNLOG.md](results_v3/RUNLOG.md).

## Repository layout

| Path | Content |
|---|---|
| `src/ssl4polyp/` | Python package: fine-tuning and evaluation (`classification/`), models and MAE (`models/`), config loading, pack utilities |
| `config/` | Layered YAML configs; campaign experiments in `config/exp/v3/`, regimes in `config/model/` |
| `data_packs/` | Split definitions (CSV) and manifests; campaign v3 packs in `data_packs/v3/` |
| `scripts/v3/` | Campaign v3 drivers and verification scripts |
| `analysis/v3/` | Statistical analysis (SAP v3.3), simulator, figures and their tests |
| `docs/v3/` | Retraining specification, SAP versions and exploratory plans (in Spanish) |
| `env/` | Conda environments and exact package lists |
| `results_v3/` | Run log, SAP deviations, verification records, analysis outputs and final figures |
| `tests/` | Package tests (`tests/legacy/`: v2 tests, skipped by default) |
| `legacy/` | v2 files kept for reference: `reporting/` helpers of the v2 reports, v2 `environment.yml` |

Weights (`checkpoints/`, `checkpoints_v3/`) and data (`data/`) are not distributed.

## Installation

Three conda environments, all described in [env/README.md](env/README.md):

| Environment | File | Use |
|---|---|---|
| `ssl4polyp-ft` | `env/environment-ft.yml` (exact: `env/pip_freeze-ft.txt`) | All fine-tuning and evaluation runs. Python 3.8, torch 2.4.1+cu121, timm 0.4.12 |
| `ssl4polyp-pretrain` | `env/environment-pretrain.yml` (exact: `env/pip_freeze-pretrain.txt`) | MAE pretraining only. timm 0.3.2 with a one-line patch (see `env/README.md`) |
| `ssl4polyp-analysis` | `env/environment-analysis.yml` (lock: `env/environment-analysis.lock.yml`, `env/r_packages-analysis.txt`) | Statistical analysis and figures. Python 3.11, R 4.4 (lme4, lmerTest, pbkrtest); no torch |

```bash
conda env create -f env/environment-ft.yml
conda activate ssl4polyp-ft
pip install --no-deps -e .            # from the repository root

conda env create -f env/environment-pretrain.yml   # then apply the timm patch in env/README.md
conda activate ssl4polyp-pretrain && pip install --no-deps -e .

conda env create -f env/environment-analysis.yml   # no package install: run with PYTHONPATH=.
```

Install the package with `--no-deps` so pip keeps the pinned versions. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before training (deterministic algorithms; the drivers set it). The
campaign ran on 2 × NVIDIA RTX 6000 Ada (48 GB); `results_v3/environment/` records drivers and packages.

## Data

### Sources

Obtain the datasets from their providers, under their terms of use:

- **SUN Colonoscopy Video Database** (training and E1/E2): <https://sundatabase.org>
- **PolypGen** (E3/E4): <https://www.synapse.org/#!Synapse:syn26376615/wiki/613312>
- **HyperKvasir**, unlabelled images (SSL-colon pretraining): <https://datasets.simula.no/hyper-kvasir>
- **ImageNet (ILSVRC-2012)**, training set (SSL-imnet-matched pretraining): <https://image-net.org>. Campaign v3
  read it from the `ILSVRC/imagenet-1k` Parquet copy on Hugging Face, revision
  `49e2ee26f3810fb5a7536bbf732a7b07389a47b5` (see *Reproduction*).

Pretrained weights go under `checkpoints/pretrained/vit_b/` with the file names and SHA-256 listed in
`config/model/*.yaml`; the training code refuses a file whose hash differs. SUP-imnet is the AugReg
`.npz` file named in `config/model/sup_imnet.yaml`; SSL-imnet is the MAE ViT-B/16 ImageNet-1k pretraining
checkpoint released with MAE; SSL-colon (`mae_hyperkvasir.pth`) and SSL-imnet-matched
(`mae_imnet_matched.pth`) were pretrained by the authors (see *Reproduction*).

### Dataset roots

Image paths in the pack CSVs are relative to named roots. Map them to local directories in
`data/roots.json` (not tracked; `data/roots.example.json` is a template):

```json
{
  "sun": "/path/to/data/sun",
  "polypgen_clean": "/path/to/data/polypgen"
}
```

### Packs

The packs in `data_packs/` were generated with the `polypdb` tool of the authors' `polyp-db-analysis`
repository, commit `d78ed76e6a70d22631aace92bfe9bf9689f2b496` (not part of this repository). Each
`manifest.yaml` records the generator commit, the exact command, the seed, the selection policy and the
SHA-256 of every split CSV.

> **Generator commit hashes.** The history of `polyp-db-analysis` was later rewritten, with identical file
> contents, so the commits cited in the manifests now have other hashes:
>
> | Cited in manifests | Current commit | Tag |
> |---|---|---|
> | `7de86f7` | `6dbcb2b` | `v2-packs` |
> | `dafb5a5` | `4486226` | `v2-subsets-fewshot` |
> | `d78ed76` | `73729bf` | `v3-packs` |
>
> Full mapping: [docs/history-rewrite.csv](https://github.com/irconde/polyp-db-analysis/blob/main/docs/history-rewrite.csv)
> in `polyp-db-analysis`.

Campaign v3 uses `data_packs/v3/`:

- `k1`…`k5/base`: SUN partitions (`polypdb sun morph`, 33 frames per case, partition seeds 47, 101, 102,
  103, 104);
- `k*/sun_subsets`: nested E1 subsets (seed 13);
- `k*/polypgen_fewshot/seed{13,29,47}/s{50,100,200,500}`: E4 packs (seed 1000·k + s);
- `_parent/`: the PolypGen parent packs; `REPORT.txt`: the generation report.

`data_packs/v3/index.yaml` lists the seeds and the SHA-256 of all 221 CSVs. The CSVs are stored byte for
byte (CRLF line endings, `-text` in `.gitattributes`). To verify them (`ssl4polyp-ft`):

```bash
python scripts/v3/verify_data_packs.py --v3 --out <dir>                 # CSVs and images
python scripts/v3/verify_data_packs.py --v3 --skip-images --out <dir>   # CSVs only (no data needed)
```

For each of the 91 pack directories of `data_packs/v3/` the script checks every CSV SHA-256 against
`index.yaml` and against the pack's `manifest.yaml`, the frame, positive and negative counts, the parent hashes
declared by the subsets and few-shot packs, and that the index and the files on disk list the same 221 CSVs.
Without `--skip-images` it also checks that every referenced image exists under the roots of `data/roots.json`,
that its size matches the CSV and, for PolypGen, its SHA-1 against `manifests/frames.csv`. It writes
`pack_hashes_v3.csv` and `pack_verification_v3.json` to `--out`, prints the summary (`"passed": true`, no
`failures`) and exits non-zero on any mismatch. Without `--v3` it checks the top-level packs of `data_packs/`
(from which v3 partition k = 1 and the PolypGen parent derive) and the configs; the default `--out` is
`results_v3/verification/`, which holds the recorded results, so pass another directory to keep them intact.

### Placeholders in the records

Absolute paths of the authors' workstation were replaced in the manifests and run records
(`scripts/v3/release_depersonalize_paths.py`): `<REPO>` is the repository root, `<DATA_ROOT>` the data
directory (cf. `data/roots.json`), `<CONDA_ROOT>` the conda installation and `<HOME>` any other home
directory path. Only the manifests changed, not the CSVs, so every CSV hash above is unchanged. The
`parent_pack.manifest_sha256` fields of the few-shot manifests keep the hash of the parent manifest as it
was when the packs were generated (before the replacement); it is a provenance record and no longer equals
the hash of the edited parent manifest. Recorded commands contain `<DATA_ROOT>`, so substitute your own
paths before running them.

## Reproduction

All commands run from the repository root. The order below is the order of the campaign; the dates,
commits and checks of each step are in `results_v3/RUNLOG.md`.

### 1. MAE pretraining (`ssl4polyp-pretrain`)

SSL-colon (`mae_hyperkvasir.pth`) was pretrained on HyperKvasir unlabelled images before campaign v3
(see `src/ssl4polyp/models/mae/run_hyperkvasir_pretraining.py`). SSL-imnet-matched reuses its stored
arguments on a 99,417-image ImageNet subset:

```bash
# ImageNet Parquet shards (Hugging Face) to class folders, then the class-stratified subset (seed 99417)
python scripts/v3/imagenet_hf_to_folders.py --parquet-dir <imagenet-1k>/data --out <DATA_ROOT>/imagenet_train \
    --records results_v3/imagenet/conversion
python scripts/v3/imagenet_matched_subset.py --source <DATA_ROOT>/imagenet_train \
    --out <DATA_ROOT>/imagenet_matched_99417 --records results_v3/imagenet/matched --seed 99417
# expected SHA-256 of subset_list.csv: 2d987bb096b9c2f0a72be1337282d66a1e176eb4fb7de047cb4a7211230a5030

CUBLAS_WORKSPACE_CONFIG=:4096:8 python src/ssl4polyp/models/mae/main_pretrain.py \
  --model mae_vit_base_patch16 --batch_size 128 --accum_iter 2 --epochs 400 --warmup_epochs 40 \
  --blr 1e-3 --lr 0.001 --min_lr 0.0 --weight_decay 0.05 --mask_ratio 0.75 --input_size 224 \
  --seed 0 --no_train_dir --num_workers 8 --pin_mem --precision amp \
  --save-freq-epochs 10 --save-freq-mins 15 --keep-last 5 --keep-every-n-epochs 50 \
  --device cuda --world_size 1 --local_rank -1 --dist_url env:// \
  --data_path <DATA_ROOT>/imagenet_matched_99417 \
  --output_dir checkpoints_v3/pretrain/mae_imnet_matched/ckpts --log_dir checkpoints_v3/pretrain/mae_imnet_matched/tb
```

The checkpoint of epoch 399 is `checkpoints/pretrained/vit_b/mae_imnet_matched.pth`
(SHA-256 `964089f6…1504c328`; one RTX 6000 Ada, about 18 h).

### 2. Fine-tuning campaign (`ssl4polyp-ft`)

```bash
bash scripts/v3/run_smoke.sh                      # optional smoke test, partition k = 1, one seed
bash scripts/v3/run_lr_sweep.sh                   # SAP section 10 learning-rate sweep (validation only)
python scripts/v3/summarize_lr_sweep.py           # decision rule; chosen rates are in config/model/*.yaml
python scripts/v3/campaign_plan.py                # dry run: resolves all runs and checks them against the SAP
for phase in sun_full e2e3 e1 e4; do              # one run per GPU, resumable
  python scripts/v3/run_campaign.py "$phase" --gpus "0 1"
  python scripts/v3/verify_phase.py "$phase"      # integrity checks (spec section 6), prints no metrics
done
```

`--regimes` restricts a phase to some regimes: campaign 4 ran the same loop with
`--regimes ssl_imnet_matched` on both commands, stopping at the first failed phase or check. Fine-tuning learning rates: SUP-imnet and SSL-imnet 3e-4,
SSL-colon and SSL-imnet-matched 1e-4 (`finetune_lr` in `config/model/`). A single run:

```bash
python -m ssl4polyp.classification.train_classification --exp-config exp/v3/sun_full.yaml \
    --partition 1 --seed 13 --model-key ssl_colon --roots data/roots.json \
    --output-dir checkpoints_v3/campaign/classification/k1/sun_full/ssl_colon_seed13
```

Each run writes its weights (best epoch only), a canonical metrics JSON, per-frame output CSVs and
`RUN_COMPLETE.json` under `checkpoints_v3/campaign/classification/`.

### 3. Statistical analysis (`ssl4polyp-analysis`)

```bash
conda activate ssl4polyp-analysis
PYTHONPATH=. python -m analysis.v3.run_analysis --out results_v3/analysis      # SAP v3.3, B = 2000, seed 12345
PYTHONPATH=. python -m pytest analysis/v3/tests                                # simulated data only (about 1 h)
```

All metrics are recomputed from the per-frame CSVs; only τ is read from the JSONs. Outputs:
`report.md`, `summary.json`, `contrast_summary.csv`, `paired_differences.csv`, `run_metrics.csv`, `tables/`,
`figures/`. Module-level documentation: [analysis/v3/README.md](analysis/v3/README.md).

Exploratory analyses (not part of the SAP; plans registered before launching, see below):
`analysis/v3/exploratory_lr_common.py` and `analysis/v3/exploratory_lr_sslimnet.py`, outputs in
`results_v3/analysis_exploratory_*/`.

### 4. Figures (`ssl4polyp-analysis`)

```bash
PYTHONPATH=. python -m analysis.v3.figures_final      # Figures 2–6 from the analysis tables
```

Writes `results_v3/figures_final/figure{2..6}.{pdf,png}` and `plotted_values.csv` (every value drawn).

### Tests

pytest is not part of the locked `ssl4polyp-ft` environment; install it outside the environment (e.g.
`pip install --target <dir> pytest` and add `<dir>` to `PYTHONPATH`) so the pinned packages stay unchanged.

```bash
python -m pytest tests                                  # ssl4polyp-ft; tests/legacy is skipped
SSL4POLYP_LEGACY_TESTS=1 python -m pytest tests         # also the v2 tests (several are known to fail)
```

## Pre-registration: SAP, amendments and tags

The SAP and its amendments were registered (SHA-256 in `results_v3/RUNLOG.md`, git tag) before the data they
govern were generated or analysed. Each change is justified in
[results_v3/SAP_deviations.md](results_v3/SAP_deviations.md). Documents are in `docs/v3/`.

| Date | Document | SHA-256 | Tag | Registered before |
|---|---|---|---|---|
| 2026-10-14 | Retraining specification v3.1 (`especificacion_reentrenamiento_v3_1.md`) | `e614ce53…76612d34` | `v3-cmpb-smoke` | Phase 1 |
| 2026-10-14 | SAP v3 (`plan_analisis_estadistico_v3.md`) | `f2120a09…15cc0474` | `v3-cmpb-smoke` | Phase 1 |
| 2026-10-15 | SAP v3.1: learning-rate selection (deviation 9.1) | `a2fc1b83…8818e7bd` | `v3-cmpb-lr`[^1] | LR sweep |
| 2026-10-15 | SAP v3.2: K = 5 partitions, E2 without training, matched LR sweep (deviation 9.2) | `38f8ab05…d2375c70` | `v3-cmpb-sap32` | Any partition |
| 2026-10-15 | SAP v3.2 revision b: section 10 scope | `d32d53ff…18d0e2c1` | `v3-cmpb-sap32b` | Packs |
| 2026-10-15 | SAP v3.2 revision c: H6/H7 without s500, E4 AP within pack | `a387aed0…36874d47` | `v3-cmpb-sap32c` | Campaign |
| 2026-10-17 | **SAP v3.3** (in force): t-test on partition means (deviation 9.3) | `fd6b0342…70f7ff56` | `v3-cmpb-sap33` | Any analysis of real data |
| 2026-10-17 | Exploratory plan, common LR 1e-3 (`plan_exploratorio_lr_comun.md`) | `739fc133…c3887c1c` | `v3-cmpb-exp-lr-plan` | Its runs |
| 2026-10-18 | Exploratory plan, SSL-imnet LR 1e-4 (`plan_exploratorio_lr_sslimnet.md`) | `93fdf34d…05fbef1c` | `v3-cmpb-exp-lr2-plan` | Its runs |

Full hashes are in `results_v3/RUNLOG.md`. The file `plan_analisis_estadistico_v3_2.md` now holds revision c;
the original v3.2 and revision b are at their tags.

[^1]: Registered in commit `213efe5`, before the sweep; `v3-cmpb-lr` is the first tag that contains it.

Other campaign tags:

| Date | Tag | Content |
|---|---|---|
| 2026-10-14 | `v3-cmpb-smoke`, `v3-cmpb-smoke2` | Smoke tests (phase 0) |
| 2026-10-15 | `v3-cmpb-packs`, `v3-cmpb-packs2` | Packs v3 and configs (`packs2`: CSVs stored byte for byte; reference packs) |
| 2026-10-15 | `v3-cmpb-campaign3` | Campaign, SUP-imnet / SSL-imnet / SSL-colon (540 runs) |
| 2026-10-16 | `v3-cmpb-analysis-code` | Analysis code validated on simulated data |
| 2026-10-17 | `v3-cmpb-campaign4` | Campaign, SSL-imnet-matched (180 runs) |
| 2026-10-17 | `v3-cmpb-analysis-results` | Analysis of the real data (SAP v3.3) |
| 2026-10-18 | `v3-cmpb-exp-lr-results`, `v3-cmpb-exp-lr2-results` | Exploratory analyses |
| 2026-10-25 | `v3-cmpb-figures`, `v3-cmpb-figures2` | Manuscript figures 2–6 (`figures2`: revised labels) |

## License

Apache License 2.0 ([LICENSE](LICENSE)), **except** `src/ssl4polyp/models/mae/`, which derives from Meta's MAE
and is under CC BY-NC 4.0 ([src/ssl4polyp/models/mae/LICENSE](src/ssl4polyp/models/mae/LICENSE)): its use is
restricted to non-commercial purposes. Attributions for SSL4GIE, MAE, timm and DPT are in [NOTICE](NOTICE).
Datasets and pretrained weights are subject to their own terms.

## Citation

```bibtex
@unpublished{ssl4polyp,
  title  = {Self-supervised pretraining for colonoscopy polyp classification: a pre-registered
            controlled evaluation of domain, scale and deployment protocol},
  author = {Yip, Vincent and Abuhalimeh, Ahmed and Campos, Celso and Fdez-Riverola, Florentino and
            Rodriguez-Conde, Ivan},
  year   = {2026},
  note   = {Submitted to Computer Methods and Programs in Biomedicine}
}
```

Please also cite SSL4GIE (Sanderson and Matuszewski, IEEE Access 2024, doi:10.1109/ACCESS.2024.3381517) and
MAE (He et al., CVPR 2022).

## Contact

Corresponding author: Ivan Rodriguez-Conde ([ivarodriguez@uvigo.gal](mailto:ivarodriguez@uvigo.gal)).
