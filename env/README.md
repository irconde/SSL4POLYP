# Environments for campaign v3

Two conda environments on the 2 × NVIDIA RTX 6000 Ada Generation (48 GB) workstation. Both have
Python 3.8.20, torch 2.4.1+cu121, torchvision 0.19.1+cu121, numpy 1.24.4 and scikit-learn 1.0.2.

| Environment | Files | Use | timm | Extra |
|---|---|---|---|---|
| `ssl4polyp-pretrain` | `environment-pretrain.yml`, `pip_freeze-pretrain.txt` | MAE pretraining of the SSL-imnet-matched regime only | 0.3.2 (patched, see below) | — |
| `ssl4polyp-ft` | `environment-ft.yml`, `pip_freeze-ft.txt` | All fine-tuning and evaluation runs (sun_full, E1–E4) | 0.4.12 | matplotlib 3.7.5 |

`ssl4polyp-ft` is a clone of `ssl4polyp-pretrain` with `pip install --no-deps timm==0.4.12` and
`pip install matplotlib==3.7.5` (which added contourpy, cycler, fonttools, importlib_resources,
kiwisolver and pyparsing). Nothing else differs (`diff pip_freeze-*.txt`).

## Installation

```bash
conda env create -f env/environment-ft.yml       # or environment-pretrain.yml
conda activate ssl4polyp-ft
pip install --no-deps -e .
```

Clone with `conda create --clone`, not `mamba create --clone`: mamba copies only the conda
packages and drops everything installed with pip (torch, timm, ...). Install the package with
`--no-deps` so pip does not resolve or replace any pinned package.

## timm 0.3.2 patch (`ssl4polyp-pretrain`)

timm 0.3.2 imports `container_abcs` from `torch._six`, which no longer exists in torch 2.x.
`timm/models/layers/helpers.py` is patched by hand:

```python
# before
from torch._six import container_abcs
# after
import collections.abc as container_abcs
```

timm 0.3.2 cannot load the SUP-imnet `.npz` weights (no Flax loader); the code stops with a
message pointing to `ssl4polyp-ft`.

## Differences from the v2 environment

The v2 results (A40, RunPod) used torch 1.9.0+cu111, torchvision 0.10.0+cu111, timm 0.4.12 and
numpy 1.21.6 (each v2 run's `pip-freeze.txt`; root `requirements-pip.txt` and
`requirements.lock.txt` are kept unchanged as the v2 record).

- **timm**: `ssl4polyp-ft` uses the v2 version (0.4.12). The networks built in `ssl4polyp-ft` by
  the v2 code (commit `d678b28`) and the current code are identical for the three regimes: same
  layer names and shapes, `qkv_bias` in all blocks, LayerNorm eps 1e-6, 85,800,194 (SUP-imnet) /
  85,901,058 (MAE) parameters and bit-identical weights after loading
  (`results_v3/verification/network_v2_vs_v3.json`). `qkv_bias=True` and eps 1e-6 are still set
  explicitly in `ssl4polyp/models/models.py`.
- **torch 2.4.1 instead of 1.9.0**: kernels, cuDNN/cuBLAS versions and AMP behaviour differ, so
  v3 runs are not bit-identical to v2.
- **TF32**: the code never set it before v3, so it followed the torch default (torch 1.9: matmul
  and cuDNN on; torch 2.4: cuDNN only). `config/base.yaml` now sets both on (as in v2) and every
  run records the values (`run.tf32` in the metrics JSON).
- **numpy 1.24**: `np.float` no longer exists; `models/mae/util/pos_embed.py` uses `np.float64`
  (the same type).
- **`torch.load`**: called with `weights_only=False` explicitly (torch ≥ 2.6 changed the default).

## Initialisation weights

Never downloaded at run time; every run checks the SHA-256 in `config/model/*.yaml`.

- SUP-imnet: `checkpoints/pretrained/vit_b/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`,
  from `https://storage.googleapis.com/vit_models/augreg/` (same file name),
  SHA-256 `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`.
- SSL-colon: `checkpoints/pretrained/vit_b/mae_hyperkvasir.pth`, `faf5d0b1c67eeaf3ab333fc4dd0da23b3d294d998aa946a150ac59cf1d1d728f`.
- SSL-imnet: `checkpoints/pretrained/vit_b/mae_imagenet.pth`, `aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d`.

## Analysis environment (`ssl4polyp-analysis`)

Separate environment for the SAP v3.2c statistical analysis (`analysis/v3/`); it has no torch and uses no
GPU. Created 2026-10-16 from conda-forge only, without touching `ssl4polyp-ft` or `ssl4polyp-pretrain`:

```bash
mamba create -n ssl4polyp-analysis -c conda-forge --override-channels python=3.11 numpy=1.26 pandas=2.2 \
    scipy matplotlib=3.9 scikit-learn=1.5 pyyaml pytest r-base=4.4 r-lme4 r-lmertest r-jsonlite
# or: conda env create -f env/environment-analysis.yml
```

| Component | Version |
|---|---|
| Python | 3.11.17 |
| numpy / pandas / scipy | 1.26.4 / 2.2.3 / 1.17.1 |
| scikit-learn / matplotlib | 1.5.2 / 3.9.4 |
| R | 4.4.3 (OpenBLAS 0.3.34) |
| lme4 / lmerTest / Matrix / numDeriv | 2.0-6 / 3.2-1 / 1.7-5 / 2016.8-1.1 |
| pbkrtest (added 2026-10-17) | 0.5.5 |
| pytest | 9.1.1 |

Records: `environment-analysis.yml` (pinned spec), `environment-analysis.lock.yml` (full solved export),
`pip_freeze-analysis.txt`, `r_packages-analysis.txt` (all installed R packages).

- 2026-10-17: `r-pbkrtest` 0.5.5 added (Kenward–Roger df for `analysis/v3/validate_variants.py`), installed with
  `r-base=4.4.3 r-lme4=2.0_6 r-lmertest=3.2_1` pinned; the solver only added packages (pbkrtest and its
  dependencies), none changed or was removed. An unpinned install would have moved the environment to R 4.5.
- The mixed models run in R (`analysis/v3/R/fit_mixed.R`, `lmerTest::lmer`, REML, Satterthwaite df), called
  from Python through the environment's `Rscript`.
- All metrics come from `src/ssl4polyp/classification/metrics/binary.py`, loaded from its file
  (`analysis/v3/metrics.py`) because the package's `metrics/__init__.py` imports torch. scikit-learn
  differs from the campaign (1.0.2 there). Check on a simulated 4,578-frame CSV (2026-10-16): all
  `binary.py` outputs (metrics at τ, counts, Brier, ECE) are bit-identical between the two environments
  except AP, which differs by 1.1e-16 (floating-point summation order).
- The package itself is not installed in this environment; run with `PYTHONPATH=.` from the repository root.
