# Environment for campaign v3

Files in this folder describe the environment used for campaign v3 on the 2 × NVIDIA RTX 6000 Ada
Generation (48 GB) workstation:

- `environment.yml`: `conda env export -n ssl4polyp` (prefix line removed).
- `pip_freeze.txt`: `pip freeze` of the same environment.

Key versions: Python 3.8.20, torch 2.4.1+cu121, torchvision 0.19.1+cu121, timm 0.3.2,
numpy 1.24.4, scikit-learn 1.0.2. Each run also stores its own `pip-freeze.txt`, `cuda.json` and
git commit next to its outputs, and `results_v3/environment/` holds `nvidia-smi` and versions.

## Installation

```bash
conda env create -f env/environment.yml        # or: pip install -r requirements.txt
conda activate ssl4polyp
pip install --no-deps -e .
```

Install the package with `--no-deps`: the environment is already complete and pip must not
resolve or replace any pinned package.

## timm 0.3.2 patch

timm 0.3.2 imports `container_abcs` from `torch._six`, which no longer exists in torch 2.x.
`timm/models/layers/helpers.py` in the environment is patched by hand:

```python
# before
from torch._six import container_abcs
# after
import collections.abc as container_abcs
```

`ssl4polyp._compat.ensure_torch_container_abcs()` provides the same shim at import time, so the
code also runs with an unpatched timm 0.3.2.

## Differences from the v2 environment

The v2 results (A40, RunPod) were produced with torch 1.9.0+cu111, torchvision 0.10.0+cu111,
timm 0.4.12 and numpy 1.21.6 (recorded in every v2 run's `pip-freeze.txt` and in the root
`requirements-pip.txt` / `requirements.lock.txt`, which are kept unchanged as the v2 record).
Consequences handled in the code:

- **TF32.** The code never set TF32 before v3, so it followed the torch default: torch 1.9 used
  TF32 for matmul and cuDNN; torch 2.4 uses it only for cuDNN. `config/base.yaml` now sets both
  explicitly (`tf32.matmul: true`, `tf32.cudnn: true`, matching the v2 runs), and every run
  records the values in its metrics JSON (`run.tf32`).
- **SUP-imnet.** timm 0.3.2 has no `.npz` loader and its ViT defaults differ from timm 0.4.12
  (`qkv_bias=False`, LayerNorm `eps=1e-5`). `ssl4polyp/models/models.py` now builds the ViT with
  `qkv_bias=True` and `eps=1e-6` explicitly and contains a port of the timm 0.4.12 `.npz` loader.
- **`torch.load`.** Checkpoints are loaded with `weights_only=False` explicitly (torch ≥ 2.6
  changed the default); this argument requires torch ≥ 1.13.

## SUP-imnet weights

Downloaded once and never at run time:

- URL: `https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`
- Local copy: `checkpoints/pretrained/vit_b/<same file name>`
- SHA-256: `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`

`config/model/sup_imnet.yaml` points to the local copy and its SHA-256; a missing file or a
different hash stops the run. The MAE initialisations are checked the same way
(`checkpoint_sha256` in `config/model/ssl_*.yaml`).
