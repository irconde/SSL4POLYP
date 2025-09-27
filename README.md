Official code repository for *Evaluating Domain-Specific Self-Supervised Pre-training for Polyp Detection: A Morphology-Aware and Sample-Efficient Perspective*,
based on "A Study on Self-Supervised Pretraining for Vision Problems in Gastrointestinal Endoscopy"

Original authors: [Edward Sanderson](https://scholar.google.com/citations?user=ea4c7r0AAAAJ&hl=en&oi=ao) and [Bogdan J. Matuszewski](https://scholar.google.co.uk/citations?user=QlUO_oAAAAAJ&hl=en)

Repository author: Ivan Rodriguez-Conde, assistant professor at the Department of Computer Science at University of Vigo ([ivarodriguez@uvigo.gal](mailto:ivarodriguez@uvigo.gal))

Links to the original paper:
+ [IEEE Access (open access)](https://ieeexplore.ieee.org/document/10478725)
+ [arXiv](https://arxiv.org/abs/2401.06278)

## Installation

The project ships a fully pinned runtime in `requirements.txt` / `requirements-pip.txt`.
To reproduce the curated environment:

1. Create and activate a fresh virtual environment (for example, `python -m venv .venv && source .venv/bin/activate`).
2. Install the locked dependencies, e.g. `pip install -r requirements.txt` (or `requirements-pip.txt` when building the
   extended tooling stack).
3. Register the package in editable mode without letting pip resolve alternatives:

   ```bash
   pip install --no-deps -e .
   ```

   ```bash
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   ```

   Configure CuBLAS determinism before launching training to avoid runtime errors when `torch.use_deterministic_algorithms(True)` is active.

Using `--no-deps` prevents pip from overwriting the pinned versions with newer releases while still exposing the
`ssl4polyp` CLI entry points.

## New functionality

We include helper scripts that automate common workflows introduced in the
paper:

- `ssl4polyp/models/mae/run_hyperkvasir_pretraining.py` – wraps MAE pretraining on the
  Hyperkvasir-unlabelled dataset with the settings used in the paper. The
  script exposes `--batch-size`, `--epochs`, `--extra-args`, `--auto-resume`,
  `--save-freq-epochs` and `--save-freq-mins` so you can tweak the training run
  or forward additional options to `main_pretrain.py`.
- `ssl4polyp/classification/run_all_pretrainings.py` – sequentially fine-tunes ViT-B/16
  under all three pretraining schemes (SUP-imnet, SSL-imnet and SSL-colon).
  Customise the architecture and batch size with `--arch` and `--batch-size`,
  and pass further arguments to `train_classification.py` via `--extra-args`.
- `scripts/run_exps.sh` – iterates over manifest-defined classification experiments,
  verifying referenced files and launching each run with a shared roots mapping.

### Script improvements

Recent updates enhance reproducibility and robustness across training and
finetuning:

- Runs now accept a `--seed` flag and dump their full configuration (including
  the Git commit) to `config.yaml` in the output directory.
- TensorBoard logs are written to `tb/` under the chosen `--output-dir`.
- Checkpoints are consistently stored and a `last.pth` symlink tracks the most
  recent save.
- Mixed precision can be toggled via `--precision {amp, fp32}`.
- Data loading behaviour is configurable with `--workers`, `--prefetch-factor`,
  `--pin-memory` and `--persistent-workers`.
- MAE pretraining exposes `--accum_iter` for gradient accumulation and can
  automatically resume, save checkpoints on a time or epoch basis, retain only
  recent checkpoints and handle termination signals gracefully.
- Dataset splits can be provided via CSV manifests (`--train-csv`, `--val-csv`,
  `--test-csv`) or a consolidated `--manifest` file together with a `--roots`
  mapping. The training script snapshots these inputs and environment details
  for reproducibility.
- Class imbalance can be addressed with `--class-weights` to override automatic
  weighting.
- Evaluation reports mean AUROC in addition to the existing metrics, and
  `ssl4polyp/classification/eval_outputs.py` offers a utility to persist logits and
  metadata for later analysis.

### Configuration layout

Configuration assets are organised into two top-level directories that live
next to the source tree.  The `config/` directory now contains layered YAML
files that capture shared defaults and experiment-specific overrides, while
`data_packs/` stores the dataset manifests and CSV splits referenced by those
configs.  A typical layout looks like:

```
config/
├─ base.yaml                  # global defaults (optimizer, scheduler, seed, …)
├─ data/                      # dataset pack descriptors
│  ├─ sun_full.yaml
│  ├─ sun_morphology.yaml
│  ├─ sun_subsets.yaml
│  ├─ polypgen_clean_test.yaml
│  └─ …
├─ model/                     # backbone definitions (checkpoints, freezing policy)
│  ├─ sup_imnet.yaml
│  ├─ ssl_imnet.yaml
│  └─ ssl_colon.yaml
└─ exp/                       # experiment manifests (include base + data + models)
   ├─ exp1.yaml
   ├─ exp2.yaml
   └─ …

data_packs/
└─ <pack_name>/manifest.yaml, *.csv, …
```

Layered configs use a `defaults:` list to compose these files; the training
scripts resolve everything automatically when pointed at an experiment manifest.

In addition, the repository seeds git-ignored directories that act as default
targets for local assets:

```
data/
checkpoints/
outputs/
results/
```

Each ships with a short README describing the expected sub-structure. Command
line interfaces default to these folders for datasets, pretrained weights, logs and
evaluation exports while still letting you override the destinations.

Command-line interfaces resolve relative paths against these directories by
default.  For example, providing `--manifest classification/hyperkvasir.yaml` to
`train_classification.py` looks for the file at `config/classification/
hyperkvasir.yaml`.  Likewise, passing `--train-csv hyperkvasir/train.csv`
expects the file under `data_packs/hyperkvasir/train.csv`.  The `polypdb`
utilities follow the same convention for corruption specifications and dataset
packs.  Absolute paths are still honoured, so you can store packs elsewhere if
desired.

The following sections describe how to invoke these utilities.

## Usage

### Pretraining

Follow the guidance in this section for obtaining the weights for pretrained models.

+ For encoders pretrained in a supervised manner with ImageNet-1k, the weights provided with the [timm](https://timm.fast.ai/) library (ViT-B) are automatically used by our code and no manual steps are required.
+ For encoders pretrained in a self-supervised manner with ImageNet-1k, using [MAE](https://github.com/facebookresearch/mae), the weights provided with the codebase should be used.
+ For encoders pretrained in a self-supervised manner with
  [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/), using
  [MAE](https://github.com/facebookresearch/mae), use the helper script
  `ssl4polyp/models/mae/run_hyperkvasir_pretraining.py` to generate the checkpoint. You can
  override the default `--batch-size` and `--epochs` values and pass additional
  options to `main_pretrain.py` via `--extra-args`. By default the script writes
  checkpoints to `checkpoints/mae/hyperkvasir_pretrain/` and TensorBoard logs to
  `outputs/mae/hyperkvasir_pretrain/`, but both can be customised. Example:

```bash
python -m ssl4polyp.models.mae.run_hyperkvasir_pretraining \
    --data-root /path/to/hyperkvasir-unlabelled \
    --output-dir checkpoints/mae/hyperkvasir_pretrain \
    --log-dir outputs/mae/hyperkvasir_pretrain \
    --batch-size 64 --epochs 400
```

### Finetuning

The finetuning scripts currently support frame-level classification. Pretrained weights produced by our experiments are available [here](https://drive.google.com/drive/folders/151BWqsjTV4PuGFxS20L0TpmUQ4DhhpU4?usp=sharing) (released under a CC BY-NC-SA 4.0 license). Place the desired checkpoint in `checkpoints/classification/<exp_seed>/` if you wish to evaluate or make predictions with an already finetuned model.  Non-downloadable ViT-B MAE backbones (ImageNet and Hyperkvasir) should be stored under `checkpoints/pretrained/vit_b/` so the model configs can reference them directly.

Prior to running finetuning, download the required data and change directory:

+ For classification, download [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/) and ensure the dataset follows the expected directory layout (`labeled-images/<split>/<class>/*.jpg`).

Datasets can also be described via CSV manifests rather than directory structures; see [Experiment definition with manifests](#experiment-definition-with-manifests) for details.

We recommend launching runs via experiment manifests, which resolve datasets,
models and hyperparameters from the layered config stack:

```
python -m ssl4polyp.classification.train_classification \
    --exp-config exp/exp1.yaml \
    --model-key sup_imnet \
    --roots data/roots.json \
    --output-dir checkpoints/classification/exp1_seed42 \
    --seed 42
```

Choose a different `--model-key` to fine-tune the MAE variants defined in
`config/model/*.yaml`.  You can still override individual knobs on the command
line (for example `--batch-size` or `--lr`) and they will overwrite values from
the manifest.

If you prefer a fully manual invocation, you can supply the dataset packs and
model settings explicitly:

```
python -m ssl4polyp.classification.train_classification \
    --architecture vit_b \
    --pretraining Hyperkvasir \
    --ss-framework mae \
    --checkpoint checkpoints/pretrained/vit_b/mae_hyperkvasir.pth \
    --dataset sun_full \
    --train-pack sun_full \
    --val-pack sun_full \
    --test-pack sun_full \
    --roots data/roots.json \
    --batch-size 32 \
    --scheduler cosine \
    --warmup-epochs 5
```

The degree of encoder fine-tuning can be controlled with the
`protocol.finetune` key inside the experiment manifest (or the corresponding
command-line overrides):

* `full` (default) trains the entire encoder.
* `none` trains only the classification head, keeping the encoder frozen as in
  prior releases.
* `head+2` updates the head together with the final two transformer blocks for a
  lightweight adaptation regime.

### Experiments

The paper’s experiments map to manifests in `config/exp/`.  Launch them with
`--exp-config` and pick a backbone via `--model-key` (`sup_imnet`, `ssl_imnet`,
`ssl_colon`).  Outputs default to
`checkpoints/classification/<exp>_seed<seed>/`.

| Experiment | Config | Description | Example command |
|-----------:|:-------|:------------|:----------------|
| 1 | `config/exp/exp1.yaml` | SUN baseline vs MAE(ImageNet) fine-tuning | `python -m ssl4polyp.classification.train_classification --exp-config exp/exp1.yaml --model-key sup_imnet --seed 42 --output-dir checkpoints/classification/exp1_seed42` |
| 2 | `config/exp/exp2.yaml` | Domain MAE vs ImageNet MAE on SUN | run with `--model-key ssl_imnet` and `ssl_colon` |
| 3 | `config/exp/exp3.yaml` | Morphology sensitivity (flat vs polypoid) | reuse Exp‑1 command but point to `exp3.yaml` |
| 4 | `config/exp/exp4.yaml` | SUN sample-efficiency subsets (5–100%) | loop over percents/seeds using overrides, e.g. `--override dataset.percent=25 --override dataset.seed=29` |
| 5A | `config/exp/exp5a.yaml` | Zero-shot transfer to PolypGen clean test | `--model-key sup_imnet` (repeat for other models) |
| 5B | `config/exp/exp5b.yaml` | SUN robustness under perturbations | `--model-key ssl_imnet` (or others) |
| 5C | `config/exp/exp5c.yaml` | Few-shot adaptation on PolypGen (50–500 frames) | set `--override dataset.size=<N> dataset.seed=<seed>` and evaluate both frozen and re-fit thresholds |

To sweep subsets/seeds, wrap the command in a shell loop. Example for Exp‑4:

```bash
for seed in 13 29 47; do
  for pct in 5 10 25 50 100; do
    python -m ssl4polyp.classification.train_classification \
      --exp-config exp/exp4.yaml \
      --model-key ssl_imnet \
      --seed "${seed}" \
      --override dataset.percent="${pct}" dataset.seed="${seed}" \
      --output-dir "checkpoints/classification/exp4_seed${seed}_p${pct}";
  done
done
```

For few-shot Exp‑5C, adjust `dataset.size` (`50`, `100`, `200`, `500`) and
optionally re-fit the decision threshold on a support fold before testing. After
training, evaluate runs with `python -m ssl4polyp.classification.eval_classification`
using the same experiment config or explicit packs.

The script expects the HyperKvasir directory structure by default. For datasets
organised more simply as `data-root/class_x/*.jpg`, either pass `--simple-dataset`
or point `--data-root` to the top-level directory and the script will infer this
layout automatically. Arrange images as:

```
data-root/
├─ class_0/
│   ├─ img001.jpg
│   └─ ...
└─ class_1/
    ├─ img050.jpg
    └─ ...
```

To run all three pretraining configurations sequentially (SUP-imnet, SSL-imnet and SSL-colon) you may use the helper script:
```
python run_all_pretrainings.py \
    --dataset [dataset] \
    --data-root [data-root] \
    --imagenet-mae /path/to/mae_pretrain_vit_base.pth \
    --hyperkvasir-mae /path/to/hyperkvasir_mae.pth
```
Optionally use `--arch` to choose a backbone (default `vit_b`) and `--batch-size`
to set the batch size for each run. Any additional options supported by
`train_classification.py` can be appended via `--extra-args`.

#### Experiment definition with manifests

1. **Prepare split CSVs** inside a folder such as
   `data_packs/hyperkvasir/pathology/` containing at least `frame_path` and
   `label` columns for `train`, `val` and `test`.
2. **Create a manifest** (for example `data_packs/hyperkvasir/pathology/manifest.yaml`)
   that references those CSVs and optionally records SHA256 hashes and root
   identifiers:

   ```yaml
   train:
     csv: train.csv
   val:
     csv: val.csv
   test:
     csv: test.csv
   roots:
     data_root: /data/hyperkvasir
   ```
3. **Provide a roots mapping** (e.g. `data/roots.json`) so identifiers resolve
   to absolute paths. A template is provided at `data/roots.example.json`:

   ```json
   {
     "data_root": "/absolute/path/to/images"
   }
   ```
4. **Verify paths** with `scripts/check_paths.py`:

   ```bash
   python scripts/check_paths.py train.csv data/roots.json
   ```
5. **Launch training** with the manifest and roots mapping.  Paths that are not
   absolute are resolved against `config/` and `data_packs/` automatically:

   ```bash
   python -m ssl4polyp.classification.train_classification \
       --manifest classification/exp.yaml \
       --roots data/roots.json \
       --output-dir checkpoints/classification/exp1
   ```

   To execute a batch of manifests `exp1.yaml`..`exp5.yaml` use:

   ```bash
   scripts/run_exps.sh MANIFEST_DIR data/roots.json checkpoints/classification
   ```

   The training script snapshots the manifest files, roots mapping and
   environment details inside the chosen output directory for reproducibility.
   Omitting the optional arguments uses the default `data/roots.json` mapping
   and writes outputs under `checkpoints/classification/`.
   When `MANIFEST_DIR` or `roots.json` are provided as relative paths they are
   interpreted with respect to `config/`.

For all finetuning runs, the following optional arguments are also available:
+ `--frozen` - to freeze the pretrained encoder and only train the decoder. We did not use this in our experiments.
+ `--epochs [epochs]` - to set the number of epochs as desired (replace `[epochs]`). We did not use this in our experiments.
+ `--learning-rate-scheduler` - to use a learning rate scheduler that halves the learning rate when the performance on the validation set does not improve for 10 epochs. We did use this in our experiments.
+ `--class-weights w1,w2,...` - comma-separated list of weights to override the automatically computed class weights.

Please also note that, when using MAE, code from the [MAE](https://github.com/facebookresearch/mae) repository is used, which is released under a CC BY-NC 4.0 license. Any results from such runs are therefore covered by a CC BY-NC 4.0 license.

### Evaluation

Ensure that the weights for the desired model are located in `checkpoints/` relative to your working directory (or pass `--checkpoint-dir` to override). This will have been done automatically if finetuning was run. Additionally, download the required data and change directory accordingly.

For evaluating classification models pretrained in a self-supervised manner, run the following:
```
python -m ssl4polyp.classification.eval_classification \
    --architecture [architecture] \
    --pretraining [pretraining] \
    --ss-framework [ss-framework] \
    --dataset [dataset] \
    --data-root [data-root]
```
For evaluating classification models pretrained in a supervised manner, or not pretrained at all, omit the `--ss-framework` argument:
```
python -m ssl4polyp.classification.eval_classification \
    --architecture [architecture] \
    --pretraining [pretraining] \
    --dataset [dataset] \
    --data-root [data-root]
```

* Replace `[architecture]` with name of encoder architecture (`vit_b`).
* Replace `[pretraining]` with general pretraining methodology (`Hyperkvasir`, `ImageNet_self`, `ImageNet_class`, or `random`).
* For models pretrained in a self-supervised manner, replace `[ss-framework]` with pretraining algorithm `mae`.
* Replace `[dataset]` with name of dataset (e.g., `Hyperkvasir_anatomical` or `Hyperkvasir_pathological`).
* Replace `[data-root]` with path to the chosen dataset.

In addition to printing the results (mean F1, precision, recall, AUROC and accuracy) the evaluation script also writes them to `eval_results.txt`. For further analysis, `ssl4polyp/classification/eval_outputs.py` provides a `write_outputs` helper to persist logits and metadata.

### Prediction

Prediction utilities are not implemented for classification models.

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ESandML/SSL4GIE/blob/main/LICENSE) file. Please however note that when using MAE, code from the [MAE](https://github.com/facebookresearch/mae) repository is used, which is released under a CC BY-NC 4.0 license, and any results from such runs are therefore covered by a CC BY-NC 4.0 license.

## Citation

If you use this work, please consider citing us:
```bibtex
@article{sanderson2024study,
  title={A Study on Self-Supervised Pretraining for Vision Problems in Gastrointestinal Endoscopy},
  author={Sanderson, Edward and Matuszewski, Bogdan J.},
  journal={IEEE Access},
  year={2024},
  volume={12},
  number={},
  pages={46181-46201},
  doi={10.1109/ACCESS.2024.3381517}
}
```

## Commercial use

We allow commercial use of this work, as permitted by the [LICENSE](https://github.com/ESandML/SSL4GIE/blob/main/LICENSE). However, where possible, please inform us of this use for the facilitation of our impact case studies.

## Acknowledgements

This work was supported by the Science and Technology Facilities Council [grant number ST/S005404/1].

This work makes use of:
+ The [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/) dataset.
+ The [MAE](https://github.com/facebookresearch/mae) codebase. In addition to using this codebase for pretraining, as well as the weights provided, we also include the [MAE](https://github.com/facebookresearch/mae) repository in `SSL4GIE/Models`, with modifications made for version compatibility.

## Additional information

Links: [AIdDeCo Project](https://www.uclan.ac.uk/research/activity/machine-learning-cancer-detection), [CVML Group](https://www.uclan.ac.uk/research/activity/cvml)

Contact: esanderson4@uclan.ac.uk
