# SSL4POLYP Agent Notes

## Repository Purpose
This repository hosts the reference implementation for the paper *Evaluating Domain-Specific Self-Supervised Pre-training for Polyp Detection: A Morphology-Aware and Sample-Efficient Perspective*. The codebase focuses on vision transformers for colonoscopy imagery, offering training, evaluation, and data preparation utilities.

## High-Level Layout
- `src/ssl4polyp/`: Python package containing all runnable modules and entry points.
- `config/`: Layered YAML configuration files that define shared defaults, data packs, model checkpoints, and experiment manifests.
- `data_packs/`: Dataset manifests and CSV splits referenced by the training scripts.
- `checkpoints/`, `data/`, `outputs/`, `results/`: Git-ignored working directories seeded with README files describing expected content. Command-line tools resolve relative paths against these folders by default.
- `scripts/`: Shell helpers for orchestrating experiments.
- `tests/`: Automated tests for key components.
- `requirements*.txt`, `environment.yml`: Locked environments for reproducibility.

## Source Tree Breakdown (`src/ssl4polyp`)
- `classification/`: Fine-tuning and evaluation pipelines for classification experiments.
  - `train_classification.py`: Main entry point for supervised training. Accepts dataset manifests, model configs, and runtime flags.
  - `finetune.py`: Utilities for orchestrating fine-tuning runs.
  - `eval_classification.py`: Evaluation CLI supporting checkpointed models.
  - `eval_outputs.py`: Tools for exporting logits and metadata.
  - `metrics/`, `data/`: Helpers for metrics computation and dataset adapters.
  - `run_all_pretrainings.py`: Convenience wrapper to launch multiple pretraining schemes sequentially.
- `models/`: Model-specific tooling, including MAE pretraining scripts under `mae/`.
- `configs/`: Programmatic helpers for loading layered YAML configs and resolving manifests.
- `polypdb/`: Utilities for working with dataset packs, corruption specifications, and metadata.
- `utils/`: Shared helpers (logging, distributed utilities, serialization, etc.).
- `_compat.py`: Compatibility shims and version guards.

## Configuration & Experiment Management
- YAML configs compose through `defaults` lists. Experiment manifests typically reside under `config/exp/` and reference shared base, data, and model files.
- Dataset packs live under `data_packs/<pack_name>/` and include manifests plus CSV splits.
- Each distributed pack is materialized once per configuration, and its manifest records the canonical generation seed that the configs expose when resolving data variants.
- Scripts assume relative paths resolve against `config/` or `data_packs/` unless absolute paths are provided.

## Running Workflows
1. Install dependencies via the locked `requirements.txt` (or `requirements-pip.txt` for tooling extras) and register the package in editable mode with `pip install --no-deps -e .`.
2. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before training when deterministic algorithms are enabled.
3. Launch experiments using the CLIs:
   - Pretraining MAE on HyperKvasir: `python -m ssl4polyp.models.mae.run_hyperkvasir_pretraining ...`
   - Classification training: `python -m ssl4polyp.classification.train_classification ...`
   - Evaluation: `python -m ssl4polyp.classification.eval_classification ...`
4. Use `scripts/run_exps.sh` for batch execution of experiment manifests.

## Contribution Notes
- Follow existing module structure when adding new training or evaluation scripts (place classification-centric utilities under `src/ssl4polyp/classification/`, MAE pretraining logic under `src/ssl4polyp/models/mae/`, etc.).
- Keep new dataset descriptors in `config/data/` and register their CSVs under `data_packs/`.
- Update this file if the repository structure or workflows change significantly.
