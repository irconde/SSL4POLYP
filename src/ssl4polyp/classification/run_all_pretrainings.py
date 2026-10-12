#!/usr/bin/env python3
"""Sequentially fine-tune a backbone under several pretraining schemes.

This script launches ``ssl4polyp.classification.train_classification`` once per
requested model key defined in an experiment config. By default it runs the
three pretraining schemes used in the paper:

* SUP-imnet  – supervised ImageNet-1k weights.
* SSL-imnet – MAE self-supervised ImageNet-1k weights.
* SSL-colon – MAE self-supervised Hyperkvasir-unlabelled weights.

The self-supervised checkpoints are referenced directly by the model configs
(``config/model/ssl_imnet.yaml`` and ``config/model/ssl_colon.yaml``), so no
checkpoint paths need to be supplied here. Outputs are written under
``<output-root>/<exp>_<model>_seed<seed>/``.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_MODEL_KEYS = ("sup_imnet", "ssl_imnet", "ssl_colon")

TRAIN_MODULE = "ssl4polyp.classification.train_classification"


def run(cmd: List[str]) -> None:
    """Run a subprocess and stream its output."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a backbone under multiple pretraining schemes"
    )
    parser.add_argument(
        "--exp-config",
        required=True,
        help="Experiment manifest (relative to config/ or an absolute path)",
    )
    parser.add_argument(
        "--roots",
        default=str(Path("data") / "roots.json"),
        help="JSON file mapping manifest roots to directories (default: data/roots.json)",
    )
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=list(DEFAULT_MODEL_KEYS),
        help=(
            "Model keys defined in the experiment config to run sequentially "
            f"(default: {' '.join(DEFAULT_MODEL_KEYS)})"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for each run"
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("checkpoints") / "classification"),
        help="Base directory for per-run outputs (default: checkpoints/classification)",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional arguments passed to train_classification.py",
    )
    args = parser.parse_args()

    exp_stem = Path(args.exp_config).stem
    extra = args.extra_args.split()

    for model_key in args.model_keys:
        out_dir = Path(args.output_root) / f"{exp_stem}_{model_key}_seed{args.seed}"
        cmd = [
            sys.executable,
            "-m",
            TRAIN_MODULE,
            "--exp-config",
            args.exp_config,
            "--model-key",
            model_key,
            "--seed",
            str(args.seed),
            "--roots",
            args.roots,
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(out_dir),
        ] + extra
        run(cmd)


if __name__ == "__main__":
    main()
