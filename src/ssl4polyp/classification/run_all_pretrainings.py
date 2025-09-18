#!/usr/bin/env python3
"""Utility to fine-tune ViT-B/16 under all pretraining schemes.

This script sequentially launches ``train_classification.py`` for the
three pretraining modes used in the paper:

* SUP-imnet  – supervised ImageNet-1k weights.
* SSL-imnet – MAE self-supervised ImageNet-1k weights.
* SSL-colon – MAE self-supervised Hyperkvasir-unlabelled weights.

Checkpoints for the self-supervised options must be supplied via the
command line. Fine-tuned models are stored in
``Trained models/`` as usual.
"""

import argparse
import os
import subprocess
import sys
from typing import List


def run(cmd: List[str]) -> None:
    """Run a subprocess and stream its output."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT-B/16 for all pretraining configurations"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset tag, e.g. Hyperkvasir_pathological"
    )
    parser.add_argument(
        "--data-root", required=True, help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--imagenet-mae",
        required=True,
        help="Checkpoint for MAE pretraining on ImageNet-1k (SSL-imnet)",
    )
    parser.add_argument(
        "--hyperkvasir-mae",
        required=True,
        help="Checkpoint for MAE pretraining on Hyperkvasir-unlabelled (SSL-colon)",
    )
    parser.add_argument("--arch", default="vit_b", help="Backbone architecture")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for each run"
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional arguments passed to train_classification.py",
    )
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(__file__), "train_classification.py")

    base_cmd = [
        sys.executable,
        script,
        "--architecture",
        args.arch,
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--batch-size",
        str(args.batch_size),
        "--learning-rate-scheduler",
    ]
    extra = args.extra_args.split()

    # SUP-imnet: supervised ImageNet baseline
    run(base_cmd + ["--pretraining", "ImageNet_class"] + extra)

    # SSL-imnet: MAE on ImageNet-1k
    run(
        base_cmd
        + [
            "--pretraining",
            "ImageNet_self",
            "--ss-framework",
            "mae",
            "--checkpoint",
            args.imagenet_mae,
        ]
        + extra
    )

    # SSL-colon: MAE on Hyperkvasir-unlabelled
    run(
        base_cmd
        + [
            "--pretraining",
            "Hyperkvasir",
            "--ss-framework",
            "mae",
            "--checkpoint",
            args.hyperkvasir_mae,
        ]
        + extra
    )


if __name__ == "__main__":
    main()
