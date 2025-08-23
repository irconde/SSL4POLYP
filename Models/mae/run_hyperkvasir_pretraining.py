#!/usr/bin/env python3
"""Launch MAE pretraining on Hyperkvasir-unlabelled.

This helper wraps ``main_pretrain.py`` with the arguments used in the
paper to produce SSL-colon weights. It assumes the Hyperkvasir-unlabelled
frames are laid out as ``<root>/<video_id>/<frame>.jpg``.
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
        description="Pretrain ViT-B/16 with MAE on Hyperkvasir-unlabelled"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to Hyperkvasir-unlabelled dataset root",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store checkpoints and logs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional arguments forwarded to main_pretrain.py",
    )
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(__file__), "main_pretrain.py")
    cmd = [
        sys.executable,
        script,
        "--model",
        "mae_vit_base_patch16",
        "--data_path",
        args.data_root,
        "--output_dir",
        args.output_dir,
        "--log_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--no_train_dir",
    ] + args.extra_args.split()
    run(cmd)


if __name__ == "__main__":
    main()
