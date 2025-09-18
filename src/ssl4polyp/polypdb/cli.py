from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from .robustness.build_variants import build_sun_test_corruptions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="polypdb")
    subparsers = parser.add_subparsers(dest="command")

    sun_parser = subparsers.add_parser("sun")
    sun_sub = sun_parser.add_subparsers(dest="sun_command")

    corrupt_parser = sun_sub.add_parser("corrupt")
    corrupt_parser.add_argument("--pack", type=Path, required=True)
    corrupt_parser.add_argument("--spec", type=Path, required=True)
    corrupt_parser.add_argument("--roots", type=Path, required=True)
    corrupt_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "sun" and args.sun_command == "corrupt":
        with open(args.spec) as f:
            spec = yaml.safe_load(f)
        with open(args.roots) as f:
            roots = json.load(f)
        # Verify that all paths referenced in the test CSV exist before processing
        check_script = Path(__file__).resolve().parents[1] / "scripts" / "check_paths.py"
        test_csv = args.pack / "test.csv"
        subprocess.run(
            [sys.executable, str(check_script), str(test_csv), str(args.roots)],
            check=True,
        )
        build_sun_test_corruptions(args.pack, spec, roots, args.out)
    else:
        parser.print_help()
