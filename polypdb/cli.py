from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        build_sun_test_corruptions(args.pack, spec, roots, args.out)
    else:
        parser.print_help()
