from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from .robustness.build_variants import build_sun_test_corruptions
from ..configs import resolve_config_path, resolve_data_pack_path


def _resolve_with_default(path: Path, resolver) -> Path:
    """Resolve ``path`` relative to a repository directory when not absolute."""

    if path.is_absolute() or path.exists():
        return path
    return resolver(path)


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
        pack_dir = _resolve_with_default(args.pack, resolve_data_pack_path)
        spec_path = _resolve_with_default(args.spec, resolve_config_path)
        roots_path = _resolve_with_default(args.roots, resolve_config_path)

        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        with open(roots_path) as f:
            roots = json.load(f)
        # Verify that all paths referenced in the test CSV exist before processing
        check_script = Path(__file__).resolve().parents[1] / "scripts" / "check_paths.py"
        test_csv = pack_dir / "test.csv"
        subprocess.run(
            [sys.executable, str(check_script), str(test_csv), str(roots_path)],
            check=True,
        )
        build_sun_test_corruptions(pack_dir, spec, roots, args.out)
    else:
        parser.print_help()
