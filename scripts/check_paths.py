#!/usr/bin/env python3
"""Verify that all files referenced in a CSV exist.

The CSV is expected to contain a column named ``frame_path`` whose entries
may begin with a root identifier (the first path component).  A JSON mapping
of root identifiers to filesystem paths is provided via ``roots.json`` and
used to resolve each entry to an absolute path.  The script exits with a
non-zero status on the first missing file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping


def resolve_path(raw: str, roots: Mapping[str, str]) -> Path:
    """Resolve ``raw`` path using ``roots`` mapping."""
    p = Path(raw)
    if p.parts:
        root = p.parts[0]
        if root in roots:
            p = Path(roots[root]) / Path(*p.parts[1:])
    return p


def check_csv(csv_path: Path, roots: Mapping[str, str], column: str) -> int:
    """Check that all entries in ``column`` of ``csv_path`` exist."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            print(
                f"CSV {csv_path} missing required column '{column}'",
                file=sys.stderr,
            )
            return 1
        for row in reader:
            raw = row.get(column, "")
            resolved = resolve_path(raw, roots)
            if not resolved.exists():
                print(f"Missing file: {resolved}", file=sys.stderr)
                return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that paths listed in a CSV exist"
    )
    parser.add_argument("csv", type=Path, help="CSV file containing paths")
    parser.add_argument(
        "roots",
        type=Path,
        help="JSON file mapping root identifiers to directories",
    )
    parser.add_argument(
        "--column",
        default="frame_path",
        help="Name of CSV column containing paths (default: frame_path)",
    )
    args = parser.parse_args()

    with open(args.roots) as f:
        roots = json.load(f)

    rc = check_csv(args.csv, roots, args.column)
    sys.exit(rc)


if __name__ == "__main__":
    main()
