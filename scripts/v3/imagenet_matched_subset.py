#!/usr/bin/env python3
"""Class-stratified ImageNet-1k train subset of 99,417 images for the SSL-imnet-matched regime.

Algorithm (seed fixed by ``--seed``, numpy ``default_rng``):
  1. ``synsets`` = the 1,000 class folders in sorted order.
  2. ``extra`` = ``rng.choice(1000, size=417, replace=False)``: classes that get 100 images.
  3. For each synset in sorted order: ``files`` = its file names sorted;
     ``take`` = 100 if the class is in ``extra`` else 99; keep ``files[rng.permutation(len(files))[:take]]``.
The subset is ``<out>/<synset>/<name>`` with absolute symbolic links to ``<source>``. Writes
``subset_list.csv`` (relative path, SHA-256 of the image recomputed from the file) and its SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

TOTAL, PER_CLASS, EXTRA = 99_417, 99, 417


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path("/workspace/data/imagenet_train"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/data/imagenet_matched_99417"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists")

    synsets = sorted(p.name for p in args.source.iterdir() if p.is_dir())
    assert len(synsets) == 1000, len(synsets)
    rng = np.random.default_rng(args.seed)
    extra = set(int(i) for i in rng.choice(len(synsets), size=EXTRA, replace=False))
    rows = []
    for idx, synset in enumerate(synsets):
        files = sorted(os.listdir(args.source / synset))
        take = PER_CLASS + (1 if idx in extra else 0)
        chosen = [files[i] for i in rng.permutation(len(files))[:take]]
        (args.out / synset).mkdir(parents=True)
        for name in sorted(chosen):
            src = (args.source / synset / name).resolve()
            os.symlink(src, args.out / synset / name)
            rows.append((f"{synset}/{name}", hashlib.sha256(src.read_bytes()).hexdigest()))
    assert len(rows) == TOTAL, len(rows)

    args.records.mkdir(parents=True, exist_ok=True)
    list_path = args.records / "subset_list.csv"
    with list_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relpath", "sha256"])
        writer.writerows(rows)
    counts = {}
    for rel, _ in rows:
        counts[rel.split("/")[0]] = counts.get(rel.split("/")[0], 0) + 1
    summary = {
        "seed": args.seed, "images": len(rows), "classes": len(counts),
        "classes_with_100": sum(1 for c in counts.values() if c == 100),
        "classes_with_99": sum(1 for c in counts.values() if c == 99),
        "extra_classes": [synsets[i] for i in sorted(extra)],
        "list_sha256": hashlib.sha256(list_path.read_bytes()).hexdigest(),
        "symlinks": sum(1 for p in args.out.rglob("*") if p.is_symlink()),
        "broken_symlinks": sum(1 for p in args.out.rglob("*") if p.is_symlink() and not p.exists()),
    }
    (args.records / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "extra_classes"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
