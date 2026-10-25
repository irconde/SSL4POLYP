#!/usr/bin/env python3
"""Replace personal absolute paths by placeholders in the run records published with the article.

Scope (tracked text files only): pack manifests under ``data_packs/`` (v2 and v3; the CSVs are not
touched), ``results_v3/RUNLOG.md``, ``results_v3/analysis/``, ``results_v3/analysis_exploratory_*/``,
``results_v3/verification/`` and ``results_v3/environment/``.

Placeholders: ``<REPO>`` (repository root), ``<DATA_ROOT>`` (data directory, cf. ``data/roots.json``),
``<CONDA_ROOT>`` (conda installation) and ``<HOME>`` (any other home-directory path).
Rules are applied in order, most specific first. Prints a summary; ``--check`` exits non-zero if any
personal path is left.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOT_DATA_PACKS = r"(?![A-Za-z0-9_])"  # ".../data" but not ".../data_packs"
RULES = [
    (r"/Users/irconde/Desktop/sssl-polyps/workspace/code/SSL4GIE", "<REPO>"),
    (r"/Users/irconde/Desktop/sssl-polyps/workspace/data" + NOT_DATA_PACKS, "<DATA_ROOT>"),
    (r"/home/irconde/workspace/code/SSL4POLYP/data" + NOT_DATA_PACKS, "<DATA_ROOT>"),
    (r"/workspace/code/SSL4POLYP/data" + NOT_DATA_PACKS, "<DATA_ROOT>"),
    (r"/home/irconde/workspace/code/SSL4POLYP", "<REPO>"),
    (r"/workspace/code/SSL4POLYP", "<REPO>"),
    (r"/workspace/code/SSL4GIE", "<REPO>"),
    (r"/workspace/data_packs_v3_staging", "<DATA_ROOT>/data_packs_v3_staging"),
    (r"/home/irconde/workspace/data" + NOT_DATA_PACKS, "<DATA_ROOT>"),
    (r"/workspace/data" + NOT_DATA_PACKS, "<DATA_ROOT>"),
    (r"/home/irconde/miniforge3", "<CONDA_ROOT>"),
    (r"/home/irconde/workspace", "<REPO>/.."),
    (r"/home/irconde", "<HOME>"),
    (r"/Users/irconde", "<HOME>"),
]
LEFT = re.compile(r"/Users/irconde|/home/irconde|/workspace/(code|data)")


def targets() -> list:
    files = subprocess.run(["git", "ls-files", "data_packs", "results_v3"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.split()
    keep = []
    for f in files:
        if f.startswith("data_packs/") and not f.endswith("manifest.yaml"):
            continue
        if f.startswith("results_v3/") and not re.match(
                r"results_v3/(RUNLOG\.md|analysis/|analysis_exploratory_[^/]+/|verification/|environment/)", f):
            continue
        keep.append(f)
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report remaining personal paths")
    args = parser.parse_args()
    changed, counts, left = [], Counter(), []
    for rel in targets():
        path = REPO / rel
        raw = path.read_bytes()
        if b"\0" in raw[:8000]:
            continue
        text = raw.decode("utf-8")
        new = text
        if not args.check:
            for pattern, repl in RULES:
                new, n = re.subn(pattern, repl, new)
                counts[repl] += n
            if new != text:
                path.write_bytes(new.encode("utf-8"))  # keeps the original line endings
                changed.append(rel)
        if LEFT.search(new):
            left.append(rel)
    print(f"files changed: {len(changed)}; replacements: {dict(counts)}; files with personal paths left: {left}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
