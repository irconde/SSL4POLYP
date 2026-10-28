#!/usr/bin/env python3
"""Per-run records and hashed results manifest of a finished batch, written before its weights are deleted.

For every run under ``--root``:
  ``<out>/<tag>_runs.csv``              one row per run: GPU, attempt, duration, commit, evaluated checkpoint, epoch,
                                        checkpoint SHA-256 (recomputed from the file) and whether it equals the SHA-256
                                        recorded in the run's metrics JSON;
  ``<out>/<tag>_results_manifest.csv``  path, size and SHA-256 of every file of every run, checkpoints included
                                        (symbolic links are listed with their target and no hash).
Exits non-zero if the number of runs differs from ``--expected`` or any checkpoint hash differs from the recorded one.
Used by the per-budget learning-rate experiment (docs/v3/exploratory_lr_per_budget_plan.md, revision 1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "results_v3/verification")
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    runs, manifest, bad = [], [], []
    # run directories: launcher runs hold RUN_COMPLETE.json; runs started outside the launcher (the 1b reruns) are found
    # by their metrics JSON, the run directory being its nearest ``<regime>_seed<n>`` ancestor
    run_dirs = {m.parent for m in root.rglob("RUN_COMPLETE.json")}
    for m in root.rglob("*_last.metrics.json"):
        if not any(d in m.parents for d in run_dirs):
            run_dirs.add(next(p for p in m.parents if re.fullmatch(r"[a-z_]+_seed\d+", p.name)))
    for run_dir in sorted(run_dirs):
        marker = run_dir / "RUN_COMPLETE.json"
        done = json.loads(marker.read_text()) if marker.exists() else {}
        metrics = [p for p in run_dir.rglob("*_last.metrics.json")]
        payload = json.loads(metrics[0].read_text()) if len(metrics) == 1 else {}
        evaluated = payload.get("evaluated_checkpoint") or {}
        ckpt = next(run_dir.rglob(evaluated.get("path", "<none>")), None) if evaluated else None
        actual = sha256(ckpt) if ckpt is not None and ckpt.is_file() else ""
        match = bool(actual) and actual == evaluated.get("sha256")
        if not match:
            bad.append(str(run_dir.relative_to(REPO)))
        runs.append({"run": done.get("run", run_dir.name), "output_dir": str(run_dir.relative_to(REPO)), "gpu": done.get("gpu"),
                     "attempt": done.get("attempt"), "elapsed_s": done.get("elapsed_s"), "commit": str(done.get("commit", ""))[:7],
                     "evaluated_checkpoint": evaluated.get("path", ""), "checkpoint_epoch": evaluated.get("epoch", ""),
                     "checkpoint_bytes": ckpt.stat().st_size if ckpt is not None and ckpt.is_file() else "",
                     "checkpoint_sha256": actual, "checkpoint_sha256_matches_metrics_json": match})
        for path in sorted(p for p in run_dir.rglob("*") if p.is_file() or p.is_symlink()):
            rel = str(path.relative_to(REPO))
            if path.is_symlink():
                manifest.append({"path": rel, "bytes": "", "sha256": "", "symlink_to": os.readlink(path)})
            else:
                manifest.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path), "symlink_to": ""})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in ((f"{args.tag}_runs.csv", runs), (f"{args.tag}_results_manifest.csv", manifest)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    ok = len(runs) == args.expected and not bad
    print(json.dumps({"root": str(args.root), "runs": len(runs), "expected": args.expected, "files": len(manifest),
                      "checkpoints_hashed": sum(bool(r["checkpoint_sha256"]) for r in runs),
                      "checkpoint_hash_mismatches": bad, "passed": ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
