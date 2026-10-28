#!/usr/bin/env python3
"""Rerun a few reused E1 runs and compare their per-frame predictions with the originals bit for bit.

Exploratory plan ``docs/v3/exploratory_lr_per_budget_plan.md`` (revision 1, check 1b): one run per reused regime and
learning rate (k = 1, seed 13, a different budget each) is trained again with the command line of the equivalent new
run (``campaign_plan.run_matrix``) under ``checkpoints_v3/exp_lr_per_budget/repro/``, with the launcher's environment
(one GPU per run, ``CUBLAS_WORKSPACE_CONFIG=:4096:8``). The validation and test output CSVs must be byte-identical to
those of the reused run.

Writes ``<out>/exp_lr_per_budget_repro_check.csv`` and ``.json``; exits non-zero on any difference.
Run from the repository root in ``ssl4polyp-ft``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_plan as cp  # noqa: E402

REPO = cp.REPO
ROOT = "checkpoints_v3/exp_lr_per_budget/repro/lr{lr}/classification"
RUNS = [  # (regime, lr, budget, root of the reused run)
    ("sup_imnet", "1e-3", "p5", "checkpoints_v3/campaign_lr_common/classification"),
    ("ssl_imnet", "1e-3", "p10", "checkpoints_v3/campaign_lr_common/classification"),
    ("ssl_colon", "1e-3", "p25", "checkpoints_v3/campaign_lr_common/classification"),
    ("ssl_imnet", "1e-4", "p50", "checkpoints_v3/campaign_lr_sslimnet_1e-4/classification"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "results_v3/verification")
    parser.add_argument("--gpus", default="0 1")
    args = parser.parse_args()
    os.chdir(REPO)
    jobs = []
    for regime, lr, budget, old_root in RUNS:
        cp.set_campaign(ROOT.format(lr=lr), lr)
        run = next(r for r in cp.run_matrix((regime,)) if r["experiment"] == "E1" and r["k"] == 1
                   and r["seed"] == 13 and r["variant"] == budget)
        new_out = run["argv"][run["argv"].index("--output-dir") + 1]
        jobs.append({"regime": regime, "lr": lr, "budget": budget, "argv": run["argv"], "new": REPO / new_out,
                     "old": REPO / new_out.replace(ROOT.format(lr=lr), old_root)})
    log_dir = REPO / "results_v3/exp_lr_per_budget/repro/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not pending:
                    return
                job = pending.pop(0)
            if job["new"].exists():
                sys.exit(f"output exists: {job['new']}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
            name = f"E1_k1_{job['regime']}_seed13_{job['budget']}_lr{job['lr']}"
            start = time.time()
            with (log_dir / f"{name}.stdout").open("w") as handle:
                job["rc"] = subprocess.run([sys.executable, "-m", "ssl4polyp.classification.train_classification",
                                            *job["argv"]], stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
            job["gpu"], job["elapsed_s"] = gpu, round(time.time() - start, 1)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = []
    for job in jobs:
        for kind in ("val_outputs", "test_outputs"):
            old = next(job["old"].rglob(f"*_{kind}.csv"))
            new = next(job["new"].rglob(f"*_{kind}.csv"), None)
            rows.append({"regime": job["regime"], "lr": job["lr"], "k": 1, "seed": 13, "budget": job["budget"],
                         "file": kind, "rc": job.get("rc"), "gpu": job.get("gpu"), "elapsed_s": job.get("elapsed_s"),
                         "original": str(old.relative_to(REPO)), "original_sha256": sha256(old),
                         "rerun": str(new.relative_to(REPO)) if new else "",
                         "rerun_sha256": sha256(new) if new else "",
                         "identical": bool(new) and old.read_bytes() == new.read_bytes()})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "exp_lr_per_budget_repro_check.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"runs": len(jobs), "files": len(rows), "identical": sum(r["identical"] for r in rows),
               "passed": all(r["identical"] for r in rows)}
    (args.out / "exp_lr_per_budget_repro_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
