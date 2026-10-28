#!/usr/bin/env python3
"""Check that existing E1 runs can stand in for the runs of the per-budget learning-rate experiment.

Exploratory plan ``docs/v3/exploratory_lr_per_budget_plan.md`` (revision 1) reuses 300 E1 runs of earlier
experiments instead of repeating them:
  * 1e-3, SUP-imnet / SSL-imnet / SSL-colon: ``checkpoints_v3/campaign_lr_common/classification`` (225 runs);
  * 1e-4, SSL-imnet: ``checkpoints_v3/campaign_lr_sslimnet_1e-4/classification`` (75 runs).

For each reused run it resolves the equivalent new run (``campaign_plan.run_matrix`` with the experiment's root
and ``--lr-override``) through ``train_classification.prepare_run``, rebuilds the ``config.yaml`` that the new run
would write and compares it with the stored one after mapping the old root to the new one. The command line
(``RUN_COMPLETE.json``) is compared the same way. The ``git_commit`` field is not compared; the fine-tuning code
of the recorded commits is compared with HEAD separately (``git diff``).

Writes ``<out>/exp_lr_per_budget_reuse_config_check.csv`` and ``.json``; exits non-zero on any difference.
Run from the repository root in ``ssl4polyp-ft``.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_plan as cp  # noqa: E402

REPO = cp.REPO
NEW_ROOT = "checkpoints_v3/exp_lr_per_budget/lr{lr}/classification"
REUSED = [  # (old root, lr as passed to --override, regimes)
    ("checkpoints_v3/campaign_lr_common/classification", "1e-3", ("sup_imnet", "ssl_imnet", "ssl_colon")),
    ("checkpoints_v3/campaign_lr_sslimnet_1e-4/classification", "1e-4", ("ssl_imnet",)),
]
FINETUNE_CODE = ["src/ssl4polyp/classification", "src/ssl4polyp/models", "src/ssl4polyp/_compat.py", "config"]


def serialize(value):
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_config(argv):
    """The ``config.yaml`` content train_classification writes for ``argv`` (without ``git_commit``)."""

    sys.argv = ["train_classification", *argv]
    args = cp.tc.get_args()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        prepared = cp.tc.prepare_run(args)
    cfg = {"args": {k: serialize(v) for k, v in vars(args).items() if k != "roots_map"}}
    if prepared["resolved_overrides"]:
        cfg["config_overrides"] = serialize(prepared["resolved_overrides"])
    if prepared["experiment_cfg"] is not None:
        cfg["experiment"] = serialize(prepared["experiment_cfg"])
    if prepared["dataset_cfg"] is not None and prepared["dataset_resolved"] is not None:
        cfg["dataset"] = {"configured": serialize(prepared["dataset_cfg"]), "resolved": serialize(prepared["dataset_resolved"])}
    if prepared["selected_model"] is not None:
        cfg["model"] = serialize(prepared["selected_model"])
    # yaml round trip: the stored file went through safe_dump/safe_load
    return yaml.safe_load(yaml.safe_dump(cfg))


def remap(value, old: str, new: str):
    if isinstance(value, dict):
        return {k: remap(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, old, new) for v in value]
    return value.replace(old, new) if isinstance(value, str) else value


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}{k}: {'missing in stored' if k not in b else 'missing in new'}")
            else:
                out += diff(a[k], b[k], f"{path}{k}.")
        return out
    return [] if a == b else [f"{path.rstrip('.')}: new={a!r} stored={b!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "results_v3/verification")
    args = parser.parse_args()
    rows, commits = [], set()
    for old_root, lr, regimes in REUSED:
        new_root = NEW_ROOT.format(lr=lr)
        cp.set_campaign(new_root, lr)
        for run in (r for r in cp.run_matrix(regimes) if r["experiment"] == "E1"):
            new_out = run["argv"][run["argv"].index("--output-dir") + 1]
            old_out = REPO / new_out.replace(new_root, old_root)
            complete = json.loads((old_out / "RUN_COMPLETE.json").read_text())
            stored_cfg = yaml.safe_load(next(old_out.rglob("config.yaml")).read_text())
            commits.add(stored_cfg.pop("git_commit"))
            argv_diff = [] if remap(complete["argv"], old_root, new_root) == run["argv"] else [
                f"argv: new={run['argv']} stored={complete['argv']}"]
            cfg_diff = diff(run_config(run["argv"]), remap(stored_cfg, old_root, new_root))
            rows.append({"regime": run["regime"], "lr": lr, "k": run["k"], "seed": run["seed"], "budget": run["variant"],
                         "stored_run": str(old_out.relative_to(REPO)), "argv_match": not argv_diff,
                         "config_match": not cfg_diff, "differences": " | ".join(argv_diff + cfg_diff)})
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    code = {}
    for c in sorted(commits):
        # recorded hashes predate the 2026-10-27 history rewrite; docs/history-rewrite.csv maps them
        with (REPO / "docs/history-rewrite.csv").open(newline="") as handle:
            current = next((r["new_commit"] for r in csv.DictReader(handle) if r["old_commit"] == c), c)
        files = subprocess.run(["git", "diff", "--name-only", current, head, "--", *FINETUNE_CODE],
                               capture_output=True, text=True, cwd=REPO).stdout.split()
        # v2 report modules (legacy.reporting import path) and the MAE demo notebook are not used by fine-tuning
        relevant = [f for f in files if "/classification/analysis/" not in f and not f.endswith(".ipynb")]
        code[c] = {"current_hash": current, "changed_files": files, "finetune_code_changed": relevant}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "exp_lr_per_budget_reuse_config_check.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"runs": len(rows), "argv_match": sum(r["argv_match"] for r in rows),
               "config_match": sum(r["config_match"] for r in rows), "head": head, "training_commits": code,
               "passed": all(r["argv_match"] and r["config_match"] for r in rows)
               and not any(v["finetune_code_changed"] for v in code.values())}
    (args.out / "exp_lr_per_budget_reuse_config_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "training_commits"}, indent=2))
    for r in rows:
        if r["differences"]:
            print(r["stored_run"], r["differences"][:400])
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
