#!/usr/bin/env python3
"""Integrity check of a finished campaign v3 phase (spec v3.1 section 6); prints no metrics.

For every run of the phase: completion marker, canonical metrics JSON, the section 6 checks of
``verify_run_outputs`` (best-checkpoint reload, threshold epoch = evaluated epoch, complete CSVs,
JSON = recomputation from the CSVs, checkpoint SHA-256), and the run's partition, seed and git commit.
Output: counts and, for failures only, the failed criterion and its evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_plan as cp  # noqa: E402
import run_campaign as rc  # noqa: E402
import verify_run_outputs as vro  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def metrics_json(run: dict) -> Path:
    out = rc.output_dir(run)
    if run["experiment"] in ("E2", "E3"):
        found = [p for p in out.rglob("*.metrics.json") if not p.name.endswith("_last.metrics.json")]
    else:
        found = list(out.rglob("*_last.metrics.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"{len(found)} canonical metrics JSON under {out}")
    return found[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=sorted(rc.PHASES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    runs = [r for r in cp.run_matrix() if r["experiment"] in rc.PHASES[args.phase]]
    results, problems = [], []
    commits = Counter()
    for run in runs:
        name = rc.run_name(run)
        if not rc.is_complete(run):
            problems.append({"run": name, "criterion": "complete", "evidence": "no RUN_COMPLETE.json"})
            continue
        try:
            path = metrics_json(run)
            checks = vro.verify(path)
            payload = json.loads(path.read_text())
        except Exception as exc:
            problems.append({"run": name, "criterion": "verify", "evidence": repr(exc)})
            continue
        run_block = payload.get("run") or {}
        commits[run_block.get("git_commit")] += 1
        meta_ok = run_block.get("partition") == run["k"] and int(payload.get("seed")) == run["seed"]
        checks.append({"run": name, "criterion": "partition_seed", "passed": meta_ok,
                       "evidence": f"partition={run_block.get('partition')} seed={payload.get('seed')}"})
        for c in checks:
            c["run"] = name
            results.append(c)
            if not c["passed"]:
                problems.append(c)
    by_criterion = Counter(c["criterion"].split(":")[0] for c in results)
    summary = {"phase": args.phase, "expected_runs": len(runs), "complete_runs": sum(rc.is_complete(r) for r in runs),
               "checks": len(results), "checks_passed": sum(c["passed"] for c in results),
               "checks_by_criterion": dict(by_criterion), "git_commits": dict(commits), "problems": problems}
    out = args.out or REPO / f"results_v3/campaign/{args.phase}/verification.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=2))
    print(f"problems: {len(problems)}")
    for p in problems[:20]:
        print(f"  {p['run']} {p['criterion']}: {p['evidence']}")
    return 0 if not problems and summary["complete_runs"] == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
