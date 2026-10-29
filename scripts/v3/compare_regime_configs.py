#!/usr/bin/env python3
"""Compare the resolved run configurations of an exploratory regime with those of a registered regime.

For one run per experiment and variant (k = 1, seed 13: sun_full, E1 p5–p100, E2, E3, E4 s50–s500), both regimes are
resolved as ``train_classification`` would (``campaign_plan.resolve``) and every argument that differs is listed. Over
the full matrix (5 partitions × 3 seeds), the planner fields that must be equal (packs, splits, learning rate and its
source, fine-tuning mode, threshold policy) are compared row by row.

    python scripts/v3/compare_regime_configs.py --regime ssl_colon_r --root checkpoints_v3/exp_sslcolon_r/classification \
        --reference ssl_colon --out results_v3/verification/exp_sslcolon_r_config_check.json

Run from the repository root in ``ssl4polyp-ft``; nothing is trained or written under the roots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_plan as cp  # noqa: E402

EQUAL_FIELDS = ("train_pack", "val_pack", "test_pack", "train_split", "val_split", "test_split", "lr", "lr_source",
                "finetune_mode", "threshold_policy", "partition", "seed", "dataset_percent", "morphology_eval", "epochs")
# Fields expected to differ: identity of the regime and its checkpoint, and paths that contain them.
EXPECTED = {"model_key", "model_tag", "model_name", "ckpt", "ckpt_sha256", "output_dir", "parent_checkpoint", "run_stem",
            "init_checkpoint_sha256", "metrics_path", "tensorboard_dir", "thresholds_root"}


def resolved(regime: str, root: str):
    cp.set_campaign(root)
    out = {}
    for run in cp.run_matrix((regime,)):
        key = (run["experiment"], run["variant"], run["k"], run["seed"])
        out[key] = run
    return out


def args_of(run) -> dict:
    """Resolved arguments; the regime name inside ``lr_source`` ("model config (<regime>)") is masked."""

    a = cp.resolve(run["argv"])["args"]
    out = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)) for k, v in vars(a).items()}
    if isinstance(out.get("lr_source"), str):
        out["lr_source"] = out["lr_source"].replace(f"({run['regime']})", "(<regime>)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-root", default=cp.CKPT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    new, ref = resolved(args.regime, args.root), resolved(args.reference, args.reference_root)
    if set(new) != set(ref):
        sys.exit("run matrices differ")
    smoke, unexpected = [], []
    for key in sorted(k for k in new if k[2] == 1 and k[3] == 13):
        cp.set_campaign(args.root)
        a = args_of(new[key])
        cp.set_campaign(args.reference_root)
        b = args_of(ref[key])
        diff = {f: {"new": a.get(f), "reference": b.get(f)} for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)}
        bad = sorted(set(diff) - EXPECTED)
        smoke.append({"experiment": key[0], "variant": key[1], "differing_fields": sorted(diff), "unexpected": bad, "diff": diff})
        unexpected += [(key, f) for f in bad]
    rows = []
    for key in sorted(new):
        cp.set_campaign(args.root)
        a = args_of(new[key])
        cp.set_campaign(args.reference_root)
        b = args_of(ref[key])
        d = [f for f in EQUAL_FIELDS if a.get(f) != b.get(f)]
        if d:
            rows.append({"run": list(key), "fields": d})
    summary = {"regime": args.regime, "root": args.root, "reference": args.reference, "runs": len(new),
               "smoke_runs": len(smoke), "smoke_unexpected_differences": [[list(k), f] for k, f in unexpected],
               "matrix_fields_compared": list(EQUAL_FIELDS), "matrix_differences": rows,
               "passed": not unexpected and not rows, "smoke": smoke}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "smoke"}, indent=2, default=str))
    for s in smoke:
        print(s["experiment"], s["variant"], s["differing_fields"])
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
