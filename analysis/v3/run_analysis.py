#!/usr/bin/env python3
"""Run the SAP v3.2c analysis on campaign v3 results (or on a simulated campaign).

    conda activate ssl4polyp-analysis
    PYTHONPATH=. python -m analysis.v3.run_analysis --out results_v3/analysis [--roots ROOT ...]

Default roots: ``checkpoints_v3/campaign/classification`` and, when it exists, the SSL-imnet-matched campaign
``checkpoints_v3/campaign_matched/classification`` (matched runs written under the main root are also found).
Outputs: ``report.md``, ``summary.json``, ``contrast_summary.csv``, ``paired_differences.csv``, ``run_metrics.csv``,
``tables/`` (CSV + Markdown) and ``figures/``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402

from .analysis import Result, run_analysis  # noqa: E402
from .figures import write_figures  # noqa: E402
from .sap import B_BOOT, BOOT_SEED, HYPOTHESES, REPO  # noqa: E402
from .tables import build_tables, reliability, to_markdown, write_tables  # noqa: E402

DEFAULT_ROOTS = [REPO / "checkpoints_v3/campaign/classification", REPO / "checkpoints_v3/campaign_matched/classification"]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_report(res: Result, out: Path, tables: List[Path], figures: List[Path], title: str) -> Path:
    info = res.info
    sap = info["sap"]
    h = res.hypotheses
    lines = [f"# {title}", ""]
    lines += [
        f"- SAP implemented: v3.2 revision c, SHA-256 `{sap['implemented']}`; file matches: {sap['file_matches']}; "
        f"last SAP registered in RUNLOG: {sap['runlog_last']['name']} (`{sap['runlog_last']['sha256'][:12]}…`), matches: {sap['runlog_matches']}.",
        f"- Roots: {', '.join(f'`{r}`' for r in info['roots'])}",
        f"- Runs: {info['n_runs']}; regimes: {', '.join(info['regimes'])}; partitions: {info['partitions']}; seeds: {info['seeds']}.",
        f"- Views evaluated: {info['n_views']}; contrasts: {info['n_contrasts']}; bootstrap B = {info['bootstrap']['B']}, seed {info['bootstrap']['seed']}.",
        f"- Integrity checks: {info['integrity_checks']['n']} run, {info['integrity_checks']['failed']} failed.",
        f"- Elapsed: {info['elapsed_s']} s.",
        "",
    ]
    if res.notes:
        lines += ["## Notes", ""] + [f"- **{k}**: {v}" for k, v in sorted(res.notes.items())] + [""]
    incomplete = [r for r in info["completeness"] if r["present"] != r["expected"]]
    lines += ["## Completeness", ""]
    lines += ([f"- {r['experiment']} / {r['regime']}: {r['present']} of {r['expected']} runs" for r in incomplete]
              if incomplete else ["- All expected runs present for the regimes found."]) + [""]
    lines += ["## Primary hypotheses", ""]
    cols = [c for c in ("hypothesis", "estimate", "model_method", "model_ci_lo", "model_ci_hi", "p", "p_holm", "boot_lo", "boot_hi",
                        "confirmed", "status", "note") if c in h.columns]
    lines.append(to_markdown(h[cols], "H1–H7"))
    confirmed = [r["hypothesis"] for _, r in h.iterrows() if r.get("confirmed") is True]
    lines += [f"Confirmed (Holm p < 0.05 and bootstrap CI excluding 0): {', '.join(confirmed) if confirmed else 'none'}.", ""]
    lines += ["## Outputs", "", "Tables:"] + [f"- `{p.relative_to(out)}`" for p in tables] + ["", "Figures:"]
    lines += [f"- `{p.relative_to(out)}`" for p in figures] + [""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", nargs="+", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--title", default="Campaign v3 analysis (SAP v3.2c)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    res = run_analysis(args.roots, B=args.B, boot_seed=args.boot_seed, workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    tables = write_tables(res, args.out)
    figures = [] if args.no_figures else write_figures(res, reliability(res), args.out)
    report = write_report(res, args.out, tables, figures, args.title)
    summary = {"info": res.info, "notes": res.notes,
               "hypotheses": res.hypotheses.to_dict(orient="records")}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
