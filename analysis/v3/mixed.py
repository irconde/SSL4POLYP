"""Mixed models (SAP section 4) through ``Rscript`` and Holm's adjustment."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parent / "R" / "fit_mixed.R"


def rscript() -> str:
    """``Rscript`` of the running environment (``ssl4polyp-analysis``), else the one on PATH."""

    local = Path(sys.executable).parent / "Rscript"
    if local.exists():
        return str(local)
    found = shutil.which("Rscript")
    if not found:
        raise RuntimeError("Rscript not found: activate ssl4polyp-analysis")
    return found


def fit_many(per_run: Mapping[str, Mapping[Tuple[int, int], float]]) -> pd.DataFrame:
    """Fit ``d_{k,s} = mu + u_k + e_{k,s}`` for every id in ``{id: {(k, s): d}}`` (one Rscript call)."""

    rows = [
        {"id": cid, "k": k, "s": s, "d": d}
        for cid, values in per_run.items()
        for (k, s), d in sorted(values.items())
    ]
    if not rows:
        return pd.DataFrame(columns=["id"])
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        pd.DataFrame(rows).to_csv(src, index=False, float_format="%.17g")
        env = dict(os.environ, LC_ALL="C")
        proc = subprocess.run([rscript(), str(R_SCRIPT), str(src), str(dst)], capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"fit_mixed.R failed:\n{proc.stderr}")
        out = pd.read_csv(dst, dtype={"id": str}, keep_default_na=True)
    out["note"] = out["note"].fillna("").astype(str).str.strip()
    return out


def holm(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values (same as R ``p.adjust(method = "holm")``)."""

    items = [(k, float(p)) for k, p in pvalues.items() if p is not None and np.isfinite(p)]
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    for key, p in pvalues.items():
        adjusted.setdefault(key, float("nan"))
    return adjusted


def spread(per_run: Mapping[Tuple[int, int], float]) -> Dict[str, float]:
    """Descriptive SDs (SAP section 7): between partitions (SD of partition means) and between seeds
    (pooled SD of the seeds within each partition)."""

    by_k: Dict[int, list] = {}
    for (k, _), d in per_run.items():
        if np.isfinite(d):
            by_k.setdefault(k, []).append(d)
    means = [np.mean(v) for v in by_k.values()]
    sd_partition = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
    ss = sum(float(np.sum((np.asarray(v) - np.mean(v)) ** 2)) for v in by_k.values())
    dof = sum(len(v) - 1 for v in by_k.values())
    sd_seed = float(np.sqrt(ss / dof)) if dof > 0 else float("nan")
    values = [d for v in by_k.values() for d in v]
    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "sd_partition": sd_partition,
        "sd_seed": sd_seed,
        "n_runs": len(values),
        "n_partitions": len(by_k),
    }
