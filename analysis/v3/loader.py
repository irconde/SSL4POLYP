"""Discover and load campaign v3 runs (per-frame CSVs and operating thresholds).

Layout written by ``scripts/v3/campaign_plan.py`` under each root (``.../classification``)::

    k{K}/sun_full/{regime}_seed{S}/sun_baselines/*_test_outputs.csv      + *_last.metrics.json
    k{K}/e1/p{P}/{regime}_seed{S}/sun_subsets/p{P}/seed{S}/...           (same files)
    k{K}/e2/{regime}_seed{S}/sun_morphology/*_test_outputs.csv           + *.metrics.json (eval only)
    k{K}/e3/{regime}_seed{S}/polypgen_clean_test/*_test_outputs.csv      + *.metrics.json (eval only)
    k{K}/e4/s{N}/{regime}_seed{S}/polypgen_fewshot/s{N}/seed{S}/
        *_zeroshot_outputs.csv (arms A, B)   *_test_outputs.csv (arms C, D)   *_last.metrics.json (e4_arms)

A run counts only when its directory holds ``RUN_COMPLETE.json``. Regimes are read from the
directory names, so a root holding the SSL-imnet-matched campaign is picked up the same way.
Only operating thresholds τ are read from the JSONs; every metric is recomputed from the CSVs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

FRAME_COLUMNS = (
    "frame_id", "prob", "label", "pred", "case_id", "sequence_id",
    "center_id", "origin", "morphology", "negative_source",
)
TEXT_COLUMNS = ("frame_id", "case_id", "sequence_id", "center_id", "origin", "morphology", "negative_source")
RUN_DIR = re.compile(r"^(?P<regime>[a-z_]+)_seed(?P<seed>\d+)$")


@dataclass
class Frames:
    path: Path
    probs: np.ndarray
    labels: np.ndarray
    text: Dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.labels.size)

    def col(self, name: str) -> np.ndarray:
        return self.text[name]


def load_frames(path: Path) -> Frames:
    df = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, keep_default_na=False)
    missing = [c for c in FRAME_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return Frames(
        path=Path(path),
        probs=df["prob"].to_numpy(dtype=np.float64),
        labels=df["label"].to_numpy(dtype=np.int64),
        text={c: df[c].str.strip().to_numpy(dtype=object) for c in TEXT_COLUMNS},
    )


@dataclass
class Run:
    experiment: str            # sun_full, e1, e2, e3, e4
    regime: str
    k: int
    seed: int
    variant: Optional[int]     # E1 percent or E4 budget
    run_dir: Path
    frames: Dict[str, Frames] = field(default_factory=dict)   # "test" (all), "zeroshot" (E4)
    tau: Dict[str, float] = field(default_factory=dict)       # "primary" or E4 arms "A".."D"
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> Tuple:
        return (self.experiment, self.variant, self.regime, self.k, self.seed)


def _one(paths: List[Path], what: str, run_dir: Path) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one {what}, found {len(paths)}")
    return paths[0]


def _parse(rel: Tuple[str, ...]) -> Optional[Tuple[str, int, Optional[int], str, int, int]]:
    """``rel`` = path parts below the root; returns (experiment, k, variant, regime, seed, depth)."""

    if len(rel) < 3 or not re.fullmatch(r"k\d+", rel[0]):
        return None
    k, exp = int(rel[0][1:]), rel[1]
    if exp in ("e1", "e4"):
        if len(rel) < 4:
            return None
        variant = int(re.sub(r"^[ps]", "", rel[2]))
        m, depth = RUN_DIR.match(rel[3]), 4
    elif exp in ("sun_full", "e2", "e3"):
        variant, m, depth = None, RUN_DIR.match(rel[2]), 3
    else:
        return None
    if not m:
        return None
    return exp, k, variant, m["regime"], int(m["seed"]), depth


def _load_run(root: Path, marker: Path) -> Optional[Run]:
    run_dir = marker.parent
    parsed = _parse(run_dir.relative_to(root).parts)
    if parsed is None:
        return None
    exp, k, variant, regime, seed, _ = parsed
    run = Run(exp, regime, k, seed, variant, run_dir)
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    last = [p for p in files if p.name.endswith("_last.metrics.json")]
    plain = [p for p in files if p.name.endswith(".metrics.json") and not p.name.endswith("_last.metrics.json")]
    test = [p for p in files if p.name.endswith("_test_outputs.csv")]
    if exp in ("sun_full", "e1"):
        payload = json.loads(_one(last, "*_last.metrics.json", run_dir).read_text())
        run.tau["primary"] = float(payload["test_primary"]["tau"])
        run.info["threshold_policy"] = (payload.get("thresholds") or {}).get("primary", {}).get("policy")
    elif exp in ("e2", "e3"):
        payload = json.loads(_one(plain, "*.metrics.json", run_dir).read_text())
        run.tau["primary"] = float(payload["test_primary"]["tau"])
    else:
        payload = json.loads(_one(last, "*_last.metrics.json", run_dir).read_text())
        arms = payload["e4_arms"]
        for arm in ("A", "B", "C", "D"):
            run.tau[arm] = float(arms[arm]["metrics"]["tau"])
        run.info["kept_inherited_tau_D"] = int(arms["D"].get("kept_inherited_tau", 0) or 0)
        zs = [p for p in files if p.name.endswith("_zeroshot_outputs.csv")]
        run.frames["zeroshot"] = load_frames(_one(zs, "*_zeroshot_outputs.csv", run_dir))
    run.frames["test"] = load_frames(_one(test, "*_test_outputs.csv", run_dir))
    return run


@dataclass
class Campaign:
    runs: List[Run]
    roots: List[Path]
    checks: List[Dict[str, object]] = field(default_factory=list)

    def index(self) -> Dict[Tuple, Run]:
        return {r.key: r for r in self.runs}

    def regimes(self) -> List[str]:
        from .sap import REGIMES

        present = {r.regime for r in self.runs}
        return [g for g in REGIMES if g in present] + sorted(present - set(REGIMES))

    def partitions(self) -> List[int]:
        return sorted({r.k for r in self.runs})

    def seeds(self) -> List[int]:
        return sorted({r.seed for r in self.runs})

    def get(self, experiment: str, variant: Optional[int], regime: str, k: int, seed: int) -> Optional[Run]:
        return self.index().get((experiment, variant, regime, k, seed))


def load_campaign(roots: Iterable[Path]) -> Campaign:
    roots = [Path(r) for r in roots]
    runs: List[Run] = []
    seen: Dict[Tuple, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for marker in sorted(root.rglob("RUN_COMPLETE.json")):
            if "_aborted" in marker.parts:
                continue
            run = _load_run(root, marker)
            if run is None:
                continue
            if run.key in seen:
                raise ValueError(f"run {run.key} found twice: {seen[run.key]} and {run.run_dir}")
            seen[run.key] = run.run_dir
            runs.append(run)
    campaign = Campaign(runs, roots)
    campaign.checks = integrity_checks(campaign)
    return campaign


def integrity_checks(c: Campaign) -> List[Dict[str, object]]:
    """Structural checks only (frames, thresholds and counts); no metric is computed."""

    idx = c.index()
    out: List[Dict[str, object]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        out.append({"check": name, "passed": bool(ok), "detail": detail})

    for run in c.runs:
        if run.experiment in ("e2", "e3", "e4"):
            parent = idx.get(("sun_full", None, run.regime, run.k, run.seed))
            if parent is None:
                add("parent_present", False, f"{run.key}: no sun_full parent")
                continue
            inherited = [run.tau["primary"]] if run.experiment != "e4" else [run.tau["A"], run.tau["C"]]
            ok = all(t == parent.tau["primary"] for t in inherited)
            add("inherited_tau", ok, f"{run.key}: {'equal' if ok else 'differs from'} parent sun_full τ")
        if run.experiment == "e2":
            parent = idx.get(("sun_full", None, run.regime, run.k, run.seed))
            if parent is not None:
                same = np.array_equal(parent.frames["test"].col("frame_id"), run.frames["test"].col("frame_id"))
                add("e2_frames_equal_sun_full_test", same, str(run.key))
        if run.experiment == "e4":
            zs, ad = run.frames["zeroshot"], run.frames["test"]
            same = np.array_equal(zs.col("frame_id"), ad.col("frame_id")) and np.array_equal(zs.labels, ad.labels)
            add("e4_zeroshot_and_adapted_same_split", same, str(run.key))
    # Paired runs evaluate the same frames (same test set across regimes within (experiment, variant, k, s)).
    groups: Dict[Tuple, List[Run]] = {}
    for run in c.runs:
        groups.setdefault((run.experiment, run.variant, run.k, run.seed), []).append(run)
    for key, members in sorted(groups.items(), key=lambda kv: repr(kv[0])):
        ref = members[0].frames["test"]
        ok = all(np.array_equal(m.frames["test"].col("frame_id"), ref.col("frame_id"))
                 and np.array_equal(m.frames["test"].labels, ref.labels) for m in members)
        add("paired_runs_same_frames", ok, f"{key}: {len(members)} regimes")
    return out


def completeness(c: Campaign, regimes: Iterable[str], partitions: Iterable[int], seeds: Iterable[int]) -> pd.DataFrame:
    """Expected vs present runs per experiment and regime."""

    from .sap import E1_PERCENTS, E4_BUDGETS

    idx = c.index()
    rows = []
    for exp, variants in (("sun_full", [None]), ("e1", E1_PERCENTS), ("e2", [None]), ("e3", [None]), ("e4", E4_BUDGETS)):
        for regime in regimes:
            expected = [(exp, v, regime, k, s) for v in variants for k in partitions for s in seeds]
            present = sum(1 for key in expected if key in idx)
            rows.append({"experiment": exp, "regime": regime, "expected": len(expected), "present": present})
    return pd.DataFrame(rows)
