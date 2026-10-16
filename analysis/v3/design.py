"""Evaluation views, paired contrasts and the cluster bootstrap (SAP sections 4.1 and 5).

* A **view** is one run's per-frame outputs restricted to some rows (a stratum, a negative source)
  at one operating threshold, with its bootstrap units.
* A **contrast** defines one paired difference per run (partition k, seed s) as a linear combination
  of view metrics, ``d_{k,s} = Σ w · metric(view)``. Differences between regimes, arm differences,
  AULC (trapezoid weights) and means over regimes or budgets are all linear, so one representation
  serves the point estimate, the mixed model and every bootstrap replicate.
* **Bootstrap** (section 5): B = 2000, seed 12345, one sample of units per replicate and per group
  (group = common test set: the SUN partition for E1/E2, one group for PolypGen), shared by all the
  runs of that group; drawn per stratum with replacement. Estimate = mean of the per-run differences;
  interval = 95 % percentile. The samples are drawn sequentially in the parent process, so the result
  does not depend on the number of worker processes.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .loader import Campaign, Frames
from .metrics import evaluate
from .sap import B_BOOT, BOOT_SEED, NEGSOURCE_BUDGET

ViewKey = Tuple  # (family, variant, regime, k, seed, slot)
RunId = Tuple[int, int]  # (k, seed)


# --------------------------------------------------------------------------- units
def units_case(f: Frames, rows: np.ndarray) -> List[Tuple[str, str]]:
    return [(f"y{f.labels[i]}", f.col("case_id")[i]) for i in rows]


def units_case_morph(f: Frames, rows: np.ndarray) -> List[Tuple[str, str]]:
    out = []
    for i in rows:
        stratum = "y0" if f.labels[i] == 0 else f"y1:{f.col('morphology')[i].lower()}"
        out.append((stratum, f.col("case_id")[i]))
    return out


def units_center(f: Frames, rows: np.ndarray) -> List[Tuple[str, str]]:
    return [(f"y{f.labels[i]}", f.col("center_id")[i] or "NA") for i in rows]


def units_e4(f: Frames, rows: np.ndarray) -> List[Tuple[str, str]]:
    out = []
    for i in rows:
        if f.labels[i] == 1:
            out.append(("pos", f.col("center_id")[i] or "NA"))
        elif f.col("sequence_id")[i]:
            out.append(("neg", "seq:" + f.col("sequence_id")[i]))
        else:
            out.append(("neg", "center:" + (f.col("center_id")[i] or "NA")))
    return out


UNIT_FNS = {"case": units_case, "case_morph": units_case_morph, "center": units_center, "e4": units_e4}


@dataclass
class View:
    key: ViewKey
    frames: Frames
    tau: float
    rows: np.ndarray
    unit_kind: str
    group: Hashable
    per_polyp: bool = False
    units: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = UNIT_FNS[self.unit_kind](self.frames, self.rows)
        grouped: Dict[Tuple[str, str], List[int]] = {}
        for row, unit in zip(self.rows.tolist(), names):
            grouped.setdefault(unit, []).append(row)
        self.units = {u: np.asarray(r, dtype=np.int64) for u, r in grouped.items()}

    def metrics(self, rows: Optional[np.ndarray] = None, copies: Optional[np.ndarray] = None) -> Dict[str, float]:
        rows = self.rows if rows is None else rows
        codes = None
        if self.per_polyp:
            cases = self.frames.col("case_id")[rows]
            codes = cases if copies is None else np.char.add(np.char.add(cases.astype(str), "#"), copies.astype(str))
        return evaluate(self.frames.probs, self.frames.labels, self.tau, rows, codes)


def _strata_rows(f: Frames) -> Dict[str, np.ndarray]:
    pos = f.labels == 1
    morph = np.char.lower(f.col("morphology").astype(str))
    return {
        "overall": np.arange(len(f)),
        "flat": np.flatnonzero(~pos | (pos & (morph == "flat"))),
        "polypoid": np.flatnonzero(~pos | (pos & (morph == "polypoid"))),
    }


def negsource_rows(zeroshot: Frames, clean_ids: set, source: str, exclude_center: Optional[str] = None) -> np.ndarray:
    """Table 7 / H5: common positives (also in polypgen_clean_test) + negatives of one source."""

    pos = (zeroshot.labels == 1) & np.isin(zeroshot.col("frame_id"), list(clean_ids))
    neg = (zeroshot.labels == 0) & (zeroshot.col("negative_source") == source)
    keep = pos | neg
    if exclude_center is not None:
        keep &= zeroshot.col("center_id") != exclude_center
    return np.flatnonzero(keep)


def build_views(c: Campaign) -> Dict[ViewKey, View]:
    views: Dict[ViewKey, View] = {}
    idx = c.index()
    for run in c.runs:
        k, s, g = run.k, run.seed, run.regime
        test = run.frames["test"]
        if run.experiment in ("sun_full", "e1"):
            key = (run.experiment, run.variant, g, k, s, "all")
            views[key] = View(key, test, run.tau["primary"], np.arange(len(test)), "case", ("sun", k), per_polyp=True)
        elif run.experiment == "e2":
            for stratum, rows in _strata_rows(test).items():
                key = ("e2", None, g, k, s, stratum)
                views[key] = View(key, test, run.tau["primary"], rows, "case_morph", ("sun", k), per_polyp=True)
        elif run.experiment == "e3":
            key = ("e3", None, g, k, s, "all")
            views[key] = View(key, test, run.tau["primary"], np.arange(len(test)), "center", "polypgen")
        elif run.experiment == "e4":
            for arm in ("A", "B", "C", "D"):
                frames = run.frames["zeroshot"] if arm in ("A", "B") else test
                key = ("e4", run.variant, g, k, s, arm)
                views[key] = View(key, frames, run.tau[arm], np.arange(len(frames)), "e4", "polypgen")
            if run.variant == NEGSOURCE_BUDGET:
                parent_e3 = idx.get(("e3", None, g, k, s))
                if parent_e3 is None:
                    continue
                clean_ids = set(parent_e3.frames["test"].col("frame_id").tolist())
                for source in ("isolated", "sequence"):
                    rows = negsource_rows(run.frames["zeroshot"], clean_ids, source)
                    key = ("neg", NEGSOURCE_BUDGET, g, k, s, source)
                    views[key] = View(key, run.frames["zeroshot"], run.tau["A"], rows, "e4", "polypgen")
    return views


# --------------------------------------------------------------------------- contrasts
Term = Tuple[ViewKey, str, float]


@dataclass
class Contrast:
    id: str
    family: str
    meta: Dict[str, Any]
    terms: Dict[RunId, List[Term]]

    def views(self) -> List[ViewKey]:
        return sorted({t[0] for terms in self.terms.values() for t in terms}, key=repr)

    def per_run(self, metric_of: Callable[[ViewKey, str], float]) -> Dict[RunId, float]:
        out = {}
        for rid, terms in self.terms.items():
            out[rid] = float(sum(w * metric_of(v, m) for v, m, w in terms))
        return out


def complete_terms(
    views: Dict[ViewKey, View], runs: Sequence[RunId], make: Callable[[int, int], Optional[List[Term]]]
) -> Dict[RunId, List[Term]]:
    """Terms for every run whose views all exist (missing runs are left out of the pairing)."""

    out: Dict[RunId, List[Term]] = {}
    for k, s in runs:
        terms = make(k, s)
        if terms and all(v in views for v, _, _ in terms):
            out[(k, s)] = terms
    return out


# --------------------------------------------------------------------------- bootstrap
class Space:
    """Units per stratum of one group, from the union of the views of that group."""

    def __init__(self, views: Sequence[View]):
        by: Dict[str, set] = {}
        for v in views:
            for stratum, name in v.units:
                by.setdefault(stratum, set()).add(name)
        self.strata = {s: sorted(n) for s, n in sorted(by.items())}
        self.flat: List[Tuple[str, str]] = [(s, n) for s, names in self.strata.items() for n in names]
        self.offset: Dict[str, int] = {}
        pos = 0
        for s, names in self.strata.items():
            self.offset[s] = pos
            pos += len(names)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for s, names in self.strata.items():
            parts.append(self.offset[s] + rng.integers(0, len(names), size=len(names)))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def full(self) -> np.ndarray:
        return np.arange(len(self.flat))

    def counts(self) -> Dict[str, int]:
        return {s: len(n) for s, n in self.strata.items()}


_STATE: Dict[str, object] = {}


def _rows_for(view: View, lookup: List[Optional[np.ndarray]], sample: np.ndarray):
    parts, copies = [], []
    for j, u in enumerate(sample.tolist()):
        r = lookup[u]
        if r is not None:
            parts.append(r)
            copies.append(np.full(r.size, j, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(parts), np.concatenate(copies)


def _replicate_values(rep_indices: Sequence[int]) -> np.ndarray:
    views: Dict[ViewKey, View] = _STATE["views"]
    lookups = _STATE["lookups"]
    contrasts: List[Contrast] = _STATE["contrasts"]
    samples = _STATE["samples"]
    out = np.full((len(rep_indices), len(contrasts)), np.nan)
    for i, b in enumerate(rep_indices):
        memo: Dict[ViewKey, Dict[str, float]] = {}

        def metric_of(vk: ViewKey, metric: str) -> float:
            if vk not in memo:
                v = views[vk]
                rows, copies = _rows_for(v, lookups[vk], samples[b][v.group])
                memo[vk] = v.metrics(rows, copies)
            return memo[vk][metric]

        for j, c in enumerate(contrasts):
            d = list(c.per_run(metric_of).values())
            if d and not any(math.isnan(x) for x in d):
                out[i, j] = float(np.mean(d))
    return out


def bootstrap(
    views: Dict[ViewKey, View],
    contrasts: List[Contrast],
    B: int = B_BOOT,
    seed: int = BOOT_SEED,
    workers: int = 1,
) -> Tuple[np.ndarray, Dict[Hashable, Dict[str, int]]]:
    """Replicate estimates ``[B, n_contrasts]`` (NaN when undefined) and the unit counts per group."""

    needed = sorted({vk for c in contrasts for vk in c.views()}, key=repr)
    by_group: Dict[Hashable, List[View]] = {}
    for vk in needed:
        by_group.setdefault(views[vk].group, []).append(views[vk])
    spaces = {g: Space(vs) for g, vs in by_group.items()}
    order = sorted(spaces, key=repr)
    index_of = {g: {u: i for i, u in enumerate(spaces[g].flat)} for g in order}
    lookups: Dict[ViewKey, List[Optional[np.ndarray]]] = {}
    for vk in needed:
        v = views[vk]
        lut: List[Optional[np.ndarray]] = [None] * len(spaces[v.group].flat)
        for u, rows in v.units.items():
            lut[index_of[v.group][u]] = rows
        lookups[vk] = lut
    rng = np.random.default_rng(seed)
    samples = [{g: spaces[g].draw(rng) for g in order} for _ in range(B)]
    _STATE.update(views=views, lookups=lookups, contrasts=contrasts, samples=samples)
    chunks = [list(range(i, min(B, i + 25))) for i in range(0, B, 25)]
    if workers > 1 and B > 25:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_replicate_values, chunks)
    else:
        parts = [_replicate_values(ch) for ch in chunks]
    _STATE.clear()
    return np.vstack(parts) if parts else np.empty((0, len(contrasts))), {g: spaces[g].counts() for g in order}


def ci95(values: np.ndarray) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def aulc_weights(xs: Sequence[float]) -> List[float]:
    """Weights w with Σ w_i y_i = trapezoid(y, x) / (x_max − x_min) (SAP section 2)."""

    x = np.asarray(xs, dtype=float)
    width = x[-1] - x[0]
    w = np.zeros(x.size)
    for i in range(x.size - 1):
        h = (x[i + 1] - x[i]) / 2.0
        w[i] += h
        w[i + 1] += h
    return (w / width).tolist()
