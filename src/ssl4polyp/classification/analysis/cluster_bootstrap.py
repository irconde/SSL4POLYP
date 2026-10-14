"""Cluster bootstrap for paired comparisons (promoted from ``analysis_v2/bootstrap_v2.py``).

SAP v3, section 5:

* ``B = 2000`` replicates, seed ``12345``.
* One sample of units per replicate, shared by every run of the same *group*.  A group is a
  common test set: one group for E3/E4 (PolypGen test sets are fixed), one group per SUN
  partition for E1/E2.  Every replicate draws one sample per group from the same generator.
* Stratified: units are drawn separately inside each stratum (class; for E2 also the
  morphology stratum), each with replacement and as many draws as units.
* Point estimate = ``derive`` evaluated on the full data (normally the mean over runs of the
  per-run differences); interval = 95 % percentile of the replicate values.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Hashable, List, Sequence, Tuple

import numpy as np

from ssl4polyp.classification.metrics.binary import metrics_at

from .frame_outputs import FrameOutputs, Unit, UnitFn

__all__ = [
    "B_DEFAULT",
    "SEED_DEFAULT",
    "Space",
    "aulc",
    "ci95",
    "gather",
    "paired_mean",
    "run_bootstrap",
    "unit_rows",
]

B_DEFAULT = 2000
SEED_DEFAULT = 12345

Getter = Callable[[], Dict[str, float]]


def unit_rows(outputs: FrameOutputs, unit_fn: UnitFn) -> Dict[Unit, np.ndarray]:
    groups: Dict[Unit, List[int]] = defaultdict(list)
    for i in range(len(outputs)):
        unit = unit_fn(outputs, i)
        if unit is not None:
            groups[unit].append(i)
    return {u: np.asarray(rows, dtype=int) for u, rows in groups.items()}


def gather(units: Dict[Unit, np.ndarray], sample: Sequence[Unit]) -> np.ndarray:
    parts = [units[u] for u in sample if u in units]
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


class Space:
    """Universe of units per stratum drawn from the union of the runs of one group."""

    def __init__(self, all_units: Sequence[Dict[Unit, np.ndarray]]):
        by_stratum: Dict[str, set] = defaultdict(set)
        for units in all_units:
            for stratum, name in units:
                by_stratum[stratum].add(name)
        self.strata = {s: sorted(names) for s, names in sorted(by_stratum.items())}

    def draw(self, rng: np.random.Generator) -> List[Unit]:
        sample: List[Unit] = []
        for stratum, names in self.strata.items():
            picks = rng.integers(0, len(names), size=len(names))
            sample.extend((stratum, names[int(i)]) for i in picks)
        return sample

    def full(self) -> List[Unit]:
        return [(s, n) for s, names in self.strata.items() for n in names]

    def counts(self) -> Dict[str, int]:
        return {s: len(n) for s, n in self.strata.items()}


def run_bootstrap(
    runs: Dict[Hashable, Tuple[FrameOutputs, float]],
    units: Dict[Hashable, Dict[Unit, np.ndarray]],
    derive: Callable[[Dict[Hashable, Getter]], Dict[Hashable, float]],
    group_of: Callable[[Hashable], Hashable] = lambda key: None,
    B: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> Tuple[Dict[Hashable, float], Dict[Hashable, np.ndarray], Dict[Hashable, Space]]:
    """Bootstrap ``derive`` over runs ``{key: (outputs, tau)}``.

    ``derive`` receives ``{run_key: lambda -> metrics dict}`` evaluated on the current
    replicate and returns ``{name: value}``.  ``group_of(run_key)`` names the common test set
    of a run (e.g. the SUN partition); runs of the same group share each replicate's sample.
    """

    by_group: Dict[Hashable, List[Hashable]] = defaultdict(list)
    for key in runs:
        by_group[group_of(key)].append(key)
    spaces = {g: Space([units[k] for k in keys]) for g, keys in by_group.items()}
    group_order = sorted(spaces, key=repr)

    def evaluate(samples: Dict[Hashable, Sequence[Unit]]) -> Dict[Hashable, float]:
        memo: Dict[Hashable, Dict[str, float]] = {}

        def getter(key: Hashable) -> Getter:
            def _get() -> Dict[str, float]:
                if key not in memo:
                    outputs, tau = runs[key]
                    rows = gather(units[key], samples[group_of(key)])
                    memo[key] = metrics_at(outputs.probs, outputs.labels, tau, rows)
                return memo[key]

            return _get

        return derive({key: getter(key) for key in runs})

    point = evaluate({g: spaces[g].full() for g in group_order})
    rng = np.random.default_rng(seed)
    reps: Dict[Hashable, List[float]] = {name: [] for name in point}
    for _ in range(B):
        values = evaluate({g: spaces[g].draw(rng) for g in group_order})
        for name, value in values.items():
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                reps.setdefault(name, []).append(float(value))
    return point, {k: np.asarray(v, dtype=float) for k, v in reps.items()}, spaces


def ci95(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_mean(
    getters: Dict[Hashable, Getter],
    metric: str,
    pairs: Sequence[Tuple[Hashable, Hashable]],
) -> float:
    """Mean over ``(treat_key, base_key)`` pairs of ``treat - base`` for ``metric``."""

    diffs = [getters[a]()[metric] - getters[b]()[metric] for a, b in pairs]
    if not diffs or any(math.isnan(d) for d in diffs):
        return float("nan")
    return float(np.mean(diffs))


def aulc(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Trapezoidal area over a linear budget axis, normalised by the axis width (SAP 2)."""

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    width = x[-1] - x[0]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y, x) / width)
