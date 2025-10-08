from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["BootstrapDeltaResult", "Bootstrapper"]


AggregateFn = Callable[[np.ndarray], float]


def _default_rng() -> np.random.Generator:
    return np.random.default_rng(1337)


@dataclass(frozen=True)
class BootstrapDeltaResult:
    """Summary of a bootstrapped delta distribution."""

    mean: float
    per_seed: Mapping[int, float]
    samples: Tuple[float, ...]
    ci_lower: Optional[float]
    ci_upper: Optional[float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "mean": float(self.mean),
            "per_seed": {int(seed): float(delta) for seed, delta in self.per_seed.items()},
            "samples": list(self.samples),
            "ci_lower": float(self.ci_lower) if self.ci_lower is not None else None,
            "ci_upper": float(self.ci_upper) if self.ci_upper is not None else None,
        }


class Bootstrapper:
    """Bootstrap paired or unpaired deltas with deterministic seeding."""

    def __init__(
        self,
        metrics: Mapping[Hashable, Mapping[int, Sequence[float]]],
        *,
        clusters: Optional[Mapping[Hashable, Mapping[int, Sequence[Hashable]]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._rng = rng or _default_rng()
        self._values: Dict[Hashable, Dict[int, np.ndarray]] = {}
        self._clusters: Dict[Hashable, Dict[int, Tuple[Hashable, ...]]] = {}
        self._cluster_members: Dict[Tuple[Hashable, int], Dict[Hashable, np.ndarray]] = {}
        for condition, seed_map in metrics.items():
            condition_values: Dict[int, np.ndarray] = {}
            condition_clusters: Dict[int, Tuple[Hashable, ...]] = {}
            for seed, values in seed_map.items():
                array = np.asarray(list(values), dtype=float)
                if array.size == 0:
                    raise ValueError(f"Seed {seed} for condition '{condition}' has no metric values")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"Non-finite metric values detected for seed {seed} condition '{condition}'")
                condition_values[int(seed)] = array
                cluster_source: Optional[Sequence[Hashable]] = None
                if clusters and condition in clusters and seed in clusters[condition]:
                    cluster_source = clusters[condition][seed]
                cluster_ids = self._normalise_clusters(array, cluster_source)
                condition_clusters[int(seed)] = cluster_ids
                self._cluster_members[(condition, int(seed))] = self._build_cluster_members(cluster_ids)
            self._values[condition] = condition_values
            self._clusters[condition] = condition_clusters
        self._seed_ids = self._resolve_seeds()

    def _resolve_seeds(self) -> Tuple[int, ...]:
        seeds = set()
        for seed_map in self._values.values():
            seeds.update(seed_map.keys())
        if not seeds:
            raise ValueError("Bootstrapper requires at least one seed")
        return tuple(sorted(int(seed) for seed in seeds))

    def _normalise_clusters(
        self, values: np.ndarray, clusters: Optional[Sequence[Hashable]]
    ) -> Tuple[Hashable, ...]:
        if clusters is None:
            return tuple(range(values.size))
        if len(clusters) != values.size:
            raise ValueError("Cluster assignments must match metric value count")
        return tuple(clusters)

    def _build_cluster_members(
        self, cluster_ids: Sequence[Hashable]
    ) -> Dict[Hashable, np.ndarray]:
        members: Dict[Hashable, list[int]] = {}
        for index, cluster in enumerate(cluster_ids):
            members.setdefault(cluster, []).append(index)
        return {cluster: np.array(indices, dtype=int) for cluster, indices in members.items()}

    def _get_condition_seed_values(
        self, condition: Hashable, seed: int
    ) -> Tuple[np.ndarray, Tuple[Hashable, ...], Dict[Hashable, np.ndarray]]:
        if condition not in self._values or seed not in self._values[condition]:
            raise KeyError(f"Missing metrics for condition '{condition}' seed {seed}")
        values = self._values[condition][seed]
        clusters = self._clusters[condition][seed]
        members = self._cluster_members[(condition, seed)]
        return values, clusters, members

    def _paired_cluster_ids(
        self, treatment: Hashable, baseline: Hashable, seed: int
    ) -> Tuple[Hashable, ...]:
        _, treatment_clusters, _ = self._get_condition_seed_values(treatment, seed)
        _, baseline_clusters, _ = self._get_condition_seed_values(baseline, seed)
        if treatment_clusters != baseline_clusters:
            raise ValueError(
                f"Cluster assignments for seed {seed} do not align between {treatment!r} and {baseline!r}"
            )
        return treatment_clusters

    def _resample_condition(
        self,
        condition: Hashable,
        seed: int,
        *,
        sampled_clusters: Optional[Sequence[Hashable]] = None,
        aggregate: AggregateFn,
    ) -> float:
        values, cluster_ids, members = self._get_condition_seed_values(condition, seed)
        if sampled_clusters is None:
            sampled_clusters = self._rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        gathered = []
        for cluster in sampled_clusters:
            cluster_indices = members.get(cluster)
            if cluster_indices is None:
                raise KeyError(
                    f"Unknown cluster '{cluster}' for condition '{condition}' seed {seed}"
                )
            gathered.append(values[cluster_indices])
        sample = np.concatenate(gathered) if gathered else values
        return float(aggregate(sample))

    def paired_delta(
        self,
        treatment: Hashable,
        baseline: Hashable,
        *,
        bootstrap: int = 2000,
        aggregate: AggregateFn = np.mean,
        seed_reduction: AggregateFn = np.mean,
        ci: float = 0.95,
    ) -> BootstrapDeltaResult:
        shared_clusters = {
            seed: self._paired_cluster_ids(treatment, baseline, seed) for seed in self._seed_ids
        }
        return self._bootstrap_delta(
            treatment,
            baseline,
            bootstrap=bootstrap,
            aggregate=aggregate,
            seed_reduction=seed_reduction,
            ci=ci,
            shared_clusters=shared_clusters,
        )

    def unpaired_delta(
        self,
        treatment: Hashable,
        baseline: Hashable,
        *,
        bootstrap: int = 2000,
        aggregate: AggregateFn = np.mean,
        seed_reduction: AggregateFn = np.mean,
        ci: float = 0.95,
    ) -> BootstrapDeltaResult:
        return self._bootstrap_delta(
            treatment,
            baseline,
            bootstrap=bootstrap,
            aggregate=aggregate,
            seed_reduction=seed_reduction,
            ci=ci,
            shared_clusters=None,
        )

    def _bootstrap_delta(
        self,
        treatment: Hashable,
        baseline: Hashable,
        *,
        bootstrap: int,
        aggregate: AggregateFn,
        seed_reduction: AggregateFn,
        ci: float,
        shared_clusters: Optional[Mapping[int, Sequence[Hashable]]],
    ) -> BootstrapDeltaResult:
        per_seed: Dict[int, float] = {}
        for seed in self._seed_ids:
            treatment_stat = self._resample_condition(
                treatment,
                seed,
                sampled_clusters=shared_clusters.get(seed) if shared_clusters else None,
                aggregate=aggregate,
            )
            baseline_stat = self._resample_condition(
                baseline,
                seed,
                sampled_clusters=shared_clusters.get(seed) if shared_clusters else None,
                aggregate=aggregate,
            )
            per_seed[seed] = float(treatment_stat - baseline_stat)
        samples: list[float] = []
        for _ in range(max(0, int(bootstrap))):
            seed_deltas = []
            for seed in self._seed_ids:
                clusters = None
                if shared_clusters:
                    shared = shared_clusters[seed]
                    clusters = self._rng.choice(shared, size=len(shared), replace=True)
                treatment_stat = self._resample_condition(
                    treatment,
                    seed,
                    sampled_clusters=clusters,
                    aggregate=aggregate,
                )
                baseline_stat = self._resample_condition(
                    baseline,
                    seed,
                    sampled_clusters=clusters,
                    aggregate=aggregate,
                )
                seed_deltas.append(treatment_stat - baseline_stat)
            samples.append(float(seed_reduction(np.asarray(seed_deltas, dtype=float))))
        samples_tuple = tuple(samples)
        mean_delta = float(seed_reduction(np.asarray(list(per_seed.values()), dtype=float)))
        ci_lower: Optional[float]
        ci_upper: Optional[float]
        if samples_tuple and bootstrap > 0:
            lower_pct = (1.0 - ci) / 2.0 * 100.0
            upper_pct = (1.0 + ci) / 2.0 * 100.0
            ci_lower = float(np.percentile(samples_tuple, lower_pct))
            ci_upper = float(np.percentile(samples_tuple, upper_pct))
        else:
            ci_lower = None
            ci_upper = None
        return BootstrapDeltaResult(
            mean=mean_delta,
            per_seed=per_seed,
            samples=samples_tuple,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
