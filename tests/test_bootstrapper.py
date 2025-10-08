from __future__ import annotations

import numpy as np

from ssl4polyp.classification.analysis.bootstrapper import Bootstrapper


def _manual_paired_bootstrap(metrics, clusters, bootstrap, rng_seed=1337):
    rng = np.random.default_rng(rng_seed)
    seeds = sorted({seed for condition in metrics.values() for seed in condition})
    cluster_map = {
        seed: tuple(clusters["treatment"][seed]) for seed in seeds
    }
    samples = []
    for _ in range(bootstrap):
        seed_deltas = []
        for seed in seeds:
            chosen = rng.choice(cluster_map[seed], size=len(cluster_map[seed]), replace=True)
            indices = [np.where(np.array(cluster_map[seed]) == c)[0] for c in chosen]
            flat = np.concatenate(indices)
            treat_values = np.asarray(metrics["treatment"][seed])[flat]
            base_values = np.asarray(metrics["baseline"][seed])[flat]
            seed_deltas.append(treat_values.mean() - base_values.mean())
        samples.append(float(np.mean(seed_deltas)))
    return tuple(samples)


def test_bootstrapper_paired_delta_reproducible() -> None:
    metrics = {
        "treatment": {13: [0.6, 0.7, 0.8], 29: [0.5, 0.55, 0.6]},
        "baseline": {13: [0.5, 0.62, 0.55], 29: [0.45, 0.5, 0.52]},
    }
    clusters = {key: {seed: ["a", "b", "c"] for seed in values} for key, values in metrics.items()}
    bootstrapper = Bootstrapper(metrics, clusters=clusters)
    result = bootstrapper.paired_delta("treatment", "baseline", bootstrap=8)
    expected_samples = _manual_paired_bootstrap(metrics, clusters, bootstrap=8)
    assert result.samples == expected_samples
    expected_mean = np.mean(
        [
            np.mean(metrics["treatment"][seed]) - np.mean(metrics["baseline"][seed])
            for seed in sorted(metrics["treatment"].keys())
        ]
    )
    assert np.isclose(result.mean, expected_mean)

    bootstrapper_again = Bootstrapper(metrics, clusters=clusters)
    repeat = bootstrapper_again.paired_delta("treatment", "baseline", bootstrap=8)
    assert repeat.samples == result.samples


def test_bootstrapper_unpaired_allows_mismatched_clusters() -> None:
    metrics = {
        "treatment": {13: [0.2, 0.4], 29: [0.5, 0.7]},
        "baseline": {13: [0.1, 0.3, 0.35], 29: [0.4, 0.6, 0.65]},
    }
    clusters = {
        "treatment": {13: ["x", "y"], 29: ["u", "v"]},
        "baseline": {13: ["x", "y", "z"], 29: ["u", "v", "w"]},
    }
    bootstrapper = Bootstrapper(metrics, clusters=clusters)
    result = bootstrapper.unpaired_delta("treatment", "baseline", bootstrap=5)
    assert len(result.samples) == 5
    assert result.ci_lower is not None and result.ci_upper is not None
