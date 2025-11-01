from __future__ import annotations

import json
import copy
import csv
import math
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("numpy")

from ssl4polyp.classification.analysis.common_loader import CommonFrame
from ssl4polyp.classification.analysis.exp5a_report import (  # type: ignore[import]
    EXPECTED_SEEDS,
    PAIRWISE_METRICS,
    PRIMARY_METRICS,
    _build_cluster_set,
    _frames_to_eval,
    discover_runs,
    load_run,
    summarize_runs,
    write_domain_shift_csv,
    write_pairwise_csv,
    write_seed_metrics_csv,
)


def _write_outputs(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame_id",
        "prob",
        "label",
        "center_id",
        "sequence_id",
        "origin",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    variant = path.with_name(path.name.replace("_test_outputs", ".metrics_test_outputs"))
    for output_path in {path, variant}:
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(fieldnames) + "\n")
            for row in rows:
                values = [str(row.get(field, "")) for field in fieldnames]
                handle.write(",".join(values) + "\n")


def _write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _data_block() -> dict[str, dict[str, str]]:
    return {
        "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
        "val": {"path": "sun_full/val.csv", "sha256": "val-digest"},
        "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
    }


def _polyp_rows() -> list[dict[str, object]]:
    return [
        {"frame_id": "f1", "prob": 0.9, "label": 1, "center_id": "A", "sequence_id": "seq1", "origin": "A"},
        {"frame_id": "f2", "prob": 0.2, "label": 0, "center_id": "A", "sequence_id": "seq1", "origin": "A"},
        {"frame_id": "f3", "prob": 0.8, "label": 1, "center_id": "B", "sequence_id": "seq2", "origin": "B"},
        {"frame_id": "f4", "prob": 0.1, "label": 0, "center_id": "B", "sequence_id": "seq2", "origin": "B"},
    ]


def _sun_rows() -> list[dict[str, object]]:
    return [
        {"frame_id": "s1", "prob": 0.7, "label": 1, "center_id": "A", "sequence_id": "seq1", "origin": "A"},
        {"frame_id": "s2", "prob": 0.3, "label": 0, "center_id": "A", "sequence_id": "seq1", "origin": "A"},
        {"frame_id": "s3", "prob": 0.6, "label": 1, "center_id": "B", "sequence_id": "seq2", "origin": "B"},
        {"frame_id": "s4", "prob": 0.4, "label": 0, "center_id": "B", "sequence_id": "seq2", "origin": "B"},
    ]


def _sun_payload(tau: float, offset: float) -> dict[str, object]:
    return {
        "test_primary": {
            "tau": tau,
            "auprc": 0.6 + offset,
            "auroc": 0.7 + offset,
            "f1": 0.55 + offset,
            "recall": 0.6 + offset,
            "precision": 0.62 + offset,
            "balanced_accuracy": 0.66 + offset,
            "mcc": 0.4 + offset,
            "loss": 0.3 - offset,
            "prevalence": 0.5,
        }
    }


def _build_summary(
    tmp_path: Path, *, bootstrap: int = 10, include_sha: bool = True
) -> dict[str, object]:
    root = tmp_path / "results"
    sun_root = root / "sun_parent"
    for seed in EXPECTED_SEEDS:
        colon_dir = root / "ssl_colon" / f"seed{seed}"
        sup_dir = root / "sup_imnet" / f"seed{seed}"
        ssl_dir = root / "ssl_imnet" / f"seed{seed}"
        colon_dir.mkdir(parents=True, exist_ok=True)
        sup_dir.mkdir(parents=True, exist_ok=True)
        ssl_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(colon_dir / f"ssl_colon__seed{seed}_last_test_outputs.csv", _polyp_rows())
        _write_outputs(sup_dir / f"sup_imnet__seed{seed}_last_test_outputs.csv", _polyp_rows())
        _write_outputs(ssl_dir / f"ssl_imnet__seed{seed}_last_test_outputs.csv", _polyp_rows())
        sun_outputs_path = sun_root / f"seed{seed}_test_outputs.csv"
        _write_outputs(sun_outputs_path, _sun_rows())
        _write_outputs((colon_dir.parent / "sun_parent" / f"seed{seed}_test_outputs.csv"), _sun_rows())
        _write_outputs((sup_dir.parent / "sun_parent" / f"seed{seed}_test_outputs.csv"), _sun_rows())
        _write_outputs((ssl_dir.parent / "sun_parent" / f"seed{seed}_test_outputs.csv"), _sun_rows())
    parent_payload = _sun_payload(tau=0.45, offset=-0.08)
    val_path = _data_block()["val"]["path"]
    colon_provenance = {"model": "ssl_colon"}
    if include_sha:
        colon_provenance["test_csv_sha256"] = "deadbeef"
    colon_metrics_template = {
        "seed": 0,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.45,
            "auprc": 0.8,
            "auroc": 0.9,
            "f1": 0.75,
            "recall": 0.78,
            "precision": 0.82,
            "balanced_accuracy": 0.85,
            "mcc": 0.67,
            "loss": 0.15,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.45,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "run": {"exp": "exp5a"},
        "provenance": colon_provenance,
    }
    sup_provenance = {"model": "sup_imnet"}
    if include_sha:
        sup_provenance["test_csv_sha256"] = "deadbeef"
    sup_metrics_template = {
        "seed": 0,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.45,
            "auprc": 0.7,
            "auroc": 0.82,
            "f1": 0.68,
            "recall": 0.7,
            "precision": 0.74,
            "balanced_accuracy": 0.78,
            "mcc": 0.6,
            "loss": 0.22,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.45,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "run": {"exp": "exp5a"},
        "provenance": sup_provenance,
    }
    ssl_provenance = {"model": "ssl_imnet"}
    if include_sha:
        ssl_provenance["test_csv_sha256"] = "deadbeef"
    ssl_metrics_template = {
        "seed": 0,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.45,
            "auprc": 0.72,
            "auroc": 0.85,
            "f1": 0.7,
            "recall": 0.72,
            "precision": 0.76,
            "balanced_accuracy": 0.8,
            "mcc": 0.62,
            "loss": 0.21,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.45,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "run": {"exp": "exp5a"},
        "provenance": ssl_provenance,
    }
    for seed in EXPECTED_SEEDS:
        colon_metrics_seed = copy.deepcopy(colon_metrics_template)
        colon_metrics_seed["seed"] = seed
        colon_metrics_seed["provenance"]["parent_run"] = {
            "metrics": {"path": f"../sun_parent/seed{seed}.metrics.json", "payload": parent_payload},
            "outputs": {"path": f"../sun_parent/seed{seed}_test_outputs.csv"},
        }
        sup_metrics_seed = copy.deepcopy(sup_metrics_template)
        sup_metrics_seed["seed"] = seed
        sup_metrics_seed["provenance"]["parent_run"] = {
            "metrics": {"path": f"../sun_parent/seed{seed}.metrics.json", "payload": parent_payload},
            "outputs": {"path": f"../sun_parent/seed{seed}_test_outputs.csv"},
        }
        ssl_metrics_seed = copy.deepcopy(ssl_metrics_template)
        ssl_metrics_seed["seed"] = seed
        ssl_metrics_seed["provenance"]["parent_run"] = {
            "metrics": {"path": f"../sun_parent/seed{seed}.metrics.json", "payload": parent_payload},
            "outputs": {"path": f"../sun_parent/seed{seed}_test_outputs.csv"},
        }
        colon_dir = root / "ssl_colon" / f"seed{seed}"
        sup_dir = root / "sup_imnet" / f"seed{seed}"
        ssl_dir = root / "ssl_imnet" / f"seed{seed}"
        _write_metrics(colon_dir / f"ssl_colon__seed{seed}_last.metrics.json", colon_metrics_seed)
        _write_metrics(sup_dir / f"sup_imnet__seed{seed}_last.metrics.json", sup_metrics_seed)
        _write_metrics(ssl_dir / f"ssl_imnet__seed{seed}_last.metrics.json", ssl_metrics_seed)
    runs = discover_runs(root)
    return summarize_runs(runs, bootstrap=bootstrap, rng_seed=123)


def test_load_run_consumes_parent_metadata(tmp_path: Path) -> None:
    runs_root = tmp_path / "results"
    colon_dir = runs_root / "ssl_colon" / "seed1"
    sun_dir = runs_root / "sun_parent"
    outputs_path = colon_dir / "ssl_colon__seed1_last_test_outputs.csv"
    sun_outputs_path = sun_dir / "seed1_test_outputs.csv"
    _write_outputs(outputs_path, _polyp_rows())
    _write_outputs(sun_outputs_path, _sun_rows())
    alt_sun_outputs_path = colon_dir.parent / "sun_parent" / "seed1_test_outputs.csv"
    _write_outputs(alt_sun_outputs_path, _sun_rows())
    val_path = _data_block()["val"]["path"]
    metrics_payload = {
        "seed": 1,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.4,
            "auprc": 0.75,
            "auroc": 0.88,
            "f1": 0.7,
            "recall": 0.72,
            "precision": 0.78,
            "balanced_accuracy": 0.83,
            "mcc": 0.61,
            "loss": 0.2,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.4,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "domain_shift_delta": {
            "metrics": {
                "auprc": 0.1,
                "f1": 0.05,
            }
        },
        "run": {"exp": "exp5a"},
        "provenance": {
            "model": "ssl_colon",
            "test_csv_sha256": "deadbeef",
            "parent_run": {
                "metrics": {
                    "path": "../sun_parent/seed1.metrics.json",
                    "payload": _sun_payload(tau=0.4, offset=-0.05),
                },
                "outputs": {
                    "path": "../sun_parent/seed1_test_outputs.csv",
                },
            },
        },
    }
    metrics_path = colon_dir / "ssl_colon__seed1_last.metrics.json"
    _write_metrics(metrics_path, metrics_payload)
    run = load_run(metrics_path)
    assert run.model == "ssl_colon"
    assert run.seed == 1
    assert math.isclose(run.metrics["auprc"], 0.75)
    assert math.isclose(run.delta["auprc"], 0.1)
    assert "precision" in run.delta
    assert "balanced_accuracy" in run.delta
    assert "mcc" in run.delta
    assert "loss" in run.delta
    assert math.isclose(run.delta["precision"], 0.21)
    assert math.isclose(run.delta["balanced_accuracy"], 0.22)
    assert math.isclose(run.delta["mcc"], 0.26)
    assert math.isclose(run.delta["loss"], -0.15)
    assert run.sun_tau == 0.4
    assert run.sun_frames is not None
    assert len(run.frames) == 4


def test_load_run_accepts_legacy_outputs_sha(tmp_path: Path) -> None:
    runs_root = tmp_path / "results"
    colon_dir = runs_root / "ssl_colon" / "seed1"
    sun_dir = runs_root / "sun_parent"
    outputs_path = colon_dir / "ssl_colon__seed1_last_test_outputs.csv"
    sun_outputs_path = sun_dir / "seed1_test_outputs.csv"
    _write_outputs(outputs_path, _polyp_rows())
    _write_outputs(sun_outputs_path, _sun_rows())
    alt_sun_outputs_path = colon_dir.parent / "sun_parent" / "seed1_test_outputs.csv"
    _write_outputs(alt_sun_outputs_path, _sun_rows())
    val_path = _data_block()["val"]["path"]
    metrics_payload = {
        "seed": 1,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.4,
            "auprc": 0.75,
            "auroc": 0.88,
            "f1": 0.7,
            "recall": 0.72,
            "precision": 0.78,
            "balanced_accuracy": 0.83,
            "mcc": 0.61,
            "loss": 0.2,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.4,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "run": {"exp": "exp5a"},
        "provenance": {
            "model": "ssl_colon",
            "test_outputs_csv_sha256": "cafebabe",
            "parent_run": {
                "metrics": {
                    "path": "../sun_parent/seed1.metrics.json",
                    "payload": _sun_payload(tau=0.4, offset=-0.05),
                },
                "outputs": {
                    "path": "../sun_parent/seed1_test_outputs.csv",
                },
            },
        },
    }
    metrics_path = colon_dir / "ssl_colon__seed1_last.metrics.json"
    _write_metrics(metrics_path, metrics_payload)
    run = load_run(metrics_path)
    assert run.outputs_sha256 == "cafebabe"


def test_load_run_allows_missing_outputs_sha(tmp_path: Path) -> None:
    runs_root = tmp_path / "results"
    colon_dir = runs_root / "ssl_colon" / "seed1"
    sun_dir = runs_root / "sun_parent"
    outputs_path = colon_dir / "ssl_colon__seed1_last_test_outputs.csv"
    sun_outputs_path = sun_dir / "seed1_test_outputs.csv"
    _write_outputs(outputs_path, _polyp_rows())
    _write_outputs(sun_outputs_path, _sun_rows())
    alt_sun_outputs_path = colon_dir.parent / "sun_parent" / "seed1_test_outputs.csv"
    _write_outputs(alt_sun_outputs_path, _sun_rows())
    val_path = _data_block()["val"]["path"]
    metrics_payload = {
        "seed": 1,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.4,
            "auprc": 0.75,
            "auroc": 0.88,
            "f1": 0.7,
            "recall": 0.72,
            "precision": 0.78,
            "balanced_accuracy": 0.83,
            "mcc": 0.61,
            "loss": 0.2,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.4,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "run": {"exp": "exp5a"},
        "provenance": {
            "model": "ssl_colon",
            "parent_run": {
                "metrics": {
                    "path": "../sun_parent/seed1.metrics.json",
                    "payload": _sun_payload(tau=0.4, offset=-0.05),
                },
                "outputs": {
                    "path": "../sun_parent/seed1_test_outputs.csv",
                },
            },
        },
    }
    metrics_path = colon_dir / "ssl_colon__seed1_last.metrics.json"
    _write_metrics(metrics_path, metrics_payload)
    run = load_run(metrics_path)
    assert run.outputs_sha256 is None


def test_load_run_raises_when_tau_mismatch(tmp_path: Path) -> None:
    runs_root = tmp_path / "results"
    colon_dir = runs_root / "ssl_colon" / "seed1"
    sun_dir = runs_root / "sun_parent"
    outputs_path = colon_dir / "ssl_colon__seed1_last_test_outputs.csv"
    sun_outputs_path = sun_dir / "seed1_test_outputs.csv"
    _write_outputs(outputs_path, _polyp_rows())
    _write_outputs(sun_outputs_path, _sun_rows())
    val_path = _data_block()["val"]["path"]
    metrics_payload = {
        "seed": 1,
        "data": _data_block(),
        "test_primary": {
            "tau": 0.5,
            "auprc": 0.8,
            "auroc": 0.9,
            "f1": 0.75,
            "recall": 0.78,
            "precision": 0.82,
            "balanced_accuracy": 0.85,
            "mcc": 0.67,
            "loss": 0.15,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.5,
                "split": val_path,
                "source_split": "sun_full/val",
                "source_checkpoint": "sun_parent/seed1.pth",
            },
        },
        "val": {"path": val_path},
        "run": {"exp": "exp5a"},
        "provenance": {
            "model": "ssl_colon",
            "test_csv_sha256": "deadbeef",
            "parent_run": {
                "metrics": {
                    "path": "../sun_parent/seed1.metrics.json",
                    "payload": _sun_payload(tau=0.4, offset=-0.05),
                },
                "outputs": {
                    "path": "../sun_parent/seed1_test_outputs.csv",
                },
            },
        },
    }
    metrics_path = colon_dir / "ssl_colon__seed1_last.metrics.json"
    _write_metrics(metrics_path, metrics_payload)
    with pytest.raises(ValueError, match="tau"):
        load_run(metrics_path)


def test_summarize_runs_builds_expected_blocks(tmp_path: Path) -> None:
    summary = _build_summary(tmp_path, bootstrap=10)
    assert "models" in summary
    assert "ssl_colon" in summary["models"]
    assert summary["validated_seeds"] == list(EXPECTED_SEEDS)
    composition = summary.get("composition")
    assert isinstance(composition, dict)
    assert composition["n_pos"] == 2
    assert composition["n_neg"] == 2
    assert composition["total"] == 4
    assert math.isclose(composition["prevalence"], 0.5)
    assert composition["sha256"] == "deadbeef"
    per_center = composition.get("per_center")
    assert isinstance(per_center, dict)
    assert per_center["A"]["n_pos"] == 1
    assert per_center["A"]["n_neg"] == 1
    colon_entry = summary["models"]["ssl_colon"]
    assert "performance" in colon_entry
    assert "seeds" in colon_entry
    metadata = summary.get("metadata")
    assert isinstance(metadata, dict)
    delta_metadata = metadata.get("delta_metrics")
    assert isinstance(delta_metadata, list)
    for metric in PRIMARY_METRICS:
        assert metric in delta_metadata
    pairwise_metadata = metadata.get("pairwise_metrics")
    assert isinstance(pairwise_metadata, list)
    for metric in PAIRWISE_METRICS:
        assert metric in pairwise_metadata
    seed_payload = colon_entry["seeds"][EXPECTED_SEEDS[0]]
    delta_block = seed_payload.get("delta")
    assert isinstance(delta_block, dict)
    for metric in ("precision", "balanced_accuracy", "mcc", "loss"):
        assert metric in delta_block
    domain_shift_summary = colon_entry.get("domain_shift")
    assert isinstance(domain_shift_summary, dict)
    for metric in ("precision", "balanced_accuracy", "mcc", "loss"):
        assert metric in domain_shift_summary
    domain_shift_ci = colon_entry.get("domain_shift_ci")
    assert isinstance(domain_shift_ci, dict)
    for metric in ("precision", "balanced_accuracy", "mcc", "loss"):
        assert metric in domain_shift_ci
    pairwise = summary.get("pairwise")
    assert isinstance(pairwise, dict)
    for metric in PAIRWISE_METRICS:
        assert metric in pairwise
        for baseline in ("sup_imnet", "ssl_imnet"):
            assert baseline in pairwise[metric]


def test_summarize_runs_handles_missing_outputs_sha(tmp_path: Path) -> None:
    summary = _build_summary(tmp_path, bootstrap=8, include_sha=False)
    composition = summary.get("composition")
    assert isinstance(composition, dict)
    assert "sha256" not in composition


def test_extended_delta_metrics_propagate_to_csv(tmp_path: Path) -> None:
    summary = _build_summary(tmp_path, bootstrap=8)
    domain_shift_path = tmp_path / "domain_shift.csv"
    seed_metrics_path = tmp_path / "seed_metrics.csv"
    pairwise_path = tmp_path / "pairwise.csv"
    write_domain_shift_csv(summary, domain_shift_path)
    write_seed_metrics_csv(summary, seed_metrics_path)
    write_pairwise_csv(summary, pairwise_path)

    with domain_shift_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        domain_metrics = {row["metric"] for row in reader}
    for metric in ("precision", "balanced_accuracy", "mcc", "loss"):
        assert metric in domain_metrics

    with seed_metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    domain_rows = [row for row in reader if row.get("kind") == "domain_shift"]
    domain_seed_metrics = {row["metric"] for row in domain_rows}
    for metric in ("precision", "balanced_accuracy", "mcc", "loss"):
        assert metric in domain_seed_metrics

    with pairwise_path.open(newline="", encoding="utf-8") as handle:
        pairwise_rows = list(csv.DictReader(handle))
    pairwise_metrics = {row.get("metric") for row in pairwise_rows}
    for metric in PAIRWISE_METRICS:
        assert metric in pairwise_metrics
    metric_baseline_pairs = {
        (row.get("metric"), row.get("baseline"))
        for row in pairwise_rows
        if row.get("baseline") is not None
    }
    for metric in PAIRWISE_METRICS:
        for baseline in ("sup_imnet", "ssl_imnet"):
            assert (metric, baseline) in metric_baseline_pairs


def test_sun_bootstrap_clusters_group_positive_frames_by_case() -> None:
    frames = [
        CommonFrame(
            frame_id="f_pos_caseA_1",
            case_id="caseA",
            prob=0.9,
            label=1,
            pred=1,
            row=MappingProxyType({"sequence_id": "caseA"}),
        ),
        CommonFrame(
            frame_id="f_pos_caseA_2",
            case_id="caseA",
            prob=0.8,
            label=1,
            pred=1,
            row=MappingProxyType({"sequence_id": "caseA"}),
        ),
        CommonFrame(
            frame_id="f_pos_caseB",
            case_id="caseB",
            prob=0.85,
            label=1,
            pred=1,
            row=MappingProxyType({"sequence_id": "caseB"}),
        ),
        CommonFrame(
            frame_id="f_neg_caseA",
            case_id="caseA",
            prob=0.1,
            label=0,
            pred=0,
            row=MappingProxyType({"sequence_id": "caseA"}),
        ),
    ]
    eval_frames = _frames_to_eval(frames)
    clusters = _build_cluster_set(eval_frames, domain="sun")
    positive_clusters = [set(cluster) for cluster in clusters.positives]
    assert any({"f_pos_caseA_1", "f_pos_caseA_2"} == cluster for cluster in positive_clusters)


def test_polypgen_clusters_group_positive_frames_by_center() -> None:
    frames = [
        CommonFrame(
            frame_id="f_pos_centerA_1",
            case_id="caseA",
            prob=0.95,
            label=1,
            pred=1,
            row=MappingProxyType({"center_id": "centerA", "sequence_id": "seq1", "origin": "centerA"}),
        ),
        CommonFrame(
            frame_id="f_pos_centerA_2",
            case_id="caseB",
            prob=0.9,
            label=1,
            pred=1,
            row=MappingProxyType({"center_id": "centerA", "sequence_id": "seq2", "origin": "centerA"}),
        ),
        CommonFrame(
            frame_id="f_pos_centerB",
            case_id="caseC",
            prob=0.85,
            label=1,
            pred=1,
            row=MappingProxyType({"center_id": "centerB", "sequence_id": "seq3", "origin": "centerB"}),
        ),
        CommonFrame(
            frame_id="f_neg_centerA",
            case_id="caseA",
            prob=0.2,
            label=0,
            pred=0,
            row=MappingProxyType({"center_id": "centerA", "sequence_id": "seq1", "origin": "centerA"}),
        ),
    ]
    eval_frames = _frames_to_eval(frames)
    clusters = _build_cluster_set(eval_frames, domain="polypgen")
    positive_clusters = [set(cluster) for cluster in clusters.positives]
    assert any({"f_pos_centerA_1", "f_pos_centerA_2"} == cluster for cluster in positive_clusters)


def test_polypgen_clusters_group_positive_frames_by_sequence_when_center_missing() -> None:
    frames = [
        CommonFrame(
            frame_id="f_pos_seq_shared_1",
            case_id="caseX",
            prob=0.92,
            label=1,
            pred=1,
            row=MappingProxyType({"center_id": "", "sequence_id": "seq_shared", "origin": ""}),
        ),
        CommonFrame(
            frame_id="f_pos_seq_shared_2",
            case_id="caseY",
            prob=0.91,
            label=1,
            pred=1,
            row=MappingProxyType({"center_id": None, "sequence_id": "seq_shared", "origin": None}),
        ),
        CommonFrame(
            frame_id="f_pos_seq_other",
            case_id="caseZ",
            prob=0.88,
            label=1,
            pred=1,
            row=MappingProxyType({"center_id": "", "sequence_id": "seq_other", "origin": ""}),
        ),
    ]
    eval_frames = _frames_to_eval(frames)
    clusters = _build_cluster_set(eval_frames, domain="polypgen")
    positive_clusters = [set(cluster) for cluster in clusters.positives]
    assert any({"f_pos_seq_shared_1", "f_pos_seq_shared_2"} == cluster for cluster in positive_clusters)
