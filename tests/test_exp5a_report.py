from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ssl4polyp.classification.analysis.exp5a_report import (  # type: ignore[import]
    discover_runs,
    load_run,
    summarize_runs,
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
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            values = [str(row.get(field, "")) for field in fieldnames]
            handle.write(",".join(values) + "\n")


def _write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
        }
    }


def test_load_run_consumes_parent_metadata(tmp_path: Path) -> None:
    runs_root = tmp_path / "results"
    colon_dir = runs_root / "ssl_colon" / "seed1"
    sun_dir = runs_root / "sun_parent"
    outputs_path = colon_dir / "ssl_colon__seed1_last_test_outputs.csv"
    sun_outputs_path = sun_dir / "seed1_test_outputs.csv"
    _write_outputs(outputs_path, _polyp_rows())
    _write_outputs(sun_outputs_path, _sun_rows())
    metrics_payload = {
        "seed": 1,
        "test_primary": {
            "tau": 0.5,
            "auprc": 0.75,
            "auroc": 0.88,
            "f1": 0.7,
            "recall": 0.72,
        },
        "domain_shift_delta": {
            "metrics": {
                "auprc": 0.1,
                "f1": 0.05,
            }
        },
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
    assert run.model == "ssl_colon"
    assert run.seed == 1
    assert np.isclose(run.metrics["auprc"], 0.75)
    assert np.isclose(run.delta["auprc"], 0.1)
    assert run.sun_tau == 0.4
    assert run.sun_frames is not None
    assert len(run.frames) == 4


def test_summarize_runs_builds_expected_blocks(tmp_path: Path) -> None:
    root = tmp_path / "results"
    colon_dir = root / "ssl_colon" / "seed1"
    sup_dir = root / "sup_imnet" / "seed1"
    colon_dir.mkdir(parents=True, exist_ok=True)
    sup_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(colon_dir / "ssl_colon__seed1_last_test_outputs.csv", _polyp_rows())
    _write_outputs(sup_dir / "sup_imnet__seed1_last_test_outputs.csv", _polyp_rows())
    parent_payload = _sun_payload(tau=0.45, offset=-0.08)
    colon_metrics = {
        "seed": 1,
        "test_primary": {
            "tau": 0.55,
            "auprc": 0.8,
            "auroc": 0.9,
            "f1": 0.75,
            "recall": 0.78,
        },
        "provenance": {
            "model": "ssl_colon",
            "parent_run": {
                "metrics": {
                    "path": "../sun_parent/seed1.metrics.json",
                    "payload": parent_payload,
                },
            },
        },
    }
    sup_metrics = {
        "seed": 1,
        "test_primary": {
            "tau": 0.6,
            "auprc": 0.7,
            "auroc": 0.82,
            "f1": 0.68,
            "recall": 0.7,
        },
        "provenance": {
            "model": "sup_imnet",
            "parent_run": {
                "metrics": {
                    "path": "../sun_parent/seed1.metrics.json",
                    "payload": parent_payload,
                },
            },
        },
    }
    _write_metrics(colon_dir / "ssl_colon__seed1_last.metrics.json", colon_metrics)
    _write_metrics(sup_dir / "sup_imnet__seed1_last.metrics.json", sup_metrics)
    runs = discover_runs(root)
    summary = summarize_runs(runs, bootstrap=10, rng_seed=123)
    assert "models" in summary
    assert "ssl_colon" in summary["models"]
    colon_entry = summary["models"]["ssl_colon"]
    assert "performance" in colon_entry
    assert "seeds" in colon_entry
    pairwise = summary.get("pairwise")
    assert isinstance(pairwise, dict)
    assert "auprc" in pairwise