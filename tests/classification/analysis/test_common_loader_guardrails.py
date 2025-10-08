import csv
import json
from pathlib import Path

import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.common_loader import (  # type: ignore[import]
    get_default_loader,
    load_common_run,
)
from ssl4polyp.classification.analysis.result_loader import (  # type: ignore[import]
    GuardrailViolation,
)


def _write_outputs(path: Path) -> None:
    rows = [
        {"frame_id": "f1", "case_id": "c1", "prob": 0.9, "label": 1},
        {"frame_id": "f2", "case_id": "c2", "prob": 0.1, "label": 0},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "case_id", "prob", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _base_payload(*, tau: float = 0.5, digest: str = "deadbeef", policy: str = "f1_opt_on_val") -> dict[str, object]:
    return {
        "seed": 1,
        "test_primary": {
            "tau": tau,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
        },
        "test_sensitivity": {
            "tau": tau,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
        },
        "thresholds": {
            "primary": {"policy": policy, "tau": tau},
            "sensitivity": {"policy": "youden_on_val", "tau": tau},
        },
        "provenance": {
            "model": "ssl_colon",
            "test_outputs_csv_sha256": digest,
        },
    }


def _write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _materialise_run(tmp_path: Path, payload: dict[str, object], stem: str = "run") -> Path:
    metrics_path = tmp_path / f"{stem}.metrics.json"
    outputs_path = tmp_path / f"{stem}_test_outputs.csv"
    _write_outputs(outputs_path)
    _write_metrics(metrics_path, payload)
    return metrics_path


def test_missing_thresholds_triggers_guardrail(tmp_path: Path) -> None:
    payload = _base_payload()
    payload.pop("thresholds", None)
    metrics_path = _materialise_run(tmp_path, payload, stem="missing_thresholds")
    loader = get_default_loader()
    with pytest.raises(GuardrailViolation):
        load_common_run(metrics_path, loader=loader)


def test_digest_mismatch_across_runs(tmp_path: Path) -> None:
    loader = get_default_loader()
    first_path = _materialise_run(tmp_path, _base_payload(digest="aaaa"), stem="first")
    load_common_run(first_path, loader=loader)
    second_path = _materialise_run(tmp_path, _base_payload(digest="bbbb"), stem="second")
    with pytest.raises(GuardrailViolation):
        load_common_run(second_path, loader=loader)


def test_primary_policy_mismatch(tmp_path: Path) -> None:
    payload = _base_payload(policy="max_recall")
    metrics_path = _materialise_run(tmp_path, payload, stem="bad_policy")
    loader = get_default_loader()
    with pytest.raises(GuardrailViolation):
        load_common_run(metrics_path, loader=loader)
