from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl4polyp.utils.reporting_inputs import copy_reporting_inputs


def _write_metrics_payload(path: Path) -> None:
    payload = {"accuracy": 1.0}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_copy_reporting_inputs_supports_nested_run_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    nested_dir = run_dir / "dataset" / "p10" / "seed13"
    nested_dir.mkdir(parents=True)

    metrics_path = nested_dir / "foo_last.metrics.json"
    outputs_path = nested_dir / "foo_test_outputs.csv"
    _write_metrics_payload(metrics_path)
    outputs_path.write_text("id,pred\n1,0\n", encoding="utf-8")

    reporting_root = tmp_path / "reporting"

    copies = copy_reporting_inputs(
        run_dir,
        reporting_root,
        reporting_subdir="exp/foo",
    )

    assert {result.destination.name for result in copies} == {
        metrics_path.name,
        outputs_path.name,
    }

    dest_metrics = reporting_root / "exp" / "foo" / metrics_path.name
    dest_outputs = reporting_root / "exp" / "foo" / outputs_path.name
    assert dest_metrics.exists()
    assert dest_outputs.exists()


def test_copy_reporting_inputs_collects_all_leaf_directories(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first_leaf = run_dir / "dataset_a" / "seed13"
    second_leaf = run_dir / "dataset_b" / "s5" / "seed29"
    first_leaf.mkdir(parents=True)
    second_leaf.mkdir(parents=True)

    first_metrics = first_leaf / "first_last.metrics.json"
    first_outputs = first_leaf / "first_test_outputs.csv"
    _write_metrics_payload(first_metrics)
    first_outputs.write_text("id,pred\n1,0\n", encoding="utf-8")

    # Second leaf only emits non-``_last`` metrics to cover the fallback path.
    second_metrics = second_leaf / "second.metrics.json"
    second_outputs = second_leaf / "second_test_outputs.csv"
    _write_metrics_payload(second_metrics)
    second_outputs.write_text("id,pred\n2,1\n", encoding="utf-8")

    reporting_root = tmp_path / "reporting"

    copies = copy_reporting_inputs(
        run_dir,
        reporting_root,
        reporting_subdir="exp/bar",
    )

    copied_names = {result.destination.name for result in copies}
    assert copied_names == {
        first_metrics.name,
        first_outputs.name,
        second_metrics.name,
        second_outputs.name,
    }

    for path in (
        reporting_root / "exp" / "bar" / first_metrics.name,
        reporting_root / "exp" / "bar" / first_outputs.name,
        reporting_root / "exp" / "bar" / second_metrics.name,
        reporting_root / "exp" / "bar" / second_outputs.name,
    ):
        assert path.exists()

