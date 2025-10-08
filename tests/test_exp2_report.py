from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, cast

import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.exp2_report import (  # type: ignore[import]
    EvalFrame,
    PRIMARY_METRICS,
    load_run,
    summarize_runs,
    discover_runs,
    _metrics_from_frames,
)


def _write_run(
    root: Path,
    *,
    model: str,
    seed: int,
    tau: float,
    rows: Iterable[dict[str, object]],
    csv_digest: str = "deadbeefcafebabe",
) -> Path:
    run_name = f"{model}__sun_full_s{seed}"
    metrics_path = root / f"{run_name}.metrics.json"
    rows_list = list(rows)
    frames = []
    for idx, row in enumerate(rows_list):
        frame_id = str(row.get("frame_id", f"idx_{idx}"))
        case_id = str(row.get("case_id", f"case_{idx}"))
        prob = float(cast(float, row["prob"]))
        label = int(cast(int, row["label"]))
        frames.append(EvalFrame(frame_id=frame_id, case_id=case_id, prob=prob, label=label))
    metrics = _metrics_from_frames(frames, tau)
    primary_block = {
        metric: metrics.get(metric, float("nan")) for metric in PRIMARY_METRICS
    }
    primary_block.update(
        {
            "tau": tau,
            "tp": metrics.get("tp", 0),
            "fp": metrics.get("fp", 0),
            "tn": metrics.get("tn", 0),
            "fn": metrics.get("fn", 0),
            "n_pos": metrics.get("n_pos", 0),
            "n_neg": metrics.get("n_neg", 0),
        }
    )
    payload = {
        "seed": seed,
        "test_primary": primary_block,
        "test_sensitivity": dict(primary_block),
        "thresholds": {
            "policy": "f1_opt_on_val",
            "primary": {"policy": "f1_opt_on_val", "tau": tau},
            "sensitivity": {"policy": "youden_on_val", "tau": tau},
        },
        "provenance": {
            "model": model,
            "sun_full_test_csv_sha256": csv_digest,
            "splits": {"sun_full_test": {"csv_sha256": csv_digest}},
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    outputs_path = root / f"{run_name}_test_outputs.csv"
    with outputs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "case_id", "prob", "label"])
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
    return metrics_path


def test_load_run_parses_metrics(tmp_path: Path) -> None:
    rows = [
        {"frame_id": "f1", "case_id": "c1", "prob": 0.9, "label": 1},
        {"frame_id": "f2", "case_id": "c1", "prob": 0.2, "label": 0},
    ]
    metrics_path = _write_run(tmp_path, model="ssl_colon", seed=3, tau=0.5, rows=rows)
    run = load_run(metrics_path)
    assert run.model == "ssl_colon"
    assert run.seed == 3
    assert pytest.approx(run.tau) == 0.5
    assert len(run.frames) == 2
    assert set(run.cases.keys()) == {"c1"}
    assert run.metrics["auprc"] == pytest.approx(1.0)


def test_summarize_runs_and_deltas(tmp_path: Path) -> None:
    colon_seed1 = [
        {"frame_id": "c1_f1", "case_id": "caseA", "prob": 0.95, "label": 1},
        {"frame_id": "c1_f2", "case_id": "caseA", "prob": 0.05, "label": 0},
        {"frame_id": "c1_f3", "case_id": "caseB", "prob": 0.90, "label": 1},
        {"frame_id": "c1_f4", "case_id": "caseB", "prob": 0.10, "label": 0},
    ]
    colon_seed2 = [
        {"frame_id": "c2_f1", "case_id": "caseC", "prob": 0.92, "label": 1},
        {"frame_id": "c2_f2", "case_id": "caseC", "prob": 0.08, "label": 0},
        {"frame_id": "c2_f3", "case_id": "caseD", "prob": 0.88, "label": 1},
        {"frame_id": "c2_f4", "case_id": "caseD", "prob": 0.12, "label": 0},
    ]
    imnet_seed1 = [
        {"frame_id": "i1_f1", "case_id": "caseA", "prob": 0.70, "label": 1},
        {"frame_id": "i1_f2", "case_id": "caseA", "prob": 0.65, "label": 0},
        {"frame_id": "i1_f3", "case_id": "caseB", "prob": 0.45, "label": 1},
        {"frame_id": "i1_f4", "case_id": "caseB", "prob": 0.30, "label": 0},
    ]
    imnet_seed2 = [
        {"frame_id": "i2_f1", "case_id": "caseC", "prob": 0.60, "label": 1},
        {"frame_id": "i2_f2", "case_id": "caseC", "prob": 0.58, "label": 0},
        {"frame_id": "i2_f3", "case_id": "caseD", "prob": 0.40, "label": 1},
        {"frame_id": "i2_f4", "case_id": "caseD", "prob": 0.28, "label": 0},
    ]
    for seed, rows in ((1, colon_seed1), (2, colon_seed2)):
        _write_run(tmp_path, model="ssl_colon", seed=seed, tau=0.5, rows=rows)
    for seed, rows in ((1, imnet_seed1), (2, imnet_seed2)):
        _write_run(tmp_path, model="ssl_imnet", seed=seed, tau=0.5, rows=rows)

    runs = discover_runs(tmp_path)
    summary = summarize_runs(runs, bootstrap=100, rng_seed=123)

    colon_metrics = summary.model_metrics["ssl_colon"]
    imnet_metrics = summary.model_metrics["ssl_imnet"]
    assert colon_metrics["recall"].mean == pytest.approx(1.0)
    assert imnet_metrics["recall"].mean == pytest.approx(0.5)
    recall_delta = summary.paired_deltas["recall"]
    assert recall_delta.mean == pytest.approx(0.5)
    assert set(recall_delta.per_seed.keys()) == {1, 2}
    assert recall_delta.ci_lower is not None
    assert recall_delta.ci_upper is not None
    assert recall_delta.ci_lower <= recall_delta.ci_upper
    assert recall_delta.samples  # Bootstrap samples collected


def test_discover_runs_csv_guardrail(tmp_path: Path) -> None:
    rows = [
        {"frame_id": "f1", "case_id": "c1", "prob": 0.9, "label": 1},
        {"frame_id": "f2", "case_id": "c1", "prob": 0.1, "label": 0},
    ]
    _write_run(tmp_path, model="ssl_colon", seed=1, tau=0.5, rows=rows, csv_digest="aaaa")
    _write_run(tmp_path, model="ssl_colon", seed=2, tau=0.5, rows=rows, csv_digest="bbbb")
    with pytest.raises(RuntimeError):
        discover_runs(tmp_path)
