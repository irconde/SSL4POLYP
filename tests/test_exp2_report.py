from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, cast

import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.exp2_report import (  # type: ignore[import]
    ALL_METRICS,
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    EvalFrame,
    collect_summary,
    discover_runs,
    load_run,
    render_markdown,
    summarize_runs,
    write_csv_tables,
    _metrics_from_frames,
)


def _write_run(
    root: Path,
    *,
    model: str,
    seed: int,
    tau_primary: float,
    tau_sensitivity: float,
    rows: Iterable[dict[str, object]],
    csv_digest: str = "deadbeefcafebabe",
) -> Path:
    run_name = f"{model}__sun_full_s{seed}"
    metrics_path = root / f"{run_name}_last.metrics.json"
    rows_list = list(rows)
    frames: list[EvalFrame] = []
    for idx, row in enumerate(rows_list):
        frame_id = str(row.get("frame_id", f"idx_{idx}"))
        case_id = str(row.get("case_id", f"case_{idx}"))
        prob = float(cast(float, row["prob"]))
        label = int(cast(int, row["label"]))
        frames.append(EvalFrame(frame_id=frame_id, case_id=case_id, prob=prob, label=label))
    primary_metrics = _metrics_from_frames(frames, tau_primary)
    sensitivity_metrics = _metrics_from_frames(frames, tau_sensitivity)
    common_counts = {
        "tp": int(primary_metrics.get("tp", 0.0)),
        "fp": int(primary_metrics.get("fp", 0.0)),
        "tn": int(primary_metrics.get("tn", 0.0)),
        "fn": int(primary_metrics.get("fn", 0.0)),
        "n_pos": int(primary_metrics.get("n_pos", 0.0)),
        "n_neg": int(primary_metrics.get("n_neg", 0.0)),
    }
    primary_block = {
        metric: primary_metrics.get(metric, float("nan"))
        for metric in ALL_METRICS
    }
    primary_block.update(common_counts)
    primary_block["prevalence"] = float(primary_metrics.get("prevalence", float("nan")))
    primary_block["tau"] = tau_primary
    sensitivity_block = {
        metric: sensitivity_metrics.get(metric, float("nan"))
        for metric in ALL_METRICS
    }
    sensitivity_block.update(common_counts)
    sensitivity_block["prevalence"] = float(sensitivity_metrics.get("prevalence", float("nan")))
    sensitivity_block["tau"] = tau_sensitivity
    payload = {
        "seed": seed,
        "test_primary": primary_block,
        "test_sensitivity": sensitivity_block,
        "thresholds": {
            "primary": {"policy": "f1_opt_on_val", "tau": tau_primary, "split": "val", "epoch": 7},
            "sensitivity": {
                "policy": "youden_on_val",
                "tau": tau_sensitivity,
                "split": "val",
                "epoch": 7,
            },
        },
        "provenance": {
            "model": model,
            "sun_full_test_csv_sha256": csv_digest,
            "splits": {"sun_full_test": {"csv_sha256": csv_digest}},
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    outputs_candidates = {
        root / f"{run_name}_last.metrics_test_outputs.csv",
        root / f"{run_name}_last_test_outputs.csv",
    }
    for path in outputs_candidates:
        with path.open("w", encoding="utf-8", newline="") as handle:
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
    metrics_path = _write_run(
        tmp_path,
        model="ssl_colon",
        seed=13,
        tau_primary=0.5,
        tau_sensitivity=0.3,
        rows=rows,
    )
    run = load_run(metrics_path)
    assert run.model == "ssl_colon"
    assert run.seed == 13
    assert run.tau_primary == pytest.approx(0.5)
    assert run.tau_sensitivity == pytest.approx(0.3)
    assert run.primary_metrics["loss"] == pytest.approx(run.sensitivity_metrics["loss"], rel=1e-6)
    assert run.frames[0].case_id == "c1"
    assert set(run.cases.keys()) == {"c1"}


def test_summarize_runs_and_deltas(tmp_path: Path) -> None:
    colon_rows = {
        13: [
            {"frame_id": "c1_f1", "case_id": "caseA", "prob": 0.95, "label": 1},
            {"frame_id": "c1_f2", "case_id": "caseA", "prob": 0.05, "label": 0},
        ],
        29: [
            {"frame_id": "c2_f1", "case_id": "caseB", "prob": 0.92, "label": 1},
            {"frame_id": "c2_f2", "case_id": "caseB", "prob": 0.08, "label": 0},
        ],
        47: [
            {"frame_id": "c3_f1", "case_id": "caseC", "prob": 0.93, "label": 1},
            {"frame_id": "c3_f2", "case_id": "caseC", "prob": 0.07, "label": 0},
        ],
    }
    imnet_rows = {
        13: [
            {"frame_id": "i1_f1", "case_id": "caseA", "prob": 0.60, "label": 1},
            {"frame_id": "i1_f2", "case_id": "caseA", "prob": 0.55, "label": 0},
        ],
        29: [
            {"frame_id": "i2_f1", "case_id": "caseB", "prob": 0.58, "label": 1},
            {"frame_id": "i2_f2", "case_id": "caseB", "prob": 0.52, "label": 0},
        ],
        47: [
            {"frame_id": "i3_f1", "case_id": "caseC", "prob": 0.55, "label": 1},
            {"frame_id": "i3_f2", "case_id": "caseC", "prob": 0.50, "label": 0},
        ],
    }
    for seed, rows in colon_rows.items():
        _write_run(
            tmp_path,
            model="ssl_colon",
            seed=seed,
            tau_primary=0.5,
            tau_sensitivity=0.4,
            rows=rows,
        )
    for seed, rows in imnet_rows.items():
        _write_run(
            tmp_path,
            model="ssl_imnet",
            seed=seed,
            tau_primary=0.5,
            tau_sensitivity=0.4,
            rows=rows,
        )

    runs = discover_runs(tmp_path)
    summary = summarize_runs(runs, bootstrap=50, rng_seed=123)

    assert summary.composition.n_pos + summary.composition.n_neg > 0
    assert tuple(summary.seed_validation.expected_seeds) == EXPECTED_SEEDS
    assert set(summary.primary_metrics.keys()) == set(EXPECTED_MODELS)
    colon_metrics = summary.primary_metrics["ssl_colon"]
    imnet_metrics = summary.primary_metrics["ssl_imnet"]
    assert colon_metrics["recall"].mean > imnet_metrics["recall"].mean
    delta = summary.primary_deltas["recall"]
    assert set(delta.per_seed.keys()) == set(EXPECTED_SEEDS)
    assert delta.mean == pytest.approx(colon_metrics["recall"].mean - imnet_metrics["recall"].mean)
    appendix_delta = summary.sensitivity_deltas["recall"]
    assert appendix_delta.mean == pytest.approx(delta.mean)


def test_summarize_runs_missing_paired_model_raises(tmp_path: Path) -> None:
    seeds = EXPECTED_SEEDS
    for seed in seeds:
        colon_rows = [
            {"frame_id": f"colon_{seed}_f1", "case_id": f"case{seed}", "prob": 0.9, "label": 1},
            {"frame_id": f"colon_{seed}_f2", "case_id": f"case{seed}", "prob": 0.1, "label": 0},
        ]
        imnet_rows = [
            {"frame_id": f"imnet_{seed}_f1", "case_id": f"case{seed}", "prob": 0.6, "label": 1},
            {"frame_id": f"imnet_{seed}_f2", "case_id": f"case{seed}", "prob": 0.5, "label": 0},
        ]
        _write_run(
            tmp_path,
            model="ssl_colon",
            seed=seed,
            tau_primary=0.5,
            tau_sensitivity=0.4,
            rows=colon_rows,
        )
        _write_run(
            tmp_path,
            model="ssl_imnet",
            seed=seed,
            tau_primary=0.5,
            tau_sensitivity=0.4,
            rows=imnet_rows,
        )

    runs = discover_runs(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        summarize_runs(runs, paired_models=("ssl_colon", "ssl_imnet_missing"))
    assert "ssl_imnet_missing" in str(excinfo.value)


def test_collect_summary_and_outputs(tmp_path: Path) -> None:
    rows = [
        {"frame_id": "f1", "case_id": "caseA", "prob": 0.8, "label": 1},
        {"frame_id": "f2", "case_id": "caseA", "prob": 0.2, "label": 0},
    ]
    for model in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            _write_run(
                tmp_path,
                model=model,
                seed=seed,
                tau_primary=0.6,
                tau_sensitivity=0.4,
                rows=rows,
            )
    runs, summary, loader = collect_summary(tmp_path, bootstrap=10, rng_seed=999)
    assert summary is not None
    markdown = render_markdown(summary)
    assert "SUN-test composition" in markdown
    tables = write_csv_tables(summary, tmp_path / "tables")
    expected_keys = {
        "t1_composition",
        "t2_primary_metrics",
        "t3_primary_deltas",
        "appendix_sensitivity_metrics",
        "appendix_sensitivity_deltas",
    }
    assert expected_keys.issubset(tables.keys())
    for path in tables.values():
        assert path.exists()


def test_discover_runs_csv_guardrail(tmp_path: Path) -> None:
    rows = [
        {"frame_id": "f1", "case_id": "c1", "prob": 0.9, "label": 1},
        {"frame_id": "f2", "case_id": "c1", "prob": 0.1, "label": 0},
    ]
    _write_run(tmp_path, model="ssl_colon", seed=13, tau_primary=0.5, tau_sensitivity=0.4, rows=rows, csv_digest="aaaa")
    _write_run(tmp_path, model="ssl_colon", seed=29, tau_primary=0.5, tau_sensitivity=0.4, rows=rows, csv_digest="bbbb")
    with pytest.raises(RuntimeError):
        discover_runs(tmp_path)
