from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.exp3_report import (  # type: ignore[import]
    FrameRecord,
    compute_strata_metrics,
    generate_report,
)


@pytest.mark.parametrize("tau", [0.5, 0.7])
def test_compute_strata_metrics_counts(tau: float) -> None:
    records = [
        FrameRecord(prob=0.9, label=1, pred=1, case_id="case_polyp", morphology="polypoid"),
        FrameRecord(prob=0.8, label=1, pred=1, case_id="case_flat", morphology="flat"),
        FrameRecord(prob=0.2, label=0, pred=0, case_id="case_neg1", morphology="unknown"),
        FrameRecord(prob=0.3, label=0, pred=0, case_id="case_neg2", morphology="other"),
    ]
    metrics = compute_strata_metrics(records, tau=tau)
    assert metrics["overall"]["n_pos"] == 2
    assert metrics["overall"]["n_neg"] == 2
    assert metrics["flat_plus_negs"]["n_pos"] == 1
    assert metrics["flat_plus_negs"]["n_neg"] == 2
    assert metrics["polypoid_plus_negs"]["n_pos"] == 1
    assert metrics["polypoid_plus_negs"]["n_neg"] == 2


def test_compute_strata_metrics_excludes_missing_strata() -> None:
    records = [
        FrameRecord(prob=0.7, label=1, pred=1, case_id="case_polyp", morphology="polypoid"),
        FrameRecord(prob=0.2, label=0, pred=0, case_id="case_neg", morphology="unknown"),
    ]
    metrics = compute_strata_metrics(records, tau=0.5)
    assert "flat_plus_negs" not in metrics
    assert "polypoid_plus_negs" in metrics


def _write_run(root: Path, model: str, seed: int, rows: list[dict[str, object]], tau: float = 0.5) -> None:
    run_prefix = f"{model}__sun_morphology_s{seed}"
    metrics_path = root / f"{run_prefix}_last.metrics.json"
    tp = fp = tn = fn = 0
    n_pos = n_neg = 0
    for row in rows:
        label = int(row.get("label", 0))
        pred = int(row.get("pred", 0))
        if label == 1:
            n_pos += 1
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            n_neg += 1
            if pred == 1:
                fp += 1
            else:
                tn += 1
    metrics_payload = {
        "seed": seed,
        "test_primary": {
            "tau": tau,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_pos": n_pos,
            "n_neg": n_neg,
        },
        "test_sensitivity": {
            "tau": tau,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_pos": n_pos,
            "n_neg": n_neg,
        },
        "thresholds": {
            "primary": {"policy": "f1_opt_on_val", "tau": tau},
            "sensitivity": {"policy": "youden_on_val", "tau": tau},
        },
        "provenance": {
            "model": model,
            "test_outputs_csv_sha256": "deadbeef",
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    outputs_path = root / f"{run_prefix}_test_outputs.csv"
    with outputs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "prob", "label", "pred", "morphology"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_generate_report_smoke(tmp_path: Path) -> None:
    colon_rows = [
        {"case_id": "a", "prob": 0.9, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "a", "prob": 0.1, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "b", "prob": 0.8, "label": 1, "pred": 1, "morphology": "flat"},
        {"case_id": "c", "prob": 0.2, "label": 0, "pred": 0, "morphology": "other"},
    ]
    sup_rows = [
        {"case_id": "a", "prob": 0.7, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.4, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "c", "prob": 0.3, "label": 0, "pred": 0, "morphology": "other"},
    ]
    _write_run(tmp_path, "ssl_colon", seed=1, rows=colon_rows, tau=0.5)
    _write_run(tmp_path, "sup_imnet", seed=1, rows=sup_rows, tau=0.5)

    report = generate_report(tmp_path, bootstrap=10, rng_seed=7)

    assert "## Metrics at τ_F1(val-morph) — Overall" in report
    assert "Flat + Negs" in report
    assert "SSL-Colon − SUP-ImNet" in report
    assert "Interaction effect" in report