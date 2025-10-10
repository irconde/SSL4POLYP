from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.exp3_report import (  # type: ignore[import]
    FrameRecord,
    compute_strata_metrics,
    generate_report,
    EXPECTED_SEEDS,
)


_DATA_BLOCK = {
    "train": {"path": "sun_morphology/train.csv", "sha256": "train-digest"},
    "val": {"path": "sun_morphology/val.csv", "sha256": "val-digest"},
    "test": {"path": "sun_morphology/test.csv", "sha256": "test-digest"},
}
_VAL_PATH = _DATA_BLOCK["val"]["path"]


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


def test_compute_strata_metrics_includes_empty_strata() -> None:
    records = [
        FrameRecord(prob=0.7, label=1, pred=1, case_id="case_polyp", morphology="polypoid"),
        FrameRecord(prob=0.2, label=0, pred=0, case_id="case_neg", morphology="unknown"),
    ]
    metrics = compute_strata_metrics(records, tau=0.5)
    assert metrics["flat_plus_negs"]["n_pos"] == 0
    assert metrics["flat_plus_negs"]["n_neg"] == 1
    assert metrics["polypoid_plus_negs"]["n_pos"] == 1


def test_report_composition_includes_empty_flat_stratum(tmp_path: Path) -> None:
    colon_rows = [
        {"case_id": "a", "prob": 0.85, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.15, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "c", "prob": 0.18, "label": 0, "pred": 0, "morphology": "other"},
    ]
    sup_rows = [
        {"case_id": "a", "prob": 0.8, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.25, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "c", "prob": 0.22, "label": 0, "pred": 0, "morphology": "other"},
    ]
    ssl_rows = [
        {"case_id": "a", "prob": 0.78, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.3, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "c", "prob": 0.27, "label": 0, "pred": 0, "morphology": "other"},
    ]
    for seed in EXPECTED_SEEDS:
        _write_run(tmp_path, "ssl_colon", seed=seed, rows=colon_rows, tau=0.4)
        _write_run(tmp_path, "sup_imnet", seed=seed, rows=sup_rows, tau=0.4)
        _write_run(tmp_path, "ssl_imnet", seed=seed, rows=ssl_rows, tau=0.4)

    report = generate_report(tmp_path, bootstrap=0, rng_seed=3)

    composition_lines = [
        line for line in report.splitlines() if line.startswith("| Flat + Negs |")
    ]
    assert composition_lines, "Expected Flat + Negs stratum in composition table"
    assert "| Flat + Negs | 0 |" in composition_lines[0]


def test_generate_report_requires_all_deltas(tmp_path: Path) -> None:
    rows = [
        {"case_id": "a", "prob": 0.75, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.35, "label": 0, "pred": 0, "morphology": "other"},
    ]
    for seed in EXPECTED_SEEDS:
        _write_run(tmp_path, "ssl_colon", seed=seed, rows=rows, tau=0.5)
        _write_run(tmp_path, "sup_imnet", seed=seed, rows=rows, tau=0.5)
    with pytest.raises(RuntimeError, match="SSL-ImNet"):
        generate_report(tmp_path, bootstrap=0, rng_seed=19)


def _write_run(root: Path, model: str, seed: int, rows: list[dict[str, object]], tau: float = 0.5) -> None:
    run_prefix = f"{model}__sun_morphology_s{seed}"
    metrics_path = root / f"{run_prefix}_last.metrics.json"
    tp = fp = tn = fn = 0
    n_pos = n_neg = 0
    for row in rows:
        label = int(row.get("label", 0))  # type: ignore[call-overload]
        pred = int(row.get("pred", 0))  # type: ignore[call-overload]
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
    total = n_pos + n_neg
    metrics_payload = {
        "seed": seed,
        "data": _DATA_BLOCK,
        "val": {"path": _VAL_PATH, "sha256": "val-digest"},
        "test_primary": {
            "tau": tau,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": float(n_pos) / float(total) if total else 0.0,
        },
        "test_sensitivity": {
            "tau": tau,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": float(n_pos) / float(total) if total else 0.0,
        },
        "thresholds": {
            "primary": {
                "policy": "f1_opt_on_val",
                "tau": tau,
                "split": _VAL_PATH,
                "epoch": 1,
            },
            "sensitivity": {
                "policy": "youden_on_val",
                "tau": tau,
                "split": _VAL_PATH,
                "epoch": 1,
            },
        },
        "provenance": {
            "model": model,
            "test_outputs_csv_sha256": "deadbeef",
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    outputs_primary = root / f"{run_prefix}_test_outputs.csv"
    outputs_with_suffix = root / f"{run_prefix}.metrics_test_outputs.csv"
    outputs_expected = metrics_path.with_name(f"{metrics_path.stem}_test_outputs.csv")
    for path in {outputs_primary, outputs_with_suffix, outputs_expected}:
        with path.open("w", encoding="utf-8", newline="") as handle:
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
        {"case_id": "e", "prob": 0.3, "label": 0, "pred": 0, "morphology": "other"},
    ]
    sup_rows = [
        {"case_id": "a", "prob": 0.7, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.6, "label": 1, "pred": 1, "morphology": "flat"},
        {"case_id": "c", "prob": 0.7, "label": 0, "pred": 1, "morphology": "other"},
        {"case_id": "d", "prob": 0.2, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "e", "prob": 0.65, "label": 0, "pred": 1, "morphology": "other"},
    ]
    ssl_rows = [
        {"case_id": "a", "prob": 0.68, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.58, "label": 1, "pred": 1, "morphology": "flat"},
        {"case_id": "c", "prob": 0.66, "label": 0, "pred": 1, "morphology": "other"},
        {"case_id": "d", "prob": 0.21, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "e", "prob": 0.6, "label": 0, "pred": 1, "morphology": "other"},
    ]
    alt_sup_rows = [
        {"case_id": "a", "prob": 0.65, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.62, "label": 1, "pred": 1, "morphology": "flat"},
        {"case_id": "c", "prob": 0.68, "label": 0, "pred": 1, "morphology": "other"},
        {"case_id": "d", "prob": 0.22, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "e", "prob": 0.61, "label": 0, "pred": 1, "morphology": "other"},
    ]
    alt_ssl_rows = [
        {"case_id": "a", "prob": 0.64, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.6, "label": 1, "pred": 1, "morphology": "flat"},
        {"case_id": "c", "prob": 0.64, "label": 0, "pred": 1, "morphology": "other"},
        {"case_id": "d", "prob": 0.24, "label": 0, "pred": 0, "morphology": "unknown"},
        {"case_id": "e", "prob": 0.58, "label": 0, "pred": 1, "morphology": "other"},
    ]
    alt_colon_rows = [
        {"case_id": "a", "prob": 0.88, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.82, "label": 1, "pred": 1, "morphology": "flat"},
        {"case_id": "c", "prob": 0.18, "label": 0, "pred": 0, "morphology": "other"},
        {"case_id": "e", "prob": 0.28, "label": 0, "pred": 0, "morphology": "other"},
    ]
    for seed in EXPECTED_SEEDS:
        _write_run(tmp_path, "ssl_colon", seed=seed, rows=colon_rows if seed != 29 else alt_colon_rows, tau=0.5)
        sup_payload = sup_rows if seed != 47 else alt_sup_rows
        _write_run(tmp_path, "sup_imnet", seed=seed, rows=sup_payload, tau=0.5)
        ssl_payload = ssl_rows if seed != 47 else alt_ssl_rows
        _write_run(tmp_path, "ssl_imnet", seed=seed, rows=ssl_payload, tau=0.5)

    report = generate_report(tmp_path, bootstrap=2, rng_seed=7)

    assert "## Metrics at τ_F1(val-morph) — Overall" in report
    assert "Flat + Negs" in report
    assert "SSL-Colon − SUP-ImNet" in report
    assert "SSL-Colon − SSL-ImNet" in report
    assert "Interaction effect" in report
    assert "Appendix: Sensitivity operating point (τ_Youden(val-morph))" in report
    delta_rows = [
        line
        for line in report.splitlines()
        if "SSL-Colon − SUP-ImNet" in line and "95% CI:" in line
    ]
    assert delta_rows, "Expected delta table rows with 95% CI information"
    assert any("±" in line for line in delta_rows), "Expected at least one delta row with a mean ± sd display"


def test_generate_report_rejects_mismatched_primary_tau(tmp_path: Path) -> None:
    rows = [
        {"case_id": "a", "prob": 0.9, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.1, "label": 0, "pred": 0, "morphology": "other"},
    ]
    for seed in EXPECTED_SEEDS:
        _write_run(tmp_path, "ssl_colon", seed=seed, rows=rows, tau=0.5)
        _write_run(tmp_path, "sup_imnet", seed=seed, rows=rows, tau=0.5)
        _write_run(tmp_path, "ssl_imnet", seed=seed, rows=rows, tau=0.5)

    metrics_path = tmp_path / "ssl_colon__sun_morphology_s13_last.metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["thresholds"]["primary"]["tau"] = 0.42
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="primary.*tau"):
        generate_report(tmp_path, bootstrap=0, rng_seed=11)


def test_generate_report_rejects_unexpected_test_split(tmp_path: Path) -> None:
    rows = [
        {"case_id": "a", "prob": 0.9, "label": 1, "pred": 1, "morphology": "polypoid"},
        {"case_id": "b", "prob": 0.1, "label": 0, "pred": 0, "morphology": "other"},
    ]
    for seed in EXPECTED_SEEDS:
        _write_run(tmp_path, "ssl_colon", seed=seed, rows=rows, tau=0.5)
        _write_run(tmp_path, "sup_imnet", seed=seed, rows=rows, tau=0.5)
        _write_run(tmp_path, "ssl_imnet", seed=seed, rows=rows, tau=0.5)

    metrics_path = tmp_path / "ssl_colon__sun_morphology_s13_last.metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["data"]["test"]["path"] = "other_dataset/test.csv"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="sun_morphology/test"):
        generate_report(tmp_path, bootstrap=0, rng_seed=5)
