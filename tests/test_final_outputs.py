import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from ssl4polyp.classification import train_classification as tc
from ssl4polyp.classification.metrics import performance


class StaticModel(torch.nn.Module):
    def __init__(self, outputs: torch.Tensor):
        super().__init__()
        self.register_buffer("_outputs", outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - deterministic mapping
        batch_size = inputs.shape[0]
        return self._outputs[:batch_size]


@pytest.mark.parametrize("tau", [0.5])
def test_test_writes_per_frame_outputs(tmp_path: Path, tau: float):
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    metadata = [
        {
            "frame_id": f"frame-{idx}",
            "origin": "polypgen",
            "case_id": f"case-{idx // 2}",
            "morphology": "flat" if idx == 0 else ("polypoid" if idx == 2 else "unknown"),
        }
        for idx in range(len(targets))
    ]
    inputs = torch.zeros((len(targets), 3, 8, 8), dtype=torch.float32)
    test_loader = [(inputs, targets, metadata)]
    model = StaticModel(logits)
    log_path = tmp_path / "eval.log"
    outputs_path = tmp_path / "outputs.csv"

    result = tc.test(
        model,
        rank=0,
        test_loader=test_loader,
        epoch=1,
        perf_fn=performance.meanAUROC(n_class=2),
        log_path=str(log_path),
        metric_fns={},
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        loss_mode="binary_bce",
        split_name="Test",
        tau=tau,
        tau_info="fixed",
        save_outputs_path=outputs_path,
    )

    assert outputs_path.exists(), "per-frame outputs should be written"
    with outputs_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == len(targets)
    first_row = rows[0]
    assert first_row["frame_id"] == "frame-0"
    assert first_row["origin"] == "polypgen"
    assert first_row["sequence_id"] == "case-0"
    assert first_row["case_id"] == "case-0"
    assert first_row["morphology"] == "flat"
    assert int(first_row["label"]) == 1
    assert int(first_row["pred"]) == 1
    assert pytest.approx(float(first_row["prob"]), rel=1e-6) == torch.sigmoid(torch.tensor(1.0)).item()

    strata = result.get("strata")
    assert strata is not None
    overall = strata["overall"]
    flat_stratum = strata["flat_plus_negs"]
    polypoid_stratum = strata["polypoid_plus_negs"]
    assert overall["n_pos"] == 2
    assert overall["n_neg"] == 2
    assert overall["prevalence"] == pytest.approx(0.5)
    assert flat_stratum["n_pos"] == 1
    assert flat_stratum["n_neg"] == 2
    assert flat_stratum["prevalence"] == pytest.approx(1 / 3)
    assert polypoid_stratum["n_pos"] == 1
    assert polypoid_stratum["n_neg"] == 2
    assert polypoid_stratum["prevalence"] == pytest.approx(1 / 3)

    threshold_metrics = result.get("threshold_metrics")
    assert threshold_metrics is not None
    assert threshold_metrics["tp"] == 2
    assert threshold_metrics["tn"] == 2
    assert threshold_metrics["fp"] == 0
    assert threshold_metrics["fn"] == 0
    assert threshold_metrics["prevalence"] == pytest.approx(0.5)
    assert threshold_metrics["mcc"] == pytest.approx(1.0)
    assert result["prevalence"] == pytest.approx(0.5)
    assert result["mcc"] == pytest.approx(1.0)

    log_text = log_path.read_text()
    assert "confusion @τ" in log_text


def test_curve_exports_write_csvs(tmp_path: Path):
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.6, 0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    ckpt_stem = tmp_path / "demo_model"

    exports = tc._export_curve_sets(
        ckpt_stem,
        "Test",
        probabilities=probabilities,
        targets=targets,
        grid_points=5,
    )

    roc_path = exports["roc_csv"]
    pr_path = exports["pr_csv"]

    assert roc_path.exists()
    assert pr_path.exists()
    assert exports["grid_points"] == 5

    def _read_csv(path: Path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    roc_rows = _read_csv(roc_path)
    pr_rows = _read_csv(pr_path)

    assert len(roc_rows) == 5
    assert len(pr_rows) == 5

    def as_float(value: str) -> Optional[float]:
        return None if value == "" else float(value)

    first_roc = roc_rows[0]
    assert as_float(first_roc["threshold"]) == pytest.approx(0.0)
    assert as_float(first_roc["tpr"]) == pytest.approx(1.0)
    assert as_float(first_roc["fpr"]) == pytest.approx(1.0)

    last_roc = roc_rows[-1]
    assert as_float(last_roc["threshold"]) == pytest.approx(1.0)
    assert as_float(last_roc["tpr"]) == pytest.approx(0.0)
    assert as_float(last_roc["fpr"]) == pytest.approx(0.0)

    last_pr = pr_rows[-1]
    assert as_float(last_pr["threshold"]) == pytest.approx(1.0)
    assert last_pr["precision"] == ""


def test_prepare_metric_export_includes_counts_and_confusion():
    metrics = {
        "auroc": 0.75,
        "class_counts": [4, 6],
        "threshold_metrics": {
            "tp": 5,
            "tn": 3,
            "fp": 1,
            "fn": 2,
            "prevalence": 0.6,
            "mcc": 0.25,
        },
    }

    export = tc._prepare_metric_export(metrics)

    assert export["tp"] == 5
    assert export["fp"] == 1
    assert export["fn"] == 2
    assert export["tn"] == 3
    assert export["n_neg"] == 4
    assert export["n_pos"] == 6
    assert export["n_total"] == 10
    assert export["prevalence"] == pytest.approx(0.6)
    assert export["mcc"] == pytest.approx(0.25)


def test_build_metric_block_selects_primary_metrics():
    metrics = {
        "auprc": 0.82,
        "tp": 7,
        "fp": 2,
        "tn": 9,
        "fn": 1,
        "prevalence": 0.45,
        "tau": 0.37,
        "tau_info": "demo",
        "ignored": 123,
    }

    block = tc._build_metric_block(metrics)

    assert block["auprc"] == pytest.approx(0.82)
    assert block["tp"] == 7
    assert block["fp"] == 2
    assert block["tau"] == pytest.approx(0.37)
    assert block["tau_info"] == "demo"
    assert "ignored" not in block


def test_build_sensitivity_block_filters_invalid_entries():
    strata = {
        "overall": {"auprc": 0.5, "tp": 3},
        "flat_plus_negs": {"f1": 0.6, "n_pos": 2},
        "invalid": 123,
    }

    block = tc._build_sensitivity_block(strata)

    assert set(block.keys()) == {"flat_plus_negs", "overall"}
    assert block["overall"]["tp"] == 3
    assert block["flat_plus_negs"]["n_pos"] == 2


def test_build_metrics_provenance_prefers_subset_trace(tmp_path: Path):
    outputs_csv = tmp_path / "demo_outputs.csv"
    outputs_csv.write_text("frame_id,prob,label,pred\n", encoding="utf-8")
    outputs_sha = tc._compute_file_sha256(outputs_csv)

    args = SimpleNamespace(
        seed=7,
        active_seed=11,
        model_key="ssl_colon",
        model_tag="ssl_colon_tag",
        run_stem="ssl_colon_run",
        arch="vit_b",
        dataset_percent=None,
        dataset_seed=23,
        test_split="test",
        dataset_layout={"percent": 50, "dataset_seed": 99},
        output_dir=str(tmp_path),
        latest_test_outputs_path=outputs_csv,
        latest_test_outputs_sha256=outputs_sha,
    )
    trace = tc.Experiment4SubsetTrace(
        percent=25,
        seed=42,
        train_pos_cases=0,
        train_neg_cases=0,
        frames_per_case=None,
        total_frames=0,
        pos_case_ids=tuple(),
        neg_case_ids=tuple(),
        pos_digest="",
        neg_digest="",
        manifest=None,
    )

    provenance = tc._build_metrics_provenance(args, experiment4_trace=trace)

    assert provenance["model"] == "ssl_colon"
    assert provenance["arch"] == "vit_b"
    assert provenance["train_seed"] == 11
    assert provenance["subset_percent"] == pytest.approx(25.0)
    assert provenance["pack_seed"] == 42
    assert provenance["split"] == "test"
    assert provenance["test_outputs_csv_sha256"] == outputs_sha
    assert provenance["test_outputs_csv"] == "demo_outputs.csv"


def test_build_run_metadata_collects_core_fields():
    args = SimpleNamespace(
        exp_config="config/exp/exp1.yaml",
        run_stem="ssl_colon__sun_full_s7",
        run_tag="exp1_seed7",
        model_tag="ssl_colon",
        arch="vit_b",
        pretraining="ImageNet_self",
        finetune_mode="full",
        active_seed=7,
        eval_only=False,
        world_size=2,
    )

    run_info = tc._build_run_metadata(args, selection_tag="ValLoss")

    assert run_info["experiment"] == "exp1"
    assert run_info["experiment_config"] == "config/exp/exp1.yaml"
    assert run_info["stem"] == "ssl_colon__sun_full_s7"
    assert run_info["model"] == "ssl_colon"
    assert run_info["arch"] == "vit_b"
    assert run_info["pretraining"] == "ImageNet_self"
    assert run_info["selection"] == "ValLoss"
    assert run_info["seed"] == 7
    assert run_info["mode"] == "train"
    assert run_info["world_size"] == 2