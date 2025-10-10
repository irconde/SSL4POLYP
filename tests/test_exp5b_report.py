import json
import math
import sys
from pathlib import Path
from typing import Mapping

import pytest  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis import exp5b_report as module  # type: ignore[import]


def test_parse_case_metrics_block_preserves_auxiliary_fields():
    block = {
        "clean": [
            {
                "case_id": "case-001",
                "metrics": {"f1": 0.75, "precision": "0.8", "invalid": None},
                "clusters": {"patient": "subj-1"},
                "metadata": {"camera": "endo-3"},
                "note": "primary",
            },
            {
                "case": "case-002",
                "accuracy": 0.6,
                "clusters": {"patient": "subj-2"},
            },
        ],
        "blur_sigma_1p5": {
            "case-003": {"f1": 0.42},
        },
    }

    parsed = module._parse_case_metrics_block(block)

    clean_cases = parsed["clean"]
    case_one = clean_cases["case-001"]
    case_two = clean_cases["case-002"]

    assert math.isclose(case_one["f1"], 0.75)
    assert math.isclose(case_one["precision"], 0.8)
    assert case_one["clusters"] == {"patient": "subj-1"}
    assert case_one["metadata"] == {"camera": "endo-3"}
    # extra non-metric fields are also preserved
    assert case_one["note"] == "primary"
    # entries without a nested metrics block should retain their original structure
    assert math.isclose(case_two["accuracy"], 0.6)
    assert case_two["clusters"] == {"patient": "subj-2"}

    blur_cases = parsed["blur_sigma_1p5"]
    assert math.isclose(blur_cases["case-003"]["f1"], 0.42)


def _write_minimal_metrics(path: Path, *, seed: int, model: str) -> None:
    payload = {
        "seed": seed,
        "test_primary": {"tau": 0.5, "prevalence": 0.5},
        "test_perturbations": {
            "per_tag": {
                "clean": {"f1": 1.0, "tp": 1, "fp": 0, "tn": 1, "fn": 0, "n_pos": 1, "n_neg": 1}
            },
            "per_case": {},
        },
        "provenance": {
            "model": model,
            "train_seed": seed,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_case_metrics_run(
    path: Path,
    *,
    seed: int,
    model: str,
    blur_cases: Mapping[str, Mapping[str, float]],
) -> None:
    clean_cases = {case_id: {"recall": 1.0, "f1": 1.0} for case_id in blur_cases}
    per_tag = {
        "clean": {
            "recall": 1.0,
            "f1": 1.0,
            "count": len(clean_cases),
        }
    }
    blur_recall = [case_stats["recall"] for case_stats in blur_cases.values()]
    blur_f1 = [case_stats["f1"] for case_stats in blur_cases.values()]
    per_tag["blur_sigma_1p5"] = {
        "recall": float(sum(blur_recall) / len(blur_recall)),
        "f1": float(sum(blur_f1) / len(blur_f1)),
        "count": len(blur_cases),
    }
    payload = {
        "seed": seed,
        "test_primary": {"tau": 0.5, "prevalence": 0.5},
        "test_perturbations": {
            "per_tag": per_tag,
            "per_case": {
                "clean": clean_cases,
                "blur_sigma_1p5": blur_cases,
            },
        },
        "provenance": {"model": model, "train_seed": seed},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    outputs_path = module._resolve_outputs_path(path)
    outputs_path.write_text("case_id,probability,target,pred\n", encoding="utf-8")


@pytest.mark.parametrize("intermediate_suffix", ["best", "intermediate", "ema"])
def test_discover_runs_ignores_intermediate_metrics_without_outputs(tmp_path: Path, intermediate_suffix: str) -> None:
    runs_root = tmp_path / "runs"
    model = "ssl_colon"
    seed = 13

    best_path = runs_root / f"{model}_s{seed}_{intermediate_suffix}.metrics.json"
    _write_minimal_metrics(best_path, seed=seed, model=model)

    last_path = runs_root / f"{model}_s{seed}_last.metrics.json"
    _write_minimal_metrics(last_path, seed=seed, model=model)
    outputs_path = module._resolve_outputs_path(last_path)
    outputs_path.write_text("case_id,metric\n", encoding="utf-8")

    runs = module.discover_runs(runs_root)

    assert model in runs
    assert seed in runs[model]
    assert runs[model][seed].path == last_path


def test_summarize_runs_reports_bootstrap_cis(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    target_values = {
        13: {"case-1": {"recall": 0.82, "f1": 0.78}, "case-2": {"recall": 0.74, "f1": 0.7}},
        29: {"case-1": {"recall": 0.85, "f1": 0.8}, "case-2": {"recall": 0.76, "f1": 0.72}},
        47: {"case-1": {"recall": 0.88, "f1": 0.83}, "case-2": {"recall": 0.79, "f1": 0.75}},
    }
    imnet_values = {
        13: {"case-1": {"recall": 0.6, "f1": 0.55}, "case-2": {"recall": 0.5, "f1": 0.47}},
        29: {"case-1": {"recall": 0.58, "f1": 0.53}, "case-2": {"recall": 0.48, "f1": 0.45}},
        47: {"case-1": {"recall": 0.56, "f1": 0.51}, "case-2": {"recall": 0.46, "f1": 0.43}},
    }
    sup_values = {
        13: {"case-1": {"recall": 0.65, "f1": 0.6}, "case-2": {"recall": 0.55, "f1": 0.52}},
        29: {"case-1": {"recall": 0.63, "f1": 0.58}, "case-2": {"recall": 0.53, "f1": 0.5}},
        47: {"case-1": {"recall": 0.61, "f1": 0.56}, "case-2": {"recall": 0.51, "f1": 0.48}},
    }

    def _materialize_model_runs(model: str, values: Mapping[int, Mapping[str, Mapping[str, float]]]) -> None:
        for seed, case_map in values.items():
            path = runs_root / f"{model}_s{seed}_last.metrics.json"
            _write_case_metrics_run(path, seed=seed, model=model, blur_cases=case_map)

    _materialize_model_runs("ssl_colon", target_values)
    _materialize_model_runs("ssl_imnet", imnet_values)
    _materialize_model_runs("sup_imnet", sup_values)

    runs = module.discover_runs(runs_root)

    summary = module.summarize_runs(runs, bootstrap=16, rng_seed=7, cluster_key="case_id")
    tables = summary["tables"]

    t4_rows = tables.get("t4_delta_ausc")
    assert isinstance(t4_rows, list) and t4_rows
    relevant_t4 = [row for row in t4_rows if row.get("family") not in {"baseline", "aggregate", "macro"}]
    assert relevant_t4, "expected at least one non-macro T4 row"
    for row in relevant_t4:
        assert row.get("ci_lower") is not None
        assert row.get("ci_upper") is not None

    t5_rows = tables.get("t5_delta_retention_by_severity")
    assert isinstance(t5_rows, list) and t5_rows
    for row in t5_rows:
        assert row.get("ci_lower") is not None
        assert row.get("ci_upper") is not None
