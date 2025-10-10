import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sklearn_stub = ModuleType("sklearn")
sklearn_metrics_stub = ModuleType("sklearn.metrics")


def _unavailable(*args, **kwargs):  # pragma: no cover - safety guard
    raise RuntimeError("scikit-learn functionality unavailable in tests")


for _name in (
    "average_precision_score",
    "balanced_accuracy_score",
    "f1_score",
    "matthews_corrcoef",
    "precision_score",
    "precision_recall_curve",
    "recall_score",
    "roc_auc_score",
    "roc_curve",
):
    setattr(sklearn_metrics_stub, _name, _unavailable)

sklearn_stub.metrics = sklearn_metrics_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.metrics", sklearn_metrics_stub)

from ssl4polyp.classification.analysis import exp5c_report  # type: ignore[import]
from ssl4polyp.classification.analysis.common_loader import CommonFrame


def _write_metrics(path: Path) -> None:
    payload = {
        "seed": 13,
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": "sun_full/val.csv", "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "test_primary": {
            "tau": 0.5,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
            "prevalence": 0.5,
        },
        "test_sensitivity": {
            "tau": 0.4,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
            "prevalence": 0.5,
        },
        "val": {"auprc": 0.6, "auroc": 0.7, "loss": 0.4},
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.5,
                "source_split": "sun_full/val",
                "source_checkpoint": "checkpoint.pt",
            },
            "sensitivity": {
                "policy": "val_opt_youden",
                "tau": 0.4,
                "split": "sun_full/val.csv",
                "epoch": 3,
            },
        },
        "provenance": {
            "model": "ssl_colon",
            "train_seed": 13,
            "pack_seed": 29,
            "split": "test",
            "test_outputs_csv_sha256": "deadbeef",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_runs_errors_when_outputs_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    metrics_path = runs_root / "ssl_colon_s13_last.metrics.json"
    _write_metrics(metrics_path)

    with pytest.raises(RuntimeError) as excinfo:
        exp5c_report.discover_runs(runs_root)

    assert "missing per-frame outputs" in str(excinfo.value)


def test_build_cluster_set_prefers_center_then_sequence_then_case() -> None:
    frames = [
        CommonFrame(
            frame_id="pos_center_cluster_1",
            case_id="caseA",
            prob=0.9,
            label=1,
            pred=1,
            row=MappingProxyType({"case_id": "caseA", "center_id": "center1"}),
        ),
        CommonFrame(
            frame_id="pos_center_cluster_2",
            case_id="caseA",
            prob=0.85,
            label=1,
            pred=1,
            row=MappingProxyType({"case_id": "caseA", "center_id": "center1"}),
        ),
        CommonFrame(
            frame_id="neg_center_pref_1",
            case_id="caseX",
            prob=0.2,
            label=0,
            pred=0,
            row=MappingProxyType(
                {"case_id": "caseX", "center_id": "center1", "sequence_id": "seqX"}
            ),
        ),
        CommonFrame(
            frame_id="neg_center_pref_2",
            case_id="caseY",
            prob=0.25,
            label=0,
            pred=0,
            row=MappingProxyType(
                {"case_id": "caseY", "center_id": "center1", "sequence_id": "seqY"}
            ),
        ),
        CommonFrame(
            frame_id="neg_sequence_fallback_1",
            case_id="caseZ",
            prob=0.3,
            label=0,
            pred=0,
            row=MappingProxyType({"case_id": "caseZ", "sequence_id": "seq_shared"}),
        ),
        CommonFrame(
            frame_id="neg_sequence_fallback_2",
            case_id="caseW",
            prob=0.35,
            label=0,
            pred=0,
            row=MappingProxyType({"case_id": "caseW", "sequence_id": "seq_shared"}),
        ),
        CommonFrame(
            frame_id="neg_case_fallback_1",
            case_id="caseOnly",
            prob=0.4,
            label=0,
            pred=0,
            row=MappingProxyType({}),
        ),
        CommonFrame(
            frame_id="neg_case_fallback_2",
            case_id="caseOnly",
            prob=0.45,
            label=0,
            pred=0,
            row=MappingProxyType({}),
        ),
    ]
    eval_frames = exp5c_report._frames_to_eval(frames)
    run = exp5c_report.FewShotRun(
        model="ssl_colon",
        seed=13,
        budget=1,
        tau=0.5,
        sensitivity_tau=None,
        primary_metrics={},
        primary_counts={},
        sensitivity_metrics={},
        sensitivity_counts={},
        val_metrics={},
        provenance={},
        dataset={},
        frames=eval_frames,
        zero_shot=None,
        path=Path("dummy"),
    )
    clusters = exp5c_report._build_cluster_set(run)

    positive_clusters = {frozenset(cluster) for cluster in clusters.positives}
    assert positive_clusters == {
        frozenset({"pos_center_cluster_1"}),
        frozenset({"pos_center_cluster_2"}),
    }

    negative_clusters = {frozenset(cluster) for cluster in clusters.negatives}
    assert negative_clusters == {
        frozenset({"neg_center_pref_1", "neg_center_pref_2"}),
        frozenset({"neg_sequence_fallback_1", "neg_sequence_fallback_2"}),
        frozenset({"neg_case_fallback_1", "neg_case_fallback_2"}),
    }
