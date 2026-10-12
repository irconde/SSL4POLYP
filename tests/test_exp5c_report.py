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

try:  # Use the real scikit-learn stack when available.
    import sklearn.metrics as _sklearn_metrics  # noqa: F401
    import sklearn.exceptions as _sklearn_exceptions  # noqa: F401
except Exception:  # pragma: no cover - exercised only without scikit-learn
    sklearn_stub = ModuleType("sklearn")
    sklearn_metrics_stub = ModuleType("sklearn.metrics")

    def _unavailable(*args, **kwargs):
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

    class UndefinedMetricWarning(UserWarning):
        """Stand-in matching sklearn.exceptions.UndefinedMetricWarning."""

    sklearn_exceptions_stub = ModuleType("sklearn.exceptions")
    sklearn_exceptions_stub.UndefinedMetricWarning = UndefinedMetricWarning
    sklearn_stub.exceptions = sklearn_exceptions_stub

    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = sklearn_metrics_stub
    sys.modules["sklearn.exceptions"] = sklearn_exceptions_stub

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
        "dataset": {
            "test_primary": {"pack": "polypgen_fewshot_s50", "subset_seed": 13},
            "train": {"subset_seed": 13},
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
            "fewshot_budget": 50,
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


def test_build_cluster_set_prefers_center_for_positives_and_sequence_for_negatives() -> None:
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
            case_id="caseB",
            prob=0.85,
            label=1,
            pred=1,
            row=MappingProxyType({"case_id": "caseB", "center_id": "center2"}),
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
    # Negatives cluster by sequence first, then center; frames without either key
    # fall back to per-frame singletons.
    assert negative_clusters == {
        frozenset({"neg_center_pref_1"}),
        frozenset({"neg_center_pref_2"}),
        frozenset({"neg_sequence_fallback_1", "neg_sequence_fallback_2"}),
        frozenset({"neg_case_fallback_1"}),
        frozenset({"neg_case_fallback_2"}),
    }


def _make_shot_run(
    model: str,
    seed: int,
    budget: int,
    *,
    auprc: float,
    f1: float,
) -> "exp5c_report.FewShotRun":
    return exp5c_report.FewShotRun(
        model=model,
        seed=seed,
        budget=budget,
        tau=0.5,
        sensitivity_tau=None,
        primary_metrics={"auprc": auprc, "f1": f1},
        primary_counts={},
        sensitivity_metrics={},
        sensitivity_counts={},
        val_metrics={},
        provenance={},
        dataset={},
        frames={},
        zero_shot=None,
        path=Path("dummy"),
    )


def _make_runs_by_model(per_model_auprc: dict[str, float]):
    runs: dict[str, dict[int, dict[int, "exp5c_report.FewShotRun"]]] = {}
    for model, base in per_model_auprc.items():
        per_budget: dict[int, dict[int, "exp5c_report.FewShotRun"]] = {}
        for budget in (50, 100):
            seed_map: dict[int, "exp5c_report.FewShotRun"] = {}
            for seed in exp5c_report.EXPECTED_SEEDS:
                seed_map[seed] = _make_shot_run(
                    model, seed, budget, auprc=base + seed * 0.0001, f1=base
                )
            per_budget[budget] = seed_map
        runs[model] = per_budget
    return runs


def test_pairwise_deltas_target_model_selects_baselines() -> None:
    runs = _make_runs_by_model(
        {"ssl_colon": 0.8, "ssl_imnet": 0.7, "sup_imnet": 0.6}
    )
    pairwise = exp5c_report._compute_pairwise_deltas(
        runs,
        metric="auprc",
        bootstrap=0,
        rng_seed=1,
        expected_seeds=exp5c_report.EXPECTED_SEEDS,
        policy="primary",
    )
    assert set(pairwise.keys()) == {"ssl_imnet", "sup_imnet"}

    pairwise_imnet = exp5c_report._compute_pairwise_deltas(
        runs,
        metric="auprc",
        bootstrap=0,
        rng_seed=1,
        expected_seeds=exp5c_report.EXPECTED_SEEDS,
        policy="primary",
        target_model="ssl_imnet",
    )
    assert set(pairwise_imnet.keys()) == {"ssl_colon", "sup_imnet"}


def test_aulc_deltas_target_model_selects_baselines() -> None:
    runs = _make_runs_by_model(
        {"ssl_colon": 0.8, "ssl_imnet": 0.7, "sup_imnet": 0.6}
    )
    aulc = exp5c_report._compute_aulc_deltas(
        runs,
        metric="auprc",
        bootstrap=0,
        rng_seed=1,
        expected_seeds=exp5c_report.EXPECTED_SEEDS,
        policy="primary",
    )
    assert set(aulc.keys()) == {"ssl_imnet", "sup_imnet"}

    aulc_imnet = exp5c_report._compute_aulc_deltas(
        runs,
        metric="auprc",
        bootstrap=0,
        rng_seed=1,
        expected_seeds=exp5c_report.EXPECTED_SEEDS,
        policy="primary",
        target_model="ssl_imnet",
    )
    assert set(aulc_imnet.keys()) == {"ssl_colon", "sup_imnet"}
