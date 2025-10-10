from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ssl4polyp.classification.analysis.result_loader import GuardrailViolation, ResultLoader

_EXP_IDS = ("exp1", "exp2", "exp3b", "exp4", "exp5a", "exp5b", "exp5c")


def _base_payload(exp_id: str) -> dict[str, object]:
    val_path = "sun_full/val.csv"
    payload: dict[str, object] = {
        "seed": 13,
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": val_path, "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "val": {
            "loss": 0.25,
            "auroc": 0.85,
            "auprc": 0.75,
        },
        "test_primary": {
            "tau": 0.4,
            "tp": 10,
            "fp": 2,
            "tn": 20,
            "fn": 3,
            "n_pos": 13,
            "n_neg": 22,
        },
        "thresholds": {},
    }
    thresholds: dict[str, object] = payload["thresholds"]  # type: ignore[assignment]
    if exp_id in {"exp5a", "exp5b", "exp5c"}:
        thresholds["primary"] = {
            "policy": "sun_val_frozen",
            "tau": 0.4,
            "source_checkpoint": "ckpt.pt",
            "source_split": "sun_full/val",
        }
    else:
        thresholds["primary"] = {
            "policy": "f1_opt_on_val",
            "tau": 0.4,
            "split": val_path,
            "epoch": 5,
        }
    if exp_id in {"exp1", "exp2", "exp3b", "exp4", "exp5c"}:
        payload["test_sensitivity"] = {
            "tau": 0.3,
            "tp": 12,
            "fp": 4,
            "tn": 18,
            "fn": 1,
            "n_pos": 13,
            "n_neg": 22,
        }
        sensitivity_policy = "youden_on_val"
        if exp_id == "exp5c":
            sensitivity_policy = "val_opt_youden"
        thresholds["sensitivity"] = {
            "policy": sensitivity_policy,
            "tau": 0.3,
            "split": val_path,
            "epoch": 5,
        }
    return payload


@pytest.mark.parametrize("exp_id", _EXP_IDS)
def test_loader_accepts_valid_payload(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = _base_payload(exp_id)
    validated = loader.validate(Path("metrics.json"), payload)
    assert isinstance(validated, dict)


@pytest.mark.parametrize("exp_id", _EXP_IDS)
def test_loader_rejects_wrong_policy(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    payload["thresholds"]["primary"]["policy"] = "wrong"
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", ["exp1", "exp2", "exp3b", "exp4"])
def test_loader_requires_epoch(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    payload["thresholds"]["primary"].pop("epoch", None)
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", ["exp1", "exp2", "exp3b", "exp4"])
def test_loader_rejects_wrong_split(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    payload["thresholds"]["primary"]["split"] = "other"
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", ["exp1", "exp2", "exp3b", "exp4", "exp5c"])
def test_loader_requires_sensitivity_epoch(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    if "sensitivity" not in payload["thresholds"]:
        pytest.skip("Experiment does not define sensitivity thresholds")
    payload["thresholds"]["sensitivity"].pop("epoch", None)
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", ["exp1", "exp2", "exp3b", "exp4", "exp5c"])
def test_loader_rejects_wrong_sensitivity_split(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    if "sensitivity" not in payload["thresholds"]:
        pytest.skip("Experiment does not define sensitivity thresholds")
    payload["thresholds"]["sensitivity"]["split"] = "other"
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", ["exp5a", "exp5b", "exp5c"])
def test_loader_requires_frozen_sources(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    payload["thresholds"]["primary"].pop("source_checkpoint")
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", ["exp5a", "exp5b", "exp5c"])
def test_loader_rejects_wrong_source_split(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    payload["thresholds"]["primary"]["source_split"] = "sun_morphology/val"
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)


@pytest.mark.parametrize("exp_id", _EXP_IDS)
def test_loader_rejects_eval_blocks(exp_id: str) -> None:
    loader = ResultLoader(exp_id=exp_id)
    payload = copy.deepcopy(_base_payload(exp_id))
    payload["eval_primary"] = {}
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)
