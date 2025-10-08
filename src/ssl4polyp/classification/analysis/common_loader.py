from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, DefaultDict, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .result_loader import ResultLoader

__all__ = [
    "CommonFrame",
    "CommonRun",
    "get_default_loader",
    "load_common_run",
    "load_outputs_csv",
]


@dataclass(frozen=True)
class CommonFrame:
    frame_id: str
    case_id: str
    prob: float
    label: int
    pred: int
    row: Mapping[str, Any]


@dataclass
class CommonRun:
    model: str
    seed: int
    tau: float
    metrics_path: Path
    outputs_path: Path
    payload: Mapping[str, Any]
    provenance: Mapping[str, Any]
    primary_metrics: Dict[str, float]
    frames: Tuple[CommonFrame, ...]
    cases: Dict[str, Tuple[CommonFrame, ...]]


def get_default_loader(
    *,
    strict: bool = True,
    primary_policy: str | None = "f1_opt_on_val",
    sensitivity_policy: str | None = "youden_on_val",
    require_sensitivity: bool = True,
    required_curve_keys: Sequence[str] = (),
) -> ResultLoader:
    """Return a :class:`ResultLoader` configured for the reporting contract.

    Experiments share the same guardrails but differ in the threshold policies
    that should be present in each ``metrics.json`` payload. Allowing callers to
    override the policies keeps the validation logic centralised while making
    the expectations explicit at the call-site.
    """

    return ResultLoader(
        expected_primary_policy=primary_policy,
        expected_sensitivity_policy=sensitivity_policy,
        require_sensitivity=require_sensitivity,
        required_curve_keys=tuple(required_curve_keys),
        strict=strict,
    )


def load_common_run(
    metrics_path: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> CommonRun:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    normalised_payload = ResultLoader.normalise_payload(payload)
    active_loader = loader or get_default_loader()
    active_loader.validate(metrics_path, normalised_payload)
    provenance_block = normalised_payload.get("provenance")
    provenance = dict(provenance_block) if isinstance(provenance_block, Mapping) else {}
    model_name = _clean_text(provenance.get("model")) or _infer_model_from_filename(metrics_path)
    seed_value = _resolve_seed(normalised_payload, provenance, metrics_path)
    primary_metrics = _extract_metrics(normalised_payload.get("test_primary"))
    tau_value = primary_metrics.get("tau")
    if tau_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' is missing test_primary.tau")
    outputs_path = _resolve_outputs_path(metrics_path)
    frames, cases = load_outputs_csv(outputs_path, tau=float(tau_value))
    return CommonRun(
        model=model_name,
        seed=int(seed_value),
        tau=float(tau_value),
        metrics_path=metrics_path,
        outputs_path=outputs_path,
        payload=MappingProxyType(dict(normalised_payload)),
        provenance=MappingProxyType(dict(provenance)),
        primary_metrics=dict(primary_metrics),
        frames=frames,
        cases=cases,
    )


def load_outputs_csv(
    outputs_path: Path,
    *,
    tau: float,
) -> Tuple[Tuple[CommonFrame, ...], Dict[str, Tuple[CommonFrame, ...]]]:
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing test outputs CSV: {outputs_path}")
    frames: list[CommonFrame] = []
    cases: DefaultDict[str, list[CommonFrame]] = defaultdict(list)
    with outputs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            row_data: Dict[str, Any] = {key: value for key, value in row.items()}
            prob = _coerce_float(row_data.get("prob"))
            label = _coerce_int(row_data.get("label"))
            if prob is None or label is None:
                continue
            pred = _coerce_int(row_data.get("pred"))
            if pred is None:
                pred = 1 if float(prob) >= float(tau) else 0
            case_id = _normalise_case_id(row_data.get("case_id") or row_data.get("sequence_id"), index)
            frame_id = _normalise_frame_id(row_data.get("frame_id"), index)
            frame = CommonFrame(
                frame_id=frame_id,
                case_id=case_id,
                prob=float(prob),
                label=int(label),
                pred=int(pred),
                row=MappingProxyType(dict(row_data)),
            )
            frames.append(frame)
            cases[case_id].append(frame)
    if not frames:
        raise ValueError(f"No evaluation rows parsed from {outputs_path}")
    return tuple(frames), {case: tuple(items) for case, items in cases.items()}


def _extract_metrics(block: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    metrics: Dict[str, float] = {}
    for key, value in block.items():
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[str(key)] = float(numeric)
    return metrics


def _resolve_outputs_path(metrics_path: Path) -> Path:
    stem = metrics_path.stem
    base = stem[:-5] if stem.endswith("_last") else stem
    return metrics_path.with_name(f"{base}_test_outputs.csv")


def _infer_model_from_filename(metrics_path: Path) -> str:
    stem = metrics_path.stem
    if stem.endswith("_last"):
        stem = stem[:-5]
    model = stem.split("__", 1)[0]
    return model


def _resolve_seed(
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
    metrics_path: Path,
) -> int:
    for candidate in (
        _coerce_int(payload.get("seed")),
        _coerce_int(provenance.get("train_seed")),
        _seed_from_stem(metrics_path.stem),
    ):
        if candidate is not None:
            return int(candidate)
    raise ValueError(f"Metrics file '{metrics_path}' does not specify a seed")


def _seed_from_stem(stem: str) -> Optional[int]:
    match = re.search(r"_s(\d+)$", stem)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _normalise_case_id(raw: Optional[object], index: int) -> str:
    text = _clean_text(raw)
    if text:
        return text
    return f"case_{index}"


def _normalise_frame_id(raw: Optional[object], index: int) -> str:
    text = _clean_text(raw)
    if text:
        return text
    return f"frame_{index}"


def _clean_text(value: Optional[object]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None

