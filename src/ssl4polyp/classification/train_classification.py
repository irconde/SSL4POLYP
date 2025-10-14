from __future__ import annotations

import csv
import distutils.version
import sys
import os
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import time
import copy
import warnings
from pathlib import Path
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import yaml

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ssl4polyp.utils.tensorboard import SummaryWriter

from ssl4polyp import utils
from ssl4polyp.classification.data import (
    PackDataset,
    create_classification_dataloaders,
)
from ssl4polyp.classification.finetune import (
    collect_finetune_param_groups,
    configure_finetune_parameters,
    normalise_finetune_mode,
)
from ssl4polyp.classification.metrics import performance, thresholds
from ssl4polyp.configs import data_packs_root
from ssl4polyp.configs.layered import (
    extract_dataset_config,
    load_layered_config,
    resolve_model_entries,
)

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


EVAL_MAX_ADDITIONAL_BATCHES = 3


PERTURBATION_METADATA_FIELDS: Tuple[str, ...] = (
    "perturbation_id",
    "blur_sigma",
    "jpeg_q",
    "brightness",
    "contrast",
    "bbox_area_frac",
)


def _is_placeholder_perturbation_value(value: Any) -> bool:
    """Return ``True`` when ``value`` represents a placeholder sentinel."""

    if value in (None, ""):
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) == -1
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return True
        return numeric == -1.0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True
        lowered = text.lower()
        if lowered in {"nan", "none", "null"}:
            return True
        try:
            numeric = float(text)
        except ValueError:
            return False
        if not math.isfinite(numeric):
            return True
        return numeric == -1.0
    return False


@dataclass
class TrainEpochStats:
    """Aggregate statistics returned by :func:`train_epoch`."""

    mean_loss: float
    global_step: int
    samples_processed: int
    batches_processed: int


@dataclass
class ParentRunReference:
    """Metadata describing the canonical parent run used for evaluation."""

    checkpoint_path: Path
    checkpoint_sha256: Optional[str]
    metrics_path: Optional[Path]
    metrics_payload: Optional[Dict[str, Any]]
    metrics_sha256: Optional[str]
    outputs_path: Optional[Path]
    outputs_sha256: Optional[str]


def _compute_file_sha256(path: Path) -> Optional[str]:
    """Return the SHA256 digest for ``path`` if it exists."""

    try:
        with path.open("rb") as handle:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                hasher.update(chunk)
            return hasher.hexdigest()
    except FileNotFoundError:
        return None


def _record_test_outputs_digest(args, outputs_path: Optional[Path]) -> None:
    """Compute and cache the SHA-256 digest for the exported test outputs."""

    if outputs_path is None:
        return
    path = Path(outputs_path).expanduser()
    setattr(args, "latest_test_outputs_path", path)
    sha256 = _compute_file_sha256(path)
    if sha256:
        setattr(args, "latest_test_outputs_sha256", sha256)
    else:
        setattr(args, "latest_test_outputs_sha256", None)


def _safe_relpath(path: Path, base: Path) -> str:
    """Return a stable string path relative to ``base`` when possible."""

    try:
        return str(path.relative_to(base))
    except ValueError:
        try:
            return os.path.relpath(path, base)
        except ValueError:
            return str(path)


def _infer_metrics_candidates(checkpoint_path: Path) -> Sequence[Path]:
    """Return possible metrics JSON paths corresponding to ``checkpoint_path``."""

    stem = checkpoint_path.with_suffix("")
    candidates = [
        stem.with_suffix(".metrics.json"),
        stem.parent / f"{stem.name}_last.metrics.json",
        stem.parent / f"{stem.name}_best.metrics.json",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _infer_outputs_path_from_metrics(metrics_path: Path) -> Path:
    stem = metrics_path.stem
    if stem.endswith("_last"):
        base = stem[:-5]
    else:
        base = stem
    return metrics_path.with_name(f"{base}_test_outputs.csv")


def _load_reference_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        warnings.warn(
            f"Unable to parse parent metrics JSON at {path}; continuing without reference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _resolve_parent_reference(checkpoint_path: Path) -> ParentRunReference:
    """Resolve canonical SUN provenance for evaluation-only experiments."""

    checkpoint_path = checkpoint_path.expanduser()
    checkpoint_sha = _compute_file_sha256(checkpoint_path)
    metrics_path: Optional[Path] = None
    metrics_payload: Optional[Dict[str, Any]] = None
    metrics_sha: Optional[str] = None
    outputs_path: Optional[Path] = None
    outputs_sha: Optional[str] = None

    for candidate in _infer_metrics_candidates(checkpoint_path):
        if candidate.exists():
            metrics_path = candidate
            metrics_payload = _load_reference_payload(candidate)
            metrics_sha = _compute_file_sha256(candidate)
            outputs_candidate = _infer_outputs_path_from_metrics(candidate)
            if outputs_candidate.exists():
                outputs_path = outputs_candidate
                outputs_sha = _compute_file_sha256(outputs_candidate)
            break

    return ParentRunReference(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        metrics_path=metrics_path,
        metrics_payload=metrics_payload,
        metrics_sha256=metrics_sha,
        outputs_path=outputs_path,
        outputs_sha256=outputs_sha,
    )


@dataclass
class Experiment4SubsetTrace:
    """Metadata describing SUN subset selections for experiment 4 logging."""

    percent: Optional[int]
    seed: Optional[int]
    train_pos_cases: int
    train_neg_cases: int
    frames_per_case: Optional[int]
    total_frames: int
    pos_case_ids: Tuple[str, ...]
    neg_case_ids: Tuple[str, ...]
    pos_digest: str
    neg_digest: str
    manifest: Optional[str]


@dataclass(frozen=True)
class EvalLoggingContext:
    """Metadata describing how to log evaluation progress for a dataset."""

    tag: str
    start_lines: Tuple[str, ...]
    sample_display: str

    @property
    def prefix(self) -> str:
        return f"[eval: {self.tag}]"


@dataclass(frozen=True)
class FinetuneStage:
    """Description of a single fine-tuning stage in a multi-phase schedule."""

    index: int
    start_epoch: int
    end_epoch: int
    mode: str
    head_lr: float
    backbone_lr: float
    base_lr: float
    backbone_lr_scale: Optional[float]
    label: Optional[str] = None

    @property
    def epochs(self) -> int:
        return self.end_epoch - self.start_epoch + 1


def _resolve_case_identifier(row: Mapping[str, Any]) -> Optional[str]:
    """Extract a case identifier from metadata ``row`` if available."""

    if not isinstance(row, Mapping):
        return None
    candidate_keys = (
        "case_id",
        "caseid",
        "case",
        "sequence_id",
        "seq_id",
        "study_id",
    )
    for key in candidate_keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _compute_case_digest(case_ids: Sequence[str]) -> str:
    """Return a short SHA1 digest summarising ``case_ids``."""

    if not case_ids:
        return "n/a"
    hasher = hashlib.sha1()
    for case_id in case_ids:
        hasher.update(str(case_id).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:8]


def _summarize_case_counts(dataset: "PackDataset") -> Optional[Dict[str, int]]:
    """Return positive/negative case counts when metadata provides case IDs."""

    labels = getattr(dataset, "labels_list", None)
    metadata = getattr(dataset, "metadata", None)
    if labels is None or metadata is None:
        return None
    case_frames: Dict[str, int] = {}
    case_labels: Dict[str, int] = {}
    for label, row in zip(labels, metadata):
        case_id = _resolve_case_identifier(row)
        if not case_id:
            continue
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        case_frames[case_id] = case_frames.get(case_id, 0) + 1
        case_labels.setdefault(case_id, label_int)
    if not case_frames:
        return None
    pos_cases = sum(1 for cid, lbl in case_labels.items() if int(lbl) == 1)
    neg_cases = sum(1 for cid, lbl in case_labels.items() if int(lbl) == 0)
    total_frames = int(sum(case_frames.values()))
    return {
        "pos_cases": int(pos_cases),
        "neg_cases": int(neg_cases),
        "total_frames": total_frames,
    }


def _summarize_frame_counts(labels: Optional[Sequence[object]]) -> Dict[int, int]:
    """Return frame counts keyed by class label."""

    counts: Counter[int] = Counter()
    if labels is None:
        return {}
    for label in labels:
        try:
            counts[int(label)] += 1
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return {int(key): int(value) for key, value in counts.items()}


def _format_frame_summary(total_frames: int, counts: Mapping[int, int]) -> str:
    """Format ``counts`` into a human-readable frame summary string."""

    if counts and set(counts.keys()).issubset({0, 1}):
        pos = counts.get(1, 0)
        neg = counts.get(0, 0)
        return f"frames(total={total_frames}, pos={pos}, neg={neg})"
    if counts:
        per_class = ", ".join(
            f"{label}:{counts.get(label, 0)}" for label in sorted(counts.keys())
        )
        return f"frames(total={total_frames}, per-class={{ {per_class} }})"
    return f"frames(total={total_frames})"


def _format_numeric_token(value: Any) -> str:
    """Return a compact textual representation of ``value`` for tagging."""

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return str(numeric)
        formatted = f"{numeric:.4f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return str(value)


def _canonicalize_perturbation_tag(row: Mapping[str, Any]) -> Optional[str]:
    """Normalise ``row`` metadata into a canonical perturbation tag."""

    if not isinstance(row, Mapping):
        return None
    candidate = row.get("perturbation_id")
    if not _is_placeholder_perturbation_value(candidate):
        text = str(candidate).strip()
        if text:
            return text
    # Fall back to composing from known manifest fields.
    components: list[str] = []
    numeric_fields = (
        ("blur_sigma", "blur_sigma"),
        ("jpeg_q", "jpeg_q"),
        ("brightness", "brightness"),
        ("contrast", "contrast"),
        ("bbox_area_frac", "bbox_area_frac"),
    )
    for field, label in numeric_fields:
        value = row.get(field)
        if _is_placeholder_perturbation_value(value):
            continue
        components.append(f"{label}={_format_numeric_token(value)}")
    if components:
        return "|".join(components)
    variant = row.get("variant")
    if not _is_placeholder_perturbation_value(variant):
        text = str(variant).strip()
        if text:
            return text
    return None


def _dataset_supports_perturbations(dataset: Optional["PackDataset"]) -> bool:
    """Return ``True`` when ``dataset`` exposes perturbation metadata."""

    if dataset is None:
        return False
    metadata = getattr(dataset, "metadata", None)
    if not metadata:
        return False
    max_rows_to_probe = 10
    for index, row in enumerate(metadata):
        if not isinstance(row, Mapping):
            continue
        if _canonicalize_perturbation_tag(row):
            return True
        for field in PERTURBATION_METADATA_FIELDS:
            value = row.get(field)
            if not _is_placeholder_perturbation_value(value):
                return True
        if index + 1 >= max_rows_to_probe:
            break
    return False


def _format_top_perturbation_summary(
    counter: Mapping[str, int], limit: int = 3
) -> Optional[str]:
    """Return a compact summary of perturbation tag counts."""

    if not counter:
        return None
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    display_items = ordered[:limit]
    remainder = max(0, len(ordered) - len(display_items))
    parts = [f"{tag}×{count}" for tag, count in display_items]
    summary = ", ".join(parts)
    if remainder > 0:
        summary += f" (+{remainder} more)"
    return f"tags: {summary}"


def _coerce_optional_float(value: Any, *, context: str) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {context}: {value!r}") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f"Non-finite value for {context}: {value!r}")
    return float(numeric)


def _sanitize_finetune_schedule_config(
    raw_schedule: Any,
    *,
    default_mode: str,
) -> list[dict[str, Any]]:
    if raw_schedule in (None, False):
        return []
    if not isinstance(raw_schedule, (list, tuple)):
        raise TypeError(
            "Fine-tune schedule must be a list of stage dictionaries."
        )
    sanitized: list[dict[str, Any]] = []
    previous_mode = default_mode
    for index, entry in enumerate(raw_schedule):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Schedule entry #{index + 1} must be a mapping; received {type(entry)!r}."
            )
        stage_mode_raw = entry.get("mode", previous_mode)
        stage_mode = normalise_finetune_mode(stage_mode_raw, default=previous_mode)
        epochs_value = entry.get("epochs")
        if epochs_value is None:
            raise ValueError(
                f"Schedule entry #{index + 1} is missing required key 'epochs'."
            )
        try:
            epochs_int = int(epochs_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Schedule entry #{index + 1} provided non-integer epochs {epochs_value!r}."
            ) from exc
        if epochs_int <= 0:
            raise ValueError(
                f"Schedule entry #{index + 1} must have a positive epoch count; received {epochs_int}."
            )
        sanitized.append(
            {
                "index": int(index),
                "mode": stage_mode,
                "epochs": epochs_int,
                "lr": _coerce_optional_float(entry.get("lr"), context=f"schedule entry #{index + 1} lr"),
                "head_lr": _coerce_optional_float(
                    entry.get("head_lr"), context=f"schedule entry #{index + 1} head_lr"
                ),
                "backbone_lr": _coerce_optional_float(
                    entry.get("backbone_lr"), context=f"schedule entry #{index + 1} backbone_lr"
                ),
                "backbone_lr_scale": _coerce_optional_float(
                    entry.get("backbone_lr_scale"),
                    context=f"schedule entry #{index + 1} backbone_lr_scale",
                ),
                "name": entry.get("name"),
            }
        )
        previous_mode = stage_mode
    return sanitized


def _sanitize_curve_export_config(raw_config: Any) -> Dict[str, Dict[str, int]]:
    """Normalise curve export configuration into a per-split map."""

    default_points = 200
    if raw_config in (None, False):
        return {}

    def _coerce_points(value: Any) -> int:
        if value in (None, ""):
            return default_points
        try:
            points = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Curve export grid points must be an integer; received {value!r}."
            ) from exc
        if points < 2:
            raise ValueError("Curve export grid must contain at least two points.")
        return points

    spec: Dict[str, Dict[str, int]] = {}
    if raw_config is True:
        spec["test"] = {"points": default_points}
        return spec

    if isinstance(raw_config, str):
        split = raw_config.strip().lower()
        if split:
            spec[split] = {"points": default_points}
        return spec

    if isinstance(raw_config, Iterable) and not isinstance(raw_config, Mapping):
        for entry in raw_config:
            if entry in (None, ""):
                continue
            spec[str(entry).strip().lower()] = {"points": default_points}
        if spec:
            return spec

    if isinstance(raw_config, Mapping):
        declared_default = raw_config.get("points")
        if declared_default is not None:
            default_points = _coerce_points(declared_default)
        splits_value = raw_config.get("splits")
        if splits_value:
            if isinstance(splits_value, (str, bytes)):
                splits_iter = [splits_value]
            else:
                splits_iter = splits_value
            for entry in splits_iter:
                if entry in (None, ""):
                    continue
                spec[str(entry).strip().lower()] = {"points": default_points}
        for key, value in raw_config.items():
            if key in {"points", "splits"}:
                continue
            split_key = str(key).strip().lower()
            if not split_key:
                continue
            if isinstance(value, Mapping):
                entry_points = _coerce_points(value.get("points"))
            elif isinstance(value, (int, float)):
                entry_points = _coerce_points(value)
            elif isinstance(value, bool):
                if not value:
                    continue
                entry_points = default_points
            elif value in (None, ""):
                entry_points = default_points
            else:
                raise TypeError(
                    f"Unsupported curve export specification for split '{key}': {value!r}"
                )
            spec[split_key] = {"points": entry_points}
        if not spec:
            spec["test"] = {"points": default_points}
        return spec

    raise TypeError(f"Unsupported curve export configuration: {raw_config!r}")


def _materialize_finetune_schedule(
    schedule_spec: Sequence[dict[str, Any]],
    *,
    base_lr: float,
) -> list[FinetuneStage]:
    if not schedule_spec:
        return []
    start_epoch = 1
    stages: list[FinetuneStage] = []
    for spec in schedule_spec:
        stage_lr = spec.get("lr")
        stage_base_lr = float(stage_lr) if stage_lr is not None else float(base_lr)
        head_lr = spec.get("head_lr")
        resolved_head_lr = (
            float(head_lr) if head_lr is not None else float(stage_base_lr)
        )
        backbone_lr = spec.get("backbone_lr")
        lr_scale = spec.get("backbone_lr_scale")
        stage_mode = str(spec.get("mode"))
        if backbone_lr is not None:
            resolved_backbone_lr = float(backbone_lr)
        elif stage_mode == "none":
            resolved_backbone_lr = 0.0
        elif lr_scale is not None:
            resolved_backbone_lr = float(stage_base_lr) * float(lr_scale)
        else:
            resolved_backbone_lr = float(stage_base_lr)
        epochs = int(spec["epochs"])
        end_epoch = start_epoch + epochs - 1
        stages.append(
            FinetuneStage(
                index=int(spec["index"]),
                start_epoch=int(start_epoch),
                end_epoch=int(end_epoch),
                mode=stage_mode,
                head_lr=float(resolved_head_lr),
                backbone_lr=float(resolved_backbone_lr),
                base_lr=float(stage_base_lr),
                backbone_lr_scale=float(lr_scale) if lr_scale is not None else None,
                label=spec.get("name"),
            )
        )
        start_epoch = end_epoch + 1
    return stages


class FinetuneScheduleRuntime:
    """Runtime helper that applies fine-tuning stages as training progresses."""

    def __init__(self, stages: Sequence[FinetuneStage]) -> None:
        self.stages: list[FinetuneStage] = list(stages)
        self._current_stage_index: Optional[int] = None

    def is_active(self) -> bool:
        return bool(self.stages)

    def stage_for_epoch(self, epoch: int) -> Optional[FinetuneStage]:
        for stage in self.stages:
            if stage.start_epoch <= epoch <= stage.end_epoch:
                return stage
        if not self.stages:
            return None
        return self.stages[-1]

    def apply_if_needed(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        *,
        rank: int = 0,
    ) -> Optional[FinetuneStage]:
        stage = self.stage_for_epoch(epoch)
        if stage is None:
            return None
        if self._current_stage_index == stage.index:
            return stage
        target_model = _unwrap_model(model)
        configure_finetune_parameters(target_model, stage.mode)
        for group in optimizer.param_groups:
            name = group.get("name")
            if name == "head":
                group["lr"] = stage.head_lr
            elif name == "backbone":
                group["lr"] = stage.backbone_lr
        if rank == 0:
            label_suffix = f" ({stage.label})" if stage.label else ""
            print(
                f"[finetune] Stage {stage.index + 1}{label_suffix}: "
                f"epochs {stage.start_epoch}-{stage.end_epoch} | mode={stage.mode} | "
                f"head_lr={stage.head_lr:.2e} | backbone_lr={stage.backbone_lr:.2e}"
            )
        self._current_stage_index = stage.index
        return stage


def _derive_eval_tag(
    split_alias: str,
    pack_spec: Optional[str],
    dataset_layout: Mapping[str, Any],
) -> Optional[str]:
    """Infer a dataset tag for evaluation logging."""

    candidates = [pack_spec, dataset_layout.get("name") if dataset_layout else None]
    base_tag: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            continue
        lower = str(candidate).lower()
        if "polypgen_clean_test" in lower:
            base_tag = "PolypGen-clean"
            break
        if "sun_full" in lower:
            base_tag = "SUN"
            break
    if base_tag is None:
        return None
    alias = split_alias.lower()
    if base_tag == "SUN":
        if alias in {"val", "validation"}:
            return "SUN-val"
        if alias in {"train", "training"}:
            return "SUN-train"
        return "SUN-test"
    return base_tag


def _build_eval_logging_context(
    *,
    split_alias: str,
    dataset: Optional["PackDataset"],
    pack_spec: Optional[str],
    dataset_layout: Mapping[str, Any],
) -> Optional[EvalLoggingContext]:
    """Construct logging context for evaluation over ``dataset`` if recognised."""

    if dataset is None:
        return None
    dataset_tag = _derive_eval_tag(split_alias, pack_spec, dataset_layout)
    if dataset_tag is None:
        return None

    total_frames = len(dataset)
    labels = getattr(dataset, "labels_list", None)
    frame_counts = _summarize_frame_counts(labels)
    frame_summary = _format_frame_summary(total_frames, frame_counts)
    start_lines: list[str] = []
    sample_display = frame_summary

    if dataset_tag.startswith("SUN"):
        case_summary = _summarize_case_counts(dataset)
        if case_summary is not None:
            pos_cases = case_summary["pos_cases"]
            neg_cases = case_summary["neg_cases"]
            case_frame_total = case_summary["total_frames"]
            case_frame_summary = _format_frame_summary(
                case_frame_total, frame_counts
            )
            start_lines.append(
                f"Composition: cases(pos={pos_cases}, neg={neg_cases}) | {case_frame_summary}"
            )
            sample_display = (
                f"cases pos={pos_cases} neg={neg_cases} | {case_frame_summary}"
            )
        else:
            start_lines.append(f"Composition: {frame_summary}")
            sample_display = frame_summary
    elif dataset_tag.startswith("PolypGen"):
        start_lines.append(f"Composition: {frame_summary}")
        sample_display = frame_summary
    else:  # pragma: no cover - defensive future-proofing
        start_lines.append(f"Composition: {frame_summary}")
        sample_display = frame_summary

    return EvalLoggingContext(
        tag=dataset_tag,
        start_lines=tuple(start_lines),
        sample_display=sample_display,
    )


def _prepare_eval_contexts(
    args,
    datasets: Mapping[str, "PackDataset"],
) -> Dict[str, EvalLoggingContext]:
    """Build evaluation logging contexts for recognised validation/test datasets."""

    dataset_layout = getattr(args, "dataset_layout", {}) or {}
    contexts: Dict[str, EvalLoggingContext] = {}

    val_split = getattr(args, "val_split", None)
    if val_split:
        val_dataset = datasets.get(val_split)
        context = _build_eval_logging_context(
            split_alias="val",
            dataset=val_dataset,
            pack_spec=getattr(args, "val_pack", None),
            dataset_layout=dataset_layout,
        )
        if context is not None:
            contexts["Val"] = context

    test_split = getattr(args, "test_split", None)
    if test_split:
        test_dataset = datasets.get(test_split)
        context = _build_eval_logging_context(
            split_alias="test",
            dataset=test_dataset,
            pack_spec=getattr(args, "test_pack", None),
            dataset_layout=dataset_layout,
        )
        if context is not None:
            contexts["Test"] = context

    return contexts


def _summarize_dataset_for_metrics(
    alias: str,
    split_name: Optional[str],
    dataset: Optional["PackDataset"],
    pack_spec: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return provenance and composition metadata for ``dataset`` suitable for JSON export."""

    if dataset is None:
        return None
    summary: Dict[str, Any] = {"alias": alias}
    if split_name:
        summary["split"] = split_name
    if pack_spec:
        summary["pack_spec"] = str(pack_spec)

    provenance = getattr(dataset, "provenance", None)
    if isinstance(provenance, Mapping):
        for key, value in provenance.items():
            if value in (None, ""):
                continue
            summary[key] = value

    total_frames = len(dataset)
    summary["frames"] = int(total_frames)
    labels = getattr(dataset, "labels_list", None)
    frame_counts = _summarize_frame_counts(labels)
    if frame_counts:
        summary["class_counts"] = {int(k): int(v) for k, v in frame_counts.items()}
        summary["n_total"] = int(sum(frame_counts.values()))
        if 1 in frame_counts:
            summary["n_pos"] = int(frame_counts[1])
        if 0 in frame_counts:
            summary["n_neg"] = int(frame_counts[0])
    else:
        summary["n_total"] = int(total_frames)

    case_summary = _summarize_case_counts(dataset)
    if isinstance(case_summary, Mapping) and case_summary:
        summary["case_counts"] = {key: int(value) for key, value in case_summary.items()}

    return summary


def _build_result_loader_data_block(
    dataset_summary: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Transform ``dataset_summary`` into the ``data`` layout required by ``ResultLoader``."""

    if not isinstance(dataset_summary, Mapping):
        return None

    data_block: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for split in ("train", "val", "test"):
        summary = dataset_summary.get(split)
        if not isinstance(summary, Mapping):
            raise RuntimeError(
                f"Dataset summary missing mapping for split '{split}'"
            )
        csv_path = summary.get("csv_path") or summary.get("path")
        csv_sha256 = summary.get("csv_sha256") or summary.get("sha256")
        if not csv_path or not csv_sha256:
            raise RuntimeError(
                f"Dataset summary for split '{split}' missing csv path/sha256"
            )
        entry: "OrderedDict[str, Any]" = OrderedDict()
        entry["path"] = str(csv_path)
        entry["sha256"] = str(csv_sha256)
        extra_summary = {
            key: value
            for key, value in summary.items()
            if key not in {"csv_path", "csv_sha256"}
        }
        if extra_summary:
            entry["summary"] = _convert_json_compatible(extra_summary)
        data_block[split] = dict(entry)

    if len(data_block) != 3:
        missing = sorted({"train", "val", "test"} - set(data_block))
        raise RuntimeError(
            "Dataset summary missing required splits for ResultLoader data block: "
            + ", ".join(missing)
        )

    return dict(data_block)


def _update_threshold_split(
    metadata: Optional[Mapping[str, Any]],
    *,
    split_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return a copy of ``metadata`` with its ``split`` field aligned to ``split_path``."""

    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        return dict(_convert_json_compatible(metadata))
    updated = dict(metadata)
    if split_path:
        updated["split"] = split_path
    return updated


def _collect_experiment4_trace(
    args,
    datasets: Mapping[str, "PackDataset"],
) -> Optional[Experiment4SubsetTrace]:
    """Build metadata used for experiment 4 logging if applicable."""

    layout = getattr(args, "dataset_layout", {}) or {}
    if str(layout.get("name", "")).lower() != "sun_subsets":
        return None

    train_split = getattr(args, "train_split", "train")
    train_dataset = datasets.get(train_split)
    if train_dataset is None:
        return None

    labels: Optional[Sequence[int]] = getattr(train_dataset, "labels_list", None)
    metadata: Optional[Sequence[MutableMapping[str, Any]]] = getattr(
        train_dataset, "metadata", None
    )
    if labels is None or metadata is None:
        return None

    case_frames: Dict[str, int] = {}
    case_labels: Dict[str, int] = {}
    for label, row in zip(labels, metadata):
        case_id = _resolve_case_identifier(row)
        if not case_id:
            continue
        case_frames[case_id] = case_frames.get(case_id, 0) + 1
        case_labels.setdefault(case_id, int(label))

    if not case_frames:
        return None

    pos_case_ids = sorted(
        case_id for case_id, lbl in case_labels.items() if int(lbl) == 1
    )
    neg_case_ids = sorted(
        case_id for case_id, lbl in case_labels.items() if int(lbl) == 0
    )

    frame_counts = {case_frames[cid] for cid in case_frames}
    frames_per_case = frame_counts.pop() if len(frame_counts) == 1 else None
    total_frames = int(sum(case_frames.values()))

    percent = layout.get("percent")
    percent_int = int(percent) if isinstance(percent, (int, float)) else None
    seed = layout.get("dataset_seed")
    seed_int = int(seed) if isinstance(seed, (int, float)) else None

    return Experiment4SubsetTrace(
        percent=percent_int,
        seed=seed_int,
        train_pos_cases=len(pos_case_ids),
        train_neg_cases=len(neg_case_ids),
        frames_per_case=frames_per_case,
        total_frames=total_frames,
        pos_case_ids=tuple(pos_case_ids),
        neg_case_ids=tuple(neg_case_ids),
        pos_digest=_compute_case_digest(pos_case_ids),
        neg_digest=_compute_case_digest(neg_case_ids),
        manifest=getattr(args, "train_pack", None),
    )


def _format_subset_summary_lines(trace: Experiment4SubsetTrace) -> List[str]:
    """Return human-readable lines describing subset selection."""

    percent_display = (
        f"{trace.percent}%" if trace.percent is not None else "unknown%"
    )
    seed_display = trace.seed if trace.seed is not None else "unknown"
    frames_display = (
        str(trace.frames_per_case) if trace.frames_per_case is not None else "n/a"
    )
    subset_size = trace.train_pos_cases + trace.train_neg_cases
    first_line = (
        "Subset percent={percent} | seed={seed} | cases(pos/neg)={pos}/{neg} "
        "(total={total}) | frames_per_case={frames} | total_frames={frames_total}"
    ).format(
        percent=percent_display,
        seed=seed_display,
        pos=trace.train_pos_cases,
        neg=trace.train_neg_cases,
        total=subset_size,
        frames=frames_display,
        frames_total=trace.total_frames,
    )
    second_line = (
        f"Case digests: pos={trace.pos_digest} ({trace.train_pos_cases}), "
        f"neg={trace.neg_digest} ({trace.train_neg_cases})"
    )
    third_line = f"Manifest: {trace.manifest}" if trace.manifest else "Manifest: n/a"
    return [first_line, second_line, third_line]


def _format_subset_tag(trace: Experiment4SubsetTrace) -> str:
    """Return a compact subset descriptor suitable for inline logging."""

    if trace.percent is None:
        percent_token = "percent=unknown"
    else:
        percent_token = f"percent={trace.percent}%"
    seed_token = (
        f"seed={trace.seed}" if trace.seed is not None else "seed=unknown"
    )
    cases_token = (
        f"cases(pos/neg)={trace.train_pos_cases}/{trace.train_neg_cases}"
    )
    digest_token = f"digests(pos/neg)={trace.pos_digest}/{trace.neg_digest}"
    return " | ".join((percent_token, seed_token, cases_token, digest_token))


def _resolve_grad_accum_steps(optimizer: torch.optim.Optimizer) -> int:
    """Return gradient accumulation steps configured on ``optimizer``."""

    raw_value = getattr(optimizer, "grad_accum_steps", 1)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 1
    return max(value, 1)


def _compute_optimizer_steps(global_step: int, grad_accum_steps: int) -> int:
    """Compute optimizer steps after accounting for gradient accumulation."""

    if grad_accum_steps <= 1:
        return int(global_step)
    return int(global_step) // int(grad_accum_steps)


def _append_log_lines(log_path: Union[str, Path], lines: Sequence[str]) -> None:
    """Append ``lines`` to ``log_path`` with robust error handling."""

    if not log_path:
        return
    path_obj = Path(log_path)
    try:
        with path_obj.open("a") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        warnings.warn(
            f"Failed to append to log file {path_obj}: {exc}",
            RuntimeWarning,
        )


def _log_lines(log_path: Path, lines: Sequence[str]) -> None:
    """Print ``lines`` and append them to ``log_path``."""

    for line in lines:
        print(line)
    _append_log_lines(log_path, lines)


def _format_eta(seconds: Optional[float]) -> str:
    """Format seconds as a zero-padded HH:MM:SS string."""

    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "--:--:--"
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _collect_param_group_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    """Return all learning rates from an optimizer's parameter groups."""

    lrs: list[float] = []
    for group in optimizer.param_groups:
        lr = group.get("lr")
        if lr is not None:
            lrs.append(float(lr))
    return lrs


def _format_lr_values(lrs: Iterable[float]) -> str:
    lrs = list(lrs)
    if not lrs:
        return "--"
    if len(lrs) == 1:
        return f"{lrs[0]:.3e}"
    return f"{min(lrs):.3e}-{max(lrs):.3e}"


def _compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> Optional[float]:
    total_sq = 0.0
    has_grad = False
    for param in parameters:
        if param is None or param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce()
            values = grad._values()
            param_norm = values.norm(2)
        else:
            param_norm = grad.norm(2)
        total_sq += float(param_norm * param_norm)
        has_grad = True
    if not has_grad:
        return None
    return math.sqrt(total_sq)


def _format_train_progress(
    *,
    epoch: int,
    sample_count: int,
    dataset_size: int,
    progress_pct: float,
    loss_value: float,
    loss_label: str,
    lrs: Iterable[float],
    throughput: float,
    eta_seconds: Optional[float],
    grad_norm: Optional[float],
    loss_scale: Optional[float],
    mem_allocated: Optional[float],
    mem_reserved: Optional[float],
    accum_step: int,
    accum_steps: int,
    elapsed_time: float,
) -> str:
    lr_display = _format_lr_values(lrs)
    throughput_display = (
        f"{throughput:.1f} samples/s" if throughput > 0 else "-- samples/s"
    )
    eta_display = _format_eta(eta_seconds)
    grad_display = f"{grad_norm:.2f}" if grad_norm is not None else "--"
    parts = [
        f"Train Epoch: {epoch} [{sample_count}/{dataset_size} ({progress_pct:.1f}%)]",
        f"{loss_label}: {loss_value:.6f}",
        f"LR: {lr_display}",
        f"Throughput: {throughput_display}",
        f"ETA: {eta_display}",
        f"GradNorm: {grad_display}",
        f"Elapsed: {elapsed_time:.1f}s",
    ]
    if loss_scale is not None:
        parts.append(f"Scale: {loss_scale:.1f}")
    if mem_allocated is not None and mem_reserved is not None:
        parts.append(f"GPU Mem: {mem_allocated:.1f}/{mem_reserved:.1f} MiB")
    if accum_steps > 1:
        parts.append(f"Accum: {accum_step}/{accum_steps}")
    else:
        parts.append("Accum: 1/1")
    return "\t".join(parts)


class TensorboardLogger:
    """Centralised TensorBoard logging with failure detection."""

    def __init__(self, writer: Optional[SummaryWriter]) -> None:
        self._writer = writer

    @classmethod
    def create(cls, log_dir: Optional[str]) -> "TensorboardLogger":
        if not log_dir:
            return cls(None)
        return cls(SummaryWriter(log_dir))

    def _is_enabled(self) -> bool:
        if self._writer is None:
            return False
        enabled = getattr(self._writer, "enabled", True)
        if not enabled:
            self._writer = None
            return False
        return True

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if not self._is_enabled():
            return
        try:
            assert self._writer is not None  # for type checkers
            self._writer.add_scalar(tag, value, step)
        except (OSError, IOError) as exc:
            warnings.warn(
                "Disabling TensorBoard logging after write failure: " f"{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.close()

    def log_metrics(self, prefix: str, metrics: Dict[str, Any], step: int) -> None:
        if not self._is_enabled():
            return
        for name, value in metrics.items():
            if name in {"logits", "probabilities", "targets"}:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.item()
            if isinstance(value, (int, float, np.floating)):
                self.log_scalar(f"{prefix}/{name}", float(value), step)

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        finally:
            self._writer = None

    def __bool__(self) -> bool:
        return self._is_enabled()


def _normalize_seeds(raw: Any) -> list[int]:
    """Normalize ``raw`` seed specifications into a list of integers."""

    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        raw = raw.replace(",", " ")
        entries = [item for item in raw.split() if item]
    elif isinstance(raw, Iterable):
        entries = list(raw)
    else:
        raise TypeError(f"Unsupported seed specification: {raw!r}")

    seeds: list[int] = []
    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, str):
            entry = entry.strip()
            if not entry:
                continue
        seeds.append(int(entry))
    return seeds


def _resolve_active_seed(args) -> int:
    """Determine the seed that should drive the current run."""

    cli_seed = getattr(args, "seed", None)
    if getattr(args, "_seed_explicit", False) and cli_seed is not None:
        return int(cli_seed)

    seeds = getattr(args, "seeds", None) or []
    if seeds:
        return int(seeds[0])

    config_seed = getattr(args, "config_seed", None)
    if config_seed is not None:
        return int(config_seed)

    if cli_seed is not None:
        return int(cli_seed)
    return 0


def _get_active_seed(args) -> int:
    """Return the resolved seed for use in downstream helpers."""

    return int(getattr(args, "active_seed", _resolve_active_seed(args)))


def set_determinism(seed: int) -> None:
    """Configure deterministic behavior for reproducibility.

    warn_only=True ensures operations without deterministic implementations
    raise a warning but continue running non-deterministically.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    utils.enable_deterministic_algorithms()
    print(f"Setting deterministic mode with seed {seed}")


def _resolve_device(rank: int) -> torch.device:
    """Return the appropriate device for ``rank``."""

    if torch.cuda.is_available():
        return torch.device("cuda", rank)
    return torch.device("cpu")


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from a possible DDP wrapper."""

    return model.module if isinstance(model, DDP) else model


def _prepare_metric_export(
    metrics: Dict[str, Any], drop: Optional[Iterable[str]] = None
) -> Dict[str, float | int]:
    """Convert ``metrics`` into a JSON-serialisable mapping of floats."""

    drop = set(drop or [])
    export: Dict[str, float | int] = {}
    for key, value in metrics.items():
        if key in drop:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.item()
        if isinstance(value, np.generic):
            value = float(value)
        if isinstance(value, (float, int)):
            export[key] = float(value)
    _augment_metric_export(export, metrics)
    return export


def _augment_metric_export(
    export: Dict[str, float | int], metrics: Mapping[str, Any]
) -> None:
    """Inject confusion counts and frame statistics into ``export`` when available."""

    threshold_metrics = metrics.get("threshold_metrics")
    if isinstance(threshold_metrics, Mapping):
        for key, value in threshold_metrics.items():
            if isinstance(value, (int, np.integer)):
                export[key] = int(value)
            elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
                export[key] = float(value)

    class_counts = metrics.get("class_counts")
    if isinstance(class_counts, Sequence):
        total = 0
        for idx, count in enumerate(class_counts):
            if isinstance(count, (int, np.integer)):
                count_int = int(count)
            elif isinstance(count, (float, np.floating)) and math.isfinite(float(count)):
                count_int = int(count)
            else:
                continue
            total += count_int
            if idx == 0:
                export.setdefault("n_neg", count_int)
            elif idx == 1:
                export.setdefault("n_pos", count_int)
        if total > 0:
            export.setdefault("n_total", total)


PRIMARY_METRIC_KEYS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "tpr",
    "tnr",
    "mcc",
    "loss",
    "tp",
    "fp",
    "tn",
    "fn",
    "n_pos",
    "n_neg",
    "n_total",
    "prevalence",
    "count",
)


RETENTION_METRIC_KEYS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "tpr",
    "tnr",
)


INTEGER_METRIC_KEYS: set[str] = {
    "tp",
    "fp",
    "tn",
    "fn",
    "n_pos",
    "n_neg",
    "n_total",
    "count",
}


def _coerce_metric_value(value: Any) -> Optional[float | int]:
    """Return a JSON-serialisable numeric representation of ``value`` if possible."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    return None


def _build_metric_block(
    metrics: Optional[Mapping[str, Any]], *, include_tau: bool = True
) -> Dict[str, Any]:
    """Select a stable subset of metrics for export."""

    if not metrics:
        return {}
    block: "OrderedDict[str, Any]" = OrderedDict()
    for key in PRIMARY_METRIC_KEYS:
        if key not in metrics:
            continue
        value = _coerce_metric_value(metrics.get(key))
        if value is None:
            continue
        if key in INTEGER_METRIC_KEYS:
            block[key] = int(value)
        else:
            block[key] = float(value)
    if include_tau and "tau" in metrics:
        tau_value = _coerce_metric_value(metrics.get("tau"))
        if tau_value is not None:
            block["tau"] = float(tau_value)
    tau_info = metrics.get("tau_info") if isinstance(metrics, Mapping) else None
    if isinstance(tau_info, str) and tau_info:
        block["tau_info"] = tau_info
    return dict(block)


def _build_case_metrics_export(
    per_case_raw: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Dict[str, Dict[str, float | int]]]]:
    """Convert nested per-case metrics into a JSON-serialisable mapping."""

    if not isinstance(per_case_raw, Mapping):
        return None
    export: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for tag, case_block in per_case_raw.items():
        if not isinstance(case_block, Mapping):
            continue
        sanitized_cases: Dict[str, Dict[str, float | int]] = {}
        for case_id, metrics in case_block.items():
            if not isinstance(metrics, Mapping):
                continue
            sanitized_metrics: Dict[str, float | int] = {}
            for key, value in metrics.items():
                numeric = _coerce_metric_value(value)
                if numeric is None:
                    continue
                if key in INTEGER_METRIC_KEYS:
                    sanitized_metrics[str(key)] = int(numeric)
                else:
                    sanitized_metrics[str(key)] = float(numeric)
            if sanitized_metrics:
                sanitized_cases[str(case_id)] = dict(sorted(sanitized_metrics.items()))
        if sanitized_cases:
            export[str(tag)] = dict(sorted(sanitized_cases.items()))
    if not export:
        return None
    return export


def _build_perturbation_export(
    metrics: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Select perturbation metrics for export when available."""

    if not isinstance(metrics, Mapping):
        return None
    per_tag_raw = metrics.get("perturbation_metrics")
    per_case_raw = metrics.get("perturbation_case_metrics")
    per_tag_export: Dict[str, Dict[str, float | int]] = {}
    if isinstance(per_tag_raw, Mapping):
        for tag, stats in per_tag_raw.items():
            if not isinstance(stats, Mapping):
                continue
            sanitized: Dict[str, float | int] = {}
            for key, value in stats.items():
                numeric = _coerce_metric_value(value)
                if numeric is None:
                    continue
                if key in INTEGER_METRIC_KEYS:
                    sanitized[key] = int(numeric)
                else:
                    sanitized[key] = float(numeric)
            if sanitized:
                per_tag_export[str(tag)] = sanitized
    per_case_export = _build_case_metrics_export(per_case_raw)
    block: Dict[str, Any] = {}
    if per_tag_export:
        block["per_tag"] = dict(sorted(per_tag_export.items()))
    if per_case_export:
        block["per_case"] = per_case_export
    if not block:
        return None
    return block


def _build_morphology_block(
    strata: Optional[Mapping[str, Mapping[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Convert morphology strata metrics into exportable form."""

    if not strata:
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for key in sorted(strata):
        metrics = strata.get(key)
        if not isinstance(metrics, Mapping):
            continue
        block = _build_metric_block(metrics, include_tau=False)
        if block:
            output[key] = block
    return output


def _format_parent_reference_block(
    reference: Optional[ParentRunReference], *, output_dir: Optional[str]
) -> Dict[str, Any]:
    """Convert ``reference`` into an exportable mapping for provenance."""

    if reference is None:
        return {}

    base_dir: Optional[Path] = None
    if output_dir:
        candidate = Path(output_dir).expanduser()
        base_dir = candidate.parent if candidate.parent != candidate else candidate
    if base_dir is None:
        base_dir = Path.cwd()

    block: "OrderedDict[str, Any]" = OrderedDict()
    block["checkpoint"] = _safe_relpath(reference.checkpoint_path, base_dir)
    if reference.checkpoint_sha256:
        block["checkpoint_sha256"] = reference.checkpoint_sha256

    metrics_info: "OrderedDict[str, Any]" = OrderedDict()
    if reference.metrics_path:
        metrics_info["path"] = _safe_relpath(reference.metrics_path, base_dir)
    if reference.metrics_sha256:
        metrics_info["sha256"] = reference.metrics_sha256
    if reference.metrics_payload and isinstance(reference.metrics_payload, Mapping):
        metrics_info["payload"] = reference.metrics_payload
    if metrics_info:
        block["metrics"] = dict(metrics_info)

    outputs_info: "OrderedDict[str, Any]" = OrderedDict()
    if reference.outputs_path:
        outputs_info["path"] = _safe_relpath(reference.outputs_path, base_dir)
    if reference.outputs_sha256:
        outputs_info["sha256"] = reference.outputs_sha256
    if outputs_info:
        block["outputs"] = dict(outputs_info)

    return dict(block)


def _build_run_metadata(
    args,
    *,
    selection_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect high-level run metadata for metrics exports."""

    run_block: "OrderedDict[str, Any]" = OrderedDict()

    exp_config = getattr(args, "exp_config", None)
    if exp_config:
        run_block["experiment_config"] = str(exp_config)
        run_block.setdefault("experiment", Path(str(exp_config)).stem)

    experiment_label = getattr(args, "experiment_name", None) or getattr(args, "experiment", None)
    if experiment_label:
        run_block.setdefault("experiment", str(experiment_label))

    run_stem = getattr(args, "run_stem", None)
    if run_stem:
        run_block["stem"] = str(run_stem)

    display_tag = getattr(args, "run_tag", None) or getattr(args, "run_name", None)
    if display_tag:
        run_block["tag"] = str(display_tag)

    model_identifier = getattr(args, "model_tag", None) or getattr(args, "model_key", None)
    if model_identifier:
        run_block["model"] = str(model_identifier)

    arch = getattr(args, "arch", None)
    if arch:
        run_block["arch"] = str(arch)

    variant = getattr(args, "variant_key", None) or getattr(args, "variant", None)
    if variant:
        run_block["variant"] = str(variant)

    pretraining = getattr(args, "pretraining", None)
    if pretraining:
        run_block["pretraining"] = str(pretraining)

    finetune_mode = getattr(args, "finetune_mode", None)
    if finetune_mode:
        run_block["finetune_mode"] = str(finetune_mode)

    if selection_tag:
        run_block["selection"] = str(selection_tag)

    seed = _get_active_seed(args)
    if seed is not None:
        run_block["seed"] = int(seed)

    finetune_mode_value = getattr(args, "finetune_mode", None)
    if isinstance(finetune_mode_value, str):
        finetune_mode_value = finetune_mode_value.strip().lower()

    if getattr(args, "eval_only", False):
        mode = "eval"
    elif getattr(args, "frozen", False) and (not finetune_mode_value or finetune_mode_value == "none"):
        mode = "inference"
    else:
        mode = "train"
    run_block["mode"] = mode

    world_size = getattr(args, "world_size", None)
    if world_size:
        run_block["world_size"] = int(world_size)

    return dict(run_block)


def _build_metrics_provenance(
    args, *, experiment4_trace: Optional[Experiment4SubsetTrace] = None
) -> Dict[str, Any]:
    """Assemble provenance information for metrics exports."""

    provenance: "OrderedDict[str, Any]" = OrderedDict()
    model_key = getattr(args, "model_key", None)
    model_tag = getattr(args, "model_tag", None)
    run_stem = getattr(args, "run_stem", None)
    model_identifier = next(
        (candidate for candidate in (model_key, model_tag, run_stem) if candidate),
        None,
    )
    if model_identifier:
        provenance["model"] = str(model_identifier)
    arch = getattr(args, "arch", None)
    if arch:
        provenance["arch"] = str(arch)
    provenance["train_seed"] = int(_get_active_seed(args))

    dataset_summary = getattr(args, "dataset_summary", None)
    train_summary = dataset_summary.get("train") if isinstance(dataset_summary, Mapping) else None
    manifest_payload: Optional[Mapping[str, Any]] = None
    train_pack_spec: Optional[str] = None
    if isinstance(train_summary, Mapping):
        train_pack_spec = train_summary.get("pack_spec")
        if train_pack_spec:
            provenance["train_pack"] = str(train_pack_spec)
            provenance.setdefault("train_pack_name", Path(str(train_pack_spec)).name)
        csv_hash = train_summary.get("csv_sha256")
        if csv_hash:
            provenance["train_csv_sha256"] = str(csv_hash)
        manifest_path = train_summary.get("manifest_path")
        if manifest_path:
            provenance.setdefault("train_manifest" , str(manifest_path))
            try:
                manifest_text = Path(str(manifest_path)).read_text(encoding="utf-8")
                manifest_payload = yaml.safe_load(manifest_text) or {}
            except OSError:
                manifest_payload = None
    val_summary = dataset_summary.get("val") if isinstance(dataset_summary, Mapping) else None
    if isinstance(val_summary, Mapping):
        csv_hash = val_summary.get("csv_sha256")
        if csv_hash:
            provenance["val_csv_sha256"] = str(csv_hash)
    test_summary = dataset_summary.get("test") if isinstance(dataset_summary, Mapping) else None
    if isinstance(test_summary, Mapping):
        csv_hash = test_summary.get("csv_sha256")
        if csv_hash:
            provenance["test_csv_sha256"] = str(csv_hash)

    latest_outputs_path = getattr(args, "latest_test_outputs_path", None)
    latest_outputs_sha = getattr(args, "latest_test_outputs_sha256", None)
    base_output_dir = getattr(args, "output_dir", None)
    base_dir = Path(str(base_output_dir)).expanduser() if base_output_dir else Path.cwd()
    if latest_outputs_path:
        try:
            outputs_path_obj = Path(latest_outputs_path)
        except (TypeError, ValueError):
            outputs_path_obj = None
        if outputs_path_obj:
            provenance["test_outputs_csv"] = _safe_relpath(outputs_path_obj, base_dir)
    if latest_outputs_sha:
        provenance["test_outputs_csv_sha256"] = str(latest_outputs_sha)

    fewshot_budget: Optional[int] = None
    pack_seed_value: Optional[int] = None
    if isinstance(manifest_payload, Mapping):
        policy_block = manifest_payload.get("policy")
        if isinstance(policy_block, Mapping):
            budget_value = policy_block.get("fewshot_budget_S") or policy_block.get("target_train_size")
            if isinstance(budget_value, (int, float)):
                fewshot_budget = int(budget_value)
        generator_block = manifest_payload.get("generator")
        if isinstance(generator_block, Mapping):
            seed_value = generator_block.get("seed")
            if isinstance(seed_value, (int, float)):
                pack_seed_value = int(seed_value)
    if fewshot_budget is None and isinstance(train_pack_spec, str):
        match = re.search(r"_s(\d+)", train_pack_spec)
        if match:
            try:
                fewshot_budget = int(match.group(1))
            except ValueError:
                fewshot_budget = None
    if fewshot_budget is not None:
        provenance["fewshot_budget"] = int(fewshot_budget)
    if pack_seed_value is not None:
        provenance.setdefault("pack_seed", int(pack_seed_value))

    subset_percent: Optional[float] = None
    if experiment4_trace and experiment4_trace.percent is not None:
        subset_percent = float(experiment4_trace.percent)
    else:
        dataset_percent = getattr(args, "dataset_percent", None)
        if isinstance(dataset_percent, (int, float)) and dataset_percent > 0:
            subset_percent = float(dataset_percent)
        else:
            layout = getattr(args, "dataset_layout", {}) or {}
            layout_percent = layout.get("percent")
            if isinstance(layout_percent, (int, float)):
                subset_percent = float(layout_percent)
    if subset_percent is None:
        subset_percent = 100.0
    provenance["subset_percent"] = subset_percent

    pack_seed: Optional[int] = None
    if experiment4_trace and experiment4_trace.seed is not None:
        pack_seed = int(experiment4_trace.seed)
    else:
        layout = getattr(args, "dataset_layout", {}) or {}
        layout_seed = layout.get("dataset_seed")
        if isinstance(layout_seed, (int, float)):
            pack_seed = int(layout_seed)
        else:
            dataset_seed = getattr(args, "dataset_seed", None)
            if isinstance(dataset_seed, (int, float)) and dataset_seed >= 0:
                pack_seed = int(dataset_seed)
    if pack_seed is not None:
        provenance["pack_seed"] = pack_seed

    split = getattr(args, "test_split", None)
    if split:
        provenance["split"] = str(split)

    parent_reference = getattr(args, "parent_reference", None)
    if isinstance(parent_reference, ParentRunReference):
        parent_block = _format_parent_reference_block(
            parent_reference, output_dir=getattr(args, "output_dir", None)
        )
        if parent_block:
            provenance["parent_run"] = parent_block

    return dict(provenance)


POLICY_LABELS: Mapping[str, str] = {
    "f1_opt_on_val": "F1-optimal",
    "youden_on_val": "Youden J",
    "val_opt_youden": "Youden J (validation-optimal)",
    "sun_val_frozen": "SUN validation τ (frozen)",
    "f1-morph": "F1 (morphology)",
    "f1": "F1",
    "youden": "Youden J",
}

POLICY_IMPLIED_SPLITS: Mapping[str, str] = {
    "f1_opt_on_val": "val",
    "youden_on_val": "val",
}


def _parse_threshold_key(key: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (dataset, split, policy) parsed from a threshold key."""

    if not key:
        return None, None, None

    text = str(key)
    lowered = text.lower()
    matched_policy: Optional[str] = None
    for candidate in sorted(POLICY_LABELS, key=len, reverse=True):
        if lowered.endswith(candidate):
            matched_policy = candidate
            break

    if not matched_policy:
        return None, None, text

    prefix = lowered[: -len(matched_policy)] if matched_policy else lowered
    prefix = prefix.rstrip("_")

    dataset: Optional[str] = None
    split: Optional[str] = None
    if prefix:
        dataset_part, sep, split_part = prefix.rpartition("_")
        if sep:
            dataset = dataset_part or None
            split = split_part or None
        else:
            dataset = None
            split = prefix or None

    return dataset, split, matched_policy


def _build_thresholds_block(
    thresholds_map: Optional[Mapping[str, Any]],
    *,
    policy: Optional[str] = None,
    sources: Optional[Mapping[str, str]] = None,
    primary: Optional[Mapping[str, Any]] = None,
    sensitivity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a structured representation of threshold provenance."""

    block: "OrderedDict[str, Any]" = OrderedDict()
    if primary:
        block["primary"] = _convert_json_compatible(primary)
    if sensitivity:
        block["sensitivity"] = _convert_json_compatible(sensitivity)
    if policy:
        block["policy"] = policy
    if thresholds_map:
        values = {
            key: float(value)
            for key, value in sorted(thresholds_map.items())
            if isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(float(value))
        }
        if values:
            block["values"] = values
    if sources:
        filtered_sources = {
            key: value
            for key, value in sources.items()
            if isinstance(value, str) and value
        }
        if filtered_sources:
            block["sources"] = filtered_sources
    return dict(block)


def _convert_json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: "OrderedDict[str, Any]" = OrderedDict()
        for key, item in value.items():
            result[str(key)] = _convert_json_compatible(item)
        return dict(result)
    if isinstance(value, (list, tuple)):
        return [_convert_json_compatible(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return str(value)


def _normalize_threshold_records_map(records: Any) -> Dict[str, Dict[str, Any]]:
    """Return a copy of ``records`` with JSON-compatible values."""

    normalised: Dict[str, Dict[str, Any]] = {}
    if isinstance(records, Mapping):
        for key, value in records.items():
            if isinstance(value, Mapping):
                normalised[str(key)] = _convert_json_compatible(dict(value))
    return normalised


def _resolve_primary_threshold_record(
    *,
    threshold_key: Optional[str],
    threshold_records: Optional[Mapping[str, Mapping[str, Any]]],
    frozen_record: Optional[Mapping[str, Any]],
    parent_reference: Optional[ParentRunReference],
) -> Optional[Dict[str, Any]]:
    """Return the richest available primary threshold record."""

    candidates: list[Mapping[str, Any]] = []
    if threshold_key and isinstance(threshold_records, Mapping):
        direct = threshold_records.get(threshold_key)
        if isinstance(direct, Mapping):
            candidates.append(direct)
        else:
            key_lower = str(threshold_key).lower()
            for stored_key, stored_record in threshold_records.items():
                if isinstance(stored_record, Mapping) and str(stored_key).lower() == key_lower:
                    candidates.append(stored_record)
                    break
    if isinstance(frozen_record, Mapping):
        candidates.append(frozen_record)
    if isinstance(parent_reference, ParentRunReference):
        payload = parent_reference.metrics_payload or {}
        thresholds_block = payload.get("thresholds")
        if isinstance(thresholds_block, Mapping):
            primary_candidate = thresholds_block.get("primary")
            if isinstance(primary_candidate, Mapping):
                candidates.append(primary_candidate)
    for candidate in candidates:
        converted = _convert_json_compatible(dict(candidate))
        if converted:
            return converted
    return None


def _compute_domain_shift_delta(
    polyp_metrics: Optional[Mapping[str, Any]],
    parent_reference: Optional[ParentRunReference],
    *,
    metrics: Sequence[str] = ("recall", "f1", "auprc", "auroc"),
) -> Optional[Dict[str, Any]]:
    """Return domain-shift deltas (PolypGen minus SUN) when parent metrics are available."""

    if not polyp_metrics or not isinstance(polyp_metrics, Mapping):
        return None
    if not isinstance(parent_reference, ParentRunReference):
        return None
    payload = parent_reference.metrics_payload
    if not isinstance(payload, Mapping):
        return None
    sun_block = payload.get("test_primary")
    if not isinstance(sun_block, Mapping):
        return None
    deltas: Dict[str, float] = {}
    sun_values: Dict[str, float] = {}
    for metric_name in metrics:
        polyp_value = polyp_metrics.get(metric_name)
        sun_value = sun_block.get(metric_name)
        if not isinstance(polyp_value, (int, float, np.integer, np.floating)):
            continue
        if not isinstance(sun_value, (int, float, np.integer, np.floating)):
            continue
        polyp_numeric = float(polyp_value)
        sun_numeric = float(sun_value)
        if not (math.isfinite(polyp_numeric) and math.isfinite(sun_numeric)):
            continue
        delta_value = polyp_numeric - sun_numeric
        deltas[metric_name] = delta_value
        sun_values[metric_name] = sun_numeric
    if not deltas:
        return None
    result: Dict[str, Any] = {"metrics": deltas}
    if sun_values:
        result["sun"] = sun_values
    sun_tau = sun_block.get("tau")
    if isinstance(sun_tau, (int, float, np.integer, np.floating)) and math.isfinite(float(sun_tau)):
        result["sun_tau"] = float(sun_tau)
    return result


def _sanitize_path_segment(raw: Any, *, default: str = "default") -> str:
    """Return a filesystem-friendly representation of ``raw``."""

    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    text = text.strip("/ ")
    if "/" in text:
        text = text.split("/")[-1]
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._-")
    return cleaned.lower() if cleaned else default


def _resolve_thresholds_subdir(args) -> str:
    """Determine the sub-directory used to persist threshold metadata."""

    dataset_layout = getattr(args, "dataset_layout", {}) or {}
    val_pack = getattr(args, "val_pack", None)
    if not val_pack:
        resolved = getattr(args, "dataset_resolved", {}) or {}
        val_pack = resolved.get("val_pack")
    if not val_pack:
        val_pack = getattr(args, "train_pack", None)
    if val_pack:
        segment = _sanitize_path_segment(val_pack)
        if segment and segment != "default":
            return segment

    dataset_name = dataset_layout.get("name") or getattr(args, "dataset", None)
    dataset_name = "sun" if dataset_name == "sun_full" else dataset_name
    dataset_name = _sanitize_path_segment(dataset_name, default="dataset")
    dataset_seed = dataset_layout.get("dataset_seed")
    size = dataset_layout.get("size")
    percent = dataset_layout.get("percent")

    parts: list[str] = [dataset_name]
    if dataset_name.startswith("polypgen_fewshot") and size:
        parts.append(f"c{int(size)}")
    elif dataset_name.startswith("sun_subsets") and percent:
        parts.append(f"p{int(percent)}")
    if dataset_seed is not None:
        parts.append(f"s{int(dataset_seed)}")
    if len(parts) == 1:
        split = getattr(args, "val_split", None) or "val"
        parts.append(_sanitize_path_segment(split, default="val"))
    return "_".join(parts)


def _compute_threshold_statistics(
    logits: Optional[torch.Tensor], targets: Optional[torch.Tensor], tau: float
) -> Dict[str, Optional[float]]:
    """Compute confusion-derived metrics at threshold ``tau``."""

    if logits is None or targets is None:
        return {}
    logits_cpu = logits.detach().cpu()
    targets_cpu = targets.detach().cpu().to(dtype=torch.long)

    if logits_cpu.ndim == 1:
        scores = torch.sigmoid(logits_cpu)
    elif logits_cpu.ndim == 2:
        if logits_cpu.size(1) == 1:
            scores = torch.sigmoid(logits_cpu.squeeze(1))
        elif logits_cpu.size(1) == 2:
            scores = torch.softmax(logits_cpu, dim=1)[:, 1]
        else:
            raise ValueError(
                "Threshold computation is only supported for binary logits"
            )
    else:
        raise ValueError(
            "Threshold computation expects logits with 1 or 2 dimensions"
        )

    preds = (scores >= tau).to(dtype=torch.long)
    targets_long = targets_cpu.to(dtype=torch.long)

    tp = int(((preds == 1) & (targets_long == 1)).sum().item())
    tn = int(((preds == 0) & (targets_long == 0)).sum().item())
    fp = int(((preds == 1) & (targets_long == 0)).sum().item())
    fn = int(((preds == 0) & (targets_long == 1)).sum().item())
    total = tp + tn + fp + fn

    def _safe_ratio(num: int, denom: int) -> Optional[float]:
        if denom == 0:
            return None
        return float(num) / float(denom)

    tpr = _safe_ratio(tp, tp + fn)
    tnr = _safe_ratio(tn, tn + fp)
    prevalence = _safe_ratio(tp + fn, total)

    denom_product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_product > 0:
        mcc_value = ((tp * tn) - (fp * fn)) / math.sqrt(float(denom_product))
    else:
        mcc_value = None

    metrics: Dict[str, Optional[float]] = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "ppv": _safe_ratio(tp, tp + fp),
        "npv": _safe_ratio(tn, tn + fn),
        "accuracy": _safe_ratio(tp + tn, total),
        "youden_j": None if tpr is None or tnr is None else tpr + tnr - 1.0,
        "prevalence": prevalence,
        "mcc": mcc_value,
    }
    return metrics


def _compute_metrics_for_probability_threshold(
    probabilities: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    tau: Optional[float],
    *,
    base_metrics: Optional[Mapping[str, Any]] = None,
    tau_info: Optional[str] = None,
) -> Dict[str, Any]:
    """Return aggregate metrics after applying ``tau`` to ``probabilities``."""

    if tau is None or probabilities is None or targets is None:
        return {}

    try:
        scores_tensor = _extract_positive_probabilities(probabilities)
    except Exception:
        return {}

    if scores_tensor.numel() == 0:
        return {}

    targets_tensor = torch.as_tensor(targets)
    if targets_tensor.numel() != scores_tensor.numel():
        return {}

    scores_np = scores_tensor.detach().cpu().numpy().astype(float, copy=False)
    labels_np = targets_tensor.detach().cpu().numpy().astype(int, copy=False)

    preds_np = (scores_np >= float(tau)).astype(int, copy=False)
    positives = labels_np == 1
    negatives = labels_np == 0

    tp = int(np.count_nonzero(preds_np & positives))
    fp = int(np.count_nonzero(preds_np & negatives))
    tn = int(np.count_nonzero((1 - preds_np) & negatives))
    fn = int(np.count_nonzero((1 - preds_np) & positives))

    total = tp + fp + tn + fn
    n_pos = int(np.count_nonzero(positives))
    n_neg = int(np.count_nonzero(negatives))

    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return float("nan")
        return float(numerator) / float(denominator)

    recall = _safe_ratio(tp, tp + fn)
    precision = _safe_ratio(tp, tp + fp)
    specificity = _safe_ratio(tn, tn + fp)

    if math.isfinite(recall) and math.isfinite(precision) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    if math.isfinite(recall) and math.isfinite(specificity):
        balanced_accuracy = 0.5 * (recall + specificity)
    else:
        balanced_accuracy = float("nan")

    denom = math.sqrt(
        float(tp + fp)
        * float(tp + fn)
        * float(tn + fp)
        * float(tn + fn)
    )
    if denom > 0:
        mcc = ((tp * tn) - (fp * fn)) / denom
    else:
        mcc = float("nan")

    prevalence = _safe_ratio(n_pos, n_pos + n_neg)

    metrics: Dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": total,
        "count": total,
        "prevalence": prevalence,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "tau": float(tau),
    }
    if tau_info:
        metrics["tau_info"] = str(tau_info)

    if base_metrics:
        for key in ("loss", "auroc", "auprc"):
            value = base_metrics.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                metrics[key] = float(value)

    return metrics


def _resolve_policy_threshold(
    *,
    policy: Optional[str],
    dataset: str,
    split: str,
    epoch: int,
    scores: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    previous_tau: Optional[float] = None,
    parent_reference: Optional[ParentRunReference] = None,
    source_key: Optional[str] = None,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Compute or retrieve a threshold and provenance record for ``policy``."""

    if not policy:
        return None, None

    policy_normalised = str(policy).strip().lower()
    if policy_normalised in {"", "none"}:
        return None, None

    if policy_normalised == "sun_val_frozen":
        if not isinstance(parent_reference, ParentRunReference):
            raise ValueError(
                "Policy 'sun_val_frozen' requires a parent run reference with stored thresholds."
            )
        thresholds_block = (parent_reference.metrics_payload or {}).get("thresholds") or {}
        tau_value, record = thresholds.resolve_frozen_sun_threshold(
            thresholds_block,
            source_key=source_key or "primary",
            expected_split_substring=f"{dataset}/{split}",
            checkpoint_path=parent_reference.checkpoint_path,
        )
        return float(tau_value), dict(_convert_json_compatible(record))

    if scores is None or labels is None:
        return None, None

    try:
        result = thresholds.compute_policy_threshold(
            scores,
            labels,
            policy=policy_normalised,
            split_name=f"{dataset}/{split}",
            epoch=int(epoch),
            previous_tau=previous_tau,
        )
    except ValueError:
        return None, None

    record = dict(_convert_json_compatible(result.record))
    if "policy" not in record:
        record["policy"] = policy_normalised
    record.setdefault("split", f"{dataset}/{split}")
    if result.metrics:
        record.setdefault("metrics", _convert_json_compatible(result.metrics))
    if result.candidates:
        record.setdefault("candidates", [_convert_json_compatible(value) for value in result.candidates])
    return float(result.tau), record


def _resolve_sensitivity_threshold(
    *,
    policy: Optional[str],
    threshold_key: Optional[str],
    dataset: str,
    split: str,
    epoch: int,
    val_probabilities: Optional[torch.Tensor],
    val_logits: Optional[torch.Tensor],
    val_targets: Optional[torch.Tensor],
    thresholds_map: MutableMapping[str, float],
    parent_reference: Optional[ParentRunReference] = None,
    source_key: Optional[str] = None,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Compute and persist the sensitivity threshold if applicable."""

    if not policy:
        return None, None

    previous_tau = None
    if threshold_key and threshold_key in thresholds_map:
        try:
            previous_tau = float(thresholds_map[threshold_key])
        except (TypeError, ValueError):
            previous_tau = None

    scores_np: Optional[np.ndarray] = None
    labels_np: Optional[np.ndarray] = None

    if policy not in {"sun_val_frozen"}:
        scores_tensor: Optional[torch.Tensor] = None
        if val_probabilities is not None:
            try:
                scores_tensor = _extract_positive_probabilities(val_probabilities)
            except Exception:
                scores_tensor = None
        if scores_tensor is None and val_logits is not None:
            try:
                scores_tensor = _extract_positive_probabilities(
                    _prepare_binary_probabilities(val_logits)
                )
            except Exception:
                scores_tensor = None
        if scores_tensor is None or val_targets is None:
            return None, None
        scores_np = scores_tensor.detach().cpu().numpy().astype(float, copy=False)
        labels_np = (
            torch.as_tensor(val_targets).detach().cpu().numpy().astype(int, copy=False)
        )

    tau_value, record = _resolve_policy_threshold(
        policy=policy,
        dataset=dataset,
        split=split,
        epoch=epoch,
        scores=scores_np,
        labels=labels_np,
        previous_tau=previous_tau,
        parent_reference=parent_reference,
        source_key=source_key,
    )

    if tau_value is not None and threshold_key:
        thresholds_map[threshold_key] = float(tau_value)

    return tau_value, record


def _coerce_metadata_row(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return {}


def _resolve_metadata_value(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _export_frame_outputs(
    path: Path,
    *,
    metadata_rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    targets: Sequence[int],
    preds: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "prob",
        "label",
        "pred",
        "case_id",
        "origin",
        "center_id",
        "sequence_id",
        "morphology",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        total = len(probabilities)
        for idx in range(total):
            row = metadata_rows[idx] if idx < len(metadata_rows) else {}
            frame_id = _resolve_metadata_value(
                row,
                (
                    "frame_id",
                    "orig_frame_id",
                    "frame",
                    "frame_path",
                    "image_id",
                ),
            ) or f"idx_{idx}"
            case_id = _resolve_metadata_value(
                row,
                (
                    "case_id",
                    "sequence_id",
                    "case",
                    "study_id",
                ),
            )
            origin = _resolve_metadata_value(
                row,
                (
                    "origin",
                    "store_id",
                    "dataset",
                    "source_dataset",
                ),
            )
            center_id = _resolve_metadata_value(
                row,
                (
                    "center_id",
                    "centre_id",
                    "center",
                    "centre",
                    "hospital_id",
                    "hospital",
                    "origin",
                    "store_id",
                ),
            )
            sequence_id = _resolve_metadata_value(
                row,
                (
                    "sequence_id",
                    "sequence",
                    "case_id",
                    "case",
                    "study_id",
                ),
            )
            morphology = None
            if isinstance(row, Mapping):
                value = row.get("morphology")
                if value not in (None, ""):
                    morphology = str(value).strip()
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "prob": float(probabilities[idx]),
                    "label": int(targets[idx]) if idx < len(targets) else None,
                    "pred": int(preds[idx]) if idx < len(preds) else None,
                    "case_id": case_id,
                    "origin": origin,
                    "center_id": center_id,
                    "sequence_id": sequence_id,
                    "morphology": morphology,
                }
            )


def _extract_positive_probabilities(probabilities: Any) -> torch.Tensor:
    tensor = torch.as_tensor(probabilities)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Curve export requires tensor-like probabilities")
    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2:
        if tensor.size(1) == 1:
            return tensor[:, 0]
        if tensor.size(1) == 2:
            return tensor[:, 1]
    raise ValueError(
        "Curve export expects binary probabilities with shape (N,), (N,1) or (N,2)."
    )


def _binary_subset_metrics(
    *,
    positive_scores: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    sample_losses: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if indices.size == 0:
        return {}
    subset_scores = positive_scores[indices]
    subset_preds = preds[indices]
    subset_labels = labels[indices]
    total = subset_labels.size
    n_pos = int(np.sum(subset_labels == 1))
    n_neg = int(np.sum(subset_labels == 0))
    prevalence = (float(n_pos) / float(total)) if total > 0 else float("nan")
    tp = int(np.sum((subset_preds == 1) & (subset_labels == 1)))
    fp = int(np.sum((subset_preds == 1) & (subset_labels == 0)))
    tn = int(np.sum((subset_preds == 0) & (subset_labels == 0)))
    fn = int(np.sum((subset_preds == 0) & (subset_labels == 1)))
    try:
        auprc = float(average_precision_score(subset_labels, subset_scores))
    except ValueError:
        auprc = float("nan")
    try:
        auroc = float(roc_auc_score(subset_labels, subset_scores))
    except ValueError:
        auroc = float("nan")
    recall = float(recall_score(subset_labels, subset_preds, zero_division=0))
    precision = float(precision_score(subset_labels, subset_preds, zero_division=0))
    f1 = float(f1_score(subset_labels, subset_preds, zero_division=0))
    try:
        balanced_acc = float(balanced_accuracy_score(subset_labels, subset_preds))
    except ValueError:
        balanced_acc = float("nan")
    try:
        mcc = float(matthews_corrcoef(subset_labels, subset_preds))
    except ValueError:
        mcc = float("nan")
    if sample_losses is not None:
        subset_losses = sample_losses[indices]
        loss_value = float(np.mean(subset_losses)) if subset_losses.size > 0 else float("nan")
    else:
        eps = 1e-12
        clipped = np.clip(subset_scores, eps, 1.0 - eps)
        loss_value = float(
            np.mean(
                -(subset_labels * np.log(clipped) + (1 - subset_labels) * np.log(1 - clipped))
            )
        )
    return {
        "count": int(total),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auprc": auprc,
        "auroc": auroc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "loss": loss_value,
    }


def _build_morphology_strata_metrics(
    *,
    positive_scores: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    morphology: Sequence[str],
    sample_losses: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    total = labels.size
    if total == 0:
        return {}
    indices_all = np.arange(total, dtype=int)
    strata: Dict[str, Dict[str, Any]] = {
        "overall": _binary_subset_metrics(
            positive_scores=positive_scores,
            preds=preds,
            labels=labels,
            indices=indices_all,
            sample_losses=sample_losses,
        )
    }
    if len(morphology) != total:
        return strata
    morph_array = np.array([str(value).strip().lower() if value not in (None, "") else "" for value in morphology])
    labels_pos = labels == 1
    labels_neg = labels == 0
    flat_pos_count = int(np.sum(labels_pos & (morph_array == "flat")))
    polypoid_pos_count = int(np.sum(labels_pos & (morph_array == "polypoid")))
    if flat_pos_count > 0:
        flat_mask = labels_neg | (labels_pos & (morph_array == "flat"))
        flat_indices = np.flatnonzero(flat_mask)
        strata["flat_plus_negs"] = _binary_subset_metrics(
            positive_scores=positive_scores,
            preds=preds,
            labels=labels,
            indices=flat_indices,
            sample_losses=sample_losses,
        )
    if polypoid_pos_count > 0:
        polypoid_mask = labels_neg | (labels_pos & (morph_array == "polypoid"))
        polypoid_indices = np.flatnonzero(polypoid_mask)
        strata["polypoid_plus_negs"] = _binary_subset_metrics(
            positive_scores=positive_scores,
            preds=preds,
            labels=labels,
            indices=polypoid_indices,
            sample_losses=sample_losses,
        )
    return strata


def _compute_f1_morph_threshold(
    *,
    positive_scores: np.ndarray,
    labels: np.ndarray,
    morphology: Sequence[str],
    grid_points: int = 512,
) -> Optional[float]:
    if positive_scores.size == 0 or labels.size == 0:
        return None
    if len(morphology) != labels.size:
        return None
    morph_array = np.array([str(value).strip().lower() if value not in (None, "") else "" for value in morphology])
    labels_pos = labels == 1
    labels_neg = labels == 0
    flat_mask = labels_neg | (labels_pos & (morph_array == "flat"))
    polypoid_mask = labels_neg | (labels_pos & (morph_array == "polypoid"))
    flat_pos_count = int(np.sum(labels_pos & (morph_array == "flat")))
    polypoid_pos_count = int(np.sum(labels_pos & (morph_array == "polypoid")))
    if flat_pos_count == 0 or polypoid_pos_count == 0:
        return None
    thresholds = np.linspace(0.0, 1.0, num=max(2, int(grid_points)), endpoint=True)
    best_tau: Optional[float] = None
    best_score = -1.0
    for tau in thresholds:
        preds = (positive_scores >= tau).astype(int)
        try:
            f1_flat = f1_score(labels[flat_mask], preds[flat_mask], zero_division=0)
            f1_polypoid = f1_score(labels[polypoid_mask], preds[polypoid_mask], zero_division=0)
        except ValueError:
            continue
        macro_f1 = 0.5 * (f1_flat + f1_polypoid)
        if macro_f1 > best_score or (math.isclose(macro_f1, best_score) and best_tau is not None and tau < best_tau):
            best_score = float(macro_f1)
            best_tau = float(tau)
    return best_tau


def _export_curve_sets(
    ckpt_stem: Path,
    split_name: str,
    *,
    probabilities: Any,
    targets: Any,
    grid_points: int = 200,
) -> Dict[str, Any]:
    if grid_points is None or int(grid_points) < 2:
        raise ValueError("Curve export requires at least two grid points.")
    if probabilities is None or targets is None:
        raise ValueError("Curve export requires probabilities and targets.")

    scores = _extract_positive_probabilities(probabilities)
    labels = torch.as_tensor(targets).detach().cpu().to(dtype=torch.long)
    if scores.numel() != labels.numel():
        raise ValueError("Mismatch between probability and target counts for curve export.")
    if scores.numel() == 0:
        raise ValueError("Curve export received no samples.")

    scores_np = scores.numpy()
    labels_np = labels.numpy()
    thresholds = np.linspace(0.0, 1.0, num=int(grid_points), endpoint=True)
    positive_mask = labels_np == 1
    negative_mask = labels_np == 0

    def _safe_fraction(numerator: int, denominator: int) -> Optional[float]:
        if denominator <= 0:
            return None
        return float(numerator) / float(denominator)

    def _normalise(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    roc_rows: list[Dict[str, Any]] = []
    pr_rows: list[Dict[str, Any]] = []
    for tau in thresholds:
        preds = scores_np >= tau
        tp = int(np.count_nonzero(preds & positive_mask))
        fp = int(np.count_nonzero(preds & negative_mask))
        tn = int(np.count_nonzero((~preds) & negative_mask))
        fn = int(np.count_nonzero((~preds) & positive_mask))

        tpr = _safe_fraction(tp, tp + fn)
        fpr = _safe_fraction(fp, fp + tn)
        precision = _safe_fraction(tp, tp + fp)
        recall = tpr
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)

        roc_rows.append(
            {
                "threshold": round(float(tau), 10),
                "tpr": _normalise(tpr),
                "fpr": _normalise(fpr),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
        pr_rows.append(
            {
                "threshold": round(float(tau), 10),
                "precision": _normalise(precision),
                "recall": _normalise(recall),
                "f1": _normalise(f1),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    split_segment = _sanitize_path_segment(split_name, default=str(split_name).lower() or "split")
    base_name = f"{ckpt_stem.name}_{split_segment}"
    roc_path = ckpt_stem.with_name(f"{base_name}_roc_curve.csv")
    pr_path = ckpt_stem.with_name(f"{base_name}_pr_curve.csv")
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    with roc_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "tpr", "fpr", "tp", "fp", "tn", "fn"])
        writer.writeheader()
        for row in roc_rows:
            writer.writerow(row)

    with pr_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"],
        )
        writer.writeheader()
        for row in pr_rows:
            writer.writerow(row)

    return {
        "roc_csv": roc_path,
        "pr_csv": pr_path,
        "grid_points": int(grid_points),
    }


def _maybe_export_curves_for_split(
    args,
    *,
    ckpt_stem: Path,
    split_name: str,
    metrics: Mapping[str, Any],
    log_path: Path,
) -> Optional[Dict[str, Any]]:
    spec_map = getattr(args, "curve_export_spec", {}) or {}
    split_key = str(split_name).strip().lower()
    entry = spec_map.get(split_key)
    if not entry:
        return None

    probabilities = metrics.get("probabilities")
    targets = metrics.get("targets")
    if probabilities is None or targets is None:
        return None

    grid_points = int(entry.get("points", 200) or 200)
    try:
        exports = _export_curve_sets(
            ckpt_stem,
            split_name,
            probabilities=probabilities,
            targets=targets,
            grid_points=grid_points,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        warning = (
            f"Warning: failed to export {split_name} curves ({grid_points} points): {exc}"
        )
        _log_lines(Path(log_path), [warning])
        return None

    metadata: Dict[str, Any] = {"points": int(exports["grid_points"])}
    parent_dir = ckpt_stem.parent
    for label, path in (("roc_csv", exports["roc_csv"]), ("pr_csv", exports["pr_csv"])):
        path_obj = Path(path)
        try:
            metadata[label] = str(path_obj.relative_to(parent_dir))
        except ValueError:
            metadata[label] = str(path_obj)

    lines = [
        f"{split_name} curve export ({metadata['points']} thresholds):",
        f"ROC CSV: {metadata['roc_csv']}",
        f"PR CSV: {metadata['pr_csv']}",
    ]
    _log_lines(Path(log_path), lines)
    return metadata


def _safe_min(delta: float) -> float:
    """Return a finite minimum delta suitable for comparisons."""

    if math.isnan(delta):
        return 0.0
    return float(delta)


def _resolve_monitor_mode(monitor: Optional[str], mode: Optional[str]) -> str:
    """Infer the comparison mode for early stopping."""

    if mode:
        resolved = mode.lower()
        if resolved not in {"min", "max", "auto"}:
            raise ValueError(f"Unsupported early-stop mode: {mode!r}")
    else:
        resolved = "auto"
    if resolved != "auto":
        return resolved
    monitor = (monitor or "").lower()
    if monitor.endswith("loss") or monitor.endswith("_loss"):
        return "min"
    if monitor.startswith("loss"):
        return "min"
    return "max"


def _improved(
    current: float,
    best: Optional[float],
    *,
    mode: str,
    min_delta: float,
) -> bool:
    """Return ``True`` when ``current`` improves over ``best``."""

    if best is None or math.isnan(best):
        return True
    if math.isnan(current):
        return False
    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError(f"Unexpected monitor mode: {mode}")


def _resolve_monitor_key(raw_key: Optional[str]) -> str:
    """Normalise monitor names to match metric dictionary keys."""

    if not raw_key:
        return "loss"
    key = raw_key.lower()
    if key.startswith("val_"):
        key = key[4:]
    return key


def _binary_logits_from_multiclass(logits: torch.Tensor) -> torch.Tensor:
    """Collapse two-class logits into a single positive-class logit."""

    if logits.ndim == 1:
        return logits
    if logits.ndim != 2:
        raise ValueError("Binary BCE loss expects logits with shape (N,) or (N, 2)")
    if logits.size(1) == 1:
        return logits.squeeze(1)
    if logits.size(1) == 2:
        return logits[:, 1] - logits[:, 0]
    raise ValueError("Binary BCE loss received logits with more than two classes")


def _compute_supervised_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Evaluate ``loss_fn`` handling binary BCE special cases."""

    if mode == "binary_bce":
        binary_logits = _binary_logits_from_multiclass(logits)
        target_float = targets.to(dtype=binary_logits.dtype)
        return loss_fn(binary_logits, target_float)
    return loss_fn(logits, targets)


def _compute_sample_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Return per-sample losses compatible with :func:`_compute_supervised_loss`."""

    if mode == "binary_bce":
        binary_logits = _binary_logits_from_multiclass(logits)
        target_float = targets.to(dtype=binary_logits.dtype)
        return F.binary_cross_entropy_with_logits(
            binary_logits,
            target_float,
            reduction="none",
        )
    if mode == "multiclass_ce":
        return F.cross_entropy(logits, targets, reduction="none")
    raise ValueError(f"Unsupported loss mode '{mode}' for per-sample loss computation")


def _prepare_binary_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return class probabilities for a binary classifier from raw logits."""

    if logits.ndim == 1:
        pos = torch.sigmoid(logits)
        neg = 1.0 - pos
        return torch.stack((neg, pos), dim=1)
    if logits.ndim != 2:
        raise ValueError("Binary probability preparation expects rank-1 or rank-2 logits")
    if logits.size(1) == 1:
        pos = torch.sigmoid(logits.squeeze(1))
        neg = 1.0 - pos
        return torch.stack((neg, pos), dim=1)
    if logits.size(1) == 2:
        binary_logits = _binary_logits_from_multiclass(logits)
        pos = torch.sigmoid(binary_logits)
        neg = 1.0 - pos
        return torch.stack((neg, pos), dim=1)
    return torch.softmax(logits, dim=1)


def _build_threshold_payload(
    args,
    *,
    threshold_key: Optional[str],
    tau: float,
    metrics_at_tau: Dict[str, Optional[float]],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    val_perf: float,
    test_perf: Optional[float],
    model_tag: Optional[str] = None,
    subdir: Optional[str] = None,
    policy_record: Optional[Mapping[str, Any]] = None,
    snapshot_epoch: Optional[int] = None,
    snapshot_tau: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a structured payload capturing threshold metadata."""

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    val_pack = getattr(args, "val_pack", None)
    if not val_pack:
        resolved = getattr(args, "dataset_resolved", {}) or {}
        val_pack = resolved.get("val_pack")
    if not val_pack:
        val_pack = getattr(args, "train_pack", None)

    payload: Dict[str, Any] = {
        "policy": getattr(args, "threshold_policy", None),
        "selected_threshold": float(tau),
        "split_used": getattr(args, "val_split", None) or "val",
        "seed": int(getattr(args, "seed", 0)),
        # Authoritative τ-based validation metrics live only in policy_record.metrics_at_tau
        # (mirrored below to metrics_at_threshold). The debug/* block is non-canonical and may
        # reflect a different τ; downstream consumers must ignore it for headline results.
        "metrics_at_threshold": metrics_at_tau,
        "auc_val": float(val_perf),
        "auc_test": float(test_perf) if test_perf is not None else None,
        "timestamp": timestamp,
        "code_git_commit": getattr(args, "git_commit", None),
        "data_pack": val_pack,
        "notes": "",
    }
    if threshold_key:
        payload["threshold_key"] = threshold_key

    def _build_debug_snapshot(
        metrics: Mapping[str, float], _label: str
    ) -> Optional[Dict[str, Any]]:
        if not metrics:
            return None
        scalar_metrics: Dict[str, float] = {}
        macro_keys = {
            "recall",
            "precision",
            "f1",
            "balanced_accuracy",
            "auprc",
            "auroc",
        }
        for key in macro_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
                float(value)
            ):
                scalar_metrics[f"{key}_macro"] = float(value)

        extra_keys = {
            "loss",
        }
        for key in extra_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
                float(value)
            ):
                scalar_metrics[key] = float(value)

        if not scalar_metrics:
            return None

        snapshot_block: Dict[str, Any] = {
            "averaging": "macro",
            "metrics": scalar_metrics,
        }
        if snapshot_epoch is not None:
            snapshot_block["epoch"] = int(snapshot_epoch)

        tau_value: Optional[float] = None
        if snapshot_tau is not None and math.isfinite(float(snapshot_tau)):
            tau_value = float(snapshot_tau)
        else:
            fallback_tau = metrics.get("tau") if isinstance(metrics, Mapping) else None
            if isinstance(fallback_tau, (int, float, np.integer, np.floating)) and math.isfinite(
                float(fallback_tau)
            ):
                tau_value = float(fallback_tau)
        if tau_value is not None:
            snapshot_block["tau_used"] = tau_value

        return snapshot_block

    debug_payload: Dict[str, Any] = {}
    val_snapshot = _build_debug_snapshot(val_metrics, "val")
    if val_snapshot:
        debug_payload["val_snapshot_prev"] = val_snapshot
    test_snapshot = _build_debug_snapshot(test_metrics, "test")
    if test_snapshot:
        debug_payload["test_snapshot_prev"] = test_snapshot
    if debug_payload:
        payload.setdefault("debug", {}).update(debug_payload)
    if threshold_key:
        payload["thresholds"] = {threshold_key: float(tau)}
    if model_tag:
        payload["model_tag"] = model_tag
    if subdir:
        payload["directory"] = subdir
    if policy_record:
        payload["policy_record"] = json.loads(
            json.dumps(policy_record, default=str)
        )
    return payload


_TOKEN_OVERRIDES = {
    "sun": "SUN",
    "sup": "SUP",
    "ssl": "SSL",
    "imnet": "ImNet",
    "imagenet": "ImageNet",
    "colon": "Colon",
    "hyperkvasir": "HyperKvasir",
    "mae": "MAE",
    "vit": "ViT",
    "polypgen": "PolypGen",
    "fewshot": "FewShot",
    "subsets": "Subset",
    "subset": "Subset",
    "full": "Full",
    "morphology": "Morph",
    "test": "Test",
    "perturbations": "Perturb",
    "clean": "Clean",
    "baseline": "Baseline",
    "baselines": "Baseline",
    "random": "Random",
}


def _canonicalize_tag(raw: Any) -> str:
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", str(raw)) if t]
    if not tokens:
        return "run"
    parts: list[str] = []
    for idx, token in enumerate(tokens):
        lower = token.lower()
        if lower in _TOKEN_OVERRIDES:
            piece = _TOKEN_OVERRIDES[lower]
        elif token.isupper():
            piece = token
        elif idx == 0:
            piece = token.capitalize()
        else:
            piece = token.capitalize()
        parts.append(piece)
    return "".join(parts)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_int(pattern: str, text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    match = re.search(pattern, text)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    return None


def _resolve_dataset_layout(
    args,
    dataset_cfg: Optional[Dict[str, Any]],
    dataset_resolved: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    dataset_cfg = dataset_cfg or {}
    dataset_resolved = dataset_resolved or {}
    dataset_name = dataset_cfg.get("name") or getattr(args, "dataset", None) or "dataset"
    dataset_name = str(dataset_name)
    dataset_key = dataset_name.lower()

    percent = _as_int(dataset_cfg.get("percent"))
    dataset_seed = _as_int(dataset_cfg.get("seed"))
    size = _as_int(dataset_cfg.get("size"))
    if percent is None:
        percent = _as_int(dataset_resolved.get("percent"))
    if dataset_seed is None:
        dataset_seed = _as_int(dataset_resolved.get("seed"))
    if size is None:
        size = _as_int(dataset_resolved.get("size"))
    train_pack = dataset_resolved.get("train_pack") if dataset_resolved else None
    if not train_pack:
        train_pack = getattr(args, "train_pack", None)
    train_pack = str(train_pack) if train_pack is not None else None

    if dataset_key == "sun_subsets":
        if percent is None:
            percent = _extract_int(r"p(\d+)", train_pack)
        if dataset_seed is None:
            dataset_seed = _extract_int(r"seed(\d+)", train_pack)
    elif dataset_key == "polypgen_fewshot":
        if size is None:
            size = _extract_int(r"_s(\d+)", train_pack)
        if dataset_seed is None:
            dataset_seed = _extract_int(r"seed(\d+)", train_pack)

    segments: list[str] = []
    data_tag = _canonicalize_tag(dataset_name)
    default_parent_tag: Optional[str] = None
    default_parent_seed: Optional[int] = None

    if dataset_key == "sun_full":
        segments = ["sun_baselines"]
        data_tag = "SUNFull"
    elif dataset_key == "sun_morphology":
        segments = ["sun_morphology"]
        data_tag = "SUNMorph"
    elif dataset_key == "sun_subsets":
        segments = ["sun_subsets"]
        if percent is not None:
            segments.append(f"p{int(percent):02d}")
            data_tag = f"SUNP{int(percent):02d}"
        else:
            data_tag = "SUNSubset"
        if dataset_seed is not None:
            segments.append(f"seed{int(dataset_seed)}")
    elif dataset_key == "polypgen_fewshot":
        segments = ["polypgen_fewshot"]
        if size is not None:
            segments.append(f"s{int(size)}")
            data_tag = f"PolypGenFew{int(size)}"
        else:
            data_tag = "PolypGenFewShot"
        if dataset_seed is not None:
            segments.append(f"seed{int(dataset_seed)}")
        default_parent_tag = "SUN"
        default_parent_seed = dataset_seed
    elif dataset_key == "polypgen_clean_test":
        segments = ["polypgen_clean_test"]
        data_tag = "PolypGenClean"
    elif dataset_key == "sun_test_perturbations":
        segments = ["sun_test_perturbations"]
        data_tag = "SUNPerturb"
    else:
        sanitized = re.sub(r"[^0-9A-Za-z]+", "_", dataset_key).strip("_")
        segments = [sanitized or "dataset"]

    return {
        "name": dataset_key,
        "segments": tuple(segments),
        "data_tag": data_tag,
        "dataset_seed": dataset_seed,
        "percent": percent,
        "size": size,
        "default_parent_tag": default_parent_tag,
        "default_parent_seed": default_parent_seed,
    }


def _resolve_model_tag(args, selected_model: Optional[Dict[str, Any]]) -> str:
    raw: Optional[str] = None
    if selected_model:
        for key in ("key", "name"):
            candidate = selected_model.get(key)
            if candidate:
                raw = str(candidate)
                break
    if not raw:
        raw = getattr(args, "model_key", None)
    if not raw:
        arch = getattr(args, "arch", None)
        pretraining = getattr(args, "pretraining", None)
        parts = [str(part) for part in (arch, pretraining) if part]
        raw = "_".join(parts)
    if not raw:
        raw = "model"
    return _canonicalize_tag(raw)


def _normalise_lineage_tag(tag: str) -> str:
    if tag.lower() == "sunfull":
        return "SUN"
    return tag


def _compose_lineage(tag: Optional[str], seed: Optional[int]) -> Optional[str]:
    if not tag:
        return None
    canonical = _canonicalize_tag(tag)
    canonical = _normalise_lineage_tag(canonical)
    qualifier = f"from{canonical}"
    if seed is not None:
        qualifier += f"_s{int(seed)}"
    return qualifier


def _extract_parent_metadata(reference: str) -> tuple[Optional[str], Optional[int]]:
    stem = Path(reference).stem
    seed_match = re.search(r"_s(\d+)$", stem)
    seed = int(seed_match.group(1)) if seed_match else None
    data_match = re.search(r"__(.+)_s\d+$", stem)
    if data_match:
        data_segment = data_match.group(1)
        data_tag = data_segment.split("_")[0]
        return data_tag, seed
    return None, seed


def _resolve_lineage_qualifiers(
    args,
    dataset_layout: Dict[str, Any],
    experiment_cfg: Optional[Dict[str, Any]],
) -> list[str]:
    qualifiers: list[str] = []
    parent_reference = getattr(args, "parent_checkpoint", None)
    if parent_reference:
        parent_tag, parent_seed = _extract_parent_metadata(parent_reference)
        qualifier = _compose_lineage(parent_tag, parent_seed)
        if qualifier:
            qualifiers.append(qualifier)
    else:
        protocol_cfg = (experiment_cfg or {}).get("protocol") or {}
        default_tag = dataset_layout.get("default_parent_tag")
        if default_tag and protocol_cfg.get("init_from"):
            qualifier = _compose_lineage(
                default_tag, dataset_layout.get("default_parent_seed")
            )
            if qualifier:
                qualifiers.append(qualifier)
    return qualifiers


def _compose_stem(
    model_tag: str, data_tag: str, qualifiers: Iterable[str], seed: int
) -> str:
    seed_value = _as_int(seed) or 0
    qualifier_list = [q for q in qualifiers if q]
    qualifier_part = f"_{'_'.join(qualifier_list)}" if qualifier_list else ""
    return f"{model_tag}__{data_tag}{qualifier_part}_s{seed_value}"


def _resolve_run_layout(
    args,
    selected_model: Optional[Dict[str, Any]] = None,
    dataset_cfg: Optional[Dict[str, Any]] = None,
    dataset_resolved: Optional[Dict[str, Any]] = None,
    experiment_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_layout = _resolve_dataset_layout(args, dataset_cfg, dataset_resolved)
    base_dir = Path(getattr(args, "output_dir", "checkpoints")).expanduser()
    output_dir = base_dir.joinpath(*dataset_layout["segments"])
    model_tag = _resolve_model_tag(args, selected_model)
    data_tag = dataset_layout["data_tag"]
    qualifiers = _resolve_lineage_qualifiers(args, dataset_layout, experiment_cfg)
    stem = _compose_stem(
        model_tag, data_tag, qualifiers, _get_active_seed(args)
    )
    checkpoint_path = output_dir / f"{stem}.pth"
    log_path = output_dir / f"{stem}.log"
    metrics_path = output_dir / f"{stem}.metrics.json"
    tb_dir = output_dir / "tb" / stem
    return {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "stem": stem,
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
        "metrics_path": metrics_path,
        "tb_dir": tb_dir,
        "dataset_layout": dataset_layout,
        "model_tag": model_tag,
    }


def _format_selection_tag(monitor: Optional[str]) -> str:
    if not monitor:
        return "best"
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", monitor) if t]
    if not tokens:
        return "best"
    formatted: list[str] = []
    for idx, token in enumerate(tokens):
        lower = token.lower()
        if lower == "auroc":
            piece = "AUROC"
        elif lower == "auc":
            piece = "AUC"
        elif lower == "auprc":
            piece = "AUPRC"
        elif lower == "loss":
            piece = "loss" if idx == 0 else "Loss"
        elif idx == 0:
            piece = token.lower()
        else:
            piece = token.capitalize()
        formatted.append(piece)
    return "".join(formatted)


def _should_trigger_early_stop(
    no_improve_epochs: int,
    patience: int,
    epochs_completed: int,
    min_epochs: int,
) -> bool:
    """Return ``True`` when early stopping criteria are met."""

    if patience <= 0:
        return False
    if epochs_completed < max(min_epochs, 0):
        return False
    return no_improve_epochs >= patience


def _find_existing_checkpoint(stem_path: Path) -> tuple[Optional[Path], bool]:
    pointer = stem_path.with_suffix(".pth")
    if pointer.exists() or pointer.is_symlink():
        return pointer, True
    parent = stem_path.parent
    if not parent.exists():
        return None, False
    pattern = f"{stem_path.name}_e*_*.pth"
    candidates = sorted(parent.glob(pattern))
    if candidates:
        return candidates[-1], False
    return None, False


def _update_checkpoint_pointer(pointer_path: Path, target_path: Path) -> None:
    pointer_dir = pointer_path.parent
    pointer_dir.mkdir(parents=True, exist_ok=True)
    try:
        if pointer_path.is_symlink() or pointer_path.exists():
            pointer_path.unlink()
        pointer_path.symlink_to(target_path.name)
    except OSError:
        shutil.copy2(target_path, pointer_path)


def create_scheduler(optimizer, args):
    name = getattr(args, "scheduler", "none")
    if name is None:
        name = "none"
    name = name.lower()
    if name == "cosine":
        total_epochs = args.epochs
        warmup_epochs = getattr(args, "warmup_epochs", 0)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "plateau":
        min_lr = getattr(args, "min_lr", 1e-6)
        patience = getattr(args, "scheduler_patience", 2)
        factor = getattr(args, "scheduler_factor", 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    return None


def _select_model_config(models, desired_key: str | None):
    if not models:
        raise ValueError("Experiment configuration does not define any models")
    if desired_key:
        for model in models:
            key = model.get("key") or model.get("name")
            if key == desired_key or model.get("name") == desired_key:
                return model
        raise ValueError(
            f"Model key '{desired_key}' not found in experiment configuration. Available: {[m.get('key') or m.get('name') for m in models]}"
        )
    if len(models) == 1:
        return models[0]
    raise ValueError(
        "Experiment configuration defines multiple models; specify one with --model-key"
    )


def _resolve_dataset_specs(
    dataset_cfg: Dict[str, Any],
    *,
    percent_override: Optional[float] = None,
    seed_override: Optional[int] = None,
    size_override: Optional[int] = None,
):
    splits = dataset_cfg.get("splits", {})
    train_split = splits.get("train")
    val_split = splits.get("val")
    test_split = splits.get("test")

    base_pack = dataset_cfg.get("pack")
    fallback_pack = dataset_cfg.get("base_pack", base_pack)
    train_pack = dataset_cfg.get("train_pack")
    if train_pack is None and train_split is not None:
        train_pack = base_pack
    val_pack = dataset_cfg.get("val_pack")
    if val_pack is None and val_split is not None:
        val_pack = fallback_pack
    test_pack = dataset_cfg.get("test_pack")
    if test_pack is None:
        test_pack = fallback_pack

    percent = dataset_cfg.get("percent")
    seed = dataset_cfg.get("seed")
    size = dataset_cfg.get("size")

    if percent is None and percent_override is not None:
        percent = percent_override
    if seed is None and seed_override is not None:
        seed = seed_override
    if size is None and size_override is not None:
        size = size_override

    if percent is not None:
        if isinstance(percent, float) and not float(percent).is_integer():
            raise ValueError("Dataset percent must be an integer when resolving train patterns")
        percent = int(percent)
    if seed is not None:
        seed = int(seed)
    if size is not None:
        size = int(size)

    if "train_pattern" in dataset_cfg:
        if percent is None or seed is None:
            raise ValueError(
                "Dataset configuration requires 'percent' and 'seed' values to resolve train_pattern"
            )
        train_pack = dataset_cfg["train_pattern"].format(percent=percent, seed=seed)

    if "pack_pattern" in dataset_cfg:
        if size is None or seed is None:
            raise ValueError(
                "Dataset configuration requires 'size' and 'seed' values to resolve pack_pattern"
            )
        resolved_pack = dataset_cfg["pack_pattern"].format(size=size, seed=seed)
        train_pack = dataset_cfg.get("train_pack", resolved_pack)
        if test_pack is None:
            test_pack = resolved_pack
        dataset_cfg.setdefault("pack", resolved_pack)

    if percent is not None:
        dataset_cfg["percent"] = percent
    if seed is not None:
        dataset_cfg["seed"] = seed
    if size is not None:
        dataset_cfg["size"] = size

    return {
        "train_pack": train_pack,
        "val_pack": val_pack,
        "test_pack": test_pack,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "percent": percent,
        "seed": seed,
        "size": size,
    }


def _flatten_override_args(raw_overrides: Optional[Iterable[Any]]) -> list[str]:
    """Return a flat list of override entries from argparse output."""

    overrides: list[str] = []
    if not raw_overrides:
        return overrides

    for group in raw_overrides:
        if group is None:
            continue
        if isinstance(group, str):
            entry = group.strip()
            if entry:
                overrides.append(entry)
            continue
        for item in group:
            if item is None:
                continue
            entry = str(item).strip()
            if entry:
                overrides.append(entry)
    return overrides


def _assign_nested_mapping(
    root: Dict[str, Any], keys: Iterable[str], value: Any, *, context: str
) -> None:
    """Assign ``value`` into ``root`` following ``keys`` while validating mappings."""

    keys = list(keys)
    if not keys:
        raise ValueError("Override path must contain at least one key")

    current = root
    for key in keys[:-1]:
        existing = current.get(key)
        if existing is None:
            existing = {}
            current[key] = existing
        elif not isinstance(existing, dict):
            raise TypeError(
                f"Cannot apply override for {'.'.join(keys)}: "
                f"{context} segment '{key}' is not a mapping"
            )
        current = existing
    current[keys[-1]] = value


def _parse_override_entry(entry: str) -> tuple[list[str], Any]:
    """Return (path, value) for a raw override specification."""

    if "=" not in entry:
        raise ValueError(
            f"Invalid override '{entry}'. Expected format section.key=value"
        )
    raw_path, raw_value = entry.split("=", 1)
    path = [segment.strip() for segment in raw_path.split(".") if segment.strip()]
    if not path:
        raise ValueError(
            f"Invalid override '{entry}'. Expected non-empty dot-separated path"
        )
    value = yaml.safe_load(raw_value)
    return path, value


def _apply_config_overrides(
    config: Dict[str, Any], overrides: Iterable[str]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply CLI overrides onto ``config`` and return the applied mapping."""

    overrides = list(overrides or [])
    if not overrides:
        return config, {}

    updated = copy.deepcopy(config)
    applied: Dict[str, Any] = {}
    for entry in overrides:
        path, value = _parse_override_entry(entry)
        _assign_nested_mapping(updated, path, value, context="config")
        _assign_nested_mapping(applied, path, value, context="override")
    return updated, applied


def apply_experiment_config(
    args,
    experiment_cfg: Dict[str, Any],
    *,
    resolved_overrides: Optional[Dict[str, Any]] = None,
):
    dataset_cfg = extract_dataset_config(experiment_cfg)
    dataset_cfg = copy.deepcopy(dataset_cfg)

    cli_percent = getattr(args, "dataset_percent", None)
    cli_seed = getattr(args, "dataset_seed", None)
    cli_size = getattr(args, "dataset_size", None)
    if cli_percent is not None and "percent" not in dataset_cfg:
        dataset_cfg["percent"] = cli_percent
    if cli_size is not None and "size" not in dataset_cfg:
        dataset_cfg["size"] = cli_size

    protocol_cfg = (experiment_cfg or {}).get("protocol") or {}
    config_training_seeds = _normalize_seeds(experiment_cfg.get("seeds"))
    protocol_training_seeds = _normalize_seeds(protocol_cfg.get("seeds"))
    cli_training_seeds = list(getattr(args, "seeds", []) or [])
    if cli_training_seeds:
        resolved_training_seeds = [int(seed) for seed in cli_training_seeds]
    elif protocol_training_seeds:
        resolved_training_seeds = protocol_training_seeds
    else:
        resolved_training_seeds = config_training_seeds

    cli_seed_value = getattr(args, "seed", None)
    if getattr(args, "_seed_explicit", False) and cli_seed_value is not None:
        cli_seed_int = int(cli_seed_value)
        if resolved_training_seeds:
            remaining = [seed for seed in resolved_training_seeds if seed != cli_seed_int]
            resolved_training_seeds = [cli_seed_int, *remaining]
        else:
            resolved_training_seeds = [cli_seed_int]
    args.training_seeds = resolved_training_seeds
    if resolved_training_seeds:
        args.seeds = resolved_training_seeds

    dataset_seed_candidates = _normalize_seeds(dataset_cfg.get("seeds"))
    args.dataset_seeds = dataset_seed_candidates
    dataset_seed_value = _as_int(dataset_cfg.get("seed"))
    if cli_seed is not None:
        dataset_seed_value = int(cli_seed)
    elif dataset_seed_value is None and resolved_training_seeds:
        preferred_seed = resolved_training_seeds[0]
        if not dataset_seed_candidates or preferred_seed in dataset_seed_candidates:
            dataset_seed_value = preferred_seed
    if dataset_seed_value is None and dataset_seed_candidates:
        dataset_seed_value = dataset_seed_candidates[0]
    if dataset_seed_value is not None:
        dataset_cfg["seed"] = int(dataset_seed_value)

    model_cfgs = resolve_model_entries(experiment_cfg.get("models", []))
    selected_model = _select_model_config(model_cfgs, getattr(args, "model_key", None))

    if "optimizer" in experiment_cfg and experiment_cfg["optimizer"].lower() != "adamw":
        raise ValueError("Only AdamW optimizer is currently supported")

    args.lr = experiment_cfg.get("lr", args.lr)
    args.weight_decay = experiment_cfg.get("weight_decay", getattr(args, "weight_decay", 0.05))
    args.batch_size = experiment_cfg.get("batch_size", args.batch_size)
    args.epochs = experiment_cfg.get("epochs", args.epochs)
    config_seed = _as_int(experiment_cfg.get("seed"))
    if config_seed is not None:
        args.config_seed = config_seed
    elif resolved_training_seeds:
        args.config_seed = int(resolved_training_seeds[0])
    args.image_size = experiment_cfg.get("image_size", args.image_size)
    args.workers = experiment_cfg.get("num_workers", args.workers)
    args.prefetch_factor = experiment_cfg.get("prefetch_factor", args.prefetch_factor)
    args.pin_memory = experiment_cfg.get("pin_memory", args.pin_memory)
    args.persistent_workers = experiment_cfg.get("persistent_workers", args.persistent_workers)
    args.log_interval = experiment_cfg.get("log_interval", args.log_interval)

    scheduler_cfg = experiment_cfg.get("scheduler")
    if isinstance(scheduler_cfg, str):
        args.scheduler = scheduler_cfg
        args.warmup_epochs = 0
    elif isinstance(scheduler_cfg, dict):
        args.scheduler = scheduler_cfg.get("name", getattr(args, "scheduler", "none"))
        args.warmup_epochs = scheduler_cfg.get("warmup_epochs", getattr(args, "warmup_epochs", 0))
        args.min_lr = scheduler_cfg.get("min_lr", getattr(args, "min_lr", 1e-6))
        args.scheduler_patience = scheduler_cfg.get(
            "patience", getattr(args, "scheduler_patience", 2)
        )
        args.scheduler_factor = scheduler_cfg.get(
            "factor", getattr(args, "scheduler_factor", 0.5)
        )

    early_cfg = experiment_cfg.get("early_stop", {}) or {}
    args.early_stop_monitor = early_cfg.get(
        "monitor", getattr(args, "early_stop_monitor", "val_loss")
    )
    args.early_stop_patience = early_cfg.get(
        "patience", getattr(args, "early_stop_patience", 0)
    )
    args.early_stop_mode = early_cfg.get(
        "mode", getattr(args, "early_stop_mode", None)
    )
    args.early_stop_min_delta = early_cfg.get(
        "min_delta", getattr(args, "early_stop_min_delta", 0.0)
    )
    args.early_stop_min_epochs = early_cfg.get(
        "min_epochs", getattr(args, "early_stop_min_epochs", 0)
    )

    args.threshold_policy = experiment_cfg.get(
        "threshold_policy", getattr(args, "threshold_policy", None)
    )

    thresholds_cfg = (protocol_cfg.get("thresholds") or {}) if isinstance(protocol_cfg, Mapping) else {}
    primary_policy = thresholds_cfg.get("primary")
    if primary_policy:
        canonical_primary = str(primary_policy)
        args.primary_threshold_policy = canonical_primary
        args.expected_primary_threshold_policy = canonical_primary
        args.threshold_policy = canonical_primary
    else:
        args.primary_threshold_policy = getattr(args, "threshold_policy", None)
        args.expected_primary_threshold_policy = None
    sensitivity_policy = thresholds_cfg.get("sensitivity")
    if sensitivity_policy is not None:
        canonical_sensitivity = str(sensitivity_policy)
        args.sensitivity_threshold_policy = canonical_sensitivity
        args.expected_sensitivity_threshold_policy = canonical_sensitivity
    else:
        if not hasattr(args, "sensitivity_threshold_policy"):
            args.sensitivity_threshold_policy = None
        args.expected_sensitivity_threshold_policy = None

    for limit_key in ("limit_train_batches", "limit_val_batches", "limit_test_batches"):
        limit_value = experiment_cfg.get(limit_key)
        if limit_value is None:
            continue
        current_value = getattr(args, limit_key, None)
        if current_value is None:
            setattr(args, limit_key, int(limit_value))

    amp_enabled = experiment_cfg.get("amp")
    if amp_enabled is not None:
        args.precision = "amp" if amp_enabled else "fp32"

    if "output_dir" in experiment_cfg:
        args.output_dir = str(experiment_cfg["output_dir"])

    args.arch = selected_model.get("arch", args.arch)
    args.pretraining = selected_model.get("pretraining", args.pretraining)
    args.ckpt = selected_model.get("checkpoint", args.ckpt)
    args.frozen = selected_model.get("frozen", args.frozen)
    # protocol_cfg already computed above
    protocol_overrides = {}
    if resolved_overrides and isinstance(resolved_overrides, dict):
        protocol_overrides = resolved_overrides.get("protocol") or {}
    if not isinstance(protocol_overrides, dict):
        protocol_overrides = {}
    config_sources = experiment_cfg.get("__sources__", []) or []
    contains_exp3_source = any(
        Path(str(source)).name == "exp3.yaml" for source in config_sources
    )
    override_requests_morphology = "morphology_eval" in protocol_overrides
    finetune_mode = protocol_cfg.get("finetune")
    default_mode = getattr(args, "finetune_mode", None)
    if not default_mode:
        default_mode = "none" if args.frozen else "full"
    args.finetune_mode = normalise_finetune_mode(
        finetune_mode, default=default_mode
    )
    args.frozen = args.finetune_mode == "none"
    schedule_cfg = protocol_cfg.get("finetune_schedule")
    schedule_spec = _sanitize_finetune_schedule_config(
        schedule_cfg, default_mode=args.finetune_mode
    )
    if schedule_spec:
        schedule_total_epochs = sum(stage["epochs"] for stage in schedule_spec)
        explicit_epochs = experiment_cfg.get("epochs")
        if explicit_epochs is None:
            args.epochs = schedule_total_epochs
        elif int(explicit_epochs) != int(schedule_total_epochs):
            raise ValueError(
                "Experiment epochs ({} ) do not match the total epochs ({}) defined by the fine-tune schedule.".format(
                    explicit_epochs, schedule_total_epochs
                )
            )
        args.finetune_schedule_spec = schedule_spec
        initial_stage_mode = schedule_spec[0]["mode"]
        stage0_head_lr = schedule_spec[0].get("head_lr")
        stage0_base_lr = schedule_spec[0].get("lr")
        if stage0_head_lr is not None:
            args.lr = float(stage0_head_lr)
        elif stage0_base_lr is not None:
            args.lr = float(stage0_base_lr)
        args.finetune_mode = initial_stage_mode
        args.frozen = args.finetune_mode == "none"
    else:
        args.finetune_schedule_spec = []
    args.curve_export_spec = _sanitize_curve_export_config(protocol_cfg.get("export_curves"))
    morphology_eval_cfg = protocol_cfg.get("morphology_eval")
    if morphology_eval_cfg is not None:
        cli_morphology = getattr(args, "morphology_eval", None)
        cli_requested = bool(cli_morphology)
        if cli_requested:
            args.morphology_eval = list(cli_morphology)
        elif contains_exp3_source or override_requests_morphology:
            args.morphology_eval = list(morphology_eval_cfg)
    args.ss_framework = selected_model.get("ss_framework", args.ss_framework)
    args.dataset = dataset_cfg.get("name", args.dataset)

    init_from = protocol_cfg.get("init_from")
    if isinstance(init_from, str):
        init_key = init_from.strip().lower()
        if (
            init_key == "canonical_sun_models"
            and not getattr(args, "parent_checkpoint", None)
        ):
            model_key = (selected_model or {}).get("key") or getattr(
                args, "model_key", None
            )
            if not model_key:
                raise ValueError(
                    "Protocol requested canonical SUN initialisation but model key is undefined."
                )
            seed_value = getattr(args, "seed", None)
            if seed_value is None:
                raise ValueError(
                    "Protocol requested canonical SUN initialisation but seed is undefined."
                )
            try:
                seed_int = int(seed_value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "Seed must be an integer to resolve canonical SUN checkpoints."
                ) from exc
            model_key_lower = str(model_key).lower()
            checkpoint_templates = {
                "sup_imnet": "exp1_sup_imnet_seed{seed}/sup_imnet__SUNFull_s{seed}.pth",
                "ssl_imnet": "exp1_ssl_imnet_seed{seed}/ssl_imnet__SUNFull_s{seed}.pth",
                "ssl_colon": "exp2_ssl_colon_seed{seed}/ssl_colon__SUNFull_s{seed}.pth",
            }
            template = checkpoint_templates.get(model_key_lower)
            if template is None:
                raise ValueError(
                    f"Unsupported model '{model_key}' for canonical SUN initialisation."
                )
            relative_path = template.format(seed=seed_int)
            canonical_root = Path("checkpoints") / "classification"
            args.parent_checkpoint = str(canonical_root / relative_path)

    init_key_lower = str(protocol_cfg.get("init_from", "")).strip().lower()
    dataset_name_lower = str(dataset_cfg.get("name", "")).strip().lower()
    args.is_exp5a = (
        dataset_name_lower == "polypgen_clean_test"
        and args.finetune_mode == "none"
        and init_key_lower == "canonical_sun_models"
    )

    dataset_resolved = _resolve_dataset_specs(
        dataset_cfg,
        percent_override=cli_percent,
        seed_override=cli_seed,
        size_override=cli_size,
    )
    args.train_pack = (
        str(dataset_resolved["train_pack"]) if dataset_resolved["train_pack"] else None
    )
    args.val_pack = (
        str(dataset_resolved["val_pack"]) if dataset_resolved["val_pack"] else None
    )
    args.test_pack = (
        str(dataset_resolved["test_pack"]) if dataset_resolved["test_pack"] else None
    )
    if dataset_resolved["train_split"] is not None:
        args.train_split = dataset_resolved["train_split"]
    if dataset_resolved["val_split"] is not None:
        args.val_split = dataset_resolved["val_split"]
    if dataset_resolved["test_split"] is not None:
        args.test_split = dataset_resolved["test_split"]

    return selected_model, dataset_cfg, dataset_resolved

def train_epoch(
    model,
    rank,
    world_size,
    train_loader,
    train_sampler,
    optimizer,
    epoch,
    loss_fn,
    log_path,
    scaler,
    use_amp,
    tb_logger,
    log_interval,
    global_step,
    seed,
    *,
    device: torch.device,
    distributed: bool,
    max_batches: Optional[int] = None,
    max_steps: Optional[int] = None,
    loss_mode: str = "multiclass_ce",
):
    model.train()
    seed_value = int(seed) if seed is not None else None
    if train_sampler is not None:
        if seed_value is not None:
            train_sampler.set_epoch(seed_value + epoch - 1)
        else:
            train_sampler.set_epoch(epoch)

    remaining_steps = None
    if max_steps is not None:
        remaining_steps = max_steps - int(global_step)
        if remaining_steps <= 0:
            if rank == 0:
                print(
                    f"Skipping training epoch {epoch}: reached max training steps ({max_steps})."
                )
            return TrainEpochStats(
                mean_loss=float("nan"),
                global_step=int(global_step),
                samples_processed=0,
                batches_processed=0,
            )

    t = time.time()
    non_blocking = device.type == "cuda"
    use_amp = use_amp and device.type == "cuda"
    scaler = scaler if use_amp else nullcontext()
    if not isinstance(scaler, torch.cuda.amp.GradScaler):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_accumulator: list[float] = []

    dataset_size = len(train_loader.dataset)
    grad_accum_steps = int(getattr(optimizer, "grad_accum_steps", 1)) or 1

    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if remaining_steps is not None:
        total_batches = min(total_batches, max(remaining_steps, 0))

    batches_processed = 0
    sample_count = 0

    for batch_idx, batch in enumerate(train_loader):
        if batches_processed >= total_batches:
            break
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch
        data = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        if device.type == "cuda":
            autocast_context = torch.cuda.amp.autocast(enabled=use_amp)
        else:
            autocast_context = nullcontext()
        with autocast_context:
            output = model(data)
            loss = _compute_supervised_loss(loss_fn, output, target, mode=loss_mode)
        scaler.scale(loss).backward()
        if use_amp:
            scaler.unscale_(optimizer)
        grad_norm_value = _compute_grad_norm(
            param for group in optimizer.param_groups for param in group["params"]
        )
        scaler.step(optimizer)
        scaler.update()
        loss_scale_value = scaler.get_scale() if use_amp else None
        if distributed:
            dist.all_reduce(loss)
            loss /= world_size
        loss_value = loss.item()
        loss_accumulator.append(loss_value)
        batches_processed += 1
        sample_count += len(data) * world_size

        if rank == 0:
            if tb_logger and global_step % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                tb_logger.log_scalar("loss", loss_value, global_step)
                tb_logger.log_scalar("lr", lr, global_step)
            progress_pct = (
                100.0 * batches_processed / total_batches if total_batches else 0.0
            )
            elapsed = time.time() - t
            throughput = sample_count / elapsed if elapsed > 0 else 0.0
            eta_seconds = None
            if batches_processed < total_batches and batches_processed > 0:
                remaining_batches = total_batches - batches_processed
                eta_seconds = (elapsed / batches_processed) * remaining_batches
            lrs = _collect_param_group_lrs(optimizer)
            mem_allocated = None
            mem_reserved = None
            if device.type == "cuda":
                mem_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            accum_step = ((batches_processed - 1) % grad_accum_steps) + 1
            if batches_processed < total_batches:
                print(
                    "\r"
                    + _format_train_progress(
                        epoch=epoch,
                        sample_count=sample_count,
                        dataset_size=dataset_size,
                        progress_pct=progress_pct,
                        loss_value=loss_value,
                        loss_label="Loss",
                        lrs=lrs,
                        throughput=throughput,
                        eta_seconds=eta_seconds,
                        grad_norm=grad_norm_value,
                        loss_scale=loss_scale_value,
                        mem_allocated=mem_allocated,
                        mem_reserved=mem_reserved,
                        accum_step=accum_step,
                        accum_steps=grad_accum_steps,
                        elapsed_time=elapsed,
                    ),
                    end="",
                )
            else:
                mean_loss_value = (
                    np.mean(loss_accumulator) if loss_accumulator else float("nan")
                )
                printout = _format_train_progress(
                    epoch=epoch,
                    sample_count=sample_count,
                    dataset_size=dataset_size,
                    progress_pct=progress_pct,
                    loss_value=mean_loss_value,
                    loss_label="Average loss",
                    lrs=lrs,
                    throughput=throughput,
                    eta_seconds=0.0,
                    grad_norm=grad_norm_value,
                    loss_scale=loss_scale_value,
                    mem_allocated=mem_allocated,
                    mem_reserved=mem_reserved,
                    accum_step=accum_step,
                    accum_steps=grad_accum_steps,
                    elapsed_time=elapsed,
                )
                print("\r" + printout)
                _append_log_lines(log_path, [printout])
        if distributed:
            dist.barrier()
        global_step += 1
        if remaining_steps is not None and global_step >= max_steps:
            break

    mean_loss = np.mean(loss_accumulator) if loss_accumulator else float("nan")
    return TrainEpochStats(
        mean_loss=float(mean_loss),
        global_step=int(global_step),
        samples_processed=int(sample_count),
        batches_processed=int(batches_processed),
    )


@torch.no_grad()
def test(
    model,
    rank,
    test_loader,
    epoch,
    perf_fn,
    log_path,
    metric_fns: Optional[Dict[str, Any]] = None,
    loss_fn: Optional[nn.Module] = None,
    loss_mode: str = "multiclass_ce",
    split_name: str = "Test",
    return_outputs: bool = False,
    tau: Optional[float] = None,
    tau_info: Optional[str] = None,
    max_batches: Optional[int] = None,
    morphology_eval: Optional[Iterable[str]] = None,
    eval_context: Optional[EvalLoggingContext] = None,
    save_outputs_path: Optional[Path] = None,
):
    if test_loader is None:
        return float("nan")
    t = time.time()
    model.eval()
    metric_fns = metric_fns or {}
    log_path_obj = Path(log_path)
    loss_sum = 0.0
    loss_count = 0
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"
    N = 0
    logits = None
    targets = None
    dataset_batches = len(test_loader)
    dataset_size = len(test_loader.dataset)
    planned_batches = dataset_batches
    if max_batches is not None:
        planned_batches = min(planned_batches, max_batches)
    extra_batch_buffer = EVAL_MAX_ADDITIONAL_BATCHES if max_batches is not None else 0
    hard_batch_limit = min(dataset_batches, planned_batches + extra_batch_buffer)
    batches_processed = 0
    observed_labels: set[int] = set()
    expected_label_count: Optional[int] = None
    expected_labels: Optional[set[int]] = None
    warned_extra_batches = False
    morphology_eval_list = [str(m).strip() for m in (morphology_eval or []) if str(m).strip()]
    morphology_eval_lookup = [m.lower() for m in morphology_eval_list]
    track_morphology = bool(morphology_eval_list)
    morphology_labels: list[str] = []
    morphology_counter: Counter[str] = Counter()
    case_ids: list[str] = []
    dataset_obj = getattr(test_loader, "dataset", None)
    track_perturbations = _dataset_supports_perturbations(dataset_obj)
    perturbation_counter: Counter[str] = Counter()
    perturbation_metadata_rows: list[Mapping[str, Any]] = []
    perturbation_tags: list[str] = []
    perturbation_sample_losses: list[torch.Tensor] = []
    perturbation_tags_aligned: Optional[List[str]] = None
    outputs_path = save_outputs_path.expanduser() if save_outputs_path is not None else None
    metadata_records: list[Mapping[str, Any]] = []

    def _with_prefix(text: str) -> str:
        if eval_context is None:
            return text
        return f"{eval_context.prefix} {text}"

    if eval_context is not None and eval_context.start_lines:
        prefixed = [_with_prefix(line) for line in eval_context.start_lines]
        _log_lines(log_path_obj, prefixed)

    progress_label = _with_prefix(split_name)

    tau_breadcrumb: Optional[str]
    if tau is not None:
        tau_breadcrumb = f"τ={tau:.4f}"
        if tau_info:
            tau_breadcrumb += f" ({tau_info})"
        prediction_mode = "prob>=τ"
    else:
        tau_breadcrumb = "argmax"
        prediction_mode = "argmax"
    breadcrumb_text = _with_prefix(
        f"{split_name} prediction mode: {prediction_mode} [{tau_breadcrumb}]"
    )
    print(breadcrumb_text)
    _append_log_lines(log_path_obj, [breadcrumb_text])

    for batch_idx, batch in enumerate(test_loader):
        if batches_processed >= hard_batch_limit:
            break
        metadata_batch = None
        if len(batch) == 3:
            data, target, metadata_batch = batch
        elif len(batch) == 2:
            raise ValueError("Test dataloader does not provide labels; enable metadata return with labels.")
        else:  # pragma: no cover - defensive
            raise ValueError("Unexpected batch structure returned by dataloader")
        data = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        if metadata_batch is not None and len(metadata_batch) == len(data):
            batch_records = [_coerce_metadata_row(row) for row in metadata_batch]
        else:
            batch_records = [{} for _ in range(len(data))]
        metadata_records.extend(batch_records)
        for row_dict in batch_records:
            case_value = _resolve_metadata_value(
                row_dict,
                (
                    "case_id",
                    "sequence_id",
                    "case",
                    "study_id",
                ),
            )
            if case_value is None:
                case_value = f"case_{len(case_ids)}"
            case_ids.append(case_value)
            morph_raw = row_dict.get("morphology") if isinstance(row_dict, Mapping) else None
            morph_normalised = (
                str(morph_raw).strip().lower() if morph_raw not in (None, "") else "unknown"
            )
            morphology_labels.append(morph_normalised)
            if track_morphology:
                morphology_counter[morph_normalised] += 1
        if track_perturbations:
            if metadata_batch is not None and len(metadata_batch) == len(data):
                for row in metadata_batch:
                    tag_value = "clean"
                    if isinstance(row, Mapping):
                        perturbation_metadata_rows.append(row)
                        tag = _canonicalize_perturbation_tag(row)
                        if tag:
                            perturbation_counter[tag] += 1
                            tag_value = tag
                    perturbation_tags.append(tag_value)
            else:
                perturbation_tags.extend(["clean"] * len(data))
        N += len(data)
        logits_batch = model(data)
        if loss_fn is not None:
            loss_tensor = _compute_supervised_loss(loss_fn, logits_batch, target, mode=loss_mode)
            loss_sum += float(loss_tensor.item()) * len(data)
            loss_count += len(data)
        if track_perturbations:
            try:
                batch_losses = (
                    _compute_sample_losses(logits_batch.detach(), target, mode=loss_mode)
                    .detach()
                    .cpu()
                )
            except Exception:
                batch_losses = torch.full((len(data),), float("nan"))
            perturbation_sample_losses.append(batch_losses)
        output = logits_batch.detach().cpu()
        target_cpu = target.detach().cpu()
        if batch_idx == 0:
            logits = output
            targets = target_cpu
        else:
            logits = torch.cat((logits, output), 0)
            targets = torch.cat((targets, target_cpu), 0)
        batches_processed += 1

        if expected_label_count is None:
            if output.ndim == 1:
                expected_label_count = 1
            else:
                expected_label_count = int(output.size(1))
            if expected_label_count is not None and expected_label_count > 0:
                expected_labels = set(range(expected_label_count))
        observed_labels.update(target_cpu.unique().tolist())
        has_all_expected_labels = False
        if expected_labels is not None:
            has_all_expected_labels = expected_labels.issubset(observed_labels)
        elif expected_label_count is not None:
            has_all_expected_labels = len(observed_labels) >= expected_label_count
        else:
            has_all_expected_labels = True

        if (
            max_batches is not None
            and not has_all_expected_labels
            and batches_processed >= planned_batches
            and not warned_extra_batches
        ):
            extra_note = (
                f"{split_name} evaluation consumed extra batches to observe all labels "
                f"(max_batches={max_batches}, observed={sorted(observed_labels)})."
            )
            print(_with_prefix(extra_note))
            warned_extra_batches = True

        # (Removed per-batch AUROC to avoid single-class crash.)
        # Compute inexpensive interim progress metric(s):
        probs_cumulative = _prepare_binary_probabilities(logits)
        binary_progress = probs_cumulative.size(1) == 2
        if tau is not None and binary_progress:
            preds_cumulative = (probs_cumulative[:, 1] >= tau).to(dtype=torch.long)
        else:
            preds_cumulative = torch.argmax(probs_cumulative, dim=1)
        targets_cumulative = targets  # already CPU
        unique_classes = torch.unique(targets_cumulative)
        if unique_classes.numel() >= 2:
            running_bal_acc = balanced_accuracy_score(
                targets_cumulative.numpy(), preds_cumulative.numpy()
            )
            bal_display = f"BALACC: {running_bal_acc:.6f}"
        else:
            bal_display = "BALACC: (pending)"

        if binary_progress:
            try:
                running_f1 = f1_score(
                    targets_cumulative.numpy(), preds_cumulative.numpy()
                )
                f1_display = f"F1@τ: {running_f1:.6f}"
            except ValueError:
                f1_display = "F1@τ: (pending)"
            try:
                ap_value = average_precision_score(
                    targets_cumulative.numpy(),
                    probs_cumulative[:, 1].numpy(),
                )
                ap_display = f"AP: {ap_value:.6f}"
            except ValueError:
                ap_display = "AP: (pending)"
        else:
            f1_display = "F1@τ: n/a"
            ap_display = "AP: n/a"

        if loss_count > 0:
            loss_display = f"Loss: {loss_sum / loss_count:.6f}"
        elif loss_fn is not None:
            loss_display = "Loss: (pending)"
        else:
            loss_display = None
        metrics_parts = [bal_display, f1_display, ap_display]
        if loss_display is not None:
            metrics_parts.append(loss_display)
        metrics_display = "\t".join(metrics_parts)
        progress_metrics_display = metrics_display
        if track_perturbations:
            tag_summary = _format_top_perturbation_summary(perturbation_counter)
            if tag_summary:
                if progress_metrics_display:
                    progress_metrics_display = f"{progress_metrics_display}\t{tag_summary}"
                else:
                    progress_metrics_display = tag_summary

        progress_reference = planned_batches if planned_batches else max(1, hard_batch_limit)
        progress_measure = (
            min(batches_processed, planned_batches)
            if planned_batches
            else batches_processed
        )
        progress_pct = 100.0 * progress_measure / progress_reference
        reached_planned_batches = batches_processed >= planned_batches
        reached_hard_limit = batches_processed >= hard_batch_limit
        dataset_exhausted = (batch_idx + 1) >= dataset_batches
        can_stop_for_labels = max_batches is not None
        should_stop = (
            dataset_exhausted
            or reached_hard_limit
            or (can_stop_for_labels and has_all_expected_labels and reached_planned_batches)
        )
        if not should_stop:
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\t{}\tTime: {:.6f}".format(
                    progress_label,
                    epoch,
                    N,
                    dataset_size,
                    progress_pct,
                    progress_metrics_display,
                    time.time() - t,
                ),
                end="",
            )
        else:
            printout = "{}  Epoch: {} [{}/{} ({:.1f}%)]\t{}\tTime: {:.6f}".format(
                progress_label,
                epoch,
                N,
                dataset_size,
                progress_pct,
                progress_metrics_display,
                time.time() - t,
            )
            print("\r" + printout)
            _append_log_lines(log_path_obj, [printout])
            break
    probs = _prepare_binary_probabilities(logits)
    n_classes = probs.size(1) if probs.ndim == 2 else 1
    binary_tau = tau if (tau is not None and n_classes == 2) else None
    if logits is not None and targets is not None and len(observed_labels) >= 2:
        auroc_value = float(perf_fn(probs, targets).item())
    else:
        auroc_value = float("nan")
    results: Dict[str, Any] = {"auroc": auroc_value}
    if loss_fn is not None and loss_count > 0:
        mean_loss = loss_sum / loss_count
        results["loss"] = mean_loss
    else:
        mean_loss = float("nan")
    if (
        (not math.isfinite(mean_loss))
        and logits is not None
        and targets is not None
        and probs is not None
    ):
        try:
            positive_scores = _extract_positive_probabilities(probs)
        except Exception:
            positive_scores = None
        if positive_scores is not None and positive_scores.numel() == targets.numel():
            scores_np = positive_scores.detach().cpu().numpy().astype(float, copy=False)
            labels_np = targets.detach().cpu().numpy().astype(int, copy=False)
            if labels_np.size > 0:
                eps = 1e-12
                clipped = np.clip(scores_np, eps, 1.0 - eps)
                computed_loss = float(
                    np.mean(
                        -(labels_np * np.log(clipped) + (1 - labels_np) * np.log(1 - clipped))
                    )
                )
                if math.isfinite(computed_loss):
                    results["loss"] = computed_loss
                    mean_loss = computed_loss
    sample_losses_tensor: Optional[torch.Tensor] = None
    if loss_fn is not None and logits is not None and targets is not None:
        try:
            sample_losses_tensor = _compute_sample_losses(
                logits, targets, mode=loss_mode
            ).detach().cpu()
        except Exception:
            sample_losses_tensor = None

    if binary_tau is not None:
        preds = (probs[:, 1] >= binary_tau).to(dtype=torch.long)
    else:
        preds = torch.argmax(probs, dim=1)
    for name, fn in metric_fns.items():
        try:
            results[name] = fn(probs, targets, tau=binary_tau).item()
        except TypeError:
            results[name] = fn(preds, targets).item()

    class_counts_tensor = torch.bincount(
        targets.to(dtype=torch.long), minlength=n_classes
    )
    class_counts = class_counts_tensor.tolist()
    results["class_counts"] = class_counts
    results["observed_labels"] = sorted(int(x) for x in observed_labels)
    if binary_tau is not None:
        threshold_stats = _compute_threshold_statistics(
            logits, targets, float(binary_tau)
        )
    else:
        threshold_stats = {}
    if threshold_stats:
        results["threshold_metrics"] = threshold_stats
    if binary_tau is not None:
        results["tau"] = float(binary_tau)
    elif tau is not None:
        results["tau"] = float(tau)
    else:
        results["tau"] = None
    if tau_info:
        results["tau_info"] = tau_info

    positive_scores_tensor = _extract_positive_probabilities(probs)
    positive_scores_np = positive_scores_tensor.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    sample_losses_np = (
        sample_losses_tensor.numpy() if sample_losses_tensor is not None else None
    )
    strata_metrics = _build_morphology_strata_metrics(
        positive_scores=positive_scores_np,
        preds=preds_np,
        labels=targets_np,
        morphology=morphology_labels,
        sample_losses=sample_losses_np,
    )
    if strata_metrics:
        results.setdefault("strata", {}).update(strata_metrics)

    if case_ids:
        results["_case_ids"] = list(case_ids)
    if morphology_labels:
        results["_morphology_labels"] = list(morphology_labels)
    if track_perturbations and perturbation_counter:
        results["perturbation_tag_counts"] = dict(sorted(perturbation_counter.items()))
    if track_perturbations and perturbation_metadata_rows:
        results["perturbation_metadata"] = list(perturbation_metadata_rows)
    if track_perturbations and perturbation_tags and logits is not None:
        per_sample_losses_tensor = (
            torch.cat(perturbation_sample_losses, dim=0)
            if perturbation_sample_losses
            else torch.full((logits.size(0),), float("nan"))
        )
        if per_sample_losses_tensor.numel() < logits.size(0):
            padding = torch.full(
                (logits.size(0) - per_sample_losses_tensor.numel(),), float("nan")
            )
            per_sample_losses_tensor = torch.cat((per_sample_losses_tensor, padding), dim=0)
        elif per_sample_losses_tensor.numel() > logits.size(0):
            per_sample_losses_tensor = per_sample_losses_tensor[: logits.size(0)]
        if len(perturbation_tags) < logits.size(0):
            perturbation_tags.extend(["clean"] * (logits.size(0) - len(perturbation_tags)))
        perturbation_tags_aligned = list(perturbation_tags[: logits.size(0)])
        perturbation_case_ids = list(case_ids[: logits.size(0)]) if case_ids else []
        results["perturbation_samples"] = {
            "tags": perturbation_tags_aligned,
            "case_ids": perturbation_case_ids,
            "logits": logits,
            "probabilities": probs,
            "targets": targets,
            "losses": per_sample_losses_tensor,
        }

    observed_sorted = results["observed_labels"]
    if len(observed_sorted) >= 2:
        class_flag = f"classes={{" + ", ".join(map(str, observed_sorted)) + "}}"
    elif observed_sorted:
        class_flag = f"⚠ single-class (labels={{" + ", ".join(map(str, observed_sorted)) + "}})"
    else:
        class_flag = "⚠ no labels observed"
    class_presence_line = _with_prefix(f"{split_name} class presence: {class_flag}")

    metric_line_parts: list[str] = []
    recall_value = results.get("recall")
    if isinstance(recall_value, (float, np.floating)) and not math.isnan(recall_value):
        metric_line_parts.append(f"Recall@τ: {recall_value:.6f}")
    precision_value = results.get("precision")
    if isinstance(precision_value, (float, np.floating)) and not math.isnan(precision_value):
        metric_line_parts.append(f"Precision@τ: {precision_value:.6f}")
    f1_value = results.get("f1")
    if isinstance(f1_value, (float, np.floating)) and not math.isnan(f1_value):
        metric_line_parts.append(f"F1@τ: {f1_value:.6f}")
    bal_value = results.get("balanced_accuracy")
    if isinstance(bal_value, (float, np.floating)) and not math.isnan(bal_value):
        metric_line_parts.append(f"Balanced Acc: {bal_value:.6f}")
    ap_value = results.get("auprc")
    if isinstance(ap_value, (float, np.floating)) and not math.isnan(ap_value):
        metric_line_parts.append(f"AP (PR-AUC): {ap_value:.6f}")
    if not math.isnan(auroc_value):
        metric_line_parts.append(f"AUROC: {auroc_value:.6f}")
    else:
        metric_line_parts.append("AUROC: —")
    if not math.isnan(mean_loss):
        metric_line_parts.append(f"Loss: {mean_loss:.6f}")
    metric_body = " | ".join(metric_line_parts)
    if eval_context is not None:
        metrics_header = f"{split_name} metrics [{eval_context.sample_display}]"
    else:
        metrics_header = f"{split_name} metrics"
    metrics_line = _with_prefix(f"{metrics_header}: {metric_body}")

    if binary_tau is not None:
        tau_text = f"{split_name} τ info: τ={binary_tau:.4f}"
        if tau_info:
            tau_text += f" ({tau_info})"
        tau_line = _with_prefix(tau_text)
    elif tau is not None:
        tau_line = _with_prefix(f"{split_name} τ info: τ={tau:.4f}")
    else:
        tau_line = _with_prefix(f"{split_name} τ info: argmax")

    total_count = int(class_counts_tensor.sum().item())
    if n_classes == 2:
        neg_count = int(class_counts[0]) if len(class_counts) > 0 else 0
        pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
        counts_line = _with_prefix(
            f"{split_name} counts: total={total_count}, negatives={neg_count}, positives={pos_count}"
        )
    else:
        per_class = ", ".join(
            f"{idx}:{int(count)}" for idx, count in enumerate(class_counts) if count
        )
        counts_line = _with_prefix(
            f"{split_name} counts: total={total_count}, per-class={{ {per_class} }}"
        )

    morph_counts_line: Optional[str] = None
    morph_metrics_line: Optional[str] = None
    if track_morphology and morphology_labels and targets is not None:
        requested_counts: Dict[str, int] = {}
        requested_metrics: Dict[str, Dict[str, float]] = {}
        morph_array = np.array(morphology_labels)
        targets_np = targets.numpy()
        preds_np = preds.numpy()
        for display, lookup in zip(morphology_eval_list, morphology_eval_lookup):
            mask = morph_array == lookup
            count = int(mask.sum())
            requested_counts[display] = count
            if count > 0 and n_classes == 2:
                recall_val = recall_score(
                    targets_np[mask], preds_np[mask], zero_division=0
                )
                f1_val = f1_score(
                    targets_np[mask], preds_np[mask], zero_division=0
                )
                requested_metrics[display] = {
                    "recall": float(recall_val),
                    "f1": float(f1_val),
                }
            else:
                requested_metrics[display] = {"recall": float("nan"), "f1": float("nan")}
        full_counts = {name: int(count) for name, count in morphology_counter.items()}
        results["morphology_counts"] = requested_counts
        results["morphology_metrics"] = requested_metrics
        results["observed_morphology_counts"] = full_counts
        display_parts = [f"{name}:{requested_counts.get(name, 0)}" for name in morphology_eval_list]
        extra_keys = [
            key for key in sorted(morphology_counter.keys()) if key not in morphology_eval_lookup
        ]
        display_parts.extend(f"{key}:{morphology_counter[key]}" for key in extra_keys)
        morph_counts_line = _with_prefix(
            f"{split_name} morphology counts: {{ " + ", ".join(display_parts) + " }}"
        )
        metric_parts: list[str] = []
        for name in morphology_eval_list:
            metrics = requested_metrics.get(name, {})
            recall_val = metrics.get("recall")
            f1_val = metrics.get("f1")
            if recall_val is None or math.isnan(recall_val):
                recall_display = "—"
            else:
                recall_display = f"{recall_val:.6f}"
            if f1_val is None or math.isnan(f1_val):
                f1_display = "—"
            else:
                f1_display = f"{f1_val:.6f}"
            metric_parts.append(
                f"{name}: recall={recall_display}, F1={f1_display}"
            )
        morph_metrics_line = _with_prefix(
            f"{split_name} morphology metrics: " + " | ".join(metric_parts)
        )

    confusion_parts: list[str] = []
    if threshold_stats:
        tp_val = threshold_stats.get("tp")
        tn_val = threshold_stats.get("tn")
        fp_val = threshold_stats.get("fp")
        fn_val = threshold_stats.get("fn")
        if isinstance(tp_val, (int, np.integer)):
            confusion_parts.append(f"TP={int(tp_val)}")
        if isinstance(fp_val, (int, np.integer)):
            confusion_parts.append(f"FP={int(fp_val)}")
        if isinstance(fn_val, (int, np.integer)):
            confusion_parts.append(f"FN={int(fn_val)}")
        if isinstance(tn_val, (int, np.integer)):
            confusion_parts.append(f"TN={int(tn_val)}")
        mcc_val = threshold_stats.get("mcc")
        if isinstance(mcc_val, (float, np.floating)) and math.isfinite(mcc_val):
            confusion_parts.append(f"MCC={mcc_val:.6f}")
            results["mcc"] = float(mcc_val)
        prevalence_val = threshold_stats.get("prevalence")
        if isinstance(prevalence_val, (float, np.floating)) and math.isfinite(prevalence_val):
            confusion_parts.append(f"Prev={prevalence_val:.6f}")
            results["prevalence"] = float(prevalence_val)
    confusion_line = _with_prefix(
        f"{split_name} confusion @τ: "
        + (" ".join(confusion_parts) if confusion_parts else "n/a")
    )

    output_lines = [class_presence_line, metrics_line, tau_line, counts_line]
    if morph_counts_line:
        output_lines.append(morph_counts_line)
    if morph_metrics_line:
        output_lines.append(morph_metrics_line)
    output_lines.append(confusion_line)

    if outputs_path is not None and logits is not None and targets is not None:
        sample_total = int(targets.shape[0])
        if len(metadata_records) < sample_total:
            metadata_records.extend({} for _ in range(sample_total - len(metadata_records)))
        if probs.ndim == 2:
            if probs.size(1) == 2:
                prob_tensor = probs[:, 1]
            else:
                prob_tensor = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        else:
            prob_tensor = probs
        probabilities = [float(v) for v in prob_tensor.detach().cpu().tolist()]
        target_list = [int(v) for v in targets.detach().cpu().tolist()]
        pred_list = [int(v) for v in preds.detach().cpu().tolist()]
        _export_frame_outputs(
            outputs_path,
            metadata_rows=metadata_records[:sample_total],
            probabilities=probabilities,
            targets=target_list,
            preds=pred_list,
        )
        results["outputs_path"] = str(outputs_path)

    if (
        track_perturbations
        and perturbation_tags_aligned is not None
        and logits is not None
        and targets is not None
    ):
        per_sample_losses_tensor = results["perturbation_samples"]["losses"]
        metrics_lookup = {
            "recall": performance.meanRecall(n_class=n_classes),
            "precision": performance.meanPrecision(n_class=n_classes),
            "f1": performance.meanF1Score(n_class=n_classes),
            "balanced_accuracy": performance.meanBalancedAccuracy(n_class=n_classes),
            "auroc": performance.meanAUROC(n_class=n_classes),
            "auprc": performance.meanAUPRC(n_class=n_classes),
        }

        def _perturbation_tag_sort_key(tag: str) -> Tuple[Any, ...]:
            if tag == "clean":
                return (0,)
            components: list[Tuple[str, int, Any]] = []
            for segment in str(tag).split("|"):
                name, _, value = segment.partition("=")
                name = name.strip()
                value = value.strip()
                if not name and not value:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    components.append((name, 1, value))
                else:
                    components.append((name, 0, numeric))
            return (1, tuple(components))

        tags_array = np.array(perturbation_tags_aligned)
        unique_tags = sorted(set(tags_array.tolist()), key=_perturbation_tag_sort_key)
        per_tag_metrics: Dict[str, Dict[str, float]] = {}
        per_case_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        case_ids_array: Optional[np.ndarray]
        if case_ids:
            case_ids_array = np.array(case_ids[: logits.size(0)], dtype=object)
        else:
            case_ids_array = None
        preds_cpu = preds.detach().cpu()

        def compute_metrics_for_indices(indices: torch.Tensor) -> Dict[str, float]:
            subset_metrics: Dict[str, float] = {}
            if indices.numel() == 0:
                return subset_metrics
            subset_probs = probs.index_select(0, indices)
            subset_targets = targets.index_select(0, indices)
            subset_losses = per_sample_losses_tensor.index_select(0, indices)
            subset_metrics["count"] = int(indices.numel())
            for key, fn in metrics_lookup.items():
                try:
                    if key in {"recall", "precision", "f1", "balanced_accuracy"}:
                        value = fn(subset_probs, subset_targets, tau=binary_tau)
                    else:
                        value = fn(subset_probs, subset_targets)
                    subset_metrics[key] = float(value.item())
                except Exception:
                    subset_metrics[key] = float("nan")
            if subset_losses.numel() > 0:
                subset_metrics["mean_loss"] = float(subset_losses.mean().item())
            else:
                subset_metrics["mean_loss"] = float("nan")
            return subset_metrics

        clean_metrics: Optional[Dict[str, float]] = None
        for tag in unique_tags:
            indices_np = np.flatnonzero(tags_array == tag)
            if indices_np.size == 0:
                continue
            indices_tensor = torch.from_numpy(indices_np).to(dtype=torch.long)
            metrics_for_tag = compute_metrics_for_indices(indices_tensor)
            if metrics_for_tag:
                per_tag_metrics[tag] = metrics_for_tag
                if tag == "clean":
                    clean_metrics = metrics_for_tag
                if case_ids_array is not None:
                    tag_case_metrics: Dict[str, Dict[str, float]] = {}
                    tag_case_ids = case_ids_array[indices_np]
                    unique_cases = sorted({str(case_id) for case_id in tag_case_ids.tolist()})
                    for case_id in unique_cases:
                        case_mask = tag_case_ids == case_id
                        if not np.any(case_mask):
                            continue
                        case_indices_np = indices_np[case_mask]
                        case_indices_tensor = torch.from_numpy(case_indices_np).to(
                            dtype=torch.long
                        )
                        case_targets = targets.index_select(0, case_indices_tensor)
                        case_preds = preds_cpu.index_select(0, case_indices_tensor)
                        if case_targets.numel() == 0:
                            continue
                        try:
                            recall_val = recall_score(
                                case_targets.numpy(),
                                case_preds.numpy(),
                                zero_division=0,
                            )
                            f1_val = f1_score(
                                case_targets.numpy(),
                                case_preds.numpy(),
                                zero_division=0,
                            )
                        except Exception:
                            continue
                        case_metrics: Dict[str, float] = {
                            "recall": float(recall_val),
                            "f1": float(f1_val),
                            "count": float(int(case_targets.numel())),
                        }
                        tag_case_metrics[str(case_id)] = case_metrics
                    if tag_case_metrics:
                        per_case_metrics[tag] = tag_case_metrics

        non_clean_mask = tags_array != "clean"
        if np.any(non_clean_mask):
            indices_np = np.flatnonzero(non_clean_mask)
            indices_tensor = torch.from_numpy(indices_np).to(dtype=torch.long)
            metrics_for_tag = compute_metrics_for_indices(indices_tensor)
            if metrics_for_tag:
                per_tag_metrics["ALL-perturbed"] = metrics_for_tag
            if case_ids_array is not None:
                tag_case_metrics: Dict[str, Dict[str, float]] = {}
                tag_case_ids = case_ids_array[indices_np]
                unique_cases = sorted({str(case_id) for case_id in tag_case_ids.tolist()})
                for case_id in unique_cases:
                    case_mask = tag_case_ids == case_id
                    if not np.any(case_mask):
                        continue
                    case_indices_np = indices_np[case_mask]
                    case_indices_tensor = torch.from_numpy(case_indices_np).to(
                        dtype=torch.long
                    )
                    case_targets = targets.index_select(0, case_indices_tensor)
                    case_preds = preds_cpu.index_select(0, case_indices_tensor)
                    if case_targets.numel() == 0:
                        continue
                    try:
                        recall_val = recall_score(
                            case_targets.numpy(), case_preds.numpy(), zero_division=0
                        )
                        f1_val = f1_score(
                            case_targets.numpy(), case_preds.numpy(), zero_division=0
                        )
                    except Exception:
                        continue
                    case_metrics = {
                        "recall": float(recall_val),
                        "f1": float(f1_val),
                        "count": float(int(case_targets.numel())),
                    }
                    tag_case_metrics[str(case_id)] = case_metrics
                if tag_case_metrics:
                    per_case_metrics["ALL-perturbed"] = tag_case_metrics

        def _format_metric_display(
            metric_key: str,
            metric_label: str,
            value: float,
            reference: Optional[float],
        ) -> str:
            if value is None or math.isnan(value):
                value_text = "—"
            else:
                value_text = f"{value:.6f}"
            if reference is None or math.isnan(reference):
                retention_text = "ret=—"
                delta_text = "Δ=—"
            else:
                if reference == 0:
                    retention_text = "ret=—"
                else:
                    retention = value / reference if math.isfinite(value) else float("nan")
                    retention_text = (
                        f"ret={retention * 100:.1f}%" if math.isfinite(retention) else "ret=—"
                    )
                delta = value - reference if math.isfinite(value) else float("nan")
                delta_text = f"Δ={delta:+.6f}" if math.isfinite(delta) else "Δ=—"
            return f"{metric_label}={value_text} ({retention_text}, {delta_text})"

        metric_print_order = [
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("f1", "F1@τ"),
            ("balanced_accuracy", "Balanced Acc"),
            ("auroc", "AUROC"),
            ("auprc", "AP"),
            ("mean_loss", "Loss"),
        ]

        ordered_tags: list[str] = []
        if "clean" in per_tag_metrics:
            ordered_tags.append("clean")
        for tag in unique_tags:
            if tag == "clean":
                continue
            ordered_tags.append(tag)
        if "ALL-perturbed" in per_tag_metrics:
            ordered_tags.append("ALL-perturbed")

        perturbation_output_lines: list[str] = []
        for tag in ordered_tags:
            metrics_for_tag = per_tag_metrics.get(tag)
            if not metrics_for_tag:
                continue
            reference_metrics = clean_metrics or {}
            metric_parts = []
            for metric_key, metric_label in metric_print_order:
                value = metrics_for_tag.get(metric_key)
                reference = reference_metrics.get(metric_key)
                metric_parts.append(
                    _format_metric_display(metric_key, metric_label, value, reference)
                )
            count_value = int(metrics_for_tag.get("count", 0))
            line = _with_prefix(
                f"{split_name} perturbation[{tag}]: n={count_value} | "
                + " | ".join(metric_parts)
            )
            perturbation_output_lines.append(line)

        if perturbation_output_lines:
            output_lines.extend(perturbation_output_lines)
        results["perturbation_metrics"] = per_tag_metrics
        if per_case_metrics:
            results["perturbation_case_metrics"] = per_case_metrics

    for line in output_lines:
        print(line)
    _append_log_lines(log_path_obj, output_lines)

    if return_outputs:
        results["logits"] = logits
        results["probabilities"] = probs
        results["targets"] = targets

    return results


def build(args, rank, device: torch.device, distributed: bool):

    pack_root = Path(args.pack_root).expanduser() if args.pack_root else None
    val_spec = args.val_pack or args.train_pack
    test_spec = args.test_pack or val_spec

    loaders, datasets, samplers = create_classification_dataloaders(
        train_spec=args.train_pack,
        val_spec=val_spec,
        test_spec=test_spec,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        rank=rank,
        world_size=args.world_size,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        pack_root=pack_root,
        roots_map=args.roots_map,
        image_size=args.image_size,
        perturbation_splits=args.perturbation_splits,
        hmac_key=args.perturbation_key.encode("utf-8"),
        snapshot_dir=Path(args.output_dir) if rank == 0 else None,
    )

    dataset_summaries: Dict[str, Dict[str, Any]] = {}
    split_specs = {
        "train": (getattr(args, "train_split", None), getattr(args, "train_pack", None)),
        "val": (getattr(args, "val_split", None), getattr(args, "val_pack", None)),
        "test": (getattr(args, "test_split", None), getattr(args, "test_pack", None)),
    }
    for alias, (split_name, pack_spec) in split_specs.items():
        if not split_name:
            continue
        dataset_obj = datasets.get(split_name)
        summary = _summarize_dataset_for_metrics(alias, split_name, dataset_obj, pack_spec)
        if summary:
            dataset_summaries[alias] = summary
    args.dataset_summary = dataset_summaries

    experiment4_trace = _collect_experiment4_trace(args, datasets)

    train_dataloader = loaders.get("train")
    eval_only = train_dataloader is None
    args.eval_only = eval_only
    if train_dataloader is None and not getattr(args, "frozen", False):
        raise RuntimeError(
            "Training dataloader could not be constructed; check --train-pack and --train-split inputs."
        )
    val_dataloader = loaders.get("val")
    test_dataloader = loaders.get("test")
    if val_dataloader is None and not eval_only:
        raise RuntimeError("Validation dataloader missing; specify --val-pack/--val-split.")
    if test_dataloader is None:
        raise RuntimeError("Test dataloader missing; specify --test-pack/--test-split.")
    train_sampler = samplers.get("train")

    eval_context_lookup = _prepare_eval_contexts(args, datasets)
    args.eval_context_lookup = eval_context_lookup

    args.latest_test_outputs_path = None
    args.latest_test_outputs_sha256 = None

    train_dataset = (
        datasets.get(args.train_split) if getattr(args, "train_split", None) else None
    )
    if not eval_only:
        if train_dataset is None or train_dataset.labels_list is None:
            raise ValueError(
                "Training dataset does not provide labels; ensure the selected pack includes labels."
            )

    labelled_datasets = []
    for split_name in (args.train_split, args.val_split, args.test_split):
        if not split_name:
            continue
        candidate = datasets.get(split_name)
        if candidate is not None and candidate.labels_list is not None:
            labelled_datasets.append(candidate)

    label_source = labelled_datasets[0] if labelled_datasets else None
    if label_source is None:
        raise ValueError(
            "No labelled datasets available; ensure at least one split provides labels."
        )

    label_values = list(label_source.labels_list or [])
    n_class = len(set(label_values))
    if n_class == 0:
        raise ValueError("No classes found in available datasets.")

    class_weights: list[float]
    class_counts: Optional[list[int]] = None
    if train_dataset is not None and train_dataset.labels_list is not None:
        train_labels = list(train_dataset.labels_list)
        counts = np.bincount(train_labels, minlength=n_class)
        N_total = len(train_labels)
        class_weights = [
            (N_total / (n_class * count)) if count > 0 else 0.0 for count in counts
        ]
        class_counts = counts.tolist()
    else:
        class_weights = [1.0 for _ in range(n_class)]

    # Override automatically computed class weights if provided by the user
    if args.class_weights is not None:
        class_weights = [float(w) for w in args.class_weights.split(",")]
        if len(class_weights) != n_class:
            raise ValueError("Number of class weights must match number of classes")

    args.loader_limits = {
        "train": getattr(args, "limit_train_batches", None),
        "val": getattr(args, "limit_val_batches", None),
        "test": getattr(args, "limit_test_batches", None),
    }

    if class_counts is not None:
        args.class_counts = list(class_counts)
    else:
        args.class_counts = None

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        assert os.path.exists(args.ckpt)
        model = utils.get_MAE_backbone(
            args.ckpt, True, n_class, args.frozen, None
        )
    elif args.pretraining == "ImageNet_class":
        model = utils.get_ImageNet_or_random_ViT(
            True, n_class, args.frozen, None, ImageNet_weights=True
        )
    elif args.pretraining == "random":
        model = utils.get_ImageNet_or_random_ViT(
            True, n_class, args.frozen, None, ImageNet_weights=False
        )
    stem = getattr(args, "run_stem", None)
    if not stem:
        layout = _resolve_run_layout(args)
        stem = layout["stem"]
        args.run_stem = stem
    stem_path = Path(args.output_dir) / stem
    ckpt_path = stem_path.with_suffix(".pth")
    log_path = stem_path.with_suffix(".log")

    thresholds_map: Dict[str, Any] = {}
    threshold_record_cache: Dict[str, Dict[str, Any]] = {}
    existing_ckpt, pointer_valid = _find_existing_checkpoint(stem_path)
    parent_reference = getattr(args, "parent_checkpoint", None)
    parent_run_reference: Optional[ParentRunReference] = None
    resume_monitor_available = False
    if existing_ckpt is not None:
        main_dict = torch.load(existing_ckpt, map_location="cpu")
        model.load_state_dict(main_dict["model_state_dict"])
        start_epoch = main_dict["epoch"] + 1
        monitor_value = main_dict.get("monitor_value")
        if monitor_value is None and "val_loss" in main_dict:
            monitor_value = main_dict.get("val_loss")
        if monitor_value is None:
            monitor_value = main_dict.get("val_perf")
        else:
            resume_monitor_available = True
        best_val_perf = monitor_value
        random.setstate(main_dict["py_state"])
        np.random.set_state(main_dict["np_state"])
        torch.set_rng_state(main_dict["torch_state"])
        thresholds_map = dict(main_dict.get("thresholds", {}) or {})
        threshold_record_cache = _normalize_threshold_records_map(
            main_dict.get("threshold_records")
        )
        if not pointer_valid:
            _update_checkpoint_pointer(ckpt_path, Path(existing_ckpt))
    elif parent_reference:
        parent_path = Path(parent_reference).expanduser()
        if not parent_path.exists():
            raise FileNotFoundError(
                f"Parent checkpoint '{parent_reference}' does not exist."
            )
        if rank == 0:
            print(f"Loading parent checkpoint from {parent_path}")
        parent_state = torch.load(parent_path, map_location="cpu")
        if isinstance(parent_state, dict) and "model_state_dict" in parent_state:
            state_dict = parent_state["model_state_dict"]
            thresholds_map = dict(parent_state.get("thresholds", {}) or {})
            threshold_record_cache = _normalize_threshold_records_map(
                parent_state.get("threshold_records")
            )
        else:
            state_dict = parent_state
        model.load_state_dict(state_dict)
        start_epoch = 1
        best_val_perf = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        parent_run_reference = _resolve_parent_reference(parent_path)
    else:
        start_epoch = 1
        best_val_perf = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()
        threshold_record_cache = {}

    args.parent_reference = parent_run_reference
    args.cached_threshold_records = threshold_record_cache

    schedule_spec_list = getattr(args, "finetune_schedule_spec", []) or []
    schedule_stages = _materialize_finetune_schedule(
        schedule_spec_list, base_lr=float(getattr(args, "lr", 0.0) or 0.0)
    )
    schedule_runtime: Optional[FinetuneScheduleRuntime]
    if schedule_stages:
        schedule_runtime = FinetuneScheduleRuntime(schedule_stages)
    else:
        schedule_runtime = None

    initial_mode = getattr(args, "finetune_mode", "full")
    if schedule_runtime and schedule_runtime.is_active():
        stage = schedule_runtime.stage_for_epoch(max(int(start_epoch), 1))
        if stage is not None:
            initial_mode = stage.mode
    configure_finetune_parameters(model, initial_mode)
    finetune_param_groups = collect_finetune_param_groups(model)

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        ddp_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [rank]
        model = DDP(model, **ddp_kwargs)
    optimizer_param_groups: list[Dict[str, Any]] = []
    head_params = finetune_param_groups.get("head", [])
    backbone_params = finetune_param_groups.get("backbone", [])
    if head_params:
        optimizer_param_groups.append(
            {"params": head_params, "lr": args.lr, "name": "head"}
        )
    if backbone_params:
        optimizer_param_groups.append(
            {"params": backbone_params, "lr": args.lr, "name": "backbone"}
        )
    if not optimizer_param_groups:
        optimizer_param_groups.append(
            {"params": list(model.parameters()), "lr": args.lr, "name": "head"}
        )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0)
    )
    thresholds_map = dict(thresholds_map)
    raw_policy = (args.threshold_policy or "auto").strip().lower()
    alias_map = {
        "youden": "youden_on_val",
        "f1": "f1_opt_on_val",
        "f1-morph": "f1_opt_on_val",
    }
    raw_policy = alias_map.get(raw_policy, raw_policy)
    allowed_policies = {
        "",
        "auto",
        "none",
        "f1_opt_on_val",
        "youden_on_val",
        "sun_val_frozen",
        "val_opt_youden",
    }
    if raw_policy not in allowed_policies:
        raise ValueError(
            f"Unsupported threshold policy '{raw_policy}'. Use one of {sorted(allowed_policies)}."
        )

    dataset_name = args.dataset or "dataset"
    dataset_name_lower = dataset_name.lower()
    val_split = args.val_split or "val"
    binary_task = len(class_weights) == 2
    resolved_policy = raw_policy
    if resolved_policy in {"", "auto"}:
        if not binary_task:
            resolved_policy = "none"
        elif getattr(args, "is_exp5a", False):
            resolved_policy = "sun_val_frozen"
        elif dataset_name_lower.startswith("polypgen_fewshot"):
            resolved_policy = "val_opt_youden"
        else:
            resolved_policy = "f1_opt_on_val"

    expected_primary_raw = getattr(args, "expected_primary_threshold_policy", None)
    expected_primary_canonical: Optional[str]
    if expected_primary_raw:
        expected_primary_lower = str(expected_primary_raw).strip().lower()
        expected_primary_canonical = alias_map.get(
            expected_primary_lower, expected_primary_lower
        )
        if expected_primary_canonical in {"", "auto"}:
            expected_primary_canonical = None
    else:
        expected_primary_canonical = None
    if expected_primary_canonical and expected_primary_canonical != resolved_policy:
        raise ValueError(
            "Experiment configuration requires primary threshold policy "
            f"'{expected_primary_raw}' (canonical '{expected_primary_canonical}'), "
            f"but resolved policy is '{resolved_policy}'."
        )

    if resolved_policy in {"f1_opt_on_val", "youden_on_val", "val_opt_youden"} and not binary_task:
        if rank == 0:
            print(
                "Warning: threshold policy requires binary classification; disabling threshold computation."
            )
        resolved_policy = "none"

    threshold_key: Optional[str] = None
    if resolved_policy == "sun_val_frozen":
        source_key = getattr(args, "threshold_source_key", None) or "primary"
        parent_ref = getattr(args, "parent_reference", None)
        if not isinstance(parent_ref, ParentRunReference) or not parent_ref.metrics_payload:
            raise ValueError(
                "Policy 'sun_val_frozen' requires a parent run providing stored thresholds."
            )
        tau, record = thresholds.resolve_frozen_sun_threshold(
            parent_ref.metrics_payload.get("thresholds") or {},
            source_key=source_key,
            expected_split_substring="sun_full/val",
            checkpoint_path=parent_ref.checkpoint_path,
        )
        threshold_key = thresholds.format_threshold_key(
            dataset_name, val_split, resolved_policy
        )
        thresholds_map.setdefault(threshold_key, float(tau))
        args.frozen_threshold_record = dict(record)
        args.threshold_source_key = source_key
    else:
        args.frozen_threshold_record = None

    args.threshold_policy = resolved_policy

    raw_sensitivity_policy = getattr(args, "sensitivity_threshold_policy", None)
    sensitivity_policy = None
    if raw_sensitivity_policy is not None:
        raw_sensitivity_policy = str(raw_sensitivity_policy).strip().lower()
        sensitivity_policy = alias_map.get(raw_sensitivity_policy, raw_sensitivity_policy)
        if sensitivity_policy not in allowed_policies - {"", "auto"}:
            raise ValueError(
                f"Unsupported sensitivity threshold policy '{raw_sensitivity_policy}'. Use one of {sorted(allowed_policies - {'', 'auto'})}."
            )

    sensitivity_threshold_key: Optional[str] = None

    if sensitivity_policy in {"f1_opt_on_val", "youden_on_val", "val_opt_youden"} and not binary_task:
        if rank == 0:
            print(
                "Warning: sensitivity threshold policy requires binary classification; disabling sensitivity threshold computation."
            )
        sensitivity_policy = None

    if sensitivity_policy == "sun_val_frozen":
        parent_ref = getattr(args, "parent_reference", None)
        if not isinstance(parent_ref, ParentRunReference) or not parent_ref.metrics_payload:
            raise ValueError(
                "Sensitivity policy 'sun_val_frozen' requires a parent run providing stored thresholds."
            )

    if sensitivity_policy and val_dataloader is None and sensitivity_policy != "sun_val_frozen":
        if rank == 0:
            print(
                "Warning: validation loader unavailable; sensitivity threshold computation disabled."
            )
        sensitivity_policy = None

    if sensitivity_policy:
        sensitivity_threshold_key = thresholds.format_threshold_key(
            dataset_name, val_split, sensitivity_policy
        )

    expected_sensitivity_raw = getattr(args, "expected_sensitivity_threshold_policy", None)
    if expected_sensitivity_raw:
        expected_sensitivity_lower = str(expected_sensitivity_raw).strip().lower()
        expected_sensitivity_canonical = alias_map.get(
            expected_sensitivity_lower, expected_sensitivity_lower
        )
        if expected_sensitivity_canonical in {"", "auto"}:
            expected_sensitivity_canonical = None
    else:
        expected_sensitivity_canonical = None
    if expected_sensitivity_canonical:
        if not sensitivity_policy:
            raise ValueError(
                "Experiment configuration requires sensitivity threshold policy "
                f"'{expected_sensitivity_raw}' (canonical '{expected_sensitivity_canonical}'), "
                "but no sensitivity policy was resolved."
            )
        if sensitivity_policy != expected_sensitivity_canonical:
            raise ValueError(
                "Experiment configuration requires sensitivity threshold policy "
                f"'{expected_sensitivity_raw}' (canonical '{expected_sensitivity_canonical}'), "
                f"but resolved sensitivity policy is '{sensitivity_policy}'."
            )

    args.sensitivity_threshold_policy = sensitivity_policy
    args.sensitivity_threshold_key = sensitivity_threshold_key

    if rank == 0:
        if resolved_policy == "f1_opt_on_val":
            print("Threshold policy resolved to 'f1_opt_on_val' (F1 optimisation on validation split).")
        elif resolved_policy == "youden_on_val":
            print("Threshold policy resolved to 'youden_on_val' (Youden J on validation split).")
        elif resolved_policy == "val_opt_youden":
            print("Threshold policy resolved to 'val_opt_youden' (few-shot validation Youden optimisation).")
        elif resolved_policy == "sun_val_frozen":
            print("Threshold policy resolved to 'sun_val_frozen' (reusing SUN validation τ).")
        elif resolved_policy == "none":
            print("Threshold policy resolved to 'none'; no threshold computation will be performed.")
        if sensitivity_policy:
            if sensitivity_policy == "sun_val_frozen":
                policy_note = "reusing SUN validation τ"
            elif sensitivity_policy == "val_opt_youden":
                policy_note = "few-shot validation Youden optimisation"
            elif sensitivity_policy == "youden_on_val":
                policy_note = "Youden J on validation split"
            elif sensitivity_policy == "f1_opt_on_val":
                policy_note = "F1 optimisation on validation split"
            else:
                policy_note = sensitivity_policy
            print(f"Sensitivity threshold policy resolved to '{sensitivity_policy}' ({policy_note}).")

    compute_threshold = (
        resolved_policy in {"f1_opt_on_val", "youden_on_val", "val_opt_youden"}
        and val_dataloader is not None
    )
    if compute_threshold or resolved_policy == "sun_val_frozen":
        threshold_key = thresholds.format_threshold_key(
            dataset_name, val_split, resolved_policy
        )
    if sensitivity_policy == "sun_val_frozen" and sensitivity_threshold_key not in thresholds_map:
        parent_ref = getattr(args, "parent_reference", None)
        tau_value, record = _resolve_policy_threshold(
            policy=sensitivity_policy,
            dataset=dataset_name,
            split=val_split,
            epoch=start_epoch - 1 if start_epoch > 1 else 0,
            scores=None,
            labels=None,
            previous_tau=None,
            parent_reference=parent_ref,
            source_key=getattr(args, "sensitivity_threshold_source_key", None),
        )
        if tau_value is not None and sensitivity_threshold_key:
            thresholds_map.setdefault(sensitivity_threshold_key, float(tau_value))
            args.frozen_sensitivity_record = record
        else:
            args.frozen_sensitivity_record = None
    else:
        args.frozen_sensitivity_record = None
    use_amp = args.precision == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = create_scheduler(optimizer, args)
    if best_val_perf is not None:
        optimizer.load_state_dict(main_dict["optimizer_state_dict"])
        scaler.load_state_dict(main_dict["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in main_dict:
            scheduler.load_state_dict(main_dict["scheduler_state_dict"])

    args.resume_monitor_available = bool(resume_monitor_available)

    if schedule_runtime is not None and schedule_runtime.is_active():
        schedule_runtime.apply_if_needed(
            model, optimizer, max(int(start_epoch), 1), rank=rank
        )
    args.finetune_schedule_runtime = schedule_runtime

    return (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
        model,
        optimizer,
        ckpt_path,
        log_path,
        start_epoch,
        best_val_perf,
        class_weights,
        scaler,
        scheduler,
        thresholds_map,
        compute_threshold,
        threshold_key,
        experiment4_trace,
        schedule_runtime,
    )


def train(rank, args):

    device = _resolve_device(rank)
    distributed = args.world_size > 1
    backend = None
    if distributed:
        if device.type == "cuda" and torch.distributed.is_nccl_available():
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=args.world_size,
            init_method="tcp://localhost:58472",
        )
    backend_msg = f" using {backend} backend" if backend else ""
    print(
        f"Rank {rank + 1}/{args.world_size} process initialized on {device}.{backend_msg}\n"
    )

    seed = int(getattr(args, "seed", 0)) + int(rank)
    set_determinism(seed)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
        model,
        optimizer,
        ckpt_path,
        log_path,
        start_epoch,
        best_monitor_value,
        class_weights,
        scaler,
        scheduler,
        thresholds_map,
        compute_threshold,
        threshold_key,
        experiment4_trace,
        schedule_runtime,
    ) = build(args, rank, device, distributed)
    eval_context_lookup = getattr(args, "eval_context_lookup", {})
    eval_only = train_dataloader is None
    use_amp = args.precision == "amp" and device.type == "cuda"

    sensitivity_policy = getattr(args, "sensitivity_threshold_policy", None)
    compute_sensitivity_threshold = (
        sensitivity_policy in {"f1_opt_on_val", "youden_on_val", "val_opt_youden"}
        and val_dataloader is not None
    )
    need_val_outputs = compute_threshold or compute_sensitivity_threshold
    sensitivity_threshold_key = getattr(args, "sensitivity_threshold_key", None)
    cached_threshold_records = getattr(args, "cached_threshold_records", None)
    if isinstance(cached_threshold_records, MutableMapping):
        threshold_record_cache = cached_threshold_records
    else:
        threshold_record_cache = {}
        args.cached_threshold_records = threshold_record_cache

    if best_monitor_value is not None:
        try:
            best_monitor_value = float(best_monitor_value)
        except (TypeError, ValueError):
            best_monitor_value = None

    n_classes = len(class_weights)
    class_weights_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
    class_counts = getattr(args, "class_counts", None) or []
    loss_mode = "multiclass_ce"
    if n_classes == 2:
        neg_count = float(class_counts[0]) if len(class_counts) >= 1 else None
        pos_count = float(class_counts[1]) if len(class_counts) >= 2 else None
        if pos_count and pos_count > 0:
            pos_weight_value = neg_count / pos_count if neg_count is not None else 1.0
        elif class_weights_tensor.numel() >= 2 and class_weights_tensor[0] > 0 and class_weights_tensor[1] > 0:
            pos_weight_value = float(class_weights_tensor[1] / class_weights_tensor[0])
        else:
            pos_weight_value = 1.0
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
        )
        loss_mode = "binary_bce"
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    perf_fn = performance.meanAUROC(n_class=n_classes)
    aux_metric_fns = {
        "balanced_accuracy": performance.meanBalancedAccuracy(n_class=n_classes),
        "f1": performance.meanF1Score(n_class=n_classes),
        "precision": performance.meanPrecision(n_class=n_classes),
        "recall": performance.meanRecall(n_class=n_classes),
        "auprc": performance.meanAUPRC(n_class=n_classes),
    }
    morphology_eval = getattr(args, "morphology_eval", None)
    sun_threshold_key = thresholds.format_threshold_key(
        "sun_full", args.val_split or "val", "youden"
    )

    def resolve_eval_tau(*keys: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
        for key in keys:
            if not key:
                continue
            tau_value = thresholds.resolve_threshold(thresholds_map, key)
            if tau_value is not None:
                return float(tau_value), key
        return None, None

    def describe_tau_source(key: Optional[str]) -> Optional[str]:
        if not key:
            return None

        dataset_name, split_name, policy_key = _parse_threshold_key(key)
        if not policy_key:
            return None

        implied_split = POLICY_IMPLIED_SPLITS.get(policy_key)
        split_value = split_name or implied_split

        policy_label = POLICY_LABELS.get(
            policy_key,
            policy_key.replace("_", " ").title(),
        )

        split_labels = {
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
            "train": "training",
        }

        dataset_label = None
        if dataset_name:
            dataset_label = dataset_name.replace("_", " ").strip().title()

        split_label = None
        if split_value:
            split_label = split_labels.get(split_value, split_value.replace("_", " ").strip())

        location_parts = [part for part in (dataset_label, split_label) if part]
        if location_parts:
            location = " ".join(location_parts)
            return f"{policy_label} on {location}"

        return policy_label

    if rank == 0:
        tb_dir = getattr(args, "tensorboard_dir", None)
        tb_path = str(tb_dir) if tb_dir else os.path.join(args.output_dir, "tb")
        tb_logger = TensorboardLogger.create(tb_path)
        _append_log_lines(log_path, [str(args)])
    else:
        tb_logger = TensorboardLogger.create(None)
    if distributed:
        dist.barrier()

    grad_accum_steps = _resolve_grad_accum_steps(optimizer)
    total_seen_samples = 0
    total_batches_processed = 0
    epochs_run = max(0, start_epoch - 1)
    if schedule_runtime is not None and not schedule_runtime.is_active():
        schedule_runtime = None

    if rank == 0 and experiment4_trace is not None and not eval_only:
        subset_lines = ["Experiment4 subset summary (run start):"]
        subset_lines.extend(_format_subset_summary_lines(experiment4_trace))
        per_rank_batch = getattr(train_dataloader, "batch_size", None)
        global_batch = (
            per_rank_batch * args.world_size if per_rank_batch is not None else None
        )
        samples_per_opt = (
            global_batch * grad_accum_steps if global_batch is not None else None
        )
        epochs_planned = getattr(args, "epochs", "n/a")
        max_steps = getattr(args, "max_train_steps", None)
        global_arg_batch = getattr(args, "batch_size", None)
        budget_parts = [
            f"epochs={epochs_planned}",
            f"max_steps={max_steps if max_steps is not None else 'n/a'}",
            f"grad_accum={grad_accum_steps}",
        ]
        if global_arg_batch is not None:
            budget_parts.append(f"batch_size(global_arg)={global_arg_batch}")
        if per_rank_batch is not None:
            budget_parts.append(f"batch_size(per_rank)={per_rank_batch}")
        if global_batch is not None:
            budget_parts.append(f"samples_per_step={global_batch}")
        if samples_per_opt is not None:
            budget_parts.append(
                f"samples_per_optimizer_step={samples_per_opt}"
            )
        subset_lines.append("Planned optimizer budget: " + " | ".join(budget_parts))
        _log_lines(log_path, subset_lines)

    monitor_name = getattr(args, "early_stop_monitor", None) or "val_loss"
    monitor_mode = _resolve_monitor_mode(monitor_name, getattr(args, "early_stop_mode", None))
    monitor_key = _resolve_monitor_key(monitor_name)
    min_delta = _safe_min(float(getattr(args, "early_stop_min_delta", 0.0)))
    if (
        not getattr(args, "resume_monitor_available", False)
        and best_monitor_value is not None
        and monitor_mode == "min"
    ):
        # Old checkpoints tracked AUROC; fall back to reselecting best using the new monitor.
        best_monitor_value = None

    if eval_only:
        eval_epoch = max(start_epoch - 1, 0)
        ckpt_stem = Path(ckpt_path).with_suffix("")
        val_metrics: Optional[Dict[str, Any]] = None
        test_metrics: Optional[Dict[str, Any]] = None
        curve_export_metadata: Optional[Dict[str, Any]] = None
        if rank == 0:
            val_logits_tensor: Optional[torch.Tensor] = None
            val_probabilities_tensor: Optional[torch.Tensor] = None
            val_targets_tensor: Optional[torch.Tensor] = None
            test_probabilities_tensor: Optional[torch.Tensor] = None
            test_targets_tensor: Optional[torch.Tensor] = None
            sensitivity_record: Optional[Dict[str, Any]] = None
            sensitivity_tau: Optional[float] = None
            sensitivity_tau_info: Optional[str] = None
            sensitivity_metrics_raw: Dict[str, Any] = {}
            morphology_block: Dict[str, Dict[str, Any]] = {}
            frozen_sensitivity = getattr(args, "frozen_sensitivity_record", None)
            if isinstance(frozen_sensitivity, Mapping):
                sensitivity_record = dict(_convert_json_compatible(frozen_sensitivity))
            if sensitivity_threshold_key:
                existing_tau = thresholds.resolve_threshold(thresholds_map, sensitivity_threshold_key)
                if existing_tau is not None:
                    sensitivity_tau = float(existing_tau)
            print("No training data provided; running evaluation-only mode.")
            eval_tau, eval_tau_key = resolve_eval_tau(
                threshold_key, sun_threshold_key
            )
            eval_tau_info = describe_tau_source(eval_tau_key)
            if val_dataloader is not None:
                val_metrics = test(
                    _unwrap_model(model),
                    rank,
                    val_dataloader,
                    eval_epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    loss_fn=loss_fn,
                    loss_mode=loss_mode,
                    split_name="Val",
                    return_outputs=need_val_outputs,
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_val_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Val"),
                )
                if isinstance(val_metrics, Mapping):
                    raw_val_logits = val_metrics.pop("logits", None)
                    if isinstance(raw_val_logits, torch.Tensor):
                        val_logits_tensor = raw_val_logits.detach().cpu()
                    raw_val_probabilities = val_metrics.pop("probabilities", None)
                    if isinstance(raw_val_probabilities, torch.Tensor):
                        val_probabilities_tensor = raw_val_probabilities.detach().cpu()
                    raw_val_targets = val_metrics.pop("targets", None)
                    if isinstance(raw_val_targets, torch.Tensor):
                        val_targets_tensor = raw_val_targets.detach().cpu()
                if tb_logger:
                    tb_logger.log_metrics("val", val_metrics, eval_epoch)
            if test_dataloader is not None:
                test_outputs_path = ckpt_stem.parent / f"{ckpt_stem.name}_test_outputs.csv"
                test_metrics = test(
                    _unwrap_model(model),
                    rank,
                    test_dataloader,
                    eval_epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    loss_fn=loss_fn,
                    loss_mode=loss_mode,
                    split_name="Test",
                    return_outputs=True,
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_test_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Test"),
                    save_outputs_path=test_outputs_path,
                )
                _record_test_outputs_digest(args, test_outputs_path)
                curve_export_metadata = _maybe_export_curves_for_split(
                    args,
                    ckpt_stem=ckpt_stem,
                    split_name="Test",
                    metrics=test_metrics,
                    log_path=log_path,
                )
                if isinstance(test_metrics, Mapping):
                    raw_test_probabilities = test_metrics.get("probabilities")
                    if isinstance(raw_test_probabilities, torch.Tensor):
                        test_probabilities_tensor = raw_test_probabilities.detach().cpu()
                    raw_test_targets = test_metrics.get("targets")
                    if isinstance(raw_test_targets, torch.Tensor):
                        test_targets_tensor = raw_test_targets.detach().cpu()
                    morphology_block = _build_morphology_block(test_metrics.get("strata"))
                    test_metrics.pop("_case_ids", None)
                    test_metrics.pop("_morphology_labels", None)
                    if curve_export_metadata:
                        test_metrics.setdefault("curve_exports", {})["test"] = (
                            curve_export_metadata
                        )
                if tb_logger:
                    tb_logger.log_metrics("test", test_metrics, eval_epoch)

            val_metrics_export = (
                _prepare_metric_export(val_metrics) if isinstance(val_metrics, Mapping) else {}
            )
            test_metrics_export = (
                _prepare_metric_export(test_metrics) if isinstance(test_metrics, Mapping) else {}
            )
            base_metrics_for_sensitivity: Optional[Mapping[str, Any]] = (
                test_metrics if isinstance(test_metrics, Mapping) else None
            )
            if sensitivity_policy:
                tau_value, record = _resolve_sensitivity_threshold(
                    policy=sensitivity_policy,
                    threshold_key=sensitivity_threshold_key,
                    dataset=dataset_name,
                    split=val_split,
                    epoch=int(eval_epoch),
                    val_probabilities=val_probabilities_tensor,
                    val_logits=val_logits_tensor,
                    val_targets=val_targets_tensor,
                    thresholds_map=thresholds_map,
                    parent_reference=getattr(args, "parent_reference", None),
                    source_key=getattr(args, "sensitivity_threshold_source_key", None),
                )
                if tau_value is not None:
                    sensitivity_tau = float(tau_value)
                if record:
                    sensitivity_record = dict(_convert_json_compatible(record))
            sensitivity_block: Dict[str, Any]
            if (
                sensitivity_tau is not None
                and test_probabilities_tensor is not None
                and test_targets_tensor is not None
            ):
                if sensitivity_tau_info is None:
                    if isinstance(sensitivity_record, Mapping):
                        info_value = sensitivity_record.get("info") or sensitivity_record.get("tau_info")
                        if isinstance(info_value, str) and info_value:
                            sensitivity_tau_info = info_value
                    if not sensitivity_tau_info and sensitivity_threshold_key:
                        sensitivity_tau_info = describe_tau_source(sensitivity_threshold_key)
                sensitivity_metrics_raw = _compute_metrics_for_probability_threshold(
                    probabilities=test_probabilities_tensor,
                    targets=test_targets_tensor,
                    tau=sensitivity_tau,
                    base_metrics=base_metrics_for_sensitivity,
                    tau_info=sensitivity_tau_info,
                )
                sensitivity_block = _build_metric_block(sensitivity_metrics_raw)
            else:
                sensitivity_metrics_raw = {}
                sensitivity_block = {}
            val_block = _build_metric_block(val_metrics_export)
            test_primary_block = _build_metric_block(test_metrics_export)
            threshold_sources: Dict[str, str] = {}
            if isinstance(val_metrics, Mapping):
                val_tau_info = val_metrics.get("tau_info")
                if isinstance(val_tau_info, str) and val_tau_info:
                    threshold_sources["val"] = val_tau_info
            if isinstance(test_metrics, Mapping):
                test_tau_info = test_metrics.get("tau_info")
                if isinstance(test_tau_info, str) and test_tau_info:
                    threshold_sources["test"] = test_tau_info
            if sensitivity_tau_info:
                threshold_sources["sensitivity"] = sensitivity_tau_info
            cached_threshold_records = getattr(args, "cached_threshold_records", None)
            tau_value: Optional[float] = None
            if isinstance(test_primary_block, Mapping):
                tau_candidate = test_primary_block.get("tau")
                if isinstance(tau_candidate, (int, float, np.integer, np.floating)) and math.isfinite(float(tau_candidate)):
                    tau_value = float(tau_candidate)
            tau_info_value = None
            if isinstance(test_metrics, Mapping):
                tau_info_candidate = test_metrics.get("tau_info")
                if isinstance(tau_info_candidate, str) and tau_info_candidate:
                    tau_info_value = tau_info_candidate
            dataset_summary = getattr(args, "dataset_summary", None)
            try:
                data_block = _build_result_loader_data_block(dataset_summary)
            except RuntimeError as exc:
                raise RuntimeError("Unable to construct metrics data block") from exc
            val_data_path: Optional[str] = None
            if data_block:
                val_entry = data_block.get("val")
                if isinstance(val_entry, Mapping):
                    candidate_path = str(val_entry.get("path") or "").strip()
                    if candidate_path:
                        val_data_path = candidate_path

            primary_record = _resolve_primary_threshold_record(
                threshold_key=eval_tau_key,
                threshold_records=cached_threshold_records if isinstance(cached_threshold_records, Mapping) else None,
                frozen_record=getattr(args, "frozen_threshold_record", None),
                parent_reference=getattr(args, "parent_reference", None),
            )
            primary_metadata: Optional[Dict[str, Any]]
            if primary_record:
                primary_metadata = dict(primary_record)
            else:
                primary_metadata = {}
            if tau_value is not None:
                primary_metadata["tau"] = float(tau_value)
            dataset_name_from_key, split_name_from_key, policy_name_from_key = _parse_threshold_key(eval_tau_key)
            resolved_policy_label = (
                str(getattr(args, "threshold_policy", "")).strip().lower() or None
            )
            if resolved_policy_label:
                record_policy = primary_metadata.get("policy")
                record_policy_lower = (
                    str(record_policy).strip().lower() if record_policy is not None else None
                )
                if record_policy_lower and record_policy_lower != resolved_policy_label:
                    raise ValueError(
                        "Stored primary threshold record policy "
                        f"'{record_policy}' does not match resolved policy '{resolved_policy_label}'."
                    )
                if not record_policy_lower:
                    primary_metadata["policy"] = resolved_policy_label
            elif policy_name_from_key:
                primary_metadata.setdefault("policy", str(policy_name_from_key).strip().lower())
            if dataset_name_from_key and split_name_from_key:
                primary_metadata.setdefault(
                    "split", f"{dataset_name_from_key}/{split_name_from_key}"
                )
            elif split_name_from_key:
                primary_metadata.setdefault("split", str(split_name_from_key))
            if eval_tau_key:
                primary_metadata.setdefault("source_key", str(eval_tau_key))
            if tau_info_value:
                primary_metadata.setdefault("info", tau_info_value)
            if val_data_path:
                primary_metadata = _update_threshold_split(
                    primary_metadata, split_path=val_data_path
                )
            if not primary_metadata:
                primary_metadata = None

            thresholds_block = _build_thresholds_block(
                thresholds_map,
                policy=getattr(args, "threshold_policy", None),
                sources=threshold_sources,
                primary=primary_metadata,
                sensitivity=_update_threshold_split(
                    sensitivity_record, split_path=val_data_path
                ),
            )
            selection_tag = _format_selection_tag(getattr(args, "early_stop_monitor", None))
            run_block = _build_run_metadata(args, selection_tag=selection_tag)
            provenance_block = _build_metrics_provenance(
                args, experiment4_trace=experiment4_trace
            )
            metrics_payload: Dict[str, Any] = {
                "seed": _get_active_seed(args),
                "epoch": int(eval_epoch),
                "eval_only": True,
                "val": val_block,
                "test_primary": test_primary_block,
                "test_sensitivity": sensitivity_block,
                "provenance": provenance_block,
            }
            if morphology_block:
                metrics_payload["test_morphology"] = morphology_block
            if run_block:
                metrics_payload["run"] = run_block
            perturbation_block = _build_perturbation_export(test_metrics)
            if perturbation_block:
                metrics_payload["test_perturbations"] = perturbation_block
            if data_block:
                metrics_payload["data"] = data_block
            if dataset_summary:
                metrics_payload["dataset"] = dataset_summary
            if threshold_sources:
                if "val" in threshold_sources:
                    metrics_payload["val_tau_source"] = threshold_sources["val"]
                if "test" in threshold_sources:
                    metrics_payload["test_tau_source"] = threshold_sources["test"]
            if thresholds_block:
                metrics_payload["thresholds"] = thresholds_block
            if getattr(args, "threshold_policy", None):
                metrics_payload.setdefault(
                    "threshold_policy", args.threshold_policy
                )
            if sensitivity_policy:
                metrics_payload.setdefault(
                    "sensitivity_threshold_policy", sensitivity_policy
                )
            if curve_export_metadata:
                metrics_payload.setdefault("curve_exports", {})["test"] = (
                    curve_export_metadata
                )
            if getattr(args, "is_exp5a", False):
                domain_shift_block = _compute_domain_shift_delta(
                    test_primary_block if isinstance(test_primary_block, Mapping) else None,
                    getattr(args, "parent_reference", None),
                )
                if domain_shift_block:
                    metrics_payload["domain_shift_delta"] = domain_shift_block
            metrics_path = ckpt_stem.with_suffix(".metrics.json")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)

        tb_logger.close()
        if distributed:
            dist.destroy_process_group()
        return

    ckpt_pointer = Path(ckpt_path)
    ckpt_stem = ckpt_pointer.with_suffix("")
    selection_tag = _format_selection_tag(getattr(args, "early_stop_monitor", None))

    global_step = 0
    no_improve_epochs = 0
    scheduler_name = getattr(args, "scheduler", "none").lower()
    early_patience = max(0, int(getattr(args, "early_stop_patience", 0) or 0))
    early_min_epochs = max(0, int(getattr(args, "early_stop_min_epochs", 0) or 0))
    best_epoch: Optional[int] = None
    if best_monitor_value is not None:
        prior_epoch = start_epoch - 1
        if prior_epoch >= 1:
            best_epoch = int(prior_epoch)
    last_epoch: Optional[int] = None
    last_loss: Optional[float] = None
    last_val_perf: Optional[float] = None
    last_test_perf: Optional[float] = None
    last_monitor_value: Optional[float] = None
    last_val_loss: Optional[float] = None
    last_test_loss: Optional[float] = None
    last_test_monitor: Optional[float] = None
    last_val_metrics_export: Optional[Dict[str, float]] = None
    last_test_metrics_export: Optional[Dict[str, float]] = None
    last_train_lr: Optional[float] = None
    last_train_lr_groups: Dict[str, float] = {}
    last_val_tau_info: Optional[str] = None
    last_test_tau_info: Optional[str] = None
    last_sensitivity_tau_info: Optional[str] = None
    val_logits: Optional[torch.Tensor] = None
    val_targets: Optional[torch.Tensor] = None
    val_probabilities: Optional[torch.Tensor] = None
    val_case_ids: Optional[List[str]] = None
    val_morphology_labels: Optional[List[str]] = None
    latest_threshold_file_relpath: Optional[str] = None
    frozen_initial_record = getattr(args, "frozen_threshold_record", None)
    if isinstance(frozen_initial_record, Mapping):
        latest_threshold_record = dict(_convert_json_compatible(frozen_initial_record))
    else:
        latest_threshold_record = None
    latest_thresholds_root: Optional[str] = None
    frozen_sensitivity_initial = getattr(args, "frozen_sensitivity_record", None)
    if isinstance(frozen_sensitivity_initial, Mapping):
        latest_sensitivity_record = dict(_convert_json_compatible(frozen_sensitivity_initial))
    else:
        latest_sensitivity_record = None
    latest_sensitivity_tau: Optional[float] = None
    if sensitivity_threshold_key:
        try:
            existing_sensitivity_tau = thresholds.resolve_threshold(
                thresholds_map, sensitivity_threshold_key
            )
        except Exception:
            existing_sensitivity_tau = None
        if existing_sensitivity_tau is not None:
            latest_sensitivity_tau = float(existing_sensitivity_tau)

    for epoch in range(start_epoch, args.epochs + 1):
        if schedule_runtime is not None:
            stage = schedule_runtime.apply_if_needed(
                model, optimizer, epoch, rank=rank
            )
            if stage is not None:
                args.finetune_mode = stage.mode
                args.frozen = stage.mode == "none"
        try:
            prev_global_step = global_step
            epoch_stats = train_epoch(
                model,
                rank,
                args.world_size,
                train_dataloader,
                train_sampler,
                optimizer,
                epoch,
                loss_fn,
                log_path,
                scaler,
                use_amp,
                tb_logger,
                args.log_interval,
                global_step,
                _get_active_seed(args),
                device=device,
                distributed=distributed,
                max_batches=args.limit_train_batches,
                max_steps=args.max_train_steps,
                loss_mode=loss_mode,
            )
            loss = float(epoch_stats.mean_loss)
            global_step = int(epoch_stats.global_step)
            total_seen_samples += int(epoch_stats.samples_processed)
            total_batches_processed += int(epoch_stats.batches_processed)
            if epoch_stats.batches_processed > 0:
                epochs_run += 1
            if rank == 0:
                train_lr_groups_map: Dict[str, float] = {}
                train_lr_display_parts: list[str] = []
                primary_lr = None
                train_metrics_payload: Dict[str, float] = {"loss": float(loss)}
                for idx, group in enumerate(optimizer.param_groups):
                    lr_value = float(group["lr"])
                    group_name = str(group.get("name") or f"group{idx}")
                    train_metrics_payload[f"{group_name}_lr"] = lr_value
                    train_lr_groups_map[group_name] = lr_value
                    train_lr_display_parts.append(f"{group_name}={lr_value:.2e}")
                    if primary_lr is None:
                        primary_lr = lr_value
                if primary_lr is not None:
                    train_metrics_payload["lr"] = primary_lr
                if tb_logger:
                    tb_logger.log_metrics("train", train_metrics_payload, epoch)
                last_train_lr = primary_lr
                last_train_lr_groups = dict(train_lr_groups_map)
                train_lr_summary = ", ".join(train_lr_display_parts) if train_lr_display_parts else "n/a"
                eval_tau, eval_tau_key = resolve_eval_tau(
                    threshold_key, sun_threshold_key
                )
                eval_tau_info = describe_tau_source(eval_tau_key)
                val_metrics = test(
                    _unwrap_model(model),
                    rank,
                    val_dataloader,
                    epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    loss_fn=loss_fn,
                    loss_mode=loss_mode,
                    split_name="Val",
                    return_outputs=need_val_outputs,
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_val_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Val"),
                )
                val_tau_info = val_metrics.get("tau_info")
                if isinstance(val_tau_info, str) and val_tau_info:
                    last_val_tau_info = val_tau_info
                else:
                    last_val_tau_info = None
                val_metrics_export = _prepare_metric_export(
                    val_metrics, drop={"logits", "probabilities", "targets"}
                )
                test_metrics_export: Dict[str, float] = {}
                raw_val_logits = val_metrics.pop("logits", None)
                val_logits = (
                    raw_val_logits.detach().cpu() if isinstance(raw_val_logits, torch.Tensor) else None
                )
                raw_val_probabilities = val_metrics.pop("probabilities", None)
                if isinstance(raw_val_probabilities, torch.Tensor):
                    val_probabilities = raw_val_probabilities.detach().cpu()
                else:
                    val_probabilities = None
                raw_val_targets = val_metrics.pop("targets", None)
                val_targets = (
                    raw_val_targets.detach().cpu() if isinstance(raw_val_targets, torch.Tensor) else None
                )
                raw_val_case_ids = val_metrics.pop("_case_ids", None)
                val_case_ids = list(raw_val_case_ids) if raw_val_case_ids is not None else None
                raw_val_morph = val_metrics.pop("_morphology_labels", None)
                val_morphology_labels = (
                    list(raw_val_morph) if raw_val_morph is not None else None
                )
                val_loss_value = val_metrics.get("loss")
                monitor_value = val_metrics.get(monitor_key)
                if monitor_value is None:
                    available = ", ".join(sorted(val_metrics.keys()))
                    raise KeyError(
                        f"Validation metrics do not contain monitor '{monitor_key}'. Available: {available}"
                    )
                test_monitor_value = None
                test_loss_value = None
                test_perf = None
                val_perf = float(val_metrics["auroc"])
                if tb_logger:
                    tb_logger.log_metrics("val", val_metrics, epoch)
                def _format_epoch_metric(value: Any) -> str:
                    if value is None:
                        return "—"
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            return "—"
                        value = value.item()
                    if isinstance(value, np.generic):
                        value = float(value)
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        return "—"
                    if not math.isfinite(numeric):
                        return "—"
                    return f"{numeric:.6f}"

                summary_line = (
                    f"[epoch {epoch}] train: loss={_format_epoch_metric(loss)}"
                    f", lr={train_lr_summary} | val: loss={_format_epoch_metric(val_loss_value)}"
                    f", auprc={_format_epoch_metric(val_metrics.get('auprc'))}"
                    f", auroc={_format_epoch_metric(val_metrics.get('auroc'))}"
                )
                print(summary_line)
                _append_log_lines(log_path, [summary_line])
                last_epoch = int(epoch)
                last_loss = float(loss)
                last_val_perf = float(val_perf)
                last_monitor_value = float(monitor_value)
                last_val_loss = float(val_loss_value) if val_loss_value is not None else None
                last_test_loss = None
                last_test_monitor = None
                last_test_perf = None
                last_val_metrics_export = dict(val_metrics_export)
                last_test_metrics_export = dict(test_metrics_export)
            else:
                val_perf = 0.0
                test_perf = None
                monitor_value = 0.0
                test_monitor_value = None
                val_loss_value = None
                test_loss_value = None
                last_test_monitor = None

            steps_this_epoch = global_step - prev_global_step
            should_step = steps_this_epoch > 0
            if scheduler is not None:
                if scheduler_name == "plateau":
                    plateau_metric = (
                        float(monitor_value) if monitor_value is not None else float(val_perf)
                    )
                    if distributed:
                        metric_tensor = torch.tensor(
                            [plateau_metric if rank == 0 else 0.0], device=device
                        )
                        dist.broadcast(metric_tensor, src=0)
                        if should_step:
                            scheduler.step(metric_tensor.item())
                    elif should_step:
                        scheduler.step(plateau_metric)
                elif should_step:
                    scheduler.step()
            if distributed:
                dist.barrier()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)

        if rank == 0:
            improved = _improved(
                float(monitor_value),
                best_monitor_value,
                mode=monitor_mode,
                min_delta=min_delta,
            )
            checkpoint_saved = False
            if improved:
                if experiment4_trace is not None:
                    checkpoint_lines = [
                        "Experiment4 checkpoint update:",
                        "Steps: global_step={gs} | optimizer_steps={os} | seen_samples={ss}".format(
                            gs=int(global_step),
                            os=_compute_optimizer_steps(global_step, grad_accum_steps),
                            ss=int(total_seen_samples),
                        ),
                        "Subset tag: " + _format_subset_tag(experiment4_trace),
                    ]
                    _log_lines(log_path, checkpoint_lines)
                print("Saving...")
                _append_log_lines(log_path, ["Saving..."])
                dataset_label = args.dataset or "dataset"
                val_split_label = args.val_split or "val"
                updated_thresholds = dict(thresholds_map)
                threshold_file_relpath: Optional[str] = None
                thresholds_root_path: Optional[Path] = None
                threshold_record: Optional[Dict[str, Any]] = None
                frozen_record = getattr(args, "frozen_threshold_record", None)
                if isinstance(frozen_record, Mapping):
                    threshold_record = dict(_convert_json_compatible(frozen_record))
                sensitivity_threshold_record: Optional[Dict[str, Any]] = (
                    dict(latest_sensitivity_record) if latest_sensitivity_record else None
                )
                if (
                    compute_threshold
                    and threshold_key is not None
                    and val_logits is not None
                    and val_targets is not None
                ):
                    policy_result: Optional[thresholds.ThresholdPolicyResult] = None
                    try:
                        if val_probabilities is not None:
                            positive_scores = _extract_positive_probabilities(val_probabilities)
                        else:
                            positive_scores = _extract_positive_probabilities(
                                _prepare_binary_probabilities(val_logits)
                            )
                        scores_np = positive_scores.detach().cpu().numpy().astype(float)
                        labels_np = val_targets.detach().cpu().numpy().astype(int)
                        previous_tau = thresholds.resolve_threshold(updated_thresholds, threshold_key)
                        split_label = f"{dataset_label}/{val_split_label}"
                        policy_result = thresholds.compute_policy_threshold(
                            scores_np,
                            labels_np,
                            policy=args.threshold_policy,
                            split_name=split_label,
                            epoch=int(epoch),
                            previous_tau=previous_tau,
                        )
                    except ValueError as exc:
                        print(
                            f"Warning: unable to compute threshold '{threshold_key}': {exc}"
                        )
                        policy_result = None
                    if policy_result is not None:
                        tau = float(policy_result.tau)
                        updated_thresholds[threshold_key] = tau
                        threshold_record = dict(_convert_json_compatible(policy_result.record))
                        threshold_record.setdefault("source_key", threshold_key)
                        metrics_at_tau = _compute_threshold_statistics(
                            val_logits, val_targets, tau
                        )
                        metrics_at_tau = _convert_json_compatible(metrics_at_tau)
                        threshold_record.setdefault("metrics_at_tau", metrics_at_tau)
                        if threshold_key:
                            threshold_record_cache[str(threshold_key)] = dict(threshold_record)
                        thresholds_root_path = Path(
                            getattr(args, "thresholds_root", None)
                            or (Path(args.output_dir).expanduser().parent / "thresholds")
                        ).expanduser()
                        subdir = _resolve_thresholds_subdir(args)
                        model_tag = getattr(args, "model_tag", None)
                        if not model_tag:
                            stem = getattr(args, "run_stem", None)
                            if not stem:
                                stem = Path(ckpt_path).with_suffix("").name
                            model_tag = stem.split("__", 1)[0] if "__" in stem else stem
                        model_tag = _sanitize_path_segment(model_tag, default="model")
                        threshold_dir = thresholds_root_path / subdir
                        threshold_dir.mkdir(parents=True, exist_ok=True)
                        threshold_file = threshold_dir / f"{model_tag}.json"
                        threshold_record_payload = _build_threshold_payload(
                            args,
                            threshold_key=threshold_key,
                            tau=tau,
                            metrics_at_tau=metrics_at_tau,
                            val_metrics=val_metrics_export,
                            test_metrics=test_metrics_export,
                            val_perf=val_perf,
                            test_perf=test_perf,
                            model_tag=model_tag,
                            subdir=subdir,
                            policy_record=threshold_record,
                            snapshot_epoch=int(epoch),
                            snapshot_tau=previous_tau,
                        )
                        with threshold_file.open("w", encoding="utf-8") as handle:
                            json.dump(threshold_record_payload, handle, indent=2)
                        try:
                            threshold_file_relpath = str(
                                threshold_file.relative_to(thresholds_root_path.parent)
                            )
                        except ValueError:
                            threshold_file_relpath = str(threshold_file)
                        print(
                            f"Updated threshold {threshold_key} = {tau:.6f} -> {threshold_file_relpath}"
                        )
                if sensitivity_policy and sensitivity_threshold_key:
                    tau_value, record = _resolve_sensitivity_threshold(
                        policy=sensitivity_policy,
                        threshold_key=sensitivity_threshold_key,
                        dataset=dataset_label,
                        split=val_split_label,
                        epoch=int(epoch),
                        val_probabilities=val_probabilities,
                        val_logits=val_logits,
                        val_targets=val_targets,
                        thresholds_map=updated_thresholds,
                        parent_reference=getattr(args, "parent_reference", None),
                        source_key=getattr(args, "sensitivity_threshold_source_key", None),
                    )
                    if tau_value is not None:
                        latest_sensitivity_tau = float(tau_value)
                    if record:
                        sensitivity_threshold_record = dict(_convert_json_compatible(record))
                        if sensitivity_threshold_key:
                            threshold_record_cache[str(sensitivity_threshold_key)] = dict(
                                sensitivity_threshold_record
                            )
                elif sensitivity_threshold_key and sensitivity_threshold_key in updated_thresholds:
                    try:
                        latest_sensitivity_tau = float(updated_thresholds[sensitivity_threshold_key])
                    except (TypeError, ValueError):
                        latest_sensitivity_tau = latest_sensitivity_tau
                if sensitivity_threshold_record is not None:
                    latest_sensitivity_record = dict(sensitivity_threshold_record)
                    info_value = sensitivity_threshold_record.get("info") or sensitivity_threshold_record.get("tau_info")
                    if isinstance(info_value, str) and info_value:
                        last_sensitivity_tau_info = info_value
                dataset_summary = getattr(args, "dataset_summary", None)
                try:
                    data_block = _build_result_loader_data_block(dataset_summary)
                except RuntimeError as exc:
                    raise RuntimeError("Unable to construct metrics data block") from exc
                val_data_path: Optional[str] = None
                if data_block:
                    val_entry = data_block.get("val")
                    if isinstance(val_entry, Mapping):
                        candidate_path = str(val_entry.get("path") or "").strip()
                        if candidate_path:
                            val_data_path = candidate_path
                threshold_record = _update_threshold_split(
                    threshold_record, split_path=val_data_path
                )
                sensitivity_threshold_record = _update_threshold_split(
                    sensitivity_threshold_record, split_path=val_data_path
                )
                if threshold_key is not None and threshold_record is not None:
                    threshold_record_cache[str(threshold_key)] = dict(threshold_record)
                if (
                    sensitivity_threshold_key is not None
                    and sensitivity_threshold_record is not None
                ):
                    threshold_record_cache[str(sensitivity_threshold_key)] = dict(
                        sensitivity_threshold_record
                    )
                model_to_save = _unwrap_model(model)
                payload = {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": loss,
                    "val_perf": float(monitor_value),
                    "val_loss": float(val_loss_value)
                    if val_loss_value is not None
                    else None,
                    "val_auroc": float(val_perf),
                    "test_perf": float(test_monitor_value)
                    if test_monitor_value is not None
                    else (
                        float(test_perf) if test_perf is not None else None
                    ),
                    "test_loss": float(test_loss_value)
                    if test_loss_value is not None
                    else None,
                    "test_auroc": float(test_perf) if test_perf is not None else None,
                    "monitor_value": float(monitor_value),
                    "monitor_metric": monitor_name,
                    "test_monitor_value": float(test_monitor_value)
                    if test_monitor_value is not None
                    else None,
                    "py_state": random.getstate(),
                    "np_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                }
                if scheduler is not None:
                    payload["scheduler_state_dict"] = scheduler.state_dict()
                if updated_thresholds:
                    payload["thresholds"] = updated_thresholds
                if threshold_file_relpath:
                    payload.setdefault("threshold_files", {})[
                        threshold_key
                    ] = threshold_file_relpath
                    if threshold_record is not None:
                        payload.setdefault("threshold_records", {})[
                            threshold_key
                        ] = threshold_record
                elif threshold_record is not None and threshold_key is not None:
                    payload.setdefault("threshold_records", {})[
                        threshold_key
                    ] = threshold_record
                if (
                    sensitivity_threshold_record is not None
                    and sensitivity_threshold_key is not None
                ):
                    payload.setdefault("threshold_records", {})[
                        sensitivity_threshold_key
                    ] = sensitivity_threshold_record
                if thresholds_root_path is not None:
                    payload["thresholds_root"] = str(thresholds_root_path)
                    latest_thresholds_root = str(thresholds_root_path)
                candidate_name = (
                    f"{ckpt_stem.name}_e{epoch:02d}_{selection_tag}.pth"
                )
                final_path = ckpt_stem.parent / candidate_name
                if final_path.exists():
                    digest_payload = {
                        "epoch": int(epoch),
                        "seed": _get_active_seed(args),
                        "val": float(val_perf),
                        "test": float(test_perf) if test_perf is not None else None,
                    }
                    digest = hashlib.sha1(
                        json.dumps(digest_payload, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:8]
                    candidate_name = (
                        f"{ckpt_stem.name}_e{epoch:02d}_{selection_tag}+{digest}.pth"
                    )
                    final_path = ckpt_stem.parent / candidate_name
                torch.save(payload, final_path)
                _update_checkpoint_pointer(ckpt_pointer, final_path)
                val_block = _build_metric_block(val_metrics_export)
                test_primary_block = _build_metric_block(test_metrics_export)
                sensitivity_block: Dict[str, Any] = {}
                morphology_block_epoch: Dict[str, Dict[str, Any]] = {}
                provenance_block = _build_metrics_provenance(
                    args, experiment4_trace=experiment4_trace
                )
                threshold_sources: Dict[str, str] = {}
                if isinstance(val_tau_info, str) and val_tau_info:
                    threshold_sources["val"] = val_tau_info
                if last_test_tau_info:
                    threshold_sources["test"] = last_test_tau_info
                if isinstance(sensitivity_threshold_record, Mapping):
                    info_value = sensitivity_threshold_record.get("info") or sensitivity_threshold_record.get("tau_info")
                    if isinstance(info_value, str) and info_value:
                        threshold_sources["sensitivity"] = info_value
                thresholds_block = _build_thresholds_block(
                    updated_thresholds,
                    policy=getattr(args, "threshold_policy", None),
                    sources=threshold_sources,
                    primary=threshold_record,
                    sensitivity=sensitivity_threshold_record,
                )
                run_block = _build_run_metadata(args, selection_tag=selection_tag)
                metrics_payload: Dict[str, Any] = {
                    "seed": _get_active_seed(args),
                    "epoch": int(epoch),
                    "train_loss": float(loss),
                    "monitor_value": float(monitor_value),
                    "monitor_metric": monitor_name,
                    "val": val_block,
                    "test_primary": test_primary_block,
                    "test_sensitivity": sensitivity_block,
                    "provenance": provenance_block,
                }
                if morphology_block_epoch:
                    metrics_payload["test_morphology"] = morphology_block_epoch
                if run_block:
                    metrics_payload["run"] = run_block
                if data_block:
                    metrics_payload["data"] = data_block
                if dataset_summary:
                    metrics_payload["dataset"] = dataset_summary
                if isinstance(val_tau_info, str) and val_tau_info:
                    metrics_payload["val_tau_source"] = val_tau_info
                if last_train_lr is not None:
                    metrics_payload["train_lr"] = float(last_train_lr)
                if last_train_lr_groups:
                    metrics_payload["train_lr_groups"] = {
                        key: float(value) for key, value in last_train_lr_groups.items()
                    }
                if threshold_file_relpath:
                    metrics_payload["threshold_files"] = {
                        threshold_key: threshold_file_relpath
                    }
                if thresholds_block:
                    metrics_payload["thresholds"] = thresholds_block
                if getattr(args, "threshold_policy", None):
                    metrics_payload.setdefault("threshold_policy", args.threshold_policy)
                if sensitivity_policy:
                    metrics_payload.setdefault(
                        "sensitivity_threshold_policy", sensitivity_policy
                    )
                metrics_path = ckpt_stem.with_suffix(".metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics_payload, f, indent=2)
                if threshold_file_relpath:
                    latest_threshold_file_relpath = threshold_file_relpath
                    latest_threshold_record = threshold_record
                best_monitor_value = float(monitor_value)
                best_epoch = int(epoch)
                thresholds_map = updated_thresholds
                no_improve_epochs = 0
                checkpoint_saved = True
            else:
                no_improve_epochs += 1

            if best_monitor_value is not None and math.isnan(best_monitor_value):
                best_monitor_value = None
                best_epoch = None
            if best_monitor_value is not None:
                if best_epoch is not None:
                    best_display = f"{best_monitor_value:.3f} (epoch {best_epoch})"
                else:
                    best_display = f"{best_monitor_value:.3f}"
            else:
                best_display = "—"
            if early_patience > 0:
                patience_display = f"{no_improve_epochs}/{early_patience}"
            else:
                patience_display = "—"
            checkpoint_note = "checkpoint saved" if checkpoint_saved else "no checkpoint"
            patience_line = (
                f"Patience tracker: best={best_display} | patience={patience_display} | {checkpoint_note}"
            )
            print(patience_line)
            _append_log_lines(log_path, [patience_line])

        if early_patience > 0:
            if distributed:
                stop_tensor = torch.tensor(
                    [
                        1
                        if (
                            rank == 0
                            and _should_trigger_early_stop(
                                no_improve_epochs,
                                early_patience,
                                epochs_run,
                                early_min_epochs,
                            )
                        )
                        else 0
                    ],
                    device=device,
                )
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item():
                    if rank == 0:
                        print(
                            "Early stopping triggered after reaching patience limit."
                        )
                    break
            elif rank == 0 and _should_trigger_early_stop(
                no_improve_epochs,
                early_patience,
                epochs_run,
                early_min_epochs,
            ):
                print("Early stopping triggered after reaching patience limit.")
                break

        if distributed:
            dist.barrier()
    final_test_metrics: Optional[Dict[str, Any]] = None
    curve_export_metadata: Optional[Dict[str, Any]] = None
    final_tau_info: Optional[str] = None
    if (
        rank == 0
        and not eval_only
        and test_dataloader is not None
        and (last_epoch is not None or args.epochs is not None)
    ):
        final_eval_epoch = int(last_epoch) if last_epoch is not None else int(args.epochs)
        final_tau, final_tau_key = resolve_eval_tau(threshold_key, sun_threshold_key)
        final_tau_info = describe_tau_source(final_tau_key)
        if (
            final_tau is None
            and compute_threshold
            and threshold_key is not None
            and val_logits is not None
            and val_targets is not None
        ):
            computed_tau: Optional[float] = None
            if (
                args.threshold_policy == "f1-morph"
                and val_probabilities is not None
                and val_morphology_labels
            ):
                try:
                    positive_scores = _extract_positive_probabilities(val_probabilities)
                    computed_tau = _compute_f1_morph_threshold(
                        positive_scores=positive_scores.detach().cpu().numpy(),
                        labels=val_targets.detach().cpu().numpy(),
                        morphology=list(val_morphology_labels),
                    )
                except Exception:
                    computed_tau = None
            if computed_tau is None:
                try:
                    computed_tau = thresholds.compute_youden_j_threshold(
                        val_logits, val_targets
                    )
                except ValueError as exc:
                    print(
                        f"Warning: unable to compute final threshold '{threshold_key}': {exc}"
                    )
            if computed_tau is not None:
                final_tau = float(computed_tau)
                final_tau_info = describe_tau_source(threshold_key) or final_tau_info
                thresholds_map = dict(thresholds_map or {})
                thresholds_map[threshold_key] = float(final_tau)
        test_outputs_path = ckpt_stem.parent / f"{ckpt_stem.name}_test_outputs.csv"
        final_test_metrics = test(
            _unwrap_model(model),
            rank,
            test_dataloader,
            final_eval_epoch,
            perf_fn,
            log_path,
            metric_fns=aux_metric_fns,
            loss_fn=loss_fn,
            loss_mode=loss_mode,
            split_name="Test",
            return_outputs=True,
            tau=final_tau,
            tau_info=final_tau_info,
            max_batches=args.limit_test_batches,
            morphology_eval=morphology_eval,
            eval_context=eval_context_lookup.get("Test"),
            save_outputs_path=test_outputs_path,
        )
        _record_test_outputs_digest(args, test_outputs_path)
        curve_export_metadata = _maybe_export_curves_for_split(
            args,
            ckpt_stem=ckpt_stem,
            split_name="Test",
            metrics=final_test_metrics,
            log_path=log_path,
        )
        final_morphology_block: Dict[str, Dict[str, Any]] = {}
        final_test_probabilities_tensor: Optional[torch.Tensor] = None
        final_test_targets_tensor: Optional[torch.Tensor] = None
        if isinstance(final_test_metrics, Mapping):
            raw_final_probabilities = final_test_metrics.get("probabilities")
            if isinstance(raw_final_probabilities, torch.Tensor):
                final_test_probabilities_tensor = raw_final_probabilities.detach().cpu()
            raw_final_targets = final_test_metrics.get("targets")
            if isinstance(raw_final_targets, torch.Tensor):
                final_test_targets_tensor = raw_final_targets.detach().cpu()
            final_morphology_block = _build_morphology_block(final_test_metrics.get("strata"))
        sensitivity_record_final: Optional[Dict[str, Any]] = (
            dict(latest_sensitivity_record) if latest_sensitivity_record else None
        )
        sensitivity_tau_final = latest_sensitivity_tau
        if sensitivity_tau_final is None and sensitivity_policy and sensitivity_threshold_key:
            tau_value, record = _resolve_sensitivity_threshold(
                policy=sensitivity_policy,
                threshold_key=sensitivity_threshold_key,
                dataset=dataset_name,
                split=val_split,
                epoch=int(final_eval_epoch),
                val_probabilities=val_probabilities,
                val_logits=val_logits,
                val_targets=val_targets,
                thresholds_map=thresholds_map,
                parent_reference=getattr(args, "parent_reference", None),
                source_key=getattr(args, "sensitivity_threshold_source_key", None),
            )
            if tau_value is not None:
                sensitivity_tau_final = float(tau_value)
            if record:
                sensitivity_record_final = dict(_convert_json_compatible(record))
        if (
            sensitivity_record_final is not None
            and sensitivity_threshold_key
        ):
            threshold_record_cache[str(sensitivity_threshold_key)] = dict(
                sensitivity_record_final
            )
        if sensitivity_tau_final is not None and sensitivity_threshold_key:
            thresholds_map[sensitivity_threshold_key] = float(sensitivity_tau_final)
        sensitivity_tau_info_final: Optional[str] = None
        if isinstance(sensitivity_record_final, Mapping):
            info_value = sensitivity_record_final.get("info") or sensitivity_record_final.get("tau_info")
            if isinstance(info_value, str) and info_value:
                sensitivity_tau_info_final = info_value
        if not sensitivity_tau_info_final and sensitivity_threshold_key:
            sensitivity_tau_info_final = describe_tau_source(sensitivity_threshold_key)
        sensitivity_metrics_final = _compute_metrics_for_probability_threshold(
            probabilities=final_test_probabilities_tensor,
            targets=final_test_targets_tensor,
            tau=sensitivity_tau_final,
            base_metrics=final_test_metrics if isinstance(final_test_metrics, Mapping) else None,
            tau_info=sensitivity_tau_info_final,
        )
        final_sensitivity_block = _build_metric_block(sensitivity_metrics_final)
        if sensitivity_tau_info_final:
            last_sensitivity_tau_info = sensitivity_tau_info_final
        if sensitivity_record_final is not None:
            latest_sensitivity_record = dict(sensitivity_record_final)
            if sensitivity_tau_final is not None:
                latest_sensitivity_tau = float(sensitivity_tau_final)
        final_test_metrics.pop("_case_ids", None)
        final_test_metrics.pop("_morphology_labels", None)
        if curve_export_metadata:
            final_test_metrics.setdefault("curve_exports", {})["test"] = curve_export_metadata
        if tb_logger:
            tb_logger.log_metrics("test", final_test_metrics, final_eval_epoch)
        last_test_metrics_export = dict(
            _prepare_metric_export(final_test_metrics)
        )
        test_loss_value = final_test_metrics.get("loss")
        last_test_loss = float(test_loss_value) if test_loss_value is not None else None
        test_monitor_value = final_test_metrics.get(monitor_key)
        last_test_monitor = (
            float(test_monitor_value) if test_monitor_value is not None else None
        )
        final_test_auroc = final_test_metrics.get("auroc")
        last_test_perf = (
            float(final_test_auroc) if final_test_auroc is not None else None
        )
        last_test_tau_info = final_tau_info

    if rank == 0 and experiment4_trace is not None and not eval_only:
        limit_train_batches = getattr(args, "limit_train_batches", None)
        final_lines = [
            "Experiment4 final summary:",
            "Run summary: epochs_run={epochs} | final_global_step={gs} | optimizer_steps={os} | final_seen_samples={samples}".format(
                epochs=int(epochs_run),
                gs=int(global_step),
                os=_compute_optimizer_steps(global_step, grad_accum_steps),
                samples=int(total_seen_samples),
            ),
            "Train loader: actual_batches={batches} | drop_last={drop_last} | limit_train_batches={limit_batches}".format(
                batches=int(total_batches_processed),
                drop_last=getattr(train_dataloader, "drop_last", "n/a"),
                limit_batches=limit_train_batches if limit_train_batches is not None else "n/a",
            ),
            "Subset tag: " + _format_subset_tag(experiment4_trace),
        ]
        _log_lines(log_path, final_lines)

    tb_logger.close()
    if rank == 0 and last_epoch is not None:
        dataset_summary = getattr(args, "dataset_summary", None)
        try:
            data_block_final = _build_result_loader_data_block(dataset_summary)
        except RuntimeError as exc:
            raise RuntimeError("Unable to construct metrics data block") from exc
        val_data_path: Optional[str] = None
        if data_block_final:
            val_entry = data_block_final.get("val")
            if isinstance(val_entry, Mapping):
                candidate_path = str(val_entry.get("path") or "").strip()
                if candidate_path:
                    val_data_path = candidate_path
        latest_threshold_record = _update_threshold_split(
            latest_threshold_record, split_path=val_data_path
        )
        sensitivity_record_final = _update_threshold_split(
            sensitivity_record_final, split_path=val_data_path
        )
        if threshold_key is not None and latest_threshold_record is not None:
            threshold_record_cache[str(threshold_key)] = dict(latest_threshold_record)
        if (
            sensitivity_threshold_key is not None
            and sensitivity_record_final is not None
        ):
            threshold_record_cache[str(sensitivity_threshold_key)] = dict(
                sensitivity_record_final
            )
        last_payload: Dict[str, Any] = {
            "epoch": int(last_epoch),
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": float(last_loss) if last_loss is not None else None,
            "val_perf": float(last_monitor_value)
            if last_monitor_value is not None
            else None,
            "val_loss": float(last_val_loss) if last_val_loss is not None else None,
            "val_auroc": float(last_val_perf) if last_val_perf is not None else None,
            "test_perf": float(last_test_monitor)
            if last_test_monitor is not None
            else (float(last_test_perf) if last_test_perf is not None else None),
            "test_loss": float(last_test_loss) if last_test_loss is not None else None,
            "test_auroc": float(last_test_perf) if last_test_perf is not None else None,
            "monitor_value": float(last_monitor_value)
            if last_monitor_value is not None
            else None,
            "monitor_metric": monitor_name,
            "py_state": random.getstate(),
            "np_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
        }
        if scheduler is not None:
            last_payload["scheduler_state_dict"] = scheduler.state_dict()
        if thresholds_map:
            last_payload["thresholds"] = thresholds_map
        if latest_threshold_file_relpath and threshold_key is not None:
            last_payload.setdefault("threshold_files", {})[
                threshold_key
            ] = latest_threshold_file_relpath
            if latest_threshold_record is not None:
                last_payload.setdefault("threshold_records", {})[
                    threshold_key
                ] = latest_threshold_record
        elif latest_threshold_record is not None and threshold_key is not None:
            last_payload.setdefault("threshold_records", {})[
                threshold_key
            ] = latest_threshold_record
        if (
            latest_sensitivity_record is not None
            and sensitivity_threshold_key is not None
        ):
            last_payload.setdefault("threshold_records", {})[
                sensitivity_threshold_key
            ] = latest_sensitivity_record
        if latest_thresholds_root is not None:
            last_payload["thresholds_root"] = latest_thresholds_root
        last_candidate_name = (
            f"{ckpt_stem.name}_last_e{last_epoch:02d}_{selection_tag}.pth"
        )
        last_final_path = ckpt_stem.parent / last_candidate_name
        if last_final_path.exists():
            digest_payload = {
                "epoch": int(last_epoch),
                "seed": _get_active_seed(args),
                "val": float(last_monitor_value)
                if last_monitor_value is not None
                else None,
                "test": float(last_test_monitor)
                if last_test_monitor is not None
                else (float(last_test_perf) if last_test_perf is not None else None),
            }
            digest = hashlib.sha1(
                json.dumps(digest_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:8]
            last_candidate_name = (
                f"{ckpt_stem.name}_last_e{last_epoch:02d}_{selection_tag}+{digest}.pth"
            )
            last_final_path = ckpt_stem.parent / last_candidate_name
        torch.save(last_payload, last_final_path)
        last_pointer_path = ckpt_stem.parent / f"{ckpt_stem.name}_last.pth"
        _update_checkpoint_pointer(last_pointer_path, last_final_path)
        val_block = _build_metric_block(last_val_metrics_export)
        test_primary_block = _build_metric_block(last_test_metrics_export)
        sensitivity_block = final_sensitivity_block
        provenance_block = _build_metrics_provenance(
            args, experiment4_trace=experiment4_trace
        )
        threshold_sources: Dict[str, str] = {}
        if last_val_tau_info:
            threshold_sources["val"] = last_val_tau_info
        if last_test_tau_info:
            threshold_sources["test"] = last_test_tau_info
        if last_sensitivity_tau_info:
            threshold_sources["sensitivity"] = last_sensitivity_tau_info
        thresholds_block = _build_thresholds_block(
            thresholds_map,
            policy=getattr(args, "threshold_policy", None),
            sources=threshold_sources,
            primary=latest_threshold_record,
            sensitivity=sensitivity_record_final,
        )
        run_block = _build_run_metadata(args, selection_tag=selection_tag)
        last_metrics_payload: Dict[str, Any] = {
            "seed": _get_active_seed(args),
            "epoch": int(last_epoch),
            "train_loss": float(last_loss) if last_loss is not None else None,
            "monitor_value": float(last_monitor_value)
            if last_monitor_value is not None
            else None,
            "monitor_metric": monitor_name,
            "val": val_block,
            "test_primary": test_primary_block,
            "test_sensitivity": sensitivity_block,
            "provenance": provenance_block,
        }
        if final_morphology_block:
            last_metrics_payload["test_morphology"] = final_morphology_block
        if run_block:
            last_metrics_payload["run"] = run_block
        perturbation_block = _build_perturbation_export(final_test_metrics)
        if perturbation_block:
            last_metrics_payload["test_perturbations"] = perturbation_block
        if data_block_final:
            last_metrics_payload["data"] = data_block_final
        if dataset_summary:
            last_metrics_payload["dataset"] = dataset_summary
        if last_val_tau_info:
            last_metrics_payload["val_tau_source"] = last_val_tau_info
        if last_test_tau_info:
            last_metrics_payload["test_tau_source"] = last_test_tau_info
        if last_train_lr is not None:
            last_metrics_payload["train_lr"] = float(last_train_lr)
        if last_train_lr_groups:
            last_metrics_payload["train_lr_groups"] = {
                key: float(value) for key, value in last_train_lr_groups.items()
            }
        if latest_threshold_file_relpath and threshold_key is not None:
            last_metrics_payload["threshold_files"] = {
                threshold_key: latest_threshold_file_relpath
            }
        if thresholds_block:
            last_metrics_payload["thresholds"] = thresholds_block
        if getattr(args, "threshold_policy", None):
            last_metrics_payload.setdefault("threshold_policy", args.threshold_policy)
        if sensitivity_policy:
            last_metrics_payload.setdefault(
                "sensitivity_threshold_policy", sensitivity_policy
            )
        if curve_export_metadata:
            last_metrics_payload.setdefault("curve_exports", {})["test"] = curve_export_metadata
        last_metrics_path = ckpt_stem.parent / f"{ckpt_stem.name}_last.metrics.json"
        with open(last_metrics_path, "w") as handle:
            json.dump(last_metrics_payload, handle, indent=2)
    if distributed:
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained model for classification"
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        dest="exp_config",
        help="Experiment configuration reference (relative to configs/ or absolute path)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        dest="overrides",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override experiment configuration entries (e.g. dataset.percent=5). "
            "May be specified multiple times."
        ),
    )
    parser.add_argument(
        "--model-key",
        type=str,
        dest="model_key",
        help="Model key to select when an experiment defines multiple models",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["vit_b"],
        default="vit_b",
        dest="arch",
    )
    parser.add_argument(
        "--pretraining",
        type=str,
        default=None,
        choices=["Hyperkvasir", "ImageNet_class", "ImageNet_self", "random"],
    )
    parser.add_argument("--ss-framework", type=str, choices=["mae"])
    parser.add_argument("--checkpoint", type=str, dest="ckpt", default=None)
    parser.add_argument(
        "--parent-checkpoint",
        type=str,
        dest="parent_checkpoint",
        default=None,
        help="Path to the checkpoint of a parent run when fine-tuning hierarchically.",
    )
    parser.add_argument(
        "--finetune-mode",
        type=str,
        default="full",
        choices=["none", "full", "head+2"],
        help="Fine-tuning regime controlling which encoder layers receive gradients.",
    )
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--train-pack",
        type=str,
        default=None,
        help="Pack specification (name, directory or manifest) providing the training split",
    )
    parser.add_argument(
        "--val-pack",
        type=str,
        default=None,
        help="Pack specification for validation split (defaults to --train-pack)",
    )
    parser.add_argument(
        "--test-pack",
        type=str,
        default=None,
        help="Pack specification for test split (defaults to --val-pack)",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--morphology-eval",
        nargs="+",
        dest="morphology_eval",
        default=None,
        help=(
            "Optional list of morphology categories (e.g. flat polypoid) to "
            "enable morphology-aware logging."
        ),
    )
    parser.add_argument(
        "--dataset-percent",
        type=float,
        default=None,
        help=(
            "Optional percentage of the dataset to sample for constrained smoke tests. "
            "Values <= 0 disable subsetting."
        ),
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help=(
            "Seed used when randomly subsetting the dataset for smoke tests. "
            "Negative values disable deterministic subsetting."
        ),
    )
    parser.add_argument(
        "--pack-root",
        type=str,
        default=str(data_packs_root()),
        help="Base directory containing data packs (defaults to repository data_packs/)",
    )
    parser.add_argument(
        "--roots",
        type=str,
        default=str(Path("data") / "roots.json"),
        help=("JSON file mapping manifest root identifiers to directories. Defaults to data/roots.json"),
    )
    parser.add_argument(
        "--perturbation-key",
        type=str,
        default="ssl4polyp",
        help="HMAC key used for deterministic per-row perturbations",
    )
    parser.add_argument(
        "--perturbation-splits",
        nargs="*",
        default=["test"],
        help="Splits (case-insensitive) where on-the-fly perturbations should be enabled",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Comma-separated list of class weights to override automatic computation",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "plateau"],
        default="none",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help=(
            "Optional hard limit on total training steps for fast smoke tests. "
            "Values <= 0 disable the limit."
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stop-monitor", type=str, default="val_loss")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--early-stop-mode",
        type=str,
        default="auto",
        choices=["min", "max", "auto"],
        help="Direction for monitoring comparisons (default: auto)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum change in the monitored metric to reset patience.",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=0,
        help="Minimum number of full epochs before early stopping can trigger.",
    )
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="auto",
        choices=["auto", "youden", "none", "f1-morph"],
        help="Threshold policy to apply when tracking the best checkpoint (default: auto)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned memory when loading batches",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory when loading batches",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=True,
        help="Keep dataloader workers alive between epochs",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Shut down dataloader workers at the end of each epoch",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of training batches per epoch for quick smoke tests. "
            "Values <= 0 disable this cap."
        ),
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of validation batches per epoch for smoke tests. "
            "Values <= 0 disable this cap."
        ),
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of test batches per epoch for smoke tests. "
            "Values <= 0 disable this cap."
        ),
    )
    parser.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("checkpoints") / "classification"),
        dest="output_dir",
        help="Directory for checkpoints, logs and config snapshots",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Collection of seeds to evaluate; the first seed is used for the current run",
    )

    seed_flag_present = any(
        entry == "--seed" or entry.startswith("--seed=") for entry in sys.argv[1:]
    )
    parsed = parser.parse_args()
    setattr(parsed, "_seed_explicit", bool(seed_flag_present))
    return parsed


def main():
    args = get_args()
    overrides = _flatten_override_args(getattr(args, "overrides", None))
    args.overrides = overrides
    resolved_overrides: Dict[str, Any] = {}
    args.finetune_mode = normalise_finetune_mode(
        getattr(args, "finetune_mode", None), default="full"
    )
    if getattr(args, "frozen", False):
        args.finetune_mode = "none"
    args.frozen = args.finetune_mode == "none"
    args.seeds = _normalize_seeds(getattr(args, "seeds", None))
    
    def _warn_disable(flag: str, value: Any, reason: str) -> None:
        warnings.warn(
            f"{reason} (received {value!r}). Disabling --{flag}.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _coerce_positive_int(value: Optional[int], flag: str) -> Optional[int]:
        if value is None:
            return None
        if value <= 0:
            _warn_disable(flag, value, "Expected a positive integer")
            return None
        return int(value)

    dataset_percent = getattr(args, "dataset_percent", None)
    if dataset_percent is not None:
        if dataset_percent <= 0:
            _warn_disable("dataset-percent", dataset_percent, "Dataset percentage must be > 0")
            dataset_percent = None
        elif dataset_percent > 100:
            warnings.warn(
                f"--dataset-percent {dataset_percent} exceeds 100; clamping to 100 for smoke tests.",
                RuntimeWarning,
                stacklevel=2,
            )
            dataset_percent = 100.0
        else:
            dataset_percent = float(dataset_percent)
    args.dataset_percent = dataset_percent

    dataset_seed = getattr(args, "dataset_seed", None)
    if dataset_seed is not None and dataset_seed < 0:
        _warn_disable("dataset-seed", dataset_seed, "Dataset seed must be >= 0")
        dataset_seed = None
    args.dataset_seed = dataset_seed

    args.limit_train_batches = _coerce_positive_int(
        getattr(args, "limit_train_batches", None), "limit-train-batches"
    )
    args.limit_val_batches = _coerce_positive_int(
        getattr(args, "limit_val_batches", None), "limit-val-batches"
    )
    args.limit_test_batches = _coerce_positive_int(
        getattr(args, "limit_test_batches", None), "limit-test-batches"
    )
    args.max_train_steps = _coerce_positive_int(
        getattr(args, "max_train_steps", None), "max-train-steps"
    )

    experiment_cfg = None
    dataset_cfg = None
    dataset_resolved = None
    selected_model = None

    if args.exp_config:
        experiment_cfg = load_layered_config(args.exp_config)
        experiment_cfg, resolved_overrides = _apply_config_overrides(
            experiment_cfg, overrides
        )
        selected_model, dataset_cfg, dataset_resolved = apply_experiment_config(
            args,
            experiment_cfg,
            resolved_overrides=resolved_overrides,
        )

    args.active_seed = _resolve_active_seed(args)
    args.seed = args.active_seed

    for required in ("pretraining", "dataset"):
        if getattr(args, required) is None:
            raise ValueError(
                f"Missing required argument '{required}'. Provide it via --{required.replace('_', '-')} or the experiment config."
            )

    if getattr(args, "train_pack", None) is None and not getattr(args, "frozen", False):
        raise ValueError(
            "Missing required argument 'train_pack'. Provide it via --train-pack or the experiment config, or run with --frozen/finetune-mode none for evaluation-only mode."
        )

    if not args.val_pack and args.train_pack:
        args.val_pack = args.train_pack
    if not args.test_pack:
        args.test_pack = args.val_pack or args.train_pack

    args.pack_root = str(Path(args.pack_root).expanduser()) if args.pack_root else str(data_packs_root())
    args.perturbation_splits = [s.lower() for s in (args.perturbation_splits or [])]
    args.scheduler = (args.scheduler or "none").lower()

    roots_path = Path(args.roots).expanduser()
    if not roots_path.exists():
        raise FileNotFoundError(
            f"Roots mapping not found at {roots_path}. Copy data/roots.example.json to data/roots.json or provide --roots explicitly."
        )
    with open(roots_path) as f:
        roots_map = json.load(f)
    args.roots_map = roots_map

    layout = _resolve_run_layout(
        args,
        selected_model=selected_model,
        dataset_cfg=dataset_cfg,
        dataset_resolved=dataset_resolved,
        experiment_cfg=experiment_cfg,
    )
    args.run_stem = layout["stem"]
    args.output_dir = str(layout["output_dir"])
    args.tensorboard_dir = str(layout["tb_dir"])
    args.metrics_path = str(layout["metrics_path"])
    args.dataset_layout = layout.get("dataset_layout")
    args.dataset_resolved = dataset_resolved
    args.model_tag = layout.get("model_tag")
    thresholds_root = Path(layout.get("base_dir", "checkpoints")).expanduser()
    args.thresholds_root = str(thresholds_root.parent / "thresholds")

    set_determinism(_get_active_seed(args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    def _serialize(value):
        if isinstance(value, dict):
            return {k: _serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_serialize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    run_config = {
        "args": {k: _serialize(v) for k, v in vars(args).items() if k != "roots_map"},
    }
    if resolved_overrides:
        run_config["config_overrides"] = _serialize(resolved_overrides)
    if experiment_cfg is not None:
        run_config["experiment"] = _serialize(experiment_cfg)
    if dataset_cfg is not None and dataset_resolved is not None:
        run_config["dataset"] = {
            "configured": _serialize(dataset_cfg),
            "resolved": _serialize(dataset_resolved),
        }
    if selected_model is not None:
        run_config["model"] = _serialize(selected_model)
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        commit = None
    run_config["git_commit"] = commit
    args.git_commit = commit
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(run_config, f)
    device_count = torch.cuda.device_count()
    args.world_size = device_count if device_count > 0 else 1
    if (
        getattr(args, "max_train_steps", None) is not None
        and args.frozen
        and args.world_size > 1
    ):
        warnings.warn(
            "--max-train-steps is ignored for distributed evaluation-only runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        args.max_train_steps = None
    assert args.batch_size % args.world_size == 0
    if args.world_size > 1:
        mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)
    else:
        train(0, args)


if __name__ == "__main__":
    main()
