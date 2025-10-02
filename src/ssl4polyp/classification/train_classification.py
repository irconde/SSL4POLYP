from __future__ import annotations

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
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    recall_score,
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


def _log_lines(log_path: Path, lines: Sequence[str]) -> None:
    """Print ``lines`` and append them to ``log_path``."""

    for line in lines:
        print(line)
    if log_path:
        with open(log_path, "a") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")


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

    seeds = getattr(args, "seeds", None) or []
    if seeds:
        return int(seeds[0])
    config_seed = getattr(args, "config_seed", None)
    if config_seed is not None:
        return int(config_seed)
    return int(getattr(args, "seed", 0))


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
) -> Dict[str, float]:
    """Convert ``metrics`` into a JSON-serialisable mapping of floats."""

    drop = set(drop or [])
    export: Dict[str, float] = {}
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
    return export


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
    }
    return metrics


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
    test_perf: float,
    model_tag: Optional[str] = None,
    subdir: Optional[str] = None,
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
        "metrics_at_threshold": metrics_at_tau,
        "auc_val": float(val_perf),
        "auc_test": float(test_perf),
        "timestamp": timestamp,
        "code_git_commit": getattr(args, "git_commit", None),
        "data_pack": val_pack,
        "notes": "",
    }
    if threshold_key:
        payload["threshold_key"] = threshold_key
    snapshot: Dict[str, Dict[str, float]] = {}
    if val_metrics:
        snapshot["val"] = val_metrics
    if test_metrics:
        snapshot["test"] = test_metrics
    if snapshot:
        payload["metrics_snapshot"] = snapshot
    if threshold_key:
        payload["thresholds"] = {threshold_key: float(tau)}
    if model_tag:
        payload["model_tag"] = model_tag
    if subdir:
        payload["directory"] = subdir
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
    if cli_seed is not None and "seed" not in dataset_cfg:
        dataset_cfg["seed"] = cli_seed
    if cli_size is not None and "size" not in dataset_cfg:
        dataset_cfg["size"] = cli_size
    model_cfgs = resolve_model_entries(experiment_cfg.get("models", []))
    selected_model = _select_model_config(model_cfgs, getattr(args, "model_key", None))

    if "optimizer" in experiment_cfg and experiment_cfg["optimizer"].lower() != "adamw":
        raise ValueError("Only AdamW optimizer is currently supported")

    config_seeds = _normalize_seeds(experiment_cfg.get("seeds"))
    if config_seeds:
        args.seeds = config_seeds

    args.lr = experiment_cfg.get("lr", args.lr)
    args.weight_decay = experiment_cfg.get("weight_decay", getattr(args, "weight_decay", 0.05))
    args.batch_size = experiment_cfg.get("batch_size", args.batch_size)
    args.epochs = experiment_cfg.get("epochs", args.epochs)
    config_seed = _as_int(experiment_cfg.get("seed"))
    if config_seed is not None:
        args.config_seed = config_seed
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

    args.threshold_policy = experiment_cfg.get(
        "threshold_policy", getattr(args, "threshold_policy", None)
    )

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
    protocol_cfg = (experiment_cfg or {}).get("protocol") or {}
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
                with open(log_path, "a") as f:
                    f.write(printout)
                    f.write("\n")
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
    morphology_values: list[str] = []
    morphology_counter: Counter[str] = Counter()
    dataset_obj = getattr(test_loader, "dataset", None)
    track_perturbations = _dataset_supports_perturbations(dataset_obj)
    perturbation_counter: Counter[str] = Counter()
    perturbation_metadata_rows: list[Mapping[str, Any]] = []
    perturbation_tags: list[str] = []
    perturbation_sample_losses: list[torch.Tensor] = []
    perturbation_tags_aligned: Optional[List[str]] = None

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
    with open(log_path_obj, "a") as f:
        f.write(breadcrumb_text + "\n")

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
        if track_morphology and metadata_batch is not None:
            for row in metadata_batch:
                morph_label = row.get("morphology") if isinstance(row, dict) else None
                morph_value = str(morph_label).strip().lower() if morph_label not in (None, "") else "unknown"
                morphology_values.append(morph_value)
                morphology_counter[morph_value] += 1
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
            with open(log_path_obj, "a") as f:
                f.write(printout + "\n")
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
        results["perturbation_samples"] = {
            "tags": perturbation_tags_aligned,
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
    bal_value = results.get("balanced_accuracy")
    if isinstance(bal_value, (float, np.floating)) and not math.isnan(bal_value):
        metric_line_parts.append(f"Balanced Acc: {bal_value:.6f}")
    f1_value = results.get("f1")
    if isinstance(f1_value, (float, np.floating)) and not math.isnan(f1_value):
        metric_line_parts.append(f"F1@τ: {f1_value:.6f}")
    ap_value = results.get("auprc")
    if isinstance(ap_value, (float, np.floating)) and not math.isnan(ap_value):
        metric_line_parts.append(f"AP (PR-AUC): {ap_value:.6f}")
    if not math.isnan(auroc_value):
        metric_line_parts.append(f"AUROC: {auroc_value:.6f}")
    else:
        metric_line_parts.append("AUROC: —")
    if loss_fn is not None and not math.isnan(mean_loss):
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
    if track_morphology and morphology_values and targets is not None:
        requested_counts: Dict[str, int] = {}
        requested_metrics: Dict[str, Dict[str, float]] = {}
        morph_array = np.array(morphology_values)
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

    if threshold_stats and all(key in threshold_stats for key in ("tp", "fp", "fn", "tn")):
        confusion_line = _with_prefix(
            f"{split_name} confusion @τ: TP={int(threshold_stats['tp'])} "
            f"FP={int(threshold_stats['fp'])} FN={int(threshold_stats['fn'])} "
            f"TN={int(threshold_stats['tn'])}"
        )
    else:
        confusion_line = _with_prefix(f"{split_name} confusion @τ: n/a")

    output_lines = [class_presence_line, metrics_line, tau_line, counts_line]
    if morph_counts_line:
        output_lines.append(morph_counts_line)
    if morph_metrics_line:
        output_lines.append(morph_metrics_line)
    output_lines.append(confusion_line)

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

        non_clean_mask = tags_array != "clean"
        if np.any(non_clean_mask):
            indices_np = np.flatnonzero(non_clean_mask)
            indices_tensor = torch.from_numpy(indices_np).to(dtype=torch.long)
            metrics_for_tag = compute_metrics_for_indices(indices_tensor)
            if metrics_for_tag:
                per_tag_metrics["ALL-perturbed"] = metrics_for_tag

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

    for line in output_lines:
        print(line)
    with open(log_path_obj, "a") as f:
        for line in output_lines:
            f.write(line + "\n")

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

    experiment4_trace = _collect_experiment4_trace(args, datasets)

    train_dataloader = loaders.get("train")
    eval_only = train_dataloader is None
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
    existing_ckpt, pointer_valid = _find_existing_checkpoint(stem_path)
    parent_reference = getattr(args, "parent_checkpoint", None)
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
        else:
            state_dict = parent_state
        model.load_state_dict(state_dict)
        start_epoch = 1
        best_val_perf = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
    else:
        start_epoch = 1
        best_val_perf = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()

    configure_finetune_parameters(model, getattr(args, "finetune_mode", "full"))

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        ddp_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [rank]
        model = DDP(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0)
    )
    thresholds_map = dict(thresholds_map)
    raw_policy = (args.threshold_policy or "auto").strip().lower()
    if raw_policy not in {"auto", "youden", "none", ""}:
        raise ValueError(
            f"Unsupported threshold policy '{raw_policy}'. Use 'auto', 'youden' or 'none'."
        )
    resolved_policy = raw_policy
    if resolved_policy in {"", "auto"}:
        resolved_policy = "youden" if len(class_weights) == 2 else "none"
        if rank == 0:
            if resolved_policy == "youden":
                print("Auto-selecting Youden's J threshold policy for binary classification.")
            else:
                print(
                    "Threshold policy resolved to 'none' because the task is not binary."
                )
    if resolved_policy == "youden" and len(class_weights) != 2:
        if rank == 0:
            print(
                "Warning: Youden threshold policy requested but dataset is not binary; disabling threshold computation."
            )
        resolved_policy = "none"
    compute_threshold = resolved_policy == "youden" and val_dataloader is not None
    threshold_key = None
    if compute_threshold:
        dataset_name = args.dataset or "dataset"
        val_split = args.val_split or "val"
        threshold_key = thresholds.format_threshold_key(
            dataset_name, val_split, resolved_policy
        )
    args.threshold_policy = resolved_policy
    use_amp = args.precision == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = create_scheduler(optimizer, args)
    if best_val_perf is not None:
        optimizer.load_state_dict(main_dict["optimizer_state_dict"])
        scaler.load_state_dict(main_dict["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in main_dict:
            scheduler.load_state_dict(main_dict["scheduler_state_dict"])

    args.resume_monitor_available = bool(resume_monitor_available)

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
    ) = build(args, rank, device, distributed)
    eval_only = train_dataloader is None
    use_amp = args.precision == "amp" and device.type == "cuda"

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
        parts = [segment for segment in str(key).split("_") if segment]
        if not parts:
            return None
        policy_raw = parts[-1]
        split_raw = parts[-2] if len(parts) >= 2 else (args.val_split or "val")
        policy_map = {"youden": "Youden J"}
        policy_label = policy_map.get(policy_raw, policy_raw.replace("_", " ").title())
        return f"{policy_label} on {split_raw}"

    if rank == 0:
        tb_dir = getattr(args, "tensorboard_dir", None)
        tb_path = str(tb_dir) if tb_dir else os.path.join(args.output_dir, "tb")
        tb_logger = TensorboardLogger.create(tb_path)
        with open(log_path, "a") as f:
            f.write(str(args))
            f.write("\n")
    else:
        tb_logger = TensorboardLogger.create(None)
    if distributed:
        dist.barrier()

    grad_accum_steps = _resolve_grad_accum_steps(optimizer)
    total_seen_samples = 0
    total_batches_processed = 0
    epochs_run = 0

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
        if rank == 0:
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
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_val_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Val"),
                )
                if tb_logger:
                    tb_logger.log_metrics("val", val_metrics, eval_epoch)
            if test_dataloader is not None:
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
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_test_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Test"),
                )
                if tb_logger:
                    tb_logger.log_metrics("test", test_metrics, eval_epoch)
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
    early_patience = getattr(args, "early_stop_patience", 0) or 0
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
    latest_threshold_file_relpath: Optional[str] = None
    latest_threshold_record: Optional[Dict[str, Any]] = None
    latest_thresholds_root: Optional[str] = None

    for epoch in range(start_epoch, args.epochs + 1):
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
                    return_outputs=compute_threshold,
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_val_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Val"),
                )
                test_metrics = test(
                    _unwrap_model(model),
                    rank,
                    test_dataloader,
                    epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    loss_fn=loss_fn,
                    loss_mode=loss_mode,
                    split_name="Test",
                    tau=eval_tau,
                    tau_info=eval_tau_info,
                    max_batches=args.limit_test_batches,
                    morphology_eval=morphology_eval,
                    eval_context=eval_context_lookup.get("Test"),
                )
                val_metrics_export = _prepare_metric_export(
                    val_metrics, drop={"logits", "probabilities", "targets"}
                )
                test_metrics_export = _prepare_metric_export(test_metrics)
                val_logits = val_metrics.pop("logits", None)
                val_metrics.pop("probabilities", None)
                val_targets = val_metrics.pop("targets", None)
                val_loss_value = val_metrics.get("loss")
                test_loss_value = test_metrics.get("loss")
                monitor_value = val_metrics.get(monitor_key)
                if monitor_value is None:
                    available = ", ".join(sorted(val_metrics.keys()))
                    raise KeyError(
                        f"Validation metrics do not contain monitor '{monitor_key}'. Available: {available}"
                    )
                test_monitor_value = test_metrics.get(monitor_key)
                val_perf = float(val_metrics["auroc"])
                test_perf = float(test_metrics["auroc"])
                if tb_logger:
                    tb_logger.log_metrics("val", val_metrics, epoch)
                    tb_logger.log_metrics("test", test_metrics, epoch)
                last_epoch = int(epoch)
                last_loss = float(loss)
                last_val_perf = float(val_perf)
                last_test_perf = float(test_perf)
                last_monitor_value = float(monitor_value)
                last_val_loss = float(val_loss_value) if val_loss_value is not None else None
                last_test_loss = (
                    float(test_loss_value) if test_loss_value is not None else None
                )
                last_test_monitor = (
                    float(test_monitor_value)
                    if test_monitor_value is not None
                    else None
                )
                last_val_metrics_export = dict(val_metrics_export)
                last_test_metrics_export = dict(test_metrics_export)
            else:
                val_perf = 0.0
                test_perf = 0.0
                monitor_value = 0.0
                test_monitor_value = 0.0
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
                with open(log_path, "a") as f:
                    f.write("Saving...")
                    f.write("\n")
                updated_thresholds = dict(thresholds_map)
                threshold_file_relpath: Optional[str] = None
                threshold_record: Optional[Dict[str, Any]] = None
                thresholds_root_path: Optional[Path] = None
                if (
                    compute_threshold
                    and threshold_key is not None
                    and val_logits is not None
                    and val_targets is not None
                ):
                    try:
                        tau = thresholds.compute_youden_j_threshold(
                            val_logits, val_targets
                        )
                    except ValueError as exc:
                        print(
                            f"Warning: unable to compute threshold '{threshold_key}': {exc}"
                        )
                    else:
                        updated_thresholds[threshold_key] = tau
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
                        metrics_at_tau = _compute_threshold_statistics(
                            val_logits, val_targets, float(tau)
                        )
                        threshold_record = _build_threshold_payload(
                            args,
                            threshold_key=threshold_key,
                            tau=float(tau),
                            metrics_at_tau=metrics_at_tau,
                            val_metrics=val_metrics_export,
                            test_metrics=test_metrics_export,
                            val_perf=val_perf,
                            test_perf=test_perf,
                            model_tag=model_tag,
                            subdir=subdir,
                        )
                        with threshold_file.open("w", encoding="utf-8") as handle:
                            json.dump(threshold_record, handle, indent=2)
                        try:
                            threshold_file_relpath = str(
                                threshold_file.relative_to(thresholds_root_path.parent)
                            )
                        except ValueError:
                            threshold_file_relpath = str(threshold_file)
                        print(
                            f"Updated threshold {threshold_key} = {tau:.6f} -> {threshold_file_relpath}"
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
                    else float(test_perf),
                    "test_loss": float(test_loss_value)
                    if test_loss_value is not None
                    else None,
                    "test_auroc": float(test_perf),
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
                        "test": float(test_perf),
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
                metrics_payload = {
                    "seed": _get_active_seed(args),
                    "epoch": int(epoch),
                    "train_loss": float(loss),
                    "val": val_metrics_export,
                    "test": test_metrics_export,
                    "monitor_value": float(monitor_value),
                    "monitor_metric": monitor_name,
                }
                if threshold_file_relpath:
                    metrics_payload["threshold_files"] = {
                        threshold_key: threshold_file_relpath
                    }
                    metrics_payload["threshold_policy"] = args.threshold_policy
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
            with open(log_path, "a") as f:
                f.write(patience_line)
                f.write("\n")

        if early_patience > 0:
            if distributed:
                stop_tensor = torch.tensor(
                    [1 if (rank == 0 and no_improve_epochs >= early_patience) else 0],
                    device=device,
                )
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item():
                    if rank == 0:
                        print("Early stopping triggered after reaching patience limit.")
                    break
            elif rank == 0 and no_improve_epochs >= early_patience:
                print("Early stopping triggered after reaching patience limit.")
                break

        if distributed:
            dist.barrier()
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
        last_metrics_payload: Dict[str, Any] = {
            "seed": _get_active_seed(args),
            "epoch": int(last_epoch),
            "train_loss": float(last_loss) if last_loss is not None else None,
            "val": dict(last_val_metrics_export or {}),
            "test": dict(last_test_metrics_export or {}),
            "monitor_value": float(last_monitor_value)
            if last_monitor_value is not None
            else None,
            "monitor_metric": monitor_name,
        }
        if latest_threshold_file_relpath and threshold_key is not None:
            last_metrics_payload["threshold_files"] = {
                threshold_key: latest_threshold_file_relpath
            }
            last_metrics_payload["threshold_policy"] = args.threshold_policy
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
        "--threshold-policy",
        type=str,
        default="auto",
        choices=["auto", "youden", "none"],
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

    return parser.parse_args()


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
