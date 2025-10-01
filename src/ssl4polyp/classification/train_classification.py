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

import yaml

from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ssl4polyp.utils.tensorboard import SummaryWriter

from ssl4polyp import utils
from ssl4polyp.classification.data import create_classification_dataloaders
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


EVAL_MAX_ADDITIONAL_BATCHES = 3


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


def apply_experiment_config(args, experiment_cfg: Dict[str, Any]):
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
    args.early_stop_monitor = early_cfg.get("monitor", getattr(args, "early_stop_monitor", "val_auroc"))
    args.early_stop_patience = early_cfg.get(
        "patience", getattr(args, "early_stop_patience", 0)
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
    finetune_mode = protocol_cfg.get("finetune")
    default_mode = getattr(args, "finetune_mode", None)
    if not default_mode:
        default_mode = "none" if args.frozen else "full"
    args.finetune_mode = normalise_finetune_mode(
        finetune_mode, default=default_mode
    )
    args.frozen = args.finetune_mode == "none"
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
            return float("nan"), global_step

    t = time.time()
    non_blocking = device.type == "cuda"
    use_amp = use_amp and device.type == "cuda"
    scaler = scaler if use_amp else nullcontext()
    if not isinstance(scaler, torch.cuda.amp.GradScaler):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_accumulator: list[float] = []

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
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
            if batches_processed < total_batches:
                print(
                    "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        sample_count,
                        len(train_loader.dataset),
                        progress_pct,
                        loss_value,
                        time.time() - t,
                    ),
                    end="",
                )
            else:
                printout = (
                    "Train Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}"
                ).format(
                    epoch,
                    sample_count,
                    len(train_loader.dataset),
                    progress_pct,
                    np.mean(loss_accumulator) if loss_accumulator else float("nan"),
                    time.time() - t,
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
    return mean_loss, global_step


@torch.no_grad()
def test(
    model,
    rank,
    test_loader,
    epoch,
    perf_fn,
    log_path,
    metric_fns: Optional[Dict[str, Any]] = None,
    split_name: str = "Test",
    return_outputs: bool = False,
    tau: Optional[float] = None,
    max_batches: Optional[int] = None,
):
    if test_loader is None:
        return float("nan")
    t = time.time()
    model.eval()
    metric_fns = metric_fns or {}
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

    for batch_idx, batch in enumerate(test_loader):
        if batches_processed >= hard_batch_limit:
            break
        if len(batch) == 3:
            data, target, _ = batch
        elif len(batch) == 2:
            raise ValueError("Test dataloader does not provide labels; enable metadata return with labels.")
        else:  # pragma: no cover - defensive
            raise ValueError("Unexpected batch structure returned by dataloader")
        data = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        N += len(data)
        output = model(data).detach().cpu()
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
            print(
                f"{split_name} evaluation consumed extra batches to observe all labels "
                f"(max_batches={max_batches}, observed={sorted(observed_labels)})."
            )
            warned_extra_batches = True

        # (Removed per-batch AUROC to avoid single-class crash.)
        # Compute inexpensive interim progress metric(s):
        probs_cumulative = torch.softmax(logits, dim=1)
        if tau is not None and probs_cumulative.size(1) == 2:
            preds_cumulative = (probs_cumulative[:, 1] >= tau).to(dtype=torch.long)
        else:
            preds_cumulative = torch.argmax(probs_cumulative, dim=1)
        targets_cumulative = targets  # already CPU
        # Running (cumulative) balanced accuracy (safe with single class)
        unique_classes = torch.unique(targets_cumulative)
        if unique_classes.numel() >= 2:
            # You could compute balanced accuracy; re-use existing helper if desired.
            # Simple implementation here:
            from sklearn.metrics import balanced_accuracy_score
            running_bal_acc = balanced_accuracy_score(
                targets_cumulative.numpy(),
                preds_cumulative.numpy()
            )
            bal_display = f"BALACC: {running_bal_acc:.6f}"
        else:
            bal_display = "BALACC: (pending)"

        metrics_display = "\t".join([
            "AUROC: (pending)",  # final AUROC printed after loop
            bal_display,
        ])

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
                    split_name,
                    epoch,
                    N,
                    dataset_size,
                    progress_pct,
                    metrics_display,
                    time.time() - t,
                ),
                end="",
            )
        else:
            printout = "{}  Epoch: {} [{}/{} ({:.1f}%)]\t{}\tTime: {:.6f}".format(
                split_name,
                epoch,
                N,
                dataset_size,
                progress_pct,
                metrics_display,
                time.time() - t,
            )
            print("\r" + printout)
            with open(log_path, "a") as f:
                f.write(printout + "\n")
            break
    probs = torch.softmax(logits, dim=1)
    n_classes = probs.size(1) if probs.ndim == 2 else 1
    binary_tau = tau if (tau is not None and n_classes == 2) else None
    results = {
        "auroc": perf_fn(probs, targets).item(),
    }
    if binary_tau is not None:
        preds = (probs[:, 1] >= binary_tau).to(dtype=torch.long)
    else:
        preds = torch.argmax(probs, dim=1)
    for name, fn in metric_fns.items():
        try:
            results[name] = fn(probs, targets, tau=binary_tau).item()
        except TypeError:
            results[name] = fn(preds, targets).item()

    print(f"{split_name} final AUROC: {results['auroc']:.6f}")

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
    if train_dataset is not None and train_dataset.labels_list is not None:
        train_labels = list(train_dataset.labels_list)
        counts = np.bincount(train_labels, minlength=n_class)
        N_total = len(train_labels)
        class_weights = [
            (N_total / (n_class * count)) if count > 0 else 0.0 for count in counts
        ]
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
    if existing_ckpt is not None:
        main_dict = torch.load(existing_ckpt, map_location="cpu")
        model.load_state_dict(main_dict["model_state_dict"])
        start_epoch = main_dict["epoch"] + 1
        best_val_perf = main_dict.get("val_perf")
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
        best_val_perf,
        class_weights,
        scaler,
        scheduler,
        thresholds_map,
        compute_threshold,
        threshold_key,
    ) = build(args, rank, device, distributed)
    eval_only = train_dataloader is None
    use_amp = args.precision == "amp" and device.type == "cuda"

    class_weights_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    n_classes = len(class_weights)
    perf_fn = performance.meanAUROC(n_class=n_classes)
    aux_metric_fns = {
        "f1": performance.meanF1Score(n_class=n_classes),
        "precision": performance.meanPrecision(n_class=n_classes),
        "recall": performance.meanRecall(n_class=n_classes),
    }
    sun_threshold_key = thresholds.format_threshold_key(
        "sun_full", args.val_split or "val", "youden"
    )

    def resolve_eval_tau(*keys: Optional[str]) -> Optional[float]:
        for key in keys:
            if not key:
                continue
            tau_value = thresholds.resolve_threshold(thresholds_map, key)
            if tau_value is not None:
                return tau_value
        return None

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

    if eval_only:
        eval_epoch = max(start_epoch - 1, 0)
        if rank == 0:
            print("No training data provided; running evaluation-only mode.")
            eval_tau = resolve_eval_tau(threshold_key, sun_threshold_key)
            if val_dataloader is not None:
                val_metrics = test(
                    _unwrap_model(model),
                    rank,
                    val_dataloader,
                    eval_epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    split_name="Val",
                    tau=eval_tau,
                    max_batches=args.limit_val_batches,
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
                    split_name="Test",
                    tau=eval_tau,
                    max_batches=args.limit_test_batches,
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
    last_epoch: Optional[int] = None
    last_loss: Optional[float] = None
    last_val_perf: Optional[float] = None
    last_test_perf: Optional[float] = None
    last_val_metrics_export: Optional[Dict[str, float]] = None
    last_test_metrics_export: Optional[Dict[str, float]] = None
    latest_threshold_file_relpath: Optional[str] = None
    latest_threshold_record: Optional[Dict[str, Any]] = None
    latest_thresholds_root: Optional[str] = None

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            prev_global_step = global_step
            loss, global_step = train_epoch(
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
            )
            if rank == 0:
                eval_tau = resolve_eval_tau(threshold_key, sun_threshold_key)
                val_metrics = test(
                    _unwrap_model(model),
                    rank,
                    val_dataloader,
                    epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    split_name="Val",
                    return_outputs=compute_threshold,
                    tau=eval_tau,
                    max_batches=args.limit_val_batches,
                )
                test_metrics = test(
                    _unwrap_model(model),
                    rank,
                    test_dataloader,
                    epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    split_name="Test",
                    tau=eval_tau,
                    max_batches=args.limit_test_batches,
                )
                val_metrics_export = _prepare_metric_export(
                    val_metrics, drop={"logits", "probabilities", "targets"}
                )
                test_metrics_export = _prepare_metric_export(test_metrics)
                val_logits = val_metrics.pop("logits", None)
                val_metrics.pop("probabilities", None)
                val_targets = val_metrics.pop("targets", None)
                val_perf = val_metrics["auroc"]
                test_perf = test_metrics["auroc"]
                if tb_logger:
                    tb_logger.log_metrics("val", val_metrics, epoch)
                    tb_logger.log_metrics("test", test_metrics, epoch)
                last_epoch = int(epoch)
                last_loss = float(loss)
                last_val_perf = float(val_perf)
                last_test_perf = float(test_perf)
                last_val_metrics_export = dict(val_metrics_export)
                last_test_metrics_export = dict(test_metrics_export)
            else:
                val_perf = 0.0
                test_perf = 0.0

            steps_this_epoch = global_step - prev_global_step
            should_step = steps_this_epoch > 0
            if scheduler is not None:
                if scheduler_name == "plateau":
                    if distributed:
                        metric_tensor = torch.tensor(
                            [val_perf if rank == 0 else 0.0], device=device
                        )
                        dist.broadcast(metric_tensor, src=0)
                        if should_step:
                            scheduler.step(metric_tensor.item())
                    elif should_step:
                        scheduler.step(val_perf)
                elif should_step:
                    scheduler.step()
            if distributed:
                dist.barrier()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)

        if rank == 0:
            improved = best_val_perf is None or val_perf > best_val_perf
            if improved:
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
                    "val_perf": val_perf,
                    "val_auroc": val_perf,
                    "test_perf": test_perf,
                    "test_auroc": test_perf,
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
                best_val_perf = val_perf
                thresholds_map = updated_thresholds
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

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
    tb_logger.close()
    if rank == 0 and last_epoch is not None:
        last_payload: Dict[str, Any] = {
            "epoch": int(last_epoch),
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": float(last_loss) if last_loss is not None else None,
            "val_perf": last_val_perf,
            "val_auroc": last_val_perf,
            "test_perf": last_test_perf,
            "test_auroc": last_test_perf,
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
                "val": float(last_val_perf) if last_val_perf is not None else None,
                "test": float(last_test_perf) if last_test_perf is not None else None,
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
    parser.add_argument("--early-stop-monitor", type=str, default="val_auroc")
    parser.add_argument("--early-stop-patience", type=int, default=0)
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
        selected_model, dataset_cfg, dataset_resolved = apply_experiment_config(args, experiment_cfg)

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
