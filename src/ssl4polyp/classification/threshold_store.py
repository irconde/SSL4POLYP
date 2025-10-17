"""Utilities for deriving canonical threshold storage paths."""

import re
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "sanitize_path_segment",
    "canonical_threshold_directory",
    "canonical_threshold_path",
    "canonical_threshold_filename",
]


def sanitize_path_segment(raw: Any, *, default: str = "default") -> str:
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


def _format_seed(seed: Optional[int]) -> str:
    try:
        value = int(seed) if seed is not None else 0
    except (TypeError, ValueError):  # pragma: no cover - defensive
        value = 0
    return f"seed-{value}"


def canonical_threshold_directory(
    root: Path,
    *,
    val_pack: Optional[str],
    model_tag: Optional[str],
    arch: Optional[str],
    pretraining: Optional[str],
    train_pack: Optional[str] = None,
    subset: Optional[str] = None,
    seed: Optional[int],
) -> Path:
    """Return the canonical directory for storing threshold artifacts."""

    root = Path(root).expanduser()
    val_segment = sanitize_path_segment(val_pack, default="dataset")
    model_segment = sanitize_path_segment(model_tag, default="model")
    arch_segment = sanitize_path_segment(arch, default="unknown")
    pretraining_segment = sanitize_path_segment(pretraining, default="unknown")
    train_pack_segment = sanitize_path_segment(train_pack, default="full")
    subset_segment = sanitize_path_segment(subset, default="full")

    segments = [
        val_segment,
        model_segment,
        f"arch-{arch_segment}",
        f"pretrain-{pretraining_segment}",
        f"trainpack-{train_pack_segment}",
        f"subset-{subset_segment}",
        _format_seed(seed),
    ]
    return root.joinpath(*segments)


def canonical_threshold_filename(policy: Optional[str]) -> str:
    """Return the canonical filename for a stored threshold policy."""

    policy_segment = sanitize_path_segment(policy, default="policy")
    return f"policy-{policy_segment}.json"


def canonical_threshold_path(
    root: Path,
    *,
    val_pack: Optional[str],
    model_tag: Optional[str],
    arch: Optional[str],
    pretraining: Optional[str],
    train_pack: Optional[str] = None,
    subset: Optional[str] = None,
    seed: Optional[int],
    policy: Optional[str],
) -> Path:
    """Return the canonical path for a persisted threshold artifact."""

    directory = canonical_threshold_directory(
        root,
        val_pack=val_pack,
        model_tag=model_tag,
        arch=arch,
        pretraining=pretraining,
        train_pack=train_pack,
        subset=subset,
        seed=seed,
    )
    return directory / canonical_threshold_filename(policy)
