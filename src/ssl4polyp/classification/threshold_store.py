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
    seed: Optional[int],
) -> Path:
    """Return the canonical directory for storing threshold artifacts."""

    root = Path(root).expanduser()
    segments = [
        sanitize_path_segment(val_pack, default="dataset"),
        sanitize_path_segment(model_tag, default="model"),
        sanitize_path_segment(arch, default="arch"),
        sanitize_path_segment(pretraining, default="pretraining"),
        _format_seed(seed),
    ]
    return root.joinpath(*segments)


def canonical_threshold_filename(policy: Optional[str]) -> str:
    """Return the canonical filename for a stored threshold policy."""

    return f"{sanitize_path_segment(policy, default="policy")}.json"


def canonical_threshold_path(
    root: Path,
    *,
    val_pack: Optional[str],
    model_tag: Optional[str],
    arch: Optional[str],
    pretraining: Optional[str],
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
        seed=seed,
    )
    return directory / canonical_threshold_filename(policy)
