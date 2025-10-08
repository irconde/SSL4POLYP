from __future__ import annotations

import math
from typing import Optional

__all__ = ["format_decimal", "format_interval", "format_with_ci"]


def format_decimal(value: Optional[float], digits: int = 3, placeholder: str = "—") -> str:
    """Format a floating-point value to a fixed number of decimals."""

    if value is None or not isinstance(value, (int, float)):
        return placeholder
    if isinstance(value, bool):
        value = float(value)
    if not math.isfinite(float(value)):
        return placeholder
    return f"{float(value):.{digits}f}"


def format_interval(
    lower: Optional[float],
    upper: Optional[float],
    *,
    digits: int = 3,
    placeholder: str = "—",
    separator: str = ", ",
) -> str:
    if lower is None or upper is None:
        return placeholder
    return f"{format_decimal(lower, digits=digits, placeholder=placeholder)}{separator}{format_decimal(upper, digits=digits, placeholder=placeholder)}"


def format_with_ci(
    mean: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
    *,
    digits: int = 3,
    placeholder: str = "—",
    ci_label: str = "CI",
) -> str:
    mean_text = format_decimal(mean, digits=digits, placeholder=placeholder)
    if lower is None or upper is None:
        return mean_text
    interval = format_interval(lower, upper, digits=digits, placeholder=placeholder, separator="–")
    return f"{mean_text} ({ci_label}: {interval})"
