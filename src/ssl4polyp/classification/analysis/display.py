from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "PLACEHOLDER",
    "coerce_numeric",
    "format_decimal",
    "format_interval",
    "format_with_ci",
    "format_scalar",
    "format_signed",
    "format_mean_std",
    "format_ci",
    "format_value_with_ci",
    "format_percent",
    "format_percent_numeric",
    "format_interval_label",
]

PLACEHOLDER = "—"


def coerce_numeric(value: object) -> Optional[float]:
    """Return ``float(value)`` if finite; otherwise ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        try:
            value = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def format_decimal(
    value: Optional[float],
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
) -> str:
    """Format a floating-point value to a fixed number of decimals."""

    numeric = coerce_numeric(value)
    if numeric is None:
        return placeholder
    return f"{numeric:.{digits}f}"


def format_interval(
    lower: Optional[float],
    upper: Optional[float],
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
    separator: str = ", ",
) -> str:
    numeric_lower = coerce_numeric(lower)
    numeric_upper = coerce_numeric(upper)
    if numeric_lower is None or numeric_upper is None:
        return placeholder
    return (
        f"{format_decimal(numeric_lower, digits=digits, placeholder=placeholder)}"
        f"{separator}{format_decimal(numeric_upper, digits=digits, placeholder=placeholder)}"
    )


def format_with_ci(
    mean: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
    ci_label: str = "CI",
) -> str:
    mean_text = format_decimal(mean, digits=digits, placeholder=placeholder)
    numeric_lower = coerce_numeric(lower)
    numeric_upper = coerce_numeric(upper)
    if numeric_lower is None or numeric_upper is None:
        return mean_text
    interval = format_interval(
        numeric_lower,
        numeric_upper,
        digits=digits,
        placeholder=placeholder,
        separator="–",
    )
    return f"{mean_text} ({ci_label}: {interval})"


def format_scalar(
    value: object,
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
) -> str:
    numeric = coerce_numeric(value)
    if numeric is None:
        return placeholder
    return f"{numeric:.{digits}f}"


def format_signed(
    value: object,
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
) -> str:
    numeric = coerce_numeric(value)
    if numeric is None:
        return placeholder
    return f"{numeric:+.{digits}f}"


def format_mean_std(
    mean: object,
    std: object,
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
) -> str:
    numeric_mean = coerce_numeric(mean)
    if numeric_mean is None:
        return placeholder
    numeric_std = coerce_numeric(std)
    if numeric_std is None or numeric_std == 0.0:
        return f"{numeric_mean:.{digits}f}"
    return f"{numeric_mean:.{digits}f} ± {numeric_std:.{digits}f}"


def format_ci(
    lower: object,
    upper: object,
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
) -> str:
    numeric_lower = coerce_numeric(lower)
    numeric_upper = coerce_numeric(upper)
    if numeric_lower is None or numeric_upper is None:
        return placeholder
    return f"[{numeric_lower:.{digits}f}, {numeric_upper:.{digits}f}]"


def format_value_with_ci(
    value: object,
    lower: object,
    upper: object,
    *,
    digits: int = 3,
    placeholder: str = PLACEHOLDER,
) -> str:
    numeric_value = coerce_numeric(value)
    if numeric_value is None:
        return placeholder
    interval = format_ci(lower, upper, digits=digits, placeholder=placeholder)
    if interval == placeholder:
        return f"{numeric_value:.{digits}f}"
    return f"{numeric_value:.{digits}f} {interval}"


def _format_percent_core(
    value: object,
    *,
    digits: int,
) -> Optional[str]:
    numeric = coerce_numeric(value)
    if numeric is None:
        return None
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return str(int(rounded))
    return f"{numeric:.{digits}f}"


def format_percent(
    value: object,
    *,
    digits: int = 1,
    placeholder: str = PLACEHOLDER,
) -> str:
    text = _format_percent_core(value, digits=digits)
    if text is None:
        return placeholder
    return f"{text}%"


def format_percent_numeric(
    value: object,
    *,
    digits: int = 1,
    placeholder: str = PLACEHOLDER,
) -> str:
    text = _format_percent_core(value, digits=digits)
    if text is None:
        return placeholder
    return text


def format_interval_label(
    start: object,
    end: object,
    *,
    digits: int = 1,
    placeholder: str = PLACEHOLDER,
) -> str:
    start_text = format_percent_numeric(start, digits=digits, placeholder=placeholder)
    end_text = format_percent_numeric(end, digits=digits, placeholder=placeholder)
    if start_text == placeholder or end_text == placeholder:
        return placeholder
    return f"{start_text}→{end_text}"
