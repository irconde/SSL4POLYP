from __future__ import annotations

from ssl4polyp.classification.analysis.display import (
    format_decimal,
    format_interval,
    format_with_ci,
)


def test_format_decimal_defaults() -> None:
    assert format_decimal(0.12345) == "0.123"
    assert format_decimal(None) == "—"
    assert format_decimal(float("nan")) == "—"


def test_format_interval_and_ci() -> None:
    interval = format_interval(0.1234, 0.5678)
    assert interval == "0.123, 0.568"
    ci_text = format_with_ci(0.9876, 0.1111, 0.2222)
    assert ci_text == "0.988 (CI: 0.111–0.222)"
    missing_ci = format_with_ci(0.5, None, None)
    assert missing_ci == "0.500"
