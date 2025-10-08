"""Seed validation helpers for experiment analysis modules."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence, Tuple

__all__ = [
    "SeedCheckError",
    "SeedValidationResult",
    "ensure_expected_seeds",
]


class SeedCheckError(RuntimeError):
    """Raised when discovered runs do not match the expected seed configuration."""


@dataclass(frozen=True)
class SeedValidationResult:
    """Immutable record describing the validated seed layout."""

    expected_seeds: Tuple[int, ...]
    observed_seeds: Mapping[str, Tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_seeds": list(self.expected_seeds),
            "observed_seeds": {
                key: list(value) for key, value in self.observed_seeds.items()
            },
        }


def _coerce_seed(value: object) -> int:
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty string cannot be coerced to a seed value")
        return int(text)
    raise TypeError(f"Unsupported seed type: {type(value)!r}")


def _normalise_seed_collection(values: Iterable[object]) -> Tuple[int, ...]:
    seen: set[int] = set()
    for raw in values:
        seed = _coerce_seed(raw)
        seen.add(seed)
    return tuple(sorted(seen))


def ensure_expected_seeds(
    groups: Mapping[str, Mapping[int, object] | Iterable[object]],
    *,
    expected_seeds: Sequence[int],
    context: str,
) -> SeedValidationResult:
    """Validate that every group exposes exactly the expected seeds.

    Args:
        groups: Mapping of group labels (e.g. model names) to seed containers. Values
            may either be mappings keyed by seed or iterables of seed identifiers.
        expected_seeds: Iterable describing the required seed set.
        context: Human readable label used to annotate error messages.

    Returns:
        A :class:`SeedValidationResult` describing the validated seed layout.

    Raises:
        SeedCheckError: If any group is missing a required seed, exposes unexpected
            seeds, or if no groups were provided while seeds are required.
    """

    expected = tuple(sorted({_coerce_seed(seed) for seed in expected_seeds}))
    observed: dict[str, Tuple[int, ...]] = {}
    errors: list[str] = []

    if not groups:
        if expected:
            raise SeedCheckError(
                f"{context} seed validation failed: no runs discovered; expected seeds {list(expected)}"
            )
        return SeedValidationResult(expected, MappingProxyType(observed))

    expected_set = set(expected)
    for label, container in sorted(groups.items(), key=lambda item: str(item[0])):
        if isinstance(container, Mapping):
            seeds = _normalise_seed_collection(container.keys())
        else:
            seeds = _normalise_seed_collection(container)
        observed[str(label)] = seeds
        seed_set = set(seeds)
        missing = sorted(expected_set - seed_set)
        extra = sorted(seed_set - expected_set)
        if missing:
            errors.append(f"{label} missing seeds {missing}")
        if extra:
            errors.append(f"{label} has unexpected seeds {extra}")

    if errors:
        detail = "; ".join(errors)
        raise SeedCheckError(
            f"{context} seed validation failed: {detail}. Expected seeds: {list(expected)}"
        )

    return SeedValidationResult(expected, MappingProxyType(observed))
