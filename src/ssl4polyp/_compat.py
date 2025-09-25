"""Compatibility helpers for the pinned dependency stack."""

from __future__ import annotations

from collections import abc as _abc
import sys
import types


def ensure_torch_container_abcs() -> None:
    """Restore ``torch._six.container_abcs`` expected by timm==0.3.2.

    ``torch._six`` stopped exposing ``container_abcs`` in newer torch releases,
    but timm==0.3.2 still imports it.  The pinned stack therefore needs this
    shim to run before any timm modules are loaded.
    """

    try:
        import torch._six as _six  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        _six = types.ModuleType("torch._six")
        sys.modules["torch._six"] = _six

    if not hasattr(_six, "container_abcs"):
        _six.container_abcs = _abc  # type: ignore[attr-defined]

