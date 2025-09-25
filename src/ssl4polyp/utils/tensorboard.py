"""Utilities for working with TensorBoard writers.

This module centralises the import of :class:`SummaryWriter` so the rest of
the codebase can depend on a single location. If TensorBoard is unavailable we
provide a lightweight stand-in that preserves the small subset of the API used
throughout the project. This allows training scripts to continue to run while
emitting a clear warning to install TensorBoard for logging support.
"""

from __future__ import annotations

import warnings
from typing import Any

try:  # pragma: no cover - thin wrapper over PyTorch import
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
except (ImportError, AttributeError) as exc:  # pragma: no cover - import guard
    _TorchSummaryWriter = None
    _import_exception = exc
else:  # pragma: no cover - simple alias path
    _import_exception = None


class _NoOpSummaryWriter:
    """Fallback SummaryWriter implementation.

    The training scripts only rely on ``add_scalar``, ``flush`` and ``close``.
    Each method is implemented as a no-op so the rest of the code can proceed
    without TensorBoard. A warning is emitted once to alert the user that
    installing TensorBoard will enable logging output.
    """

    _warned: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self.__class__._warned:
            message = (
                "TensorBoard SummaryWriter could not be imported. "
                "Install TensorBoard to enable logging output. Original error: "
                f"{_import_exception}"
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self.__class__._warned = True

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        """No-op replacement for :meth:`SummaryWriter.add_scalar`."""

    def flush(self) -> None:
        """No-op replacement for :meth:`SummaryWriter.flush`."""

    def close(self) -> None:
        """No-op replacement for :meth:`SummaryWriter.close`."""


SummaryWriter = _TorchSummaryWriter or _NoOpSummaryWriter

__all__ = ["SummaryWriter"]

