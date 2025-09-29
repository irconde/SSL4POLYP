"""Utilities for working with TensorBoard writers.

This module centralises the import of :class:`SummaryWriter` so the rest of
the codebase can depend on a single location. If TensorBoard is unavailable we
provide a lightweight stand-in that preserves the small subset of the API used
throughout the project. This allows training scripts to continue to run while
emitting a clear warning to install TensorBoard for logging support.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

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


class _SafeSummaryWriter:
    """Runtime guard around the real TensorBoard SummaryWriter.

    When the underlying :mod:`tensorboard` writer raises an :class:`OSError`
    (typically caused by filesystem issues) the wrapper emits a warning,
    closes the writer to flush any buffered events and turns into a no-op for
    all subsequent calls. This keeps training loops running while surfacing
    the logging failure to the user.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _TorchSummaryWriter is None:  # pragma: no cover - defensive guard
            raise RuntimeError("_SafeSummaryWriter requires torch SummaryWriter")
        self._factory: Callable[..., Any] = kwargs.pop("_factory", _TorchSummaryWriter)
        self._writer: Optional[Any] = self._factory(*args, **kwargs)
        self._enabled = True

    # ``SummaryWriter`` implements ``__enter__``/``__exit__`` and some calling
    # sites rely on it. Mirroring that behaviour keeps the wrapper drop-in.
    def __enter__(self) -> "_SafeSummaryWriter":  # pragma: no cover - passthrough
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - passthrough
        self.close()

    @property
    def enabled(self) -> bool:
        """Return ``True`` while the underlying writer accepts new events."""

        return self._enabled and self._writer is not None

    def _disable(self, exc: BaseException) -> None:
        if not self._enabled:
            return
        warnings.warn(
            "TensorBoard logging disabled after failing to write events: "
            f"{exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self._writer = None
        self._enabled = False

    def _call_writer(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return None
        assert self._writer is not None  # for type checkers
        method = getattr(self._writer, name)
        try:
            return method(*args, **kwargs)
        except (OSError, IOError) as exc:
            self._disable(exc)
            return None

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        self._call_writer("add_scalar", *args, **kwargs)

    def flush(self) -> None:
        self._call_writer("flush")

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
        self._enabled = False


SummaryWriter = _SafeSummaryWriter if _TorchSummaryWriter else _NoOpSummaryWriter

__all__ = ["SummaryWriter"]

