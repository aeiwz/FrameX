"""Global configuration for FrameX runtime behavior."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator, Literal


BackendType = Literal["threads", "processes"]
SerializerType = Literal["arrow", "pickle5", "pickle"]
KernelBackendType = Literal["python", "c"]


@dataclass(frozen=True)
class Config:
    """Immutable configuration snapshot."""

    backend: BackendType = "threads"
    workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    serializer: SerializerType = "arrow"
    partition_size_rows: int = 500_000
    kernel_backend: KernelBackendType = "python"


# Module-level mutable state — guarded by accessor functions.
_current: Config = Config()


def get_config() -> Config:
    """Return the current global configuration."""
    return _current


def set_backend(backend: BackendType) -> None:
    global _current
    _current = replace(_current, backend=backend)


def set_workers(workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    global _current
    _current = replace(_current, workers=workers)


def set_serializer(serializer: SerializerType) -> None:
    global _current
    _current = replace(_current, serializer=serializer)


def set_kernel_backend(kernel_backend: KernelBackendType) -> None:
    """Switch the compute kernel backend.

    ``"python"`` (default) uses pyarrow.compute for all operations.
    ``"c"`` routes eligible operations (float64/int64 reductions and
    elementwise ops) through compiled C kernels via ctypes.  Falls back
    to the Python backend transparently when the C library is unavailable
    or the data contains nulls.
    """
    if kernel_backend not in ("python", "c"):
        raise ValueError(f"kernel_backend must be 'python' or 'c', got {kernel_backend!r}")
    global _current
    _current = replace(_current, kernel_backend=kernel_backend)


@contextmanager
def config(
    *,
    backend: BackendType | None = None,
    workers: int | None = None,
    serializer: SerializerType | None = None,
    partition_size_rows: int | None = None,
    kernel_backend: KernelBackendType | None = None,
) -> Iterator[Config]:
    """Context manager that temporarily overrides global config."""
    global _current
    prev = _current
    overrides: dict = {}
    if backend is not None:
        overrides["backend"] = backend
    if workers is not None:
        overrides["workers"] = workers
    if serializer is not None:
        overrides["serializer"] = serializer
    if partition_size_rows is not None:
        overrides["partition_size_rows"] = partition_size_rows
    if kernel_backend is not None:
        overrides["kernel_backend"] = kernel_backend
    _current = replace(prev, **overrides)
    try:
        yield _current
    finally:
        _current = prev
