"""Global configuration for FrameX runtime behavior."""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pprint import pformat
from typing import Iterator, Literal


BackendType = Literal["threads", "processes", "ray", "dask", "hpc"]
SerializerType = Literal["arrow", "pickle5", "pickle"]
KernelBackendType = Literal["python", "c"]
ArrayBackendType = Literal["auto", "numpy", "numexpr", "numba", "cupy", "torch", "jax"]


@dataclass(frozen=True)
class Config:
    """Immutable configuration snapshot."""

    backend: BackendType = "threads"
    workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    serializer: SerializerType = "arrow"
    partition_size_rows: int = 500_000
    kernel_backend: KernelBackendType = "python"
    array_backend: ArrayBackendType = "auto"


# Module-level mutable state — guarded by accessor functions.
_current: Config = Config()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _detect_total_memory_gb() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return None


def _detect_gpu_array_backend() -> ArrayBackendType | None:
    # Highest-priority path: CuPy with at least one CUDA device.
    if _module_available("cupy"):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                return "cupy"
        except Exception:
            pass

    # PyTorch CUDA path.
    if _module_available("torch"):
        try:
            import torch

            if torch.cuda.is_available():
                return "torch"
        except Exception:
            pass

    # JAX accelerated devices path (GPU/TPU).
    if _module_available("jax"):
        try:
            import jax

            if any(d.platform in {"gpu", "tpu"} for d in jax.devices()):
                return "jax"
        except Exception:
            pass

    return None


def _choose_partition_size_rows(total_memory_gb: float | None) -> int:
    if total_memory_gb is None:
        return 500_000
    if total_memory_gb >= 128:
        return 2_000_000
    if total_memory_gb >= 64:
        return 1_000_000
    if total_memory_gb >= 16:
        return 500_000
    return 250_000


def recommend_best_performance_config() -> Config:
    """Recommend a hardware-aware config tuned for best throughput."""
    workers = os.cpu_count() or 4
    memory_gb = _detect_total_memory_gb()
    partition_size_rows = _choose_partition_size_rows(memory_gb)

    # Prefer HPC backend only when a cluster entrypoint is configured.
    if (
        os.getenv("FRAMEX_DASK_SCHEDULER_ADDRESS", "").strip()
        or os.getenv("FRAMEX_RAY_ADDRESS", "").strip()
        or os.getenv("FRAMEX_DASK_SLURM", "").strip() in {"1", "true", "True"}
    ):
        backend: BackendType = "hpc"
    else:
        backend = "threads"

    kernel_backend: KernelBackendType = "python"
    try:
        from framex.backends.c_backend import C_AVAILABLE

        if C_AVAILABLE:
            kernel_backend = "c"
    except Exception:
        kernel_backend = "python"

    gpu_backend = _detect_gpu_array_backend()
    if gpu_backend is not None:
        array_backend: ArrayBackendType = gpu_backend
    elif _module_available("numexpr"):
        array_backend = "numexpr"
    elif _module_available("numba"):
        array_backend = "numba"
    else:
        array_backend = "numpy"

    return Config(
        backend=backend,
        workers=workers,
        serializer="arrow",
        partition_size_rows=partition_size_rows,
        kernel_backend=kernel_backend,
        array_backend=array_backend,
    )


def auto_configure_hardware(*, apply: bool = True) -> Config:
    """Auto-detect hardware and configure FrameX for best performance.

    Parameters
    ----------
    apply:
        When True (default), applies the recommended config globally.
        When False, returns the recommended config without mutating globals.
    """
    cfg = recommend_best_performance_config()
    if apply:
        global _current
        _current = cfg
    return cfg


def get_config() -> Config:
    """Return the current global configuration."""
    return _current


def print_config() -> None:
    """Print the current global configuration to the console."""
    config_dict = {
        "backend": _current.backend,
        "workers": _current.workers,
        "serializer": _current.serializer,
        "partition_size_rows": _current.partition_size_rows,
        "kernel_backend": _current.kernel_backend,
        "array_backend": _current.array_backend,
    }
    print("FrameX configuration:")
    print(pformat(config_dict, sort_dicts=False))


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


def set_array_backend(array_backend: ArrayBackendType) -> None:
    """Switch NDArray ufunc backend.

    Supported backends:
    - ``"auto"``: try accelerated engines in order, then NumPy fallback
    - ``"numpy"``: default NumPy execution
    - ``"numexpr"``: accelerated expressions when numexpr is installed
    - ``"numba"``: JIT-friendly execution path (fallbacks when unsupported)
    - ``"cupy"``: GPU execution when CuPy is installed
    - ``"torch"``: tensor execution path when PyTorch is installed
    - ``"jax"``: XLA-backed execution path when JAX is installed
    """
    if array_backend not in ("auto", "numpy", "numexpr", "numba", "cupy", "torch", "jax"):
        raise ValueError(
            "array_backend must be 'auto', 'numpy', 'numexpr', 'numba', 'cupy', 'torch', or 'jax', "
            f"got {array_backend!r}"
        )
    global _current
    _current = replace(_current, array_backend=array_backend)


@contextmanager
def config(
    *,
    backend: BackendType | None = None,
    workers: int | None = None,
    serializer: SerializerType | None = None,
    partition_size_rows: int | None = None,
    kernel_backend: KernelBackendType | None = None,
    array_backend: ArrayBackendType | None = None,
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
    if array_backend is not None:
        overrides["array_backend"] = array_backend
    _current = replace(prev, **overrides)
    try:
        yield _current
    finally:
        _current = prev
