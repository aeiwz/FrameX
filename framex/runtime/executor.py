"""WorkerExecutor: wraps concurrent.futures pools with auto backend detection."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Executor, Future, ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable

import pyarrow as pa


def detect_backend(schema: pa.Schema | None = None) -> str:
    """Heuristic: choose execution backend based on column types.

    Numeric / binary Arrow types release the GIL in many Arrow compute
    operations, making threads efficient.  String, large_string, list,
    struct, and dictionary types tend to involve Python object overhead
    and benefit from process isolation.

    Parameters
    ----------
    schema : pa.Schema | None
        Arrow schema of the data to be processed.  If None, defaults to
        ``"threads"``.

    Returns
    -------
    str
        ``"threads"`` or ``"processes"``.
    """
    if schema is None:
        return "threads"

    _object_types = (
        pa.types.is_string,
        pa.types.is_large_string,
        pa.types.is_binary,
        pa.types.is_large_binary,
        pa.types.is_list,
        pa.types.is_large_list,
        pa.types.is_struct,
        pa.types.is_dictionary,
    )

    for field in schema:
        if any(check(field.type) for check in _object_types):
            return "processes"

    return "threads"


class WorkerExecutor:
    """Unified interface over thread and process pool executors.

    Always uses ``spawn`` start method for ProcessPoolExecutor (safe for
    Python 3.14+).
    """

    def __init__(
        self,
        max_workers: int = 4,
        backend: str = "threads",
        schema: pa.Schema | None = None,
    ):
        self._max_workers = max_workers
        # "auto" resolves at construction time via the schema heuristic.
        self._backend = detect_backend(schema) if backend == "auto" else backend
        self._pool: Executor | None = None

    def _get_pool(self) -> Executor:
        if self._pool is None:
            if self._backend == "threads":
                self._pool = ThreadPoolExecutor(max_workers=self._max_workers)
            elif self._backend == "processes":
                ctx = multiprocessing.get_context("spawn")
                self._pool = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    mp_context=ctx,
                )
            else:
                raise ValueError(f"Unknown backend: {self._backend!r}")
        return self._pool

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a callable for execution."""
        return self._get_pool().submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None

    def __enter__(self) -> WorkerExecutor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"WorkerExecutor(workers={self._max_workers}, backend={self._backend!r})"
