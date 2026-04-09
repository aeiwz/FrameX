"""WorkerExecutor: wraps thread/process/Ray/Dask/HPC pools with auto detection."""

from __future__ import annotations

import multiprocessing
import os
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
        self._pool: Executor | _RayExecutor | _DaskExecutor | _HPCExecutor | None = None

    def _get_pool(self) -> Executor | _RayExecutor | _DaskExecutor | _HPCExecutor:
        if self._pool is None:
            if self._backend == "threads":
                self._pool = ThreadPoolExecutor(max_workers=self._max_workers)
            elif self._backend == "processes":
                ctx = multiprocessing.get_context("spawn")
                self._pool = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    mp_context=ctx,
                )
            elif self._backend == "ray":
                self._pool = _RayExecutor(max_workers=self._max_workers)
            elif self._backend == "dask":
                self._pool = _DaskExecutor(max_workers=self._max_workers)
            elif self._backend == "hpc":
                self._pool = _HPCExecutor(max_workers=self._max_workers)
            else:
                raise ValueError(f"Unknown backend: {self._backend!r}")
        return self._pool

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any] | _RayFuture | _DaskFuture:
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


def _ray_call(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return fn(*args, **kwargs)


class _RayFuture:
    def __init__(self, object_ref: Any):
        self._ref = object_ref

    def result(self, timeout: float | None = None) -> Any:
        import ray

        if timeout is None:
            return ray.get(self._ref)
        ready, _ = ray.wait([self._ref], timeout=timeout)
        if not ready:
            raise TimeoutError("Ray task timed out")
        return ray.get(self._ref)


class _RayExecutor:
    def __init__(self, max_workers: int):
        import ray

        self._max_workers = max_workers
        self._owns_runtime = False
        address = os.getenv("FRAMEX_RAY_ADDRESS", "").strip()
        if not ray.is_initialized():
            if address:
                # Connect to an existing cluster (e.g., HPC job/heads).
                ray.init(
                    address=address,
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
            else:
                ray.init(
                    num_cpus=max_workers,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
                self._owns_runtime = True
        self._remote = ray.remote(_ray_call)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> _RayFuture:
        ref = self._remote.options(num_cpus=1).remote(fn, args, kwargs)
        return _RayFuture(ref)

    def shutdown(self, wait: bool = True) -> None:
        if self._owns_runtime:
            import ray

            ray.shutdown()


class _DaskFuture:
    def __init__(self, future: Any):
        self._future = future

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)


class _DaskExecutor:
    def __init__(self, max_workers: int):
        from dask.distributed import Client, LocalCluster, get_client

        self._cluster = None
        self._owns_client = False
        scheduler_address = os.getenv("FRAMEX_DASK_SCHEDULER_ADDRESS", "").strip()
        slurm_enabled = os.getenv("FRAMEX_DASK_SLURM", "").strip() in {"1", "true", "True"}

        try:
            self._client = get_client()
        except ValueError:
            if scheduler_address:
                # Connect to an existing distributed scheduler (HPC/head node).
                self._client = Client(scheduler_address)
            elif slurm_enabled:
                # Optional SLURM bootstrap for HPC environments.
                try:
                    from dask_jobqueue import SLURMCluster
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "FRAMEX_DASK_SLURM is enabled but dask-jobqueue is not installed"
                    ) from exc

                queue = os.getenv("FRAMEX_DASK_SLURM_QUEUE", None)
                account = os.getenv("FRAMEX_DASK_SLURM_ACCOUNT", None)
                walltime = os.getenv("FRAMEX_DASK_SLURM_WALLTIME", "00:30:00")
                cores = int(os.getenv("FRAMEX_DASK_SLURM_CORES", "1"))
                memory = os.getenv("FRAMEX_DASK_SLURM_MEMORY", "2GB")

                self._cluster = SLURMCluster(
                    queue=queue,
                    account=account,
                    walltime=walltime,
                    cores=cores,
                    memory=memory,
                    processes=1,
                )
                self._cluster.scale(jobs=max_workers)
                self._client = Client(self._cluster)
            else:
                # No active client: create a local one.
                use_processes = os.getenv("FRAMEX_DASK_PROCESSES", "1").strip() not in {"0", "false", "False"}
                self._cluster = LocalCluster(
                    n_workers=max_workers,
                    threads_per_worker=1,
                    processes=use_processes,
                    dashboard_address=None,
                    silence_logs="error",
                )
                self._client = Client(self._cluster)
            self._owns_client = True

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> _DaskFuture:
        fut = self._client.submit(fn, *args, **kwargs, pure=False)
        return _DaskFuture(fut)

    def shutdown(self, wait: bool = True) -> None:
        if self._owns_client:
            self._client.close()
            if self._cluster is not None:
                self._cluster.close()


class _HPCExecutor:
    """Cluster-oriented executor.

    Backend selection is controlled by environment variable:
    - ``FRAMEX_HPC_ENGINE=dask`` (default)
    - ``FRAMEX_HPC_ENGINE=ray``
    """

    def __init__(self, max_workers: int):
        engine = os.getenv("FRAMEX_HPC_ENGINE", "dask").strip().lower()
        if engine == "ray":
            self._delegate: _RayExecutor | _DaskExecutor = _RayExecutor(max_workers=max_workers)
        else:
            self._delegate = _DaskExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> _RayFuture | _DaskFuture:
        return self._delegate.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self._delegate.shutdown(wait=wait)
