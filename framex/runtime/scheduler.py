"""LocalScheduler: executes a TaskGraph on thread or process pools."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any

from framex.config import get_config
from framex.runtime.task import Task, TaskGraph
from framex.runtime.executor import WorkerExecutor


class LocalScheduler:
    """Executes a ``TaskGraph`` respecting dependencies and ``max_workers``.

    Uses ``ThreadPoolExecutor`` by default (most Arrow/NumPy ops release the
    GIL).  Falls back to ``ProcessPoolExecutor`` with ``spawn`` start method
    when configured for processes.
    """

    def __init__(self, max_workers: int | None = None, backend: str | None = None):
        cfg = get_config()
        self._max_workers = max_workers or cfg.workers
        self._backend = backend or cfg.backend

    def execute(self, graph: TaskGraph) -> dict[str, Any]:
        """Execute the full task graph and return ``{task_id: result}``."""
        order = graph.topological_order()
        results: dict[str, Any] = {}

        executor = WorkerExecutor(
            max_workers=self._max_workers,
            backend=self._backend,
        )

        with executor:
            # Group tasks by "wave" — tasks whose dependencies are all resolved.
            remaining = list(order)
            while remaining:
                ready: list[str] = []
                not_ready: list[str] = []
                for tid in remaining:
                    task = graph.get_task(tid)
                    if all(dep in results for dep in task.dependencies):
                        ready.append(tid)
                    else:
                        not_ready.append(tid)

                if not ready:
                    raise RuntimeError("Deadlock: no tasks are ready but graph is not empty")

                # Submit the ready wave.
                futures: dict[str, Future[Any]] = {}
                for tid in ready:
                    task = graph.get_task(tid)
                    dep_results = {dep: results[dep] for dep in task.dependencies}
                    futures[tid] = executor.submit(task.fn, *task.args, **task.kwargs)

                # Collect results.
                for tid, fut in futures.items():
                    results[tid] = fut.result()

                remaining = not_ready

        return results

    def __repr__(self) -> str:
        return f"LocalScheduler(workers={self._max_workers}, backend={self._backend!r})"
