"""LocalScheduler: executes a TaskGraph on thread or process pools."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from typing import Any

from framex.config import get_config
from framex.runtime.task import TaskGraph
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
        tasks = graph.tasks
        if not tasks:
            return {}

        order = graph.topological_order()
        results: dict[str, Any] = {}

        executor = WorkerExecutor(
            max_workers=self._max_workers,
            backend=self._backend,
        )

        with executor:
            # Fast path: common case in benchmarks where tasks are independent.
            if all(not tasks[tid].dependencies for tid in order):
                futures = {
                    tid: executor.submit(tasks[tid].fn, *tasks[tid].args, **tasks[tid].kwargs)
                    for tid in order
                }
                for tid, fut in futures.items():
                    results[tid] = fut.result()
                return results

            dependents: dict[str, list[str]] = {tid: [] for tid in order}
            in_degree: dict[str, int] = {}
            for tid in order:
                deps = tasks[tid].dependencies
                in_degree[tid] = len(deps)
                for dep in deps:
                    dependents.setdefault(dep, []).append(tid)

            ready = deque([tid for tid in order if in_degree[tid] == 0])

            # Group tasks by "wave" — tasks whose dependencies are all resolved.
            while ready:
                wave = list(ready)
                ready.clear()
                if not wave:
                    raise RuntimeError("Deadlock: no tasks are ready but graph is not empty")

                # Submit the ready wave.
                futures: dict[str, Future[Any]] = {}
                for tid in wave:
                    task = tasks[tid]
                    futures[tid] = executor.submit(task.fn, *task.args, **task.kwargs)

                # Collect results.
                for tid, fut in futures.items():
                    results[tid] = fut.result()
                    for dependent in dependents.get(tid, []):
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            ready.append(dependent)

            if len(results) != len(order):
                raise RuntimeError("Deadlock: no tasks are ready but graph is not empty")

        return results

    def __repr__(self) -> str:
        return f"LocalScheduler(workers={self._max_workers}, backend={self._backend!r})"
