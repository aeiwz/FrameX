"""Task and TaskGraph definitions for the execution DAG."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Task:
    """Atomic unit of work in the execution engine.

    A task wraps a callable with its arguments and tracks which output
    partition it produces.
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    fn: Callable[..., Any] = field(default=lambda: None)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    output_partition_id: int | None = None
    dependencies: list[str] = field(default_factory=list)

    def execute(self, resolved_deps: dict[str, Any] | None = None) -> Any:
        """Run the task function with resolved dependency outputs."""
        return self.fn(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id!r}, fn={getattr(self.fn, '__name__', '?')}, "
            f"deps={self.dependencies}, partition={self.output_partition_id})"
        )


class TaskGraph:
    """Directed acyclic graph of ``Task`` objects.

    Stores tasks keyed by ``task_id`` with dependency edges.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> str:
        """Add a task and return its ID."""
        self._tasks[task.task_id] = task
        return task.task_id

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Declare that ``task_id`` depends on ``depends_on``."""
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id!r} not in graph")
        if depends_on not in self._tasks:
            raise KeyError(f"Dependency {depends_on!r} not in graph")
        self._tasks[task_id].dependencies.append(depends_on)

    def get_task(self, task_id: str) -> Task:
        return self._tasks[task_id]

    @property
    def tasks(self) -> dict[str, Task]:
        # Exposed as a mutable mapping for scheduler hot paths to avoid
        # per-execution dict copies.
        return self._tasks

    def topological_order(self) -> list[str]:
        """Return task IDs in topological order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                # dep -> task  means task has an incoming edge
                pass
            in_degree[task.task_id] = len(task.dependencies)

        queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        # Build adjacency: dep -> list of dependents
        dependents: dict[str, list[str]] = {tid: [] for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                dependents[dep].append(task.task_id)

        while queue:
            tid = queue.popleft()
            order.append(tid)
            for dependent in dependents.get(tid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._tasks):
            raise RuntimeError("TaskGraph contains a cycle")

        return order

    def __len__(self) -> int:
        return len(self._tasks)

    def __repr__(self) -> str:
        return f"TaskGraph(tasks={len(self._tasks)})"
