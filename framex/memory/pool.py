"""BufferPool: lifecycle-managed shared memory pool."""

from __future__ import annotations

import uuid
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any


class BufferPool:
    """Manages a pool of ``SharedMemory`` regions with automatic cleanup.

    Uses ``SharedMemoryManager`` under the hood to ensure all shared memory
    segments are unlinked when the pool is shut down, preventing resource leaks.
    """

    def __init__(self) -> None:
        self._manager: SharedMemoryManager | None = None
        self._segments: dict[str, SharedMemory] = {}
        self._started = False

    def start(self) -> None:
        """Start the underlying ``SharedMemoryManager``."""
        if self._started:
            return
        self._manager = SharedMemoryManager()
        self._manager.start()
        self._started = True

    def allocate(self, size: int) -> SharedMemory:
        """Allocate a new shared memory segment of ``size`` bytes."""
        if not self._started or self._manager is None:
            raise RuntimeError("BufferPool has not been started.  Call .start() first.")
        shm = self._manager.SharedMemory(size)
        self._segments[shm.name] = shm
        return shm

    def get(self, name: str) -> SharedMemory:
        """Get an existing segment by name."""
        if name in self._segments:
            return self._segments[name]
        # Try attaching to an external segment.
        shm = SharedMemory(name=name, create=False)
        self._segments[name] = shm
        return shm

    def release(self, name: str) -> None:
        """Close and remove tracking of a specific segment."""
        if name in self._segments:
            shm = self._segments.pop(name)
            shm.close()

    @property
    def active_segments(self) -> int:
        return len(self._segments)

    def shutdown(self) -> None:
        """Shut down the pool and clean up all segments."""
        for shm in list(self._segments.values()):
            try:
                shm.close()
            except Exception:
                pass
        self._segments.clear()
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
        self._started = False
        self._manager = None

    def __enter__(self) -> BufferPool:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"BufferPool(started={self._started}, segments={self.active_segments})"
