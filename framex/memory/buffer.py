"""Unified buffer abstraction over SharedMemory, mmap files, and Arrow buffers."""

from __future__ import annotations

import mmap
import os
import tempfile
import uuid
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import pyarrow as pa


class BufferBackend(Enum):
    """Supported buffer backends.

    SHARED_MEMORY
        Cross-process zero-copy via ``multiprocessing.SharedMemory``.
        Best for short-lived inter-process data sharing on a single node.
    ARROW
        In-process zero-copy via ``pyarrow.Buffer``.
        Best for within-process column transfers.
    MMAP
        Memory-mapped file via ``mmap.mmap``.
        Best for out-of-core datasets and long-lived read-only sharing
        across processes via the OS page cache.
    """

    SHARED_MEMORY = auto()
    ARROW = auto()
    MMAP = auto()


class Buffer:
    """Unified buffer abstraction.

    Wraps one of:
    - ``multiprocessing.SharedMemory`` — cross-process zero-copy
    - ``pyarrow.Buffer`` — in-process zero-copy
    - ``mmap.mmap`` over a file — out-of-core / OS-page-cache sharing

    Use as a context manager to ensure cleanup.
    """

    def __init__(
        self,
        *,
        backend: BufferBackend = BufferBackend.ARROW,
        size: int = 0,
        name: str | None = None,
        path: str | Path | None = None,
        data: bytes | None = None,
        arrow_buffer: pa.Buffer | None = None,
    ):
        self._backend = backend
        self._shm: SharedMemory | None = None
        self._arrow_buf: pa.Buffer | None = None
        self._mmap: mmap.mmap | None = None
        self._mmap_fd: int | None = None
        self._mmap_path: Path | None = None
        self._mmap_owned: bool = False
        self._name: str | None = name

        if backend == BufferBackend.SHARED_MEMORY:
            if name is not None:
                self._shm = SharedMemory(name=name, create=False)
            else:
                actual_size = len(data) if data else size
                if actual_size <= 0:
                    raise ValueError("Must provide size or data for SharedMemory buffer")
                shm_name = f"fx_{uuid.uuid4().hex[:16]}"
                self._shm = SharedMemory(name=shm_name, create=True, size=actual_size)
                if data:
                    self._shm.buf[:len(data)] = data
            self._name = self._shm.name

        elif backend == BufferBackend.ARROW:
            if arrow_buffer is not None:
                self._arrow_buf = arrow_buffer
            elif data is not None:
                self._arrow_buf = pa.py_buffer(data)
            else:
                self._arrow_buf = pa.allocate_buffer(size)

        elif backend == BufferBackend.MMAP:
            if path is not None:
                self._mmap_path = Path(path)
                self._mmap_owned = False
            else:
                # Create a temporary file.
                fd, tmp = tempfile.mkstemp(prefix="fx_mmap_")
                self._mmap_path = Path(tmp)
                self._mmap_owned = True
                actual_size = len(data) if data else size
                if actual_size <= 0:
                    raise ValueError("Must provide size or data for mmap buffer")
                os.write(fd, data if data else b"\x00" * actual_size)
                os.close(fd)

            file_size = self._mmap_path.stat().st_size
            self._mmap_fd = os.open(str(self._mmap_path), os.O_RDWR)
            self._mmap = mmap.mmap(self._mmap_fd, file_size)
            self._name = str(self._mmap_path)

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # -- Properties ----------------------------------------------------------

    @property
    def backend(self) -> BufferBackend:
        return self._backend

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def size(self) -> int:
        if self._shm is not None:
            return self._shm.size
        if self._arrow_buf is not None:
            return self._arrow_buf.size
        if self._mmap is not None:
            return self._mmap.size()
        return 0

    def as_bytes(self) -> bytes:
        if self._shm is not None:
            return bytes(self._shm.buf)
        if self._arrow_buf is not None:
            return self._arrow_buf.to_pybytes()
        if self._mmap is not None:
            self._mmap.seek(0)
            return self._mmap.read()
        return b""

    def as_memoryview(self) -> memoryview:
        if self._shm is not None:
            return self._shm.buf
        if self._arrow_buf is not None:
            return memoryview(self._arrow_buf)
        if self._mmap is not None:
            return memoryview(self._mmap)
        raise RuntimeError("No buffer available")

    def as_arrow_buffer(self) -> pa.Buffer:
        """Return a zero-copy ``pyarrow.Buffer`` view of this buffer's memory."""
        return pa.py_buffer(self.as_bytes())

    @property
    def path(self) -> Path | None:
        """File path for MMAP buffers; None otherwise."""
        return self._mmap_path

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the buffer handle (does NOT unlink shared memory or delete mmap file)."""
        if self._shm is not None:
            self._shm.close()
        if self._mmap is not None:
            self._mmap.close()
        if self._mmap_fd is not None:
            try:
                os.close(self._mmap_fd)
            except OSError:
                pass
            self._mmap_fd = None

    def unlink(self) -> None:
        """Destroy the underlying OS resource.

        - SHARED_MEMORY: unlinks the shm segment
        - MMAP: deletes the backing file (only if created by this Buffer)
        - ARROW: no-op
        """
        if self._shm is not None:
            self._shm.unlink()
        if self._mmap_owned and self._mmap_path is not None:
            try:
                self._mmap_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> Buffer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Buffer(backend={self._backend.name}, size={self.size}, name={self._name})"
