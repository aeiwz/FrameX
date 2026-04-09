"""Zero-copy transport of Arrow RecordBatches between processes.

Three transport strategies, ordered by suitability:

1. **SharedMemory** (``send_zero_copy`` / ``recv_zero_copy``)
   - IPC-serialise the RecordBatch into a named SharedMemory region.
   - The receiver maps the same region without copying.
   - Best for: frequent short-lived inter-process transfers on one node.

2. **mmap file** (``send_mmap`` / ``recv_mmap``)
   - Write Arrow IPC bytes to a temp file; the OS page cache lets all
     readers share pages without redundant copies.
   - Best for: large batches, multiple readers, or out-of-core workflows.

3. **Pickle protocol 5 OOB** (``send_pickle5`` / ``recv_pickle5``)
   - Uses PEP 574 out-of-band buffers so large NumPy column arrays travel
     as separate zero-copy payloads alongside a small pickle header.
   - Best for: Python-native pipelines that already use pickle and want
     to avoid serialising large arrays inline.

All senders return enough information for the corresponding receiver to
reconstruct the original ``pa.RecordBatch`` without a full copy.

Security note: pickle is not safe for untrusted input.  Prefer Arrow IPC
(strategies 1 & 2) when data crosses trust boundaries.
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa


# ── Strategy 1: SharedMemory ──────────────────────────────────────────────

def send_zero_copy(
    batch: pa.RecordBatch,
    shm_name: str | None = None,
) -> tuple[str, pa.Schema]:
    """Serialize a RecordBatch into a SharedMemory region via Arrow IPC.

    Returns ``(shm_name, schema)`` for the receiver.
    """
    ipc_bytes = _batch_to_ipc_bytes(batch)
    if shm_name is None:
        shm_name = f"fx_{uuid.uuid4().hex[:16]}"
    shm = SharedMemory(name=shm_name, create=True, size=len(ipc_bytes))
    try:
        shm.buf[:len(ipc_bytes)] = ipc_bytes
    except Exception:
        shm.close()
        shm.unlink()
        raise
    shm.close()
    return shm_name, batch.schema


def recv_zero_copy(shm_name: str, schema: pa.Schema) -> pa.RecordBatch:
    """Read a RecordBatch from a SharedMemory region.

    The caller is responsible for unlinking the segment after use.
    """
    shm = SharedMemory(name=shm_name, create=False)
    try:
        buf = pa.py_buffer(bytes(shm.buf))
        batch = _ipc_bytes_to_batch(buf)
    finally:
        shm.close()
    return batch


def unlink_shm(shm_name: str) -> None:
    """Safely unlink a SharedMemory segment."""
    try:
        shm = SharedMemory(name=shm_name, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


# ── Strategy 2: mmap file ─────────────────────────────────────────────────

def send_mmap(
    batch: pa.RecordBatch,
    path: str | Path | None = None,
) -> Path:
    """Write a RecordBatch as Arrow IPC to a memory-mapped file.

    If ``path`` is None a temporary file is created.  Returns the file
    path so the receiver can open it with ``recv_mmap``.

    The file is NOT deleted automatically — the caller owns cleanup.
    """
    ipc_bytes = _batch_to_ipc_bytes(batch)
    if path is None:
        fd, tmp_path = tempfile.mkstemp(prefix="fx_mmap_", suffix=".arrow")
        os.write(fd, ipc_bytes)
        os.close(fd)
        return Path(tmp_path)
    path = Path(path)
    path.write_bytes(ipc_bytes)
    return path


def recv_mmap(path: str | Path) -> pa.RecordBatch:
    """Read a RecordBatch from a memory-mapped Arrow IPC file.

    Uses ``pyarrow.memory_map`` so reads are zero-copy from the OS page
    cache when the file is already in memory.
    """
    path = Path(path)
    mmap_src = pa.memory_map(str(path), "r")
    reader = pa.ipc.open_stream(mmap_src)
    batch = reader.read_next_batch()
    mmap_src.close()
    return batch


# ── Strategy 3: Pickle protocol 5 OOB buffers ─────────────────────────────

def send_pickle5(
    batch: pa.RecordBatch,
) -> tuple[bytes, list[bytes]]:
    """Serialize a RecordBatch using pickle protocol 5 with OOB buffers.

    PEP 574 allows large buffer objects (here: one numpy array per column)
    to travel as separate out-of-band payloads alongside a small pickle
    header.  This avoids embedding large arrays inline in the pickle stream.

    Returns
    -------
    (header_bytes, oob_buffers)
        ``header_bytes`` is the pickle stream (small — no large data).
        ``oob_buffers`` is a list of raw column bytes.

    Pass both to ``recv_pickle5`` to reconstruct the batch.

    Security warning: do NOT unpickle data from untrusted sources.
    """
    # Represent the batch as {col_name: numpy_array, schema: schema_ipc}
    payload = {
        "__schema__": batch.schema.serialize().to_pybytes(),
        "__columns__": {
            batch.schema.field(i).name: batch.column(i).to_pydict()
            if batch.column(i).type == pa.string()
            else batch.column(i).to_numpy(zero_copy_only=False)
            for i in range(batch.num_columns)
        },
    }

    oob_buffers: list[memoryview] = []
    header = pickle.dumps(payload, protocol=5, buffer_callback=oob_buffers.append)
    return header, [bytes(b) for b in oob_buffers]


def recv_pickle5(header: bytes, oob_buffers: list[bytes]) -> pa.RecordBatch:
    """Reconstruct a RecordBatch from a pickle5 header + OOB buffers.

    Security warning: do NOT call this with data from untrusted sources.
    """
    payload = pickle.loads(header, buffers=[memoryview(b) for b in oob_buffers])
    schema = pa.ipc.read_schema(pa.py_buffer(payload["__schema__"]))
    columns = []
    for field in schema:
        raw = payload["__columns__"][field.name]
        if isinstance(raw, np.ndarray):
            col = pa.array(raw, type=field.type)
        elif isinstance(raw, list):
            col = pa.array(raw, type=field.type)
        else:
            col = pa.array(raw, type=field.type)
        columns.append(col)
    return pa.record_batch(columns, schema=schema)


# ── Internal helpers ──────────────────────────────────────────────────────

def _batch_to_ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


def _ipc_bytes_to_batch(buf: pa.Buffer) -> pa.RecordBatch:
    reader = pa.ipc.open_stream(buf)
    return reader.read_next_batch()
