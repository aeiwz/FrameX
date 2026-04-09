"""Benchmarks for memory subsystem: SharedMemory transport, BufferPool."""

import time

import pyarrow as pa

from framex.memory.pool import BufferPool
from framex.memory.transport import recv_zero_copy, send_zero_copy, unlink_shm


def bench_zero_copy_transport(n_rows: int = 1_000_000, n_rounds: int = 5) -> None:
    """Benchmark zero-copy send/recv of a RecordBatch through SharedMemory."""
    import numpy as np

    batch = pa.record_batch({
        "a": np.random.randn(n_rows),
        "b": np.random.randint(0, 100, n_rows),
        "c": np.random.randn(n_rows),
    })

    times: list[float] = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        shm_name, schema = send_zero_copy(batch)
        recovered = recv_zero_copy(shm_name, schema)
        t1 = time.perf_counter()
        unlink_shm(shm_name)
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    size_mb = batch.nbytes / 1e6
    print(f"Zero-copy transport ({n_rows:,} rows, {size_mb:.1f} MB):")
    print(f"  Average: {avg * 1000:.2f} ms")
    print(f"  Throughput: {size_mb / avg:.0f} MB/s")


def bench_buffer_pool(n_allocations: int = 100, size: int = 1024) -> None:
    """Benchmark BufferPool allocation and cleanup."""
    t0 = time.perf_counter()
    with BufferPool() as pool:
        for _ in range(n_allocations):
            pool.allocate(size)
    t1 = time.perf_counter()
    print(f"BufferPool ({n_allocations} allocations of {size} bytes):")
    print(f"  Total: {(t1 - t0) * 1000:.2f} ms")
    print(f"  Per allocation: {(t1 - t0) / n_allocations * 1000:.3f} ms")


if __name__ == "__main__":
    bench_zero_copy_transport()
    print()
    bench_buffer_pool()
