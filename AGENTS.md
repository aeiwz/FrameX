# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Status

FrameX is in the **design/research phase**. Currently the repo contains only `docs/deep-research-matrix-dataframe.md` — a comprehensive research document outlining the architecture. No source code exists yet.

## What FrameX Is

A high-performance Python library for parallel dataframe and array processing that combines Pandas and NumPy semantics with a multiprocessing-friendly runtime. Target workloads: ETL/analytics tables from ~100MB to ~30–100GB on a single machine.

**Viable niche**: local multiprocessing efficiency with minimal serialization/copying, and a compatibility-minded API that presents both dataframe and ndarray semantics while interoperating with Pandas/NumPy/Arrow protocols.

## Planned Architecture

```
User API (DataFrame / Series / NDArray)
    ↓
Planner (eager micro-plan or lazy DAG)
    ↓
Optimizer (fusion, predicate pushdown, shuffle planning)
    ↓
Scheduler (local DAG executor)
    ↓
Worker Process/Thread Pools
    ↓
Buffer Store (SharedMemory / mmap / Arrow buffers)
    ↓
IO Layer (Parquet / IPC / CSV)
```

**Core design decisions (from research doc):**
- **Storage**: Arrow columnar format with chunked partitions — enables zero-copy possibilities and cross-library interchange
- **Concurrency model**: Hybrid — threads for NumPy numeric ops (many release the GIL), processes for Python-heavy/object-dtype workloads
- **Execution**: Eager by default (Pandas ergonomics) with optional lazy mode (`.lazy()` / `.collect()` like Polars)
- **Zero-copy transport**: `SharedMemory`, memory-mapped files, Arrow IPC
- **Semantic compatibility is a layer**, not baked into every internal optimization (the Modin lesson)

## Planned API Surface

Core objects: `DataFrame`, `Series`, `Index` (Pandas-like) + `Array`/`NDArray` (NumPy-like)

Interop contracts to implement early:
- `__array_ufunc__` (NEP 13) and `__array_function__` (NEP 18) for NumPy dispatch
- `__dataframe__` (DataFrame interchange protocol) for cross-library handoff
- Arrow C Data Interface for in-process zero-copy exchange
- `.to_pandas()`, `.to_numpy()`, `.to_arrow()` + symmetric constructors

## Planned Tech Stack

**Core dependencies**: PyArrow, NumPy, Pandas (as reference semantics)

**Build/performance tools**: Cython or Numba for native kernels, `cibuildwheel` for cross-platform wheels, `scikit-build-core` or `meson-python` for C++ extensions, `asv` + `pyperf` for benchmarking

**Optional backends**: Dask (distributed scheduling), Ray (object store + zero-copy deserialization)

## Implementation Roadmap

| Milestone | Target |
|-----------|--------|
| API contracts + product definition | Apr–May 2026 |
| Storage layer v1 (Arrow-backed, chunked partitions) | May–Jun 2026 |
| Local execution engine v1 + zero-copy transport | Jun–Aug 2026 |
| Operator suite (groupby, join, window, sort) | Aug–Nov 2026 |
| Packaging + beta release | Dec 2026 |

## Key Design Tensions

1. **"Drop-in Pandas" vs "Pandas-like"**: full compatibility is a long-tail engineering project. Prefer explicit semantic divergence over silent differences.
2. **Threads vs processes**: default to threads for numeric ops (GIL-releasing), processes for object-heavy workloads. Python 3.14 changed POSIX default from `fork` to `forkserver`.
3. **Serialization security**: `pickle` is not secure. Arrow IPC is the preferred safe serializer; make serializers configurable.
4. **Partitioning strategy**: row-wise (Dask-style) is simplest; row+column (Modin-style) enables more parallelism but complicates semantics.
5. **Shared memory lifecycle**: use `SharedMemoryManager` to avoid DoS/leak risk.

## Testing Strategy (when implemented)

- **Correctness**: pytest with dtype/shape/partition variants
- **Performance regression**: `asv` (Airspeed Velocity)
- **Representative benchmarks**: ETL (Parquet → filter → groupby → write), analytics (joins + groupby + top-k), ML preprocessing (encoding + scaling + train-test split)

## Packaging

Target: permissive license (BSD-3, Apache-2.0, or MIT) aligned with ecosystem. Wheels via `cibuildwheel`.
