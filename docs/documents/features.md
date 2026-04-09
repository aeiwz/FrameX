---
title: Features
description: Core FrameX capabilities across DataFrame, NDArray, runtime, and interoperability.
order: 4
section: Guides
---

# Features

FrameX focuses on high-throughput local analytics with predictable behavior and practical interoperability.

## DataFrame Features

- Arrow-backed partitioned `DataFrame` and `Series`
- Filtering, projection, sorting, joins, and groupby aggregation
- Eager API with optional lazy pipelines (`.lazy().collect()`)
- Pandas-compatible fallback for unimplemented methods

## NDArray Features

- Chunked `NDArray` with NumPy protocol support:
  - `__array_ufunc__`
  - `__array_function__`
- Block-parallel operations:
  - `.apply_blocks(...)`
  - `.parallel_map(...)`
  - `.jit_apply(...)`

## Runtime Features

- Local backends: `threads`, `processes`
- Optional distributed backends: `ray`, `dask`, `hpc`
- Hardware-aware auto configuration:
  - `recommend_best_performance_config()`
  - `auto_configure_hardware()`

## Interoperability Features

- `from_pandas(...)`, `from_dask(...)`, `from_ray(...)`
- `.to_pandas()`, `.to_arrow()`, `.to_dask()`, `.to_ray()`
- DataFrame interchange protocol support

## I/O Features

- Unified `read_file(...)` / `write_file(...)`
- Formats:
  - Parquet, ORC, Arrow IPC
  - CSV/TSV/Text + fixed-width text
  - JSON/NDJSON, Feather, Pickle, Excel, SQLite
  - Export-only: HTML and XML
- Compression wrappers:
  - `.gz`, `.bz2`, `.xz`, `.zip`
  - `.zst`/`.zstd` (with `zstandard`)
