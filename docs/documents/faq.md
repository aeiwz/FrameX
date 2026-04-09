---
title: FAQ
description: Common questions about compatibility, performance, and migration.
order: 10
section: Project
---

# FAQ

## Is FrameX a drop-in replacement for Pandas?

Not fully. The API is intentionally Pandas-like, but full parity is not the current goal. Use explicit compatibility checks at critical boundaries.

## When should I use lazy mode?

Use `.lazy()` when you have long transformation chains and want to defer execution until `collect()`. For quick interactive work, eager mode is often simpler.

## Does FrameX support NumPy operations?

Yes. `NDArray` implements NumPy protocol hooks, so many ufuncs and array functions interoperate directly.
You can also choose array execution backends (`auto`, `numpy`, `numexpr`, `numba`, `torch`, `jax`, `cupy`) with `set_array_backend(...)`.

## Does FrameX support Ray or Dask?

Yes, as optional execution/runtime integrations:

- `set_backend("ray")` and `set_backend("dask")` for partition execution paths
- interchange helpers `from_dask(...)` and `from_ray(...)`
- conversions `.to_dask()` and `.to_ray()`

FrameX remains single-machine-first; distributed multi-node orchestration is not a primary design target.

## How should I migrate from Pandas?

Migrate incrementally:

1. start with the heaviest ETL stage
2. compare outputs with Pandas via `.to_pandas()`
3. expand usage once validated

## Which file formats are supported today?

FrameX supports read/write for:

- Parquet (`.parquet`)
- Arrow IPC (`.arrow`, `.ipc`)
- CSV/TSV (`.csv`, `.tsv`)
- JSON / NDJSON (`.json`, `.jsonl`, `.ndjson`)
- Feather (`.feather`)
- Pickle (`.pkl`, `.pickle`)
- Excel (`.xlsx`, `.xls`) via pandas-compatible backend

`read_file(...)` and `write_file(...)` auto-detect by extension and support compressed wrappers:
`.gz`, `.bz2`, `.xz`, `.zip`, plus `.zst`/`.zstd` when `zstandard` is installed.

## How do I tune execution?

Use runtime config APIs:

- `set_backend(...)`
- `set_workers(...)`
- `set_serializer(...)`
- `set_kernel_backend(...)`
- `set_array_backend(...)`

## Why do some tests show as skipped?

Several tests are optional-backend tests and use `pytest.importorskip(...)`.
They are skipped (not failed) when optional runtimes are missing, especially:

- `dask.distributed` / `dask.dataframe`
- `ray` / `ray.data`

Install optional deps to run those suites:

```bash
pip install "dask[distributed]" "ray[data]"
pytest -q
```
