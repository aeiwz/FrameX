# FrameX

FrameX is an Arrow-backed Python library for parallel dataframe and array processing on a single machine.

It combines:

- Pandas-like tabular APIs (`DataFrame`, `Series`, `GroupBy`)
- NumPy-compatible chunked arrays (`NDArray` with NumPy protocol support)
- Arrow-native storage/interop (`to_arrow`, Parquet/IPC I/O)
- Eager execution with optional lazy pipelines (`.lazy().collect()`)
- Runtime backends for local threads/processes plus optional Ray/Dask executors

## Why FrameX

FrameX is aimed at local analytics workflows that are bigger than comfortable single-threaded scripts but do not yet require distributed infrastructure.

Typical fit:

- ETL and analytics pipelines on medium-to-large local datasets
- feature engineering workflows that mix table and array operations
- migration paths from Pandas scripts where API familiarity matters

## Installation

From PyPI:

```bash
pip install framex
```

From source:

```bash
git clone https://github.com/aeiwz/FrameX.git
cd FrameX
pip install -e .
```

Requirements:

- Python `>=3.10`
- Core dependencies: `pyarrow`, `numpy`, `dask`, `ray`
- Optional compatibility: `pandas` (`pip install framex[pandas_compat]`)

## Quick Start

```python
import framex as fx

df = fx.DataFrame(
    {
        "group": ["a", "a", "b"],
        "value": [10, 20, 30],
        "is_refund": [False, True, False],
    }
)

result = (
    df.filter(~df["is_refund"])
      .groupby("group")
      .agg({"value": ["sum", "mean", "count"]})
      .sort("value_sum", ascending=False)
)

print(result.to_pandas())
```

## Core API

Top-level imports:

```python
import framex as fx
```

Main objects and helpers:

- `fx.DataFrame`, `fx.Series`, `fx.Index`, `fx.LazyFrame`
- `fx.NDArray`, `fx.array(...)`
- `fx.read_parquet`, `fx.write_parquet`, `fx.read_ipc`, `fx.write_ipc`, `fx.read_csv`, `fx.write_csv`
- `fx.read_json`, `fx.write_json`, `fx.read_ndjson`, `fx.write_ndjson`
- `fx.read_file`, `fx.write_file` for format auto-detection

Compression:
- transparent extension-based compression for `read_file` / `write_file`
- supported wrappers: `.gz`, `.bz2`, `.xz`, `.zip`, and `.zst`/`.zstd` (when `zstandard` is installed)
- `fx.from_pandas`, `fx.from_dask`, `fx.from_ray`, `fx.from_dataframe`
- `fx.get_config`, `fx.set_backend`, `fx.set_workers`, `fx.set_serializer`, `fx.set_kernel_backend`
- `fx.set_array_backend` for auto/NumExpr/Numba/JAX/PyTorch/CuPy acceleration modes
- `fx.StreamProcessor` for micro-batch streaming pipelines

Acceleration extras:

```bash
pip install framex[accel]      # numexpr + numba
pip install framex[gpu]        # cupy (CUDA)
pip install framex[ml_accel]   # jax + pytorch
```

Backend notes:

- `fx.set_backend("threads" | "processes" | "ray" | "dask" | "hpc")`
- Ray and Dask execution backends require their respective runtimes to be installed/available.
- HPC mode (`"hpc"`) uses cluster-oriented execution via Dask or Ray:
  - `FRAMEX_HPC_ENGINE=dask|ray`
  - `FRAMEX_DASK_SCHEDULER_ADDRESS=<tcp://...>` to connect existing Dask clusters
  - `FRAMEX_RAY_ADDRESS=<ray://...>` to connect existing Ray clusters
  - optional SLURM bootstrap: `FRAMEX_DASK_SLURM=1` (requires `dask-jobqueue`)

## Documentation

Canonical docs are in [`docs/documents`](docs/documents):

- [Overview](docs/documents/overview.md)
- [Getting Started](docs/documents/getting_started.md)
- [Installation](docs/documents/installation.md)
- [Tutorial: ETL Pipeline](docs/documents/tutorial_etl_pipeline.md)
- [Tutorial: NumPy NDArray Interop](docs/documents/tutorial_numpy_array.md)
- [Use Cases](docs/documents/use_cases.md)
- [Architecture](docs/documents/architecture.md)
- [API Reference](docs/documents/api_reference.md)
- [Roadmap](docs/documents/roadmap.md)
- [FAQ](docs/documents/faq.md)

## Website (Docs UI)

The docs website lives in [`website`](website) (Next.js App Router).

Run locally:

```bash
cd website
npm install
npm run dev
```

Production build:

```bash
npm run build
npm run start
```

## Development

Install dev dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

## Benchmarks

Benchmark code and generated reports are in [`benchmarks`](benchmarks).

## Project Status

FrameX is pre-1.0 (`0.1.0`) and in active development.

- APIs are usable and documented
- compatibility/performance behavior will continue to evolve
- pin versions for production-critical workloads

## License

[MIT](LICENSE)
