---
title: Configuration Guide
description: How to tune backend, workers, kernel, array acceleration, and cluster settings.
order: 9
section: Guides
---

# Configuration Guide

FrameX exposes global runtime configuration for balancing throughput, memory, and compatibility.

## Quick Start

```python
import framex as fx

# Inspect current config
print(fx.get_config())

# Auto-tune for this machine/cluster
fx.auto_configure_hardware()
```

## Core Runtime Controls

- `set_backend("threads"|"processes"|"ray"|"dask"|"hpc")`
- `set_workers(n)`
- `set_serializer("arrow"|"pickle5"|"pickle")`
- `set_kernel_backend("python"|"c")`
- `set_array_backend("auto"|"numpy"|"numexpr"|"numba"|"torch"|"jax"|"cupy")`

## Recommended Defaults for Performance

1. Call `auto_configure_hardware()` at process startup.
2. Keep serializer as `arrow` for safe, efficient transport.
3. Use `kernel_backend="c"` when available for numeric kernels.
4. Use `array_backend="auto"` unless you need explicit backend control.

## Temporary Overrides

```python
import framex as fx

with fx.config(backend="processes", workers=8):
    result = heavy_df.map_partitions(fn)
```

## HPC / Cluster Environment Variables

- `FRAMEX_HPC_ENGINE=dask|ray`
- `FRAMEX_DASK_SCHEDULER_ADDRESS=<tcp://host:8786>`
- `FRAMEX_RAY_ADDRESS=<ray://host:10001>`
- `FRAMEX_DASK_SLURM=1` (optional SLURM mode, requires `dask-jobqueue`)

SLURM tuning:

- `FRAMEX_DASK_SLURM_QUEUE`
- `FRAMEX_DASK_SLURM_ACCOUNT`
- `FRAMEX_DASK_SLURM_WALLTIME`
- `FRAMEX_DASK_SLURM_CORES`
- `FRAMEX_DASK_SLURM_MEMORY`
