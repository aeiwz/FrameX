---
title: Installation
description: Python requirements, optional dependencies, and local development setup.
order: 3
section: Introduction
---

# Installation

## Requirements

- Python 3.10+
- macOS, Linux, or Windows

## Install from PyPI

```bash
pip install pyframe-xpy
```

## Development Install

```bash
git clone https://github.com/aeiwz/FrameX.git
cd FrameX
pip install -e .
```

## Verify Installation

```python
import framex as fx

print(fx.__version__)
print(fx.get_config())
```

## Optional Performance Tooling

Install extras based on your workload:

```bash
pip install pyframe-xpy[bench]      # benchmark suite deps
pip install pyframe-xpy[accel]      # numexpr + numba
pip install pyframe-xpy[gpu]        # cupy (CUDA)
pip install pyframe-xpy[ml_accel]   # jax + pytorch
pip install pyframe-xpy[pandas_fast]  # modin backend
pip install pyframe-xpy[distributed]  # Dask + Ray distributed backends
pip install zstandard  # .zst/.zstd compression
```

## Optional Backend Test Coverage

If `pytest` shows skipped tests, this usually means optional runtimes are not installed.
Common optional dependencies:

- `dask.distributed` / `dask.dataframe`
- `ray` / `ray.data`

Install them to reduce skips:

```bash
pip install pyframe-xpy[distributed]
pytest -q
```

## Build and Test Locally

```bash
pytest
```

```bash
cd website
npm install
npm run dev
```
