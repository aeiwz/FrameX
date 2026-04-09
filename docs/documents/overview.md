---
title: Overview
description: What FrameX is, where it fits, and who it is for.
order: 1
section: Introduction
---

# FrameX Overview

FrameX is a Python dataframe and array library designed for **single-machine parallel analytics** with familiar Pandas and NumPy ergonomics.

It combines:

- Arrow-backed columnar storage (`pyarrow.Table`, `pyarrow.RecordBatch`)
- Pandas-like tabular APIs (`DataFrame`, `Series`, `GroupBy`)
- NumPy dispatch support for array workflows (`NDArray`, `__array_ufunc__`, `__array_function__`)
- Optional lazy execution (`.lazy().collect()`) for multi-step query planning

## Why FrameX

FrameX is built for workloads that are too large for comfortable single-threaded Pandas, but do not require a distributed cluster yet.

Typical range:

- 100MB to tens of GB on one machine
- ETL and analytics pipelines
- feature engineering and preprocessing
- mixed dataframe + numeric array workflows

## Current Position

FrameX is pre-1.0 and evolving quickly. The core interfaces exist today and are already useful for local experimentation and pipeline prototyping.

If you need strict 1:1 Pandas behavior everywhere, use `to_pandas()` at boundaries and validate behavior for critical paths.

## Core Concepts

- `DataFrame`: Arrow-backed tabular data split into partitions
- `Series`: Arrow-backed 1D column abstraction
- `NDArray`: Chunked 1D array with NumPy protocol interop
- `LazyFrame`: Deferred operation chain executed on `collect()`

## Interoperability

FrameX supports explicit interchange paths:

- `from_pandas(pdf)`
- `from_dataframe(obj)` for `__dataframe__` protocol objects
- `.to_pandas()`, `.to_arrow()`, `.to_pydict()`
- `framex.array(...)` for NDArray creation with chunking

Continue with [Getting Started](/docs/getting_started) for first-run setup and a complete walkthrough.
