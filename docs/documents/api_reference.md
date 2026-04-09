---
title: API Reference
description: Public FrameX API surface by module.
order: 8
section: Reference
---

# API Reference

## Top-Level Imports

```python
import framex as fx
```

Key exports:

- `DataFrame`, `Series`, `Index`, `LazyFrame`, `NDArray`
- IO: `read_parquet`, `write_parquet`, `read_ipc`, `write_ipc`, `read_csv`
- Interchange: `from_pandas`, `from_dask`, `from_ray`, `from_dataframe`
- Config: `get_config`, `print_config`, `set_backend`, `set_workers`, `set_serializer`, `set_kernel_backend`, `set_array_backend`
- Array constructor: `array(...)`
- Streaming: `StreamProcessor`, `StreamStats`

## DataFrame (`framex.core.dataframe.DataFrame`)

Creation:

- `DataFrame(data, schema=None)`

Shape and metadata:

- `.schema`, `.columns`, `.dtypes`, `.shape`
- `.num_rows`, `.num_columns`, `.num_partitions`

Conversion:

- `.to_arrow()`
- `.to_pandas()`
- `.to_dask(npartitions=None)`
- `.to_ray()`
- `.to_pydict()`

Selection and filtering:

- `df[col]`
- `df[[col1, col2]]`
- `.column(name)`
- `.select(columns)`
- `.filter(mask_series)`
- `.drop(columns)`

Transformations:

- `.with_column(name, series)`
- `.assign(**kwargs)`
- `.rename({old: new})`
- `.map_partitions(fn, workers=None, backend="auto")`
- `.parallel_apply(fn, workers=None, backend="auto")`
- `.fillna(value, subset=None)`
- `.dropna(subset=None, how="any"|"all")`
- `.drop_duplicates(subset=None)`

Analytics:

- `.groupby(keys).agg({...})`
- `.sort(by, ascending=True|[...])`
- `.nunique()`
- `.describe()`
- `.sample(n=None, frac=None, seed=None)`
- `.head(n=5)`, `.tail(n=5)`

Join:

- `.join(other, on, how="inner"|"left"|"right"|"outer")`

Lazy:

- `.lazy()` returns `LazyFrame`

## GroupBy

Methods:

- `.agg({"col": "sum" | ["sum", "mean", ...]})`
- `.sum()`, `.mean()`, `.count()`

Supported aggregation names in `.agg`:

- `sum`, `mean`, `min`, `max`, `count`, `std`, `count_distinct`

## LazyFrame

Methods:

- `.filter(mask_or_callable)`
- `.select(columns)`
- `.map_partitions(fn, workers=None, backend="auto")`
- `.groupby(keys).agg(...)`
- `.sort(by, ascending=...)`
- `.join(other, on, how=...)`
- `.with_column(name, value_or_callable)`
- `.drop(columns)`
- `.rename(mapping)`
- `.collect()`

## Series (`framex.core.series.Series`)

Conversions:

- `.to_pyarrow()`, `.to_numpy()`, `.to_pandas()`, `.to_pylist()`

Reductions:

- `.sum()`, `.mean()`, `.min()`, `.max()`, `.count()`, `.std()`, `.var()`, `.nunique()`

Utilities:

- `.unique()`, `.value_counts()`
- `.map(fn)`, `.apply(fn)`
- `.abs()`, `.round(decimals=0)`, `.clip(lower=None, upper=None)`
- `.dropna()`, `.is_null()`, `.fill_null(value)`, `.cast(dtype)`
- `.isin(values)`

Operators:

- arithmetic (`+`, `-`, `*`, `/`)
- comparisons (`==`, `!=`, `>`, `>=`, `<`, `<=`)
- logical (`&`, `|`, `~`)

## NDArray (`framex.core.array.NDArray`)

Creation:

- `fx.array(data, dtype=None, chunks=None)`
- `NDArray(...)`

Properties:

- `.dtype`, `.shape`, `.ndim`, `.num_chunks`

Methods:

- `.to_numpy()`, `.to_pyarrow()`
- `.sum()`, `.mean()`, `.min()`, `.max()`, `.std()`
- `.apply_blocks(fn, workers=None, backend="auto")`
- `.parallel_map(fn, workers=None, backend="auto")`
- `.jit_apply(fn, workers=None, backend="threads")`

NumPy protocol support:

- `__array__`
- `__array_ufunc__` (e.g., `np.sin`, `np.log`)
- `__array_function__` support for key functions (`np.sum`, `np.mean`, `np.concatenate`, `np.where`, etc.)

## IO

- `fx.read_parquet(path, columns=None, **kwargs)`
- `fx.write_parquet(df, path, **kwargs)`
- `fx.read_ipc(path)`
- `fx.write_ipc(df, path)`
- `fx.read_csv(path, **kwargs)`
- `fx.write_csv(df, path, **kwargs)`
- `fx.read_json(path, lines=None, **kwargs)`
- `fx.write_json(df, path, lines=False, orient="records", indent=None)`
- `fx.read_ndjson(path, **kwargs)`
- `fx.write_ndjson(df, path)`
- `fx.read_file(path, format=None, **kwargs)` (auto detect by extension)
- `fx.write_file(df, path, format=None, **kwargs)` (auto detect by extension)

Compression wrappers for `read_file` / `write_file`:
- `.gz`, `.bz2`, `.xz`, `.zip`
- `.zst` / `.zstd` when the optional `zstandard` package is available

## Interchange

- `fx.from_pandas(pdf)`
- `fx.from_dask(ddf)`
- `fx.from_ray(dataset)`
- `fx.from_dataframe(obj)`

## Config

- `fx.get_config()`
- `fx.print_config()`
- `fx.recommend_best_performance_config()`
- `fx.auto_configure_hardware(apply=True)`
- `fx.set_backend("threads"|"processes"|"ray"|"dask"|"hpc")`
- `fx.set_workers(n)`
- `fx.set_serializer("arrow"|"pickle5"|"pickle")`
- `fx.set_kernel_backend("python"|"c")`
- `fx.set_array_backend("auto"|"numpy"|"numexpr"|"numba"|"torch"|"jax"|"cupy")`
- `fx.config(...)` context manager for temporary overrides

## Streaming

- `fx.StreamProcessor(transform, sink=None)`
- `StreamProcessor.process_batch(batch)`
- `StreamProcessor.run(source) -> StreamStats`
