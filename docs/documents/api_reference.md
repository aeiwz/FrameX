# FrameX API Reference

This document outlines the core public surface of the FrameX library.

## Core Data Structures

### `DataFrame`
A partitioned, Arrow-backed dataframe. Eager by default.
- **Creation**: `DataFrame(data, schema=None)` (accepts dictionaries, pandas DataFrames, pyarrow Tables, or lists of Partitions).
- **Properties**: `.schema`, `.columns`, `.dtypes`, `.shape`, `.num_rows`, `.num_partitions`.
- **Conversion**: `.to_arrow()`, `.to_pandas()`, `.to_pydict()`.
- **Projection / Filtering**: 
  - `df.select(columns: list[str])`
  - `df.filter(mask: Series)`
  - `df.drop(columns)`
- **Aggregation / Grouping**:
  - `df.groupby(keys)` -> `GroupBy`
  - `df.sort(by, ascending=True)`
- **Mutation (Returns New df)**:
  - `df.with_column(name, series)`
  - `df.assign(**kwargs)`
  - `df.rename(columns_dict)`
  - `df.dropna(subset, how="any")`
  - `df.fillna(value)`
  - `df.drop_duplicates(subset)`
- **Join**:
  - `df.join(other_df, on, how="inner")`
- **Execution Mode**:
  - `df.lazy()` -> `LazyFrame`

### `LazyFrame`
A deferred execution query builder. Methods match `DataFrame` but return a new `LazyFrame`.
- **Execution**: `lf.collect()` executes the recorded operations sequentially/optimally and returns an eager `DataFrame`.

### `Series`
A 1D chunked Arrow-backed array representing a DataFrame column.
- Provides standard series math, string matching, and element-wise transforms.

### `NDArray` (`framex.array`)
An N-dimensional array supporting optional chunking for parallel workloads.
- Fully compatible with `__array_ufunc__` and `__array_function__`.
- Designed to handle element-wise parallel execution seamlessly.

### `Index`
A DataFrame index supporting alignment logic (inspired by Pandas but Arrow-first).

---

## IO Operations (`framex.io`)

FrameX interacts optimally with Parquet and Arrow IPC schemas.

- `read_parquet(path)`: Loads partitioned Parquet data into a `DataFrame`.
- `write_parquet(df, path)`: Dumps a `DataFrame` to partitioned Parquet files.
- `read_ipc(path)`: Fast load of Arrow IPC structured binaries.
- `write_ipc(df, path)`: Fast write.
- `read_csv(path)`: Read traditional text data using the fast PyArrow CSV bindings.

---

## Configuration (`framex.config`)

Configure your runtime preferences:
- `set_workers(n)`: Change the target core utilization.
- `set_backend(name)`: Explicitly define thread or process engine ("threads", "processes", "auto").

---

## Window Functions (`framex.ops.window`)

Analytic routines optimized for rolling operations on partitions:
- `rolling_mean`, `rolling_sum`, `rolling_std`, `rolling_min`, `rolling_max`
- `top_k`, `rank`

---

## Interchange / Interop Protocols

- `from_pandas(pdf)`: Direct zero-copy (when possible) handoff from Pandas.
- `from_dataframe(df_protocol_object)`: Accepts any object implementing the standard Python Consortium `__dataframe__` protocol.
