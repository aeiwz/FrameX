"""Arrow-backed DataFrame with partition support and lazy mode."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from framex.config import get_config
from framex.core.dtypes import DType
from framex.core.index import Index
from framex.core.series import Series
from framex.pandas_engine import get_pandas_module
from framex.runtime.executor import WorkerExecutor, detect_backend
from framex.runtime.partition import Partition, partition_table


def _is_pandas_dataframe(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "DataFrame" and cls.__module__.startswith(
        ("pandas.", "modin.pandas", "fireducks.pandas")
    )


def _is_dask_dataframe(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "DataFrame" and cls.__module__.startswith("dask.dataframe")


def _is_ray_dataset(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "Dataset" and cls.__module__.startswith("ray.data")


def _table_from_dask_dataframe(ddf: Any) -> pa.Table:
    delayed_parts = ddf.to_delayed()
    tables: list[pa.Table] = []
    for part in delayed_parts:
        pdf = part.compute()
        tables.append(pa.Table.from_pandas(pdf, preserve_index=False))
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables, promote_options="default")


def _table_from_ray_dataset(ds: Any) -> pa.Table:
    if hasattr(ds, "to_arrow"):
        return ds.to_arrow()
    if hasattr(ds, "to_arrow_refs"):
        import ray

        refs = ds.to_arrow_refs()
        tables = ray.get(refs)
        if not tables:
            return pa.table({})
        return pa.concat_tables(tables, promote_options="default")
    if hasattr(ds, "to_pandas"):
        return pa.Table.from_pandas(ds.to_pandas(), preserve_index=False)
    raise TypeError("Unsupported ray dataset conversion path; expected to_arrow or to_arrow_refs")


def _serialize_batch(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _deserialize_batch(payload: bytes) -> pa.RecordBatch:
    with pa.ipc.open_stream(payload) as reader:
        return reader.read_next_batch()


def _normalize_partition_output(
    output: Any,
) -> pa.RecordBatch:
    if isinstance(output, Partition):
        return output.record_batch
    if isinstance(output, pa.RecordBatch):
        return output
    if isinstance(output, pa.Table):
        batches = output.to_batches()
        if batches:
            if len(batches) == 1:
                return batches[0]
            merged = pa.Table.from_batches(batches, schema=output.schema).combine_chunks()
            merged_batches = merged.to_batches()
            if merged_batches:
                return merged_batches[0]
        empty_cols = [pa.array([], type=field.type) for field in output.schema]
        return pa.record_batch(empty_cols, schema=output.schema)
    if _is_pandas_dataframe(output):
        return pa.Table.from_pandas(output, preserve_index=False).to_batches()[0]
    if isinstance(output, dict):
        table = pa.table(output)
        batches = table.to_batches()
        if batches:
            return batches[0]
        empty_cols = [pa.array([], type=field.type) for field in table.schema]
        return pa.record_batch(empty_cols, schema=table.schema)
    raise TypeError(
        "map_partitions function must return Partition, RecordBatch, Table, pandas.DataFrame, or dict"
    )


def _apply_partition_local(
    batch: pa.RecordBatch,
    fn: Callable[[pa.RecordBatch], Any],
) -> pa.RecordBatch:
    return _normalize_partition_output(fn(batch))


def _apply_partition_serialized(
    payload: bytes,
    fn: Callable[[pa.RecordBatch], Any],
) -> bytes:
    batch = _deserialize_batch(payload)
    mapped = _normalize_partition_output(fn(batch))
    return _serialize_batch(mapped)


def _to_pandas_compatible(value: Any) -> Any:
    if isinstance(value, DataFrame):
        return value.to_pandas()
    if isinstance(value, Series):
        return value.to_pandas()
    if isinstance(value, dict):
        return {k: _to_pandas_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        mapped = [_to_pandas_compatible(v) for v in value]
        return tuple(mapped) if isinstance(value, tuple) else mapped
    return value


def _wrap_pandas_result(value: Any) -> Any:
    pd = get_pandas_module()
    if _is_pandas_dataframe(value):
        return DataFrame(value)
    if isinstance(value, pd.Series):
        return Series(value.to_numpy(), name=value.name)
    return value


class DataFrame:
    """Partitioned, Arrow-backed DataFrame.

    Internally stores data as a list of ``Partition`` objects (each wrapping a
    ``pyarrow.RecordBatch``).  Partitioning is transparent to the user unless
    they opt into lazy execution via ``.lazy()``.
    """

    def __init__(
        self,
        data: (
            dict[str, Any]
            | pa.Table
            | list[Partition]
            | None
        ) = None,
        *,
        schema: pa.Schema | None = None,
    ):
        self._cached_table: pa.Table | None = None
        self._row_count: int = 0

        if data is None:
            schema = schema or pa.schema([])
            self._partitions: list[Partition] = []
            self._schema: pa.Schema = schema
            self._cached_table = pa.table(
                {name: pa.array([], type=self._schema.field(name).type) for name in self._schema.names}
            )
            self._row_count = 0
        elif isinstance(data, list) and all(isinstance(p, Partition) for p in data):
            self._partitions = data
            self._schema = data[0].schema if data else (schema or pa.schema([]))
            self._row_count = sum(p.num_rows for p in data)
        elif isinstance(data, pa.Table):
            self._schema = data.schema
            self._partitions = partition_table(data)
            self._cached_table = data
            self._row_count = data.num_rows
        elif _is_pandas_dataframe(data):
            table = pa.Table.from_pandas(data, preserve_index=False)
            self._schema = table.schema
            self._partitions = partition_table(table)
            self._cached_table = table
            self._row_count = table.num_rows
        elif _is_dask_dataframe(data):
            table = _table_from_dask_dataframe(data)
            self._schema = table.schema
            self._partitions = partition_table(table)
            self._cached_table = table
            self._row_count = table.num_rows
        elif _is_ray_dataset(data):
            table = _table_from_ray_dataset(data)
            self._schema = table.schema
            self._partitions = partition_table(table)
            self._cached_table = table
            self._row_count = table.num_rows
        elif isinstance(data, dict):
            arrays: dict[str, pa.Array] = {}
            for k, v in data.items():
                if isinstance(v, Series):
                    arrays[k] = v.to_pyarrow().combine_chunks()
                elif isinstance(v, (pa.Array, pa.ChunkedArray)):
                    arrays[k] = v if isinstance(v, pa.Array) else v.combine_chunks()
                elif isinstance(v, np.ndarray):
                    arrays[k] = pa.array(v)
                else:
                    arrays[k] = pa.array(v)
            table = pa.table(arrays)
            self._schema = table.schema
            self._partitions = partition_table(table)
            self._cached_table = table
            self._row_count = table.num_rows
        else:
            raise TypeError(f"Cannot construct DataFrame from {type(data)}")

    # -- Shape / schema ------------------------------------------------------

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def columns(self) -> list[str]:
        return self._schema.names

    @property
    def dtypes(self) -> dict[str, DType]:
        return {f.name: DType.from_arrow(f.type) for f in self._schema}

    @property
    def num_rows(self) -> int:
        return self._row_count

    @property
    def num_columns(self) -> int:
        return len(self._schema)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_columns)

    @property
    def num_partitions(self) -> int:
        return len(self._partitions)

    def __len__(self) -> int:
        return self.num_rows

    # -- Materialisation -----------------------------------------------------

    def to_arrow(self) -> pa.Table:
        """Materialise all partitions into a single ``pyarrow.Table``."""
        if self._cached_table is not None:
            return self._cached_table
        if not self._partitions:
            self._cached_table = pa.table(
                {name: pa.array([], type=self._schema.field(name).type) for name in self._schema.names}
            )
            return self._cached_table
        batches = [p.record_batch for p in self._partitions]
        self._cached_table = pa.Table.from_batches(batches, schema=self._schema)
        return self._cached_table

    def to_pandas(self) -> Any:
        return self.to_arrow().to_pandas()

    def to_dask(self, npartitions: int | None = None) -> Any:
        """Convert to a Dask DataFrame."""
        import dask.dataframe as dd

        pdf = self.to_arrow().to_pandas()
        return dd.from_pandas(pdf, npartitions=npartitions or max(1, self.num_partitions))

    def to_ray(self) -> Any:
        """Convert to a Ray Dataset."""
        import ray.data as rd

        return rd.from_arrow(self.to_arrow())

    def to_pydict(self) -> dict[str, list[Any]]:
        return self.to_arrow().to_pydict()

    # -- Column access -------------------------------------------------------

    def __getitem__(self, key: str | list[str]) -> Series | DataFrame:
        if isinstance(key, str):
            return self.column(key)
        if isinstance(key, list):
            return self.select(key)
        raise TypeError(f"Invalid key type: {type(key)}")

    def column(self, name: str) -> Series:
        col_index = self._schema.get_field_index(name)
        chunks: list[pa.Array] = []
        for p in self._partitions:
            chunks.append(p.record_batch.column(col_index))
        return Series(pa.chunked_array(chunks), name=name)

    # -- Projection ----------------------------------------------------------

    def select(self, columns: list[str]) -> DataFrame:
        new_partitions: list[Partition] = []
        indices = [self._schema.get_field_index(c) for c in columns]
        for p in self._partitions:
            rb = p.record_batch
            selected = pa.record_batch(
                [rb.column(i) for i in indices],
                names=columns,
            )
            new_partitions.append(Partition(record_batch=selected, partition_id=p.partition_id))
        return DataFrame(new_partitions)

    # -- Filter --------------------------------------------------------------

    def filter(self, mask: Series) -> DataFrame:
        """Filter rows by a boolean ``Series``."""
        # Arrow can filter a whole table with a ChunkedArray mask in C++.
        # This avoids Python-level partition loops and repeated chunk combines.
        table = self.to_arrow()
        filtered = table.filter(mask.to_pyarrow())
        return DataFrame(filtered)

    def map_partitions(
        self,
        fn: Callable[[pa.RecordBatch], Any],
        *,
        workers: int | None = None,
        backend: str = "auto",
    ) -> DataFrame:
        """Apply ``fn`` to each partition, optionally in parallel.

        Parameters
        ----------
        fn : Callable[[pyarrow.RecordBatch], Any]
            Function invoked once per partition. Return types supported:
            ``RecordBatch``, ``Table``, ``Partition``, ``pandas.DataFrame``,
            or ``dict[str, sequence]``.
        workers : int | None
            Pool size. Defaults to global config workers.
        backend : "threads" | "processes" | "ray" | "dask" | "hpc" | "auto"
            Execution backend; ``"auto"`` follows schema heuristic.
        """
        if not self._partitions:
            return DataFrame([], schema=self._schema)

        cfg = get_config()
        max_workers = workers or cfg.workers
        if max_workers < 1:
            raise ValueError("workers must be >= 1")

        resolved_backend = detect_backend(self._schema) if backend == "auto" else backend
        if resolved_backend not in ("threads", "processes", "ray", "dask", "hpc"):
            raise ValueError(
                f"backend must be 'threads', 'processes', 'ray', 'dask', 'hpc', or 'auto', got {backend!r}"
            )

        mapped_by_id: dict[int, pa.RecordBatch] = {}

        if max_workers == 1 or len(self._partitions) == 1:
            for p in self._partitions:
                mapped_by_id[p.partition_id] = _apply_partition_local(p.record_batch, fn)
        elif resolved_backend == "threads":
            with WorkerExecutor(max_workers=max_workers, backend="threads") as executor:
                futures = {
                    p.partition_id: executor.submit(_apply_partition_local, p.record_batch, fn)
                    for p in self._partitions
                }
                for pid, fut in futures.items():
                    mapped_by_id[pid] = fut.result()
        elif resolved_backend in ("processes", "ray", "dask", "hpc"):
            payloads = {p.partition_id: _serialize_batch(p.record_batch) for p in self._partitions}
            with WorkerExecutor(max_workers=max_workers, backend=resolved_backend) as executor:
                futures = {
                    pid: executor.submit(_apply_partition_serialized, payload, fn)
                    for pid, payload in payloads.items()
                }
                for pid, fut in futures.items():
                    mapped_by_id[pid] = _deserialize_batch(fut.result())
        else:
            raise ValueError(f"Unsupported backend: {resolved_backend!r}")

        ordered_ids = sorted(mapped_by_id)
        if not ordered_ids:
            return DataFrame([], schema=self._schema)

        first_schema = mapped_by_id[ordered_ids[0]].schema
        for pid in ordered_ids[1:]:
            if mapped_by_id[pid].schema != first_schema:
                raise ValueError("map_partitions requires all output partitions to share the same schema")

        partitions = [
            Partition(record_batch=mapped_by_id[pid], partition_id=pid)
            for pid in ordered_ids
        ]
        return DataFrame(partitions)

    def parallel_apply(
        self,
        fn: Callable[[pa.RecordBatch], Any],
        *,
        workers: int | None = None,
        backend: str = "auto",
    ) -> DataFrame:
        """Alias for :meth:`map_partitions`."""
        return self.map_partitions(fn, workers=workers, backend=backend)

    # -- GroupBy -------------------------------------------------------------

    def groupby(self, keys: str | list[str]) -> GroupBy:
        if isinstance(keys, str):
            keys = [keys]
        return GroupBy(self, keys)

    # -- Sort ----------------------------------------------------------------

    def sort(self, by: str | list[str], ascending: bool | list[bool] = True) -> DataFrame:
        table = self.to_arrow()
        if isinstance(by, str):
            by = [by]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        sort_keys = [(col, "ascending" if asc else "descending") for col, asc in zip(by, ascending)]
        indices = pc.sort_indices(table, sort_keys=sort_keys)
        sorted_table = table.take(indices)
        return DataFrame(sorted_table)

    # -- Head / tail ---------------------------------------------------------

    def head(self, n: int = 5) -> DataFrame:
        table = self.to_arrow().slice(0, n)
        return DataFrame(table)

    def tail(self, n: int = 5) -> DataFrame:
        total = self.num_rows
        start = max(0, total - n)
        table = self.to_arrow().slice(start, n)
        return DataFrame(table)

    # -- Mutation-like ops (return new DataFrame) ----------------------------

    def with_column(self, name: str, series: Series) -> DataFrame:
        """Return a new DataFrame with an added or replaced column."""
        table = self.to_arrow()
        arrow_col = series.to_pyarrow().combine_chunks()
        if name in self._schema.names:
            idx = self._schema.get_field_index(name)
            table = table.set_column(idx, name, arrow_col)
        else:
            table = table.append_column(name, arrow_col)
        return DataFrame(table)

    def assign(self, **kwargs: Any) -> DataFrame:
        """Pandas-like assign: ``df.assign(new_col=lambda d: d["x"] + 1)``.

        Each value can be a callable ``DataFrame -> Series/list`` or a plain
        ``Series``/list.
        """
        df = self
        for name, val in kwargs.items():
            if callable(val):
                result = val(df)
            else:
                result = val
            if not isinstance(result, Series):
                result = Series(result, name=name)
            df = df.with_column(name, result)
        return df

    def rename(self, columns: dict[str, str]) -> DataFrame:
        """Rename columns by mapping ``{old: new}``."""
        new_names = [columns.get(n, n) for n in self.columns]
        table = self.to_arrow().rename_columns(new_names)
        return DataFrame(table)

    def drop(self, columns: str | list[str]) -> DataFrame:
        """Drop one or more columns by name."""
        if isinstance(columns, str):
            columns = [columns]
        keep = [c for c in self.columns if c not in columns]
        return self.select(keep)

    def dropna(
        self,
        subset: list[str] | None = None,
        how: str = "any",
    ) -> DataFrame:
        """Remove rows containing null values.

        Parameters
        ----------
        subset : list[str] | None
            Columns to inspect.  Defaults to all columns.
        how : "any" | "all"
            "any" drops a row if any inspected column is null;
            "all" drops a row only if all inspected columns are null.
        """
        table = self.to_arrow()
        cols = subset if subset is not None else self.columns
        masks: list[pa.ChunkedArray] = []
        for col in cols:
            masks.append(pc.is_null(table.column(col)))
        if how == "any":
            null_mask = masks[0]
            for m in masks[1:]:
                null_mask = pc.or_(null_mask, m)
        else:
            null_mask = masks[0]
            for m in masks[1:]:
                null_mask = pc.and_(null_mask, m)
        keep_mask = pc.invert(null_mask)
        return DataFrame(table.filter(keep_mask))

    def fillna(self, value: Any, subset: list[str] | None = None) -> DataFrame:
        """Fill null values with ``value``."""
        table = self.to_arrow()
        cols = subset if subset is not None else self.columns
        for col in cols:
            idx = table.schema.get_field_index(col)
            filled = pc.fill_null(table.column(col), value)
            table = table.set_column(idx, col, filled)
        return DataFrame(table)

    def drop_duplicates(self, subset: list[str] | None = None) -> DataFrame:
        """Return DataFrame with duplicate rows removed.

        Parameters
        ----------
        subset : list[str] | None
            Columns to consider for identifying duplicates.  Defaults to all.
        """
        table = self.to_arrow()
        cols = subset if subset is not None else self.columns
        # Build a string representation of each row for the key columns, then
        # deduplicate via a set-based approach using pyarrow groupby.
        if len(cols) == 1:
            key_col = table.column(cols[0])
        else:
            parts = [pc.cast(table.column(c), pa.string()) for c in cols]
            key_col = parts[0]
            for p in parts[1:]:
                key_col = pc.binary_join_element_wise(key_col, p, "\x00")

        # Use a Python set to track seen keys (order-preserving via enumerate).
        seen: set[Any] = set()
        keep: list[bool] = []
        for v in key_col.to_pylist():
            if v not in seen:
                seen.add(v)
                keep.append(True)
            else:
                keep.append(False)
        mask = pa.array(keep, type=pa.bool_())
        return DataFrame(table.filter(mask))

    def nunique(self) -> dict[str, int]:
        """Return the number of distinct non-null values per column."""
        table = self.to_arrow()
        return {
            name: pc.count(pc.unique(table.column(name)), count_mode="only_valid").as_py()
            for name in self.columns
        }

    def describe(self) -> DataFrame:
        """Summary statistics for numeric columns (Pandas-like)."""
        table = self.to_arrow()
        numeric_cols = [
            f.name for f in table.schema
            if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
        ]
        stats = ["count", "mean", "std", "min", "max"]
        rows: dict[str, list[Any]] = {"statistic": stats}
        for col in numeric_cols:
            c = table.column(col)
            rows[col] = [
                pc.count(c).as_py(),
                pc.mean(c).as_py(),
                pc.stddev(c).as_py(),
                pc.min(c).as_py(),
                pc.max(c).as_py(),
            ]
        return DataFrame(rows)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        seed: int | None = None,
    ) -> DataFrame:
        """Random sample of rows.

        Parameters
        ----------
        n : int | None
            Number of rows to sample.
        frac : float | None
            Fraction of rows to sample (alternative to ``n``).
        seed : int | None
            Random seed for reproducibility.
        """
        if n is None and frac is None:
            raise ValueError("Must specify n or frac")
        total = self.num_rows
        if frac is not None:
            n = max(1, int(total * frac))
        n = min(n, total)  # type: ignore[arg-type]
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=n, replace=False)
        indices.sort()
        table = self.to_arrow().take(pa.array(indices.tolist(), type=pa.int64()))
        return DataFrame(table)

    # -- Join ----------------------------------------------------------------

    def join(
        self,
        other: DataFrame,
        on: str | list[str],
        how: str = "inner",
    ) -> DataFrame:
        left = self.to_arrow()
        right = other.to_arrow()
        if isinstance(on, str):
            on = [on]

        join_type_map = {
            "inner": "inner",
            "left": "left outer",
            "right": "right outer",
            "outer": "full outer",
        }
        pa_join_type = join_type_map.get(how, how)

        # Fast path: common single-key join with no non-key name conflicts.
        # Avoids rename-map setup and synthetic-key machinery.
        if len(on) == 1:
            join_key = on[0]
            left_cols = set(left.schema.names)
            right_cols = set(right.schema.names)
            overlapping_non_key = (left_cols & right_cols) - {join_key}
            if not overlapping_non_key:
                result = left.join(
                    right,
                    keys=join_key,
                    right_keys=join_key,
                    join_type=pa_join_type,
                )
                return DataFrame(result)

        # Rename conflicting columns in right table (except join keys).
        right_names = set(right.schema.names) - set(on)
        left_names = set(left.schema.names) - set(on)
        overlapping = right_names & left_names
        rename_map: dict[str, str] = {}
        for name in overlapping:
            rename_map[name] = f"{name}_right"
        if rename_map:
            new_names = [rename_map.get(n, n) for n in right.schema.names]
            right = right.rename_columns(new_names)

        # Use first key for join (Arrow only supports single-key natively).
        # For multi-key, concatenate keys into a synthetic column.
        if len(on) == 1:
            join_key = on[0]
        else:
            # Create a synthetic join key by concatenating string representations.
            def _make_synth(table: pa.Table, keys: list[str], col_name: str) -> pa.Table:
                parts = [pc.cast(table.column(k), pa.string()) for k in keys]
                combined = parts[0]
                for p in parts[1:]:
                    combined = pc.binary_join_element_wise(combined, p, "|")
                return table.append_column(col_name, combined)

            join_key = "__synth_join_key__"
            left = _make_synth(left, on, join_key)
            right = _make_synth(right, on, join_key)

        # Determine right keys name (may have been renamed).
        right_join_key = rename_map.get(join_key, join_key)

        result = left.join(right, keys=join_key, right_keys=right_join_key, join_type=pa_join_type)

        # Drop synthetic columns if created.
        if join_key == "__synth_join_key__":
            cols_to_drop = [c for c in result.schema.names if c.startswith("__synth_join_key__")]
            for c in cols_to_drop:
                idx = result.schema.get_field_index(c)
                result = result.remove_column(idx)

        return DataFrame(result)

    # -- Lazy mode -----------------------------------------------------------

    def lazy(self) -> LazyFrame:
        return LazyFrame(self)

    # -- Display -------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"DataFrame(rows={self.num_rows}, cols={self.num_columns}, partitions={self.num_partitions})"]
        lines.append(f"Columns: {self.columns}")
        if self.num_rows > 0:
            preview = self.head(5).to_pydict()
            lines.append(str(preview))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Notebook/Jupyter rich HTML representation."""
        try:
            pdf = self.to_pandas()
            html = pdf._repr_html_() if hasattr(pdf, "_repr_html_") else pdf.to_html()
            if isinstance(html, str):
                return html
        except Exception:
            pass
        return f"<pre>{repr(self)}</pre>"

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        """Provide both plain text and HTML MIME output for notebook frontends."""
        return {
            "text/plain": repr(self),
            "text/html": self._repr_html_(),
        }

    def __getattr__(self, name: str) -> Any:
        """Pandas-compat fallback for unimplemented DataFrame APIs."""
        if name.startswith("_"):
            raise AttributeError(name)

        pdf = self.to_pandas()
        attr = getattr(pdf, name, None)
        if attr is None:
            raise AttributeError(f"'DataFrame' object has no attribute {name!r}")

        if callable(attr):
            def _call(*args: Any, **kwargs: Any) -> Any:
                out = attr(
                    *[_to_pandas_compatible(a) for a in args],
                    **{k: _to_pandas_compatible(v) for k, v in kwargs.items()},
                )
                return _wrap_pandas_result(out)

            return _call

        return _wrap_pandas_result(attr)


# ---------------------------------------------------------------------------
# GroupBy
# ---------------------------------------------------------------------------


class GroupBy:
    """GroupBy aggregation builder."""

    def __init__(self, df: DataFrame, keys: list[str]):
        self._df = df
        self._keys = keys

    def agg(self, aggregations: dict[str, str | list[str]]) -> DataFrame:
        """Aggregate with ``{column: func_or_funcs}``."""
        table = self._df.to_arrow()
        # Build pyarrow aggregation spec.
        agg_specs: list[tuple[str, str, Any]] = []
        output_names: list[str] = []

        func_map = {
            "sum": "sum",
            "mean": "mean",
            "min": "min",
            "max": "max",
            "count": "count",
            "std": "stddev",
            "count_distinct": "count_distinct",
        }

        for col, funcs in aggregations.items():
            if isinstance(funcs, str):
                funcs = [funcs]
            for func in funcs:
                pa_func = func_map.get(func)
                if pa_func is None:
                    raise ValueError(f"Unsupported aggregation function: {func!r}")
                output_name = f"{col}_{func}" if len(funcs) > 1 or len(aggregations) > 1 else col
                # If output_name duplicates a key, suffix it.
                if output_name in self._keys:
                    output_name = f"{output_name}_{func}"
                agg_specs.append((col, pa_func))
                output_names.append(output_name)

        result = table.group_by(self._keys).aggregate(agg_specs)
        # Rename the aggregated columns from Arrow's default naming.
        current_names = result.schema.names
        new_names = list(self._keys)
        for i, name in enumerate(current_names):
            if name not in self._keys:
                if i - len(self._keys) < len(output_names):
                    new_names.append(output_names[i - len(self._keys)])
                else:
                    new_names.append(name)
        result = result.rename_columns(new_names)
        return DataFrame(result)

    def sum(self) -> DataFrame:
        non_key_cols = [c for c in self._df.columns if c not in self._keys]
        return self.agg({c: "sum" for c in non_key_cols})

    def mean(self) -> DataFrame:
        non_key_cols = [c for c in self._df.columns if c not in self._keys]
        return self.agg({c: "mean" for c in non_key_cols})

    def count(self) -> DataFrame:
        non_key_cols = [c for c in self._df.columns if c not in self._keys]
        return self.agg({c: "count" for c in non_key_cols})


# ---------------------------------------------------------------------------
# LazyFrame
# ---------------------------------------------------------------------------


class LazyFrame:
    """Lazy query builder that records operations and executes on ``.collect()``."""

    def __init__(self, source: DataFrame):
        self._source = source
        self._ops: list[tuple[str, Any]] = []

    def filter(self, mask_fn: Any) -> LazyFrame:
        """Record a filter operation.

        ``mask_fn`` can be a callable ``DataFrame -> Series`` or a ``Series``.
        """
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("filter", mask_fn)]
        return clone

    def select(self, columns: list[str]) -> LazyFrame:
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("select", columns)]
        return clone

    def groupby(self, keys: str | list[str]) -> LazyGroupBy:
        return LazyGroupBy(self, keys if isinstance(keys, list) else [keys])

    def sort(self, by: str | list[str], ascending: bool | list[bool] = True) -> LazyFrame:
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("sort", (by, ascending))]
        return clone

    def map_partitions(
        self,
        fn: Callable[[pa.RecordBatch], Any],
        *,
        workers: int | None = None,
        backend: str = "auto",
    ) -> LazyFrame:
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("map_partitions", (fn, workers, backend))]
        return clone

    def join(
        self,
        other: DataFrame | LazyFrame,
        on: str | list[str],
        how: str = "inner",
    ) -> LazyFrame:
        other_df = other._source if isinstance(other, LazyFrame) else other
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("join", (other_df, on, how))]
        return clone

    def with_column(self, name: str, value: Any) -> LazyFrame:
        """Record a column add/replace for deferred execution.

        ``value`` may be a callable ``DataFrame -> Series`` or a ``Series``.
        """
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("with_column", (name, value))]
        return clone

    def drop(self, columns: str | list[str]) -> LazyFrame:
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("drop", columns)]
        return clone

    def rename(self, columns: dict[str, str]) -> LazyFrame:
        clone = LazyFrame(self._source)
        clone._ops = self._ops + [("rename", columns)]
        return clone

    def collect(self) -> DataFrame:
        """Execute the recorded plan and return an eager ``DataFrame``."""
        df = self._source
        for op, arg in self._ops:
            if op == "filter":
                if callable(arg):
                    mask = arg(df)
                else:
                    mask = arg
                df = df.filter(mask)
            elif op == "select":
                df = df.select(arg)
            elif op == "sort":
                by, ascending = arg
                df = df.sort(by, ascending)
            elif op == "map_partitions":
                fn, workers, backend = arg
                df = df.map_partitions(fn, workers=workers, backend=backend)
            elif op == "groupby_agg":
                keys, aggs = arg
                df = df.groupby(keys).agg(aggs)
            elif op == "join":
                other_df, on, how = arg
                df = df.join(other_df, on=on, how=how)
            elif op == "with_column":
                name, val = arg
                series = val(df) if callable(val) else val
                if not isinstance(series, Series):
                    series = Series(series, name=name)
                df = df.with_column(name, series)
            elif op == "drop":
                df = df.drop(arg)
            elif op == "rename":
                df = df.rename(arg)
            else:
                raise ValueError(f"Unknown lazy op: {op!r}")
        return df


class LazyGroupBy:
    """Lazy groupby that records aggregation for deferred execution."""

    def __init__(self, lazy: LazyFrame, keys: list[str]):
        self._lazy = lazy
        self._keys = keys

    def agg(self, aggregations: dict[str, str | list[str]]) -> LazyFrame:
        clone = LazyFrame(self._lazy._source)
        clone._ops = self._lazy._ops + [("groupby_agg", (self._keys, aggregations))]
        return clone
