"""FrameX ASV benchmark suite.

Run with::

    asv run                  # benchmark current commit
    asv continuous HEAD~1    # compare two commits
    asv publish && asv preview  # view results in browser

Each benchmark class has a ``setup`` method that builds reusable fixtures so
benchmark timing does NOT include data construction.  ``time_*`` methods are
timed by ASV; ``mem_*`` methods report peak memory (RSS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

import framex as fx
from framex.core.dataframe import DataFrame
from framex.core.series import Series
from framex.memory.transport import send_zero_copy, recv_zero_copy, send_mmap, recv_mmap
from framex.ops.window import rolling_mean, top_k, rank


# ── Benchmark parameters ──────────────────────────────────────────────────

SIZES = [10_000, 100_000, 1_000_000]
CHUNK_COUNTS = [1, 4, 16]


# ── Filter ────────────────────────────────────────────────────────────────

class FilterBench:
    """filter: boolean predicate on a numeric column."""

    params = SIZES
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        self.df = DataFrame({
            "a": list(range(n_rows)),
            "b": [float(i) * 0.5 for i in range(n_rows)],
            "c": ["str_" + str(i % 100) for i in range(n_rows)],
        })

    def time_filter_gt(self, n_rows: int) -> None:
        """Filter: keep rows where a > n_rows // 2."""
        threshold = n_rows // 2
        _ = self.df.filter(self.df["a"] > threshold)

    def time_filter_isin(self, n_rows: int) -> None:
        """Filter: isin with a 10-element set."""
        vals = list(range(0, 100, 10))
        _ = self.df.filter(self.df["a"].isin(vals))

    def mem_filter_gt(self, n_rows: int) -> None:
        threshold = n_rows // 2
        _ = self.df.filter(self.df["a"] > threshold)


# ── GroupBy + Aggregation ─────────────────────────────────────────────────

class GroupByBench:
    """groupby + agg: sum and mean over a categorical key."""

    params = SIZES
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        self.df = DataFrame({
            "key": [str(i % 100) for i in range(n_rows)],
            "val": [float(i) for i in range(n_rows)],
        })

    def time_groupby_sum(self, n_rows: int) -> None:
        _ = self.df.groupby("key").agg({"val": "sum"})

    def time_groupby_mean(self, n_rows: int) -> None:
        _ = self.df.groupby("key").agg({"val": "mean"})

    def time_groupby_multi_agg(self, n_rows: int) -> None:
        _ = self.df.groupby("key").agg({"val": ["sum", "mean", "count"]})


# ── Sort ──────────────────────────────────────────────────────────────────

class SortBench:
    params = SIZES
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        rng = np.random.default_rng(42)
        self.df = DataFrame({
            "a": rng.integers(0, n_rows, size=n_rows).tolist(),
            "b": rng.standard_normal(n_rows).tolist(),
        })

    def time_sort_single_col(self, n_rows: int) -> None:
        _ = self.df.sort("a")

    def time_sort_two_cols(self, n_rows: int) -> None:
        _ = self.df.sort(["a", "b"])

    def time_top_k(self, n_rows: int) -> None:
        _ = top_k(self.df, k=100, by="a")


# ── Join ──────────────────────────────────────────────────────────────────

class JoinBench:
    params = [10_000, 100_000]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        keys = list(range(n_rows))
        self.left = DataFrame({"id": keys, "val": [float(k) for k in keys]})
        self.right = DataFrame({"id": keys, "score": [k * 2 for k in keys]})

    def time_inner_join(self, n_rows: int) -> None:
        _ = self.left.join(self.right, on="id", how="inner")


# ── Reduction ─────────────────────────────────────────────────────────────

class ReductionBench:
    """Column reductions: sum, mean, min, max — Python vs C backend."""

    params = ([100_000, 1_000_000], ["python", "c"])
    param_names = ["n_rows", "kernel_backend"]

    def setup(self, n_rows: int, kernel_backend: str) -> None:
        from framex.backends.c_backend import C_AVAILABLE
        if kernel_backend == "c" and not C_AVAILABLE:
            raise NotImplementedError("C backend not available")
        self.col = Series(list(range(n_rows)), dtype="float64")
        fx.set_kernel_backend(kernel_backend)  # type: ignore[arg-type]

    def teardown(self, n_rows: int, kernel_backend: str) -> None:
        fx.set_kernel_backend("python")

    def time_sum(self, n_rows: int, kernel_backend: str) -> None:
        _ = self.col.sum()

    def time_mean(self, n_rows: int, kernel_backend: str) -> None:
        _ = self.col.mean()

    def time_min_max(self, n_rows: int, kernel_backend: str) -> None:
        _ = self.col.min()
        _ = self.col.max()


# ── Memory transport ──────────────────────────────────────────────────────

class TransportBench:
    """Zero-copy transport: SharedMemory vs mmap round-trip latency."""

    params = [10_000, 100_000, 1_000_000]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        self.batch = pa.record_batch({
            "a": pa.array(np.arange(n_rows, dtype=np.float64)),
            "b": pa.array(np.arange(n_rows, dtype=np.int64)),
        })

    def time_shm_round_trip(self, n_rows: int) -> None:
        """SharedMemory: send then receive."""
        name, schema = send_zero_copy(self.batch)
        _ = recv_zero_copy(name, schema)
        from framex.memory.transport import unlink_shm
        unlink_shm(name)

    def time_mmap_round_trip(self, n_rows: int) -> None:
        """mmap: write IPC file then read back zero-copy."""
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".arrow")
        os.close(fd)
        try:
            send_mmap(self.batch, path=path)
            _ = recv_mmap(path)
        finally:
            os.unlink(path)

    def mem_shm_round_trip(self, n_rows: int) -> None:
        name, schema = send_zero_copy(self.batch)
        _ = recv_zero_copy(name, schema)
        from framex.memory.transport import unlink_shm
        unlink_shm(name)


# ── Lazy execution ────────────────────────────────────────────────────────

class LazyBench:
    """Lazy vs eager pipeline: filter → select → groupby."""

    params = [10_000, 100_000, 1_000_000]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        self.df = DataFrame({
            "key": [str(i % 10) for i in range(n_rows)],
            "a": list(range(n_rows)),
            "b": [float(i) for i in range(n_rows)],
            "c": [i * 2 for i in range(n_rows)],
        })

    def time_eager_pipeline(self, n_rows: int) -> None:
        threshold = n_rows // 2
        df2 = self.df.filter(self.df["a"] > threshold)
        df3 = df2.select(["key", "b"])
        _ = df3.groupby("key").agg({"b": "sum"})

    def time_lazy_pipeline(self, n_rows: int) -> None:
        threshold = n_rows // 2
        _ = (
            self.df.lazy()
            .filter(lambda d: d["a"] > threshold)
            .select(["key", "b"])
            .groupby("key")
            .agg({"b": "sum"})
            .collect()
        )


# ── Window ops ────────────────────────────────────────────────────────────

class WindowBench:
    params = [10_000, 100_000]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        self.series = Series(list(range(n_rows)), dtype="float64", name="val")

    def time_rolling_mean_10(self, n_rows: int) -> None:
        _ = rolling_mean(self.series, window=10)

    def time_rolling_mean_100(self, n_rows: int) -> None:
        _ = rolling_mean(self.series, window=100)

    def time_rank(self, n_rows: int) -> None:
        _ = rank(self.series, method="average")


# ── NDArray ufunc dispatch ────────────────────────────────────────────────

class NDArrayBench:
    params = [100_000, 1_000_000]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        self.arr = fx.array(list(range(n_rows)), dtype="float64", chunks=100_000)

    def time_np_sin_dispatch(self, n_rows: int) -> None:
        """__array_ufunc__: np.sin dispatched to NDArray."""
        _ = np.sin(self.arr)

    def time_np_sum_dispatch(self, n_rows: int) -> None:
        """__array_function__: np.sum dispatched to NDArray."""
        _ = np.sum(self.arr)

    def time_arithmetic_add(self, n_rows: int) -> None:
        _ = self.arr + self.arr


# ── ETL macro benchmark ───────────────────────────────────────────────────

class ETLBench:
    """End-to-end ETL: read → filter → groupby → top_k."""

    params = [100_000, 1_000_000]
    param_names = ["n_rows"]

    def setup(self, n_rows: int) -> None:
        import tempfile, os
        rng = np.random.default_rng(42)
        table = pa.table({
            "user_id": pa.array(rng.integers(0, 1000, size=n_rows).tolist()),
            "event_type": pa.array(
                ["click" if i % 3 == 0 else "view" for i in range(n_rows)]
            ),
            "amount": pa.array(rng.standard_normal(n_rows).tolist()),
        })
        fd, self._parquet_path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        import pyarrow.parquet as pq
        pq.write_table(table, self._parquet_path)

    def teardown(self, n_rows: int) -> None:
        import os
        try:
            os.unlink(self._parquet_path)
        except FileNotFoundError:
            pass

    def time_etl_pipeline(self, n_rows: int) -> None:
        df = fx.read_parquet(self._parquet_path)
        df = df.filter(df["event_type"].isin(["click", "view"]))
        result = df.groupby("user_id").agg({"amount": "sum"})
        _ = top_k(result, k=10, by="amount_sum")
