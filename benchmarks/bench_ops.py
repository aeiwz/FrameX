"""Benchmarks for DataFrame operations: filter, groupby, join, sort."""

import time

import numpy as np
import pyarrow as pa

from framex.core.dataframe import DataFrame


def _make_dataframe(n_rows: int) -> DataFrame:
    rng = np.random.default_rng(42)
    return DataFrame({
        "key": [f"group_{i % 100}" for i in range(n_rows)],
        "val1": rng.standard_normal(n_rows).tolist(),
        "val2": rng.integers(0, 1000, n_rows).tolist(),
    })


def bench_filter(n_rows: int = 1_000_000) -> None:
    df = _make_dataframe(n_rows)
    t0 = time.perf_counter()
    result = df.filter(df["val2"] > 500)
    t1 = time.perf_counter()
    print(f"Filter ({n_rows:,} rows -> {result.num_rows:,} rows): {(t1 - t0) * 1000:.2f} ms")


def bench_groupby(n_rows: int = 1_000_000) -> None:
    df = _make_dataframe(n_rows)
    t0 = time.perf_counter()
    result = df.groupby("key").agg({"val1": "sum", "val2": "mean"})
    t1 = time.perf_counter()
    print(f"GroupBy + agg ({n_rows:,} rows, 100 groups): {(t1 - t0) * 1000:.2f} ms")


def bench_sort(n_rows: int = 1_000_000) -> None:
    df = _make_dataframe(n_rows)
    t0 = time.perf_counter()
    result = df.sort("val1")
    t1 = time.perf_counter()
    print(f"Sort ({n_rows:,} rows): {(t1 - t0) * 1000:.2f} ms")


def bench_join(n_rows: int = 100_000) -> None:
    rng = np.random.default_rng(42)
    left = DataFrame({
        "id": list(range(n_rows)),
        "val": rng.standard_normal(n_rows).tolist(),
    })
    right = DataFrame({
        "id": list(range(0, n_rows, 2)),
        "score": rng.integers(0, 100, n_rows // 2).tolist(),
    })
    t0 = time.perf_counter()
    result = left.join(right, on="id", how="inner")
    t1 = time.perf_counter()
    print(f"Inner join ({n_rows:,} x {n_rows // 2:,} rows -> {result.num_rows:,}): {(t1 - t0) * 1000:.2f} ms")


if __name__ == "__main__":
    bench_filter()
    bench_groupby()
    bench_sort()
    bench_join()
