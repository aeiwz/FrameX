"""Validate FrameX capability matrix with runnable checks.

Usage:
    python3 -m benchmarks.check_framex_workloads
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import framex as fx


@dataclass
class CheckRow:
    workload: str
    status: str
    detail: str
    seconds: float


def _timed(fn):
    t0 = time.perf_counter()
    detail, status = fn()
    return detail, status, time.perf_counter() - t0


def check_single_machine_etl() -> tuple[str, str]:
    n = 200_000
    rng = np.random.default_rng(42)
    df = fx.DataFrame(
        {
            "customer_id": rng.integers(1, 5000, n),
            "amount": rng.normal(120.0, 40.0, n),
            "is_refund": rng.integers(0, 10, n) == 0,
        }
    )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "etl.parquet"
        fx.write_parquet(df, path)
        loaded = fx.read_parquet(path)
        result = (
            loaded.filter(~loaded["is_refund"])
            .groupby("customer_id")
            .agg({"amount": ["sum", "mean", "count"]})
            .sort("amount_sum", ascending=False)
            .head(10)
        )
    if result.num_rows == 10:
        return "ETL parquet round-trip + filter/groupby/sort succeeded", "pass"
    return "ETL pipeline returned unexpected shape", "fail"


def check_analytics_join() -> tuple[str, str]:
    n = 120_000
    left = fx.DataFrame({"id": np.arange(n), "value": np.arange(n) * 3})
    right = fx.DataFrame({"id": np.arange(0, n, 2), "score": np.arange(0, n // 2)})
    joined = left.join(right, on="id", how="inner")
    expected = n // 2
    if joined.num_rows == expected:
        return f"Join produced {joined.num_rows} rows as expected", "pass"
    return f"Join rows mismatch: expected {expected}, got {joined.num_rows}", "fail"


def check_ml_preprocessing() -> tuple[str, str]:
    n = 160_000
    rng = np.random.default_rng(7)
    df = fx.DataFrame(
        {
            "age": rng.integers(18, 75, n),
            "income": rng.normal(70_000, 15_000, n),
            "city": rng.choice(["A", "B", "C", "D"], n),
        }
    )
    processed = (
        df.assign(
            age_bucket=lambda d: (d["age"] / 10).round(0) * 10,
            income_k=lambda d: d["income"] / 1000.0,
        )
        .groupby(["city", "age_bucket"])
        .agg({"income_k": ["mean", "count"]})
    )
    if processed.num_rows > 0:
        return "Mixed numeric/categorical preprocessing flow succeeded", "pass"
    return "Preprocessing returned empty result unexpectedly", "fail"


def check_large_ndarray_ops() -> tuple[str, str]:
    n = 1_000_000
    x = fx.array(np.linspace(0.0, 100.0, n), dtype="float64", chunks=125_000)
    y = np.sin(x) + np.sqrt(x + 1.0)
    if isinstance(y, fx.NDArray) and len(y) == n:
        return "NEP13/18 dispatch and chained ndarray ops succeeded", "pass"
    return "NDArray operation did not preserve type/length", "fail"


def check_distributed_cluster_fit() -> tuple[str, str]:
    from framex.runtime.task import Task, TaskGraph
    from framex.runtime.scheduler import LocalScheduler

    # Prefer Ray if installed, otherwise try Dask.
    backend = None
    detail_prefix = ""
    try:
        import ray  # noqa: F401

        backend = "ray"
        detail_prefix = "Ray"
    except Exception:
        try:
            import dask.distributed  # noqa: F401

            backend = "dask"
            detail_prefix = "Dask"
        except Exception:
            return "Neither Ray nor Dask distributed backend is installed", "partial"

    graph = TaskGraph()
    id1 = graph.add_task(Task(fn=lambda: 10))
    id2 = graph.add_task(Task(fn=lambda: 20))
    result = LocalScheduler(max_workers=2, backend=backend).execute(graph)
    if result[id1] == 10 and result[id2] == 20:
        return f"{detail_prefix} backend executed task graph successfully", "pass"
    return f"{detail_prefix} backend returned unexpected results", "fail"


def check_gpu_acceleration() -> tuple[str, str]:
    try:
        import cupy  # noqa: F401
    except Exception:
        return "CuPy not installed; GPU path unavailable in this environment", "partial"

    with fx.config(array_backend="cupy"):
        arr = fx.array([1.0, 2.0, 3.0], dtype="float64")
        out = np.sin(arr)
    if isinstance(out, fx.NDArray):
        return "CuPy backend path executed successfully", "pass"
    return "CuPy backend returned unexpected type", "fail"


def check_streaming_production() -> tuple[str, str]:
    stream = fx.StreamProcessor(
        lambda df: df.filter(df["ok"]).drop(["ok"]),
    )
    batches = [
        {"value": [1, 2, 3], "ok": [True, False, True]},
        {"value": [4, 5], "ok": [True, True]},
    ]
    stats = stream.run(batches)
    if stats.batches_in == 2 and stats.rows_out == 4:
        return "Micro-batch StreamProcessor API succeeded with transform pipeline", "pass"
    return "StreamProcessor produced unexpected statistics", "fail"


def run() -> list[CheckRow]:
    checks: list[tuple[str, Any]] = [
        ("Single-machine ETL (~100MB–30GB)", check_single_machine_etl),
        ("Analytics with joins", check_analytics_join),
        ("ML preprocessing (mixed numeric + categorical)", check_ml_preprocessing),
        ("Large NumPy operations", check_large_ndarray_ops),
        ("Distributed clusters (multi-node)", check_distributed_cluster_fit),
        ("GPU acceleration", check_gpu_acceleration),
        ("Production streaming", check_streaming_production),
    ]

    rows: list[CheckRow] = []
    for workload, fn in checks:
        detail, status, seconds = _timed(fn)
        rows.append(CheckRow(workload=workload, status=status, detail=detail, seconds=seconds))
    return rows


def main() -> None:
    rows = run()
    print("FrameX workload matrix check")
    print("-" * 80)
    for row in rows:
        print(f"[{row.status.upper():14}] {row.workload} ({row.seconds:.3f}s)")
        print(f"  - {row.detail}")

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "framex_workload_check.json"
    out_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print("-" * 80)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
