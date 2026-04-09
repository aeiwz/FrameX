"""Unified benchmark suite for FrameX vs native libraries.

Covers:
1) Performance benchmarks
2) Parallel processing benchmarks
3) Single-core benchmarks
4) Multiprocessing benchmarks
5) Memory benchmarks
6) Report generation + visualization

Usage:
    python -m benchmarks.benchmark_suite
    python -m benchmarks.benchmark_suite --rows 300000 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

import framex as fx
from framex.core.dataframe import DataFrame
from framex.runtime.scheduler import LocalScheduler
from framex.runtime.task import Task, TaskGraph
from framex.ops.reduction import sum_column, mean_column, min_column, max_column
from framex.ops.elementwise import add_arrays, scalar_mul

try:
    from framex.backends.c_backend import C_AVAILABLE
except Exception:
    C_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass
class BenchmarkRow:
    category: str
    scenario: str
    engine: str
    size: int
    workers: int
    seconds: float
    speedup_vs_native: float | None = None
    peak_rss_mb: float | None = None


def _median_time(
    fn: Callable[[], Any],
    repeats: int,
    warmups: int,
    *,
    min_repeat_seconds: float = 0.03,
) -> float:
    for _ in range(max(0, warmups)):
        fn()

    runs: list[float] = []
    for _ in range(max(1, repeats)):
        # Adaptive batching improves timing stability for ultra-fast operations.
        iterations = 1
        while True:
            t0 = time.perf_counter()
            for _i in range(iterations):
                fn()
            elapsed = time.perf_counter() - t0

            if elapsed >= min_repeat_seconds or iterations >= 1 << 20:
                runs.append(elapsed / iterations)
                break
            iterations *= 2

    runs.sort()
    return float(runs[len(runs) // 2])


def _safe_speedup(native_seconds: float, candidate_seconds: float) -> float:
    if candidate_seconds <= 0:
        return math.inf
    return native_seconds / candidate_seconds


def _make_numeric_frames(n_rows: int) -> tuple[DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    keys = rng.integers(0, 1000, n_rows)
    val1 = rng.standard_normal(n_rows)
    val2 = rng.integers(0, 1000, n_rows)

    fx_df = DataFrame({
        "key": keys,
        "val1": val1,
        "val2": val2,
    })
    pd_df = pd.DataFrame({
        "key": keys,
        "val1": val1,
        "val2": val2,
    })
    return fx_df, pd_df


def _make_join_frames(n_rows: int) -> tuple[DataFrame, DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = np.arange(n_rows)
    left_vals = np.random.default_rng(1).standard_normal(n_rows)
    right_ids = np.arange(0, n_rows, 2)
    right_vals = np.random.default_rng(2).integers(0, 1000, len(right_ids))

    fx_left = DataFrame({"id": ids, "val": left_vals})
    fx_right = DataFrame({"id": right_ids, "score": right_vals})
    pd_left = pd.DataFrame({"id": ids, "val": left_vals})
    pd_right = pd.DataFrame({"id": right_ids, "score": right_vals})
    return fx_left, fx_right, pd_left, pd_right


def _numeric_chunk_kernel(chunk: np.ndarray) -> float:
    # Numpy-heavy workload (releases GIL in many operations).
    return float(np.sqrt(chunk * 1.000001).sum())


def _object_chunk_kernel(chunk: Sequence[str]) -> int:
    # Python object workload (GIL-heavy): better candidate for processes.
    vowels = {"a", "e", "i", "o", "u"}
    total = 0
    for item in chunk:
        c = 0
        for ch in item:
            if ch in vowels:
                c += 1
        total += c
    return total


def _split_array(arr: np.ndarray, n_chunks: int) -> list[np.ndarray]:
    n_chunks = max(1, n_chunks)
    return [x for x in np.array_split(arr, n_chunks) if len(x) > 0]


def _split_list(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, n_chunks)
    size = max(1, len(items) // n_chunks)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _auto_numeric_chunk_count(elements: int, workers: int) -> int:
    """Choose chunk count to balance parallelism vs scheduler overhead.

    For high worker counts (e.g. 8), too many small chunks can make
    scheduler/task overhead dominate. This heuristic keeps enough work to
    saturate workers while biasing toward larger chunks.
    """
    workers = max(1, workers)
    elements = max(1, elements)

    if workers >= 8:
        target_chunk_size = 500_000
        max_chunks = workers * 2
    elif workers >= 4:
        target_chunk_size = 300_000
        max_chunks = workers * 3
    else:
        target_chunk_size = 250_000
        max_chunks = workers * 4

    size_based_chunks = math.ceil(elements / target_chunk_size)
    n_chunks = max(workers, size_based_chunks)
    return max(1, min(n_chunks, max_chunks))


def _run_framex_scheduler(
    backend: str,
    workers: int,
    fn: Callable[..., Any],
    chunks: Sequence[Any],
) -> Any:
    graph = TaskGraph()
    task_ids: list[str] = []
    for chunk in chunks:
        task_ids.append(graph.add_task(Task(fn=fn, args=(chunk,))))

    scheduler = LocalScheduler(max_workers=workers, backend=backend)
    results = scheduler.execute(graph)
    ordered = [results[tid] for tid in task_ids]

    first = ordered[0] if ordered else 0
    if isinstance(first, (int, float, np.number)):
        return float(sum(ordered))
    return ordered


def _run_threadpool(fn: Callable[..., Any], workers: int, chunks: Sequence[Any]) -> Any:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(fn, chunks))
    first = out[0] if out else 0
    if isinstance(first, (int, float, np.number)):
        return float(sum(out))
    return out


def _run_processpool(fn: Callable[..., Any], workers: int, chunks: Sequence[Any]) -> Any:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(fn, chunks))
    first = out[0] if out else 0
    if isinstance(first, (int, float, np.number)):
        return float(sum(out))
    return out


def _rss_monitor_start() -> tuple[dict[str, bool], list[int], Any | None]:
    if psutil is None:
        return ({"running": False}, [], None)

    import threading

    state = {"running": True}
    samples: list[int] = []
    proc = psutil.Process(os.getpid())

    def _poll() -> None:
        while state["running"]:
            try:
                samples.append(proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(0.005)

    th = threading.Thread(target=_poll, daemon=True)
    th.start()
    return state, samples, th


def _measure_peak_rss_mb(fn: Callable[[], Any]) -> tuple[float, float]:
    before = 0
    after = 0
    peak = 0

    if psutil is not None:
        proc = psutil.Process(os.getpid())
        try:
            before = proc.memory_info().rss
        except Exception:
            before = 0

    state, samples, thread_obj = _rss_monitor_start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    state["running"] = False

    if thread_obj is not None:
        thread_obj.join(timeout=0.5)

    if psutil is not None:
        proc = psutil.Process(os.getpid())
        try:
            after = proc.memory_info().rss
        except Exception:
            after = 0

    if samples:
        peak = max(samples)
    else:
        peak = max(before, after)

    peak_mb = peak / (1024 * 1024)
    return elapsed, peak_mb


def run_performance_benchmark(rows: int, repeats: int, warmups: int) -> list[BenchmarkRow]:
    fx_df, pd_df = _make_numeric_frames(rows)
    fx_arr = fx.array(pd_df["val1"].to_numpy(), dtype="float64", chunks=max(1, rows // 8))
    np_arr = pd_df["val1"].to_numpy()
    fx_left, fx_right, pd_left, pd_right = _make_join_frames(rows)

    scenarios: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        (
            "filter_val2_gt_500",
            lambda: fx_df.filter(fx_df["val2"] > 500),
            lambda: pd_df[pd_df["val2"] > 500],
        ),
        (
            "groupby_key_sum_mean",
            lambda: fx_df.groupby("key").agg({"val1": "sum", "val2": "mean"}),
            lambda: pd_df.groupby("key", as_index=False).agg({"val1": "sum", "val2": "mean"}),
        ),
        (
            "sort_val1",
            lambda: fx_df.sort("val1"),
            lambda: pd_df.sort_values("val1"),
        ),
        (
            "join_inner",
            lambda: fx_left.join(fx_right, on="id", how="inner"),
            lambda: pd_left.merge(pd_right, on="id", how="inner"),
        ),
        (
            "ndarray_sum",
            lambda: fx_arr.sum(),
            lambda: np.sum(np_arr),
        ),
        (
            "ndarray_np_sin_dispatch",
            lambda: np.sin(fx_arr),
            lambda: np.sin(np_arr),
        ),
    ]

    rows_out: list[BenchmarkRow] = []
    for scenario, framex_fn, native_fn in scenarios:
        native_sec = _median_time(native_fn, repeats=repeats, warmups=warmups)
        framex_sec = _median_time(framex_fn, repeats=repeats, warmups=warmups)

        rows_out.append(
            BenchmarkRow(
                category="performance",
                scenario=scenario,
                engine="native",
                size=rows,
                workers=1,
                seconds=native_sec,
                speedup_vs_native=1.0,
            )
        )
        rows_out.append(
            BenchmarkRow(
                category="performance",
                scenario=scenario,
                engine="framex",
                size=rows,
                workers=1,
                seconds=framex_sec,
                speedup_vs_native=_safe_speedup(native_sec, framex_sec),
            )
        )

    return rows_out


def run_single_core_benchmark(
    elements: int,
    repeats: int,
    warmups: int,
) -> list[BenchmarkRow]:
    arr = np.linspace(1.0, 10_000.0, elements, dtype=np.float64)
    chunks = _split_array(arr, 1)

    def native_seq() -> Any:
        return sum(_numeric_chunk_kernel(c) for c in chunks)

    def framex_seq() -> Any:
        return _run_framex_scheduler("threads", 1, _numeric_chunk_kernel, chunks)

    native_sec = _median_time(native_seq, repeats=repeats, warmups=warmups)
    framex_sec = _median_time(framex_seq, repeats=repeats, warmups=warmups)

    return [
        BenchmarkRow(
            category="single_core",
            scenario="numeric_kernel",
            engine="native",
            size=elements,
            workers=1,
            seconds=native_sec,
            speedup_vs_native=1.0,
        ),
        BenchmarkRow(
            category="single_core",
            scenario="numeric_kernel",
            engine="framex",
            size=elements,
            workers=1,
            seconds=framex_sec,
            speedup_vs_native=_safe_speedup(native_sec, framex_sec),
        ),
    ]


def run_parallel_benchmark(
    elements: int,
    workers_list: Sequence[int],
    repeats: int,
    warmups: int,
) -> list[BenchmarkRow]:
    arr = np.linspace(1.0, 50_000.0, elements, dtype=np.float64)
    rows_out: list[BenchmarkRow] = []

    for workers in workers_list:
        n_chunks = _auto_numeric_chunk_count(elements, workers)
        chunks = _split_array(arr, n_chunks)

        native_fn = lambda: _run_threadpool(_numeric_chunk_kernel, workers, chunks)
        framex_fn = lambda: _run_framex_scheduler("threads", workers, _numeric_chunk_kernel, chunks)

        native_sec = _median_time(native_fn, repeats=repeats, warmups=warmups)
        framex_sec = _median_time(framex_fn, repeats=repeats, warmups=warmups)

        rows_out.append(
            BenchmarkRow(
                category="parallel_processing",
                scenario="numeric_kernel_threads",
                engine="native",
                size=elements,
                workers=workers,
                seconds=native_sec,
                speedup_vs_native=1.0,
            )
        )
        rows_out.append(
            BenchmarkRow(
                category="parallel_processing",
                scenario="numeric_kernel_threads",
                engine="framex",
                size=elements,
                workers=workers,
                seconds=framex_sec,
                speedup_vs_native=_safe_speedup(native_sec, framex_sec),
            )
        )

    return rows_out


def run_multiprocessing_benchmark(
    items: int,
    workers_list: Sequence[int],
    repeats: int,
    warmups: int,
) -> list[BenchmarkRow]:
    data = [f"event_{i % 1000}_source_{i % 17}_region_{i % 13}" for i in range(items)]
    rows_out: list[BenchmarkRow] = []

    for workers in workers_list:
        chunks = _split_list(data, workers * 4)

        native_fn = lambda: _run_processpool(_object_chunk_kernel, workers, chunks)
        framex_fn = lambda: _run_framex_scheduler("processes", workers, _object_chunk_kernel, chunks)

        native_sec = _median_time(native_fn, repeats=repeats, warmups=warmups)
        framex_sec = _median_time(framex_fn, repeats=repeats, warmups=warmups)

        rows_out.append(
            BenchmarkRow(
                category="multiprocessing",
                scenario="object_kernel_processes",
                engine="native",
                size=items,
                workers=workers,
                seconds=native_sec,
                speedup_vs_native=1.0,
            )
        )
        rows_out.append(
            BenchmarkRow(
                category="multiprocessing",
                scenario="object_kernel_processes",
                engine="framex",
                size=items,
                workers=workers,
                seconds=framex_sec,
                speedup_vs_native=_safe_speedup(native_sec, framex_sec),
            )
        )

    return rows_out


def run_memory_benchmark(rows: int, repeats: int, warmups: int) -> list[BenchmarkRow]:
    fx_df, pd_df = _make_numeric_frames(rows)
    fx_arr = fx.array(pd_df["val1"].to_numpy(), dtype="float64", chunks=max(1, rows // 8))
    np_arr = pd_df["val1"].to_numpy()

    scenarios: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        (
            "filter_val2_gt_500",
            lambda: fx_df.filter(fx_df["val2"] > 500),
            lambda: pd_df[pd_df["val2"] > 500],
        ),
        (
            "groupby_key_sum",
            lambda: fx_df.groupby("key").agg({"val1": "sum"}),
            lambda: pd_df.groupby("key", as_index=False).agg({"val1": "sum"}),
        ),
        (
            "ndarray_np_sin_dispatch",
            lambda: np.sin(fx_arr),
            lambda: np.sin(np_arr),
        ),
    ]

    rows_out: list[BenchmarkRow] = []
    for scenario, framex_fn, native_fn in scenarios:
        # Use warmed execution path first to reduce one-time allocator effects.
        _median_time(native_fn, repeats=1, warmups=max(1, warmups))
        _median_time(framex_fn, repeats=1, warmups=max(1, warmups))

        native_t, native_peak = _measure_peak_rss_mb(native_fn)
        framex_t, framex_peak = _measure_peak_rss_mb(framex_fn)

        rows_out.append(
            BenchmarkRow(
                category="memory",
                scenario=scenario,
                engine="native",
                size=rows,
                workers=1,
                seconds=native_t,
                speedup_vs_native=1.0,
                peak_rss_mb=native_peak,
            )
        )
        rows_out.append(
            BenchmarkRow(
                category="memory",
                scenario=scenario,
                engine="framex",
                size=rows,
                workers=1,
                seconds=framex_t,
                speedup_vs_native=_safe_speedup(native_t, framex_t),
                peak_rss_mb=framex_peak,
            )
        )

    return rows_out


def run_c_backend_benchmark(rows: int, repeats: int, warmups: int) -> list[BenchmarkRow]:
    """Benchmark Python kernel backend vs C backend on eligible operations."""
    if not C_AVAILABLE:
        return []

    values = np.linspace(1.0, float(rows), rows, dtype=np.float64)
    # Use multiple chunks to model realistic partitioned/chunked execution.
    chunks = [values[i : i + 50_000] for i in range(0, len(values), 50_000)]
    arrow_col = pa.chunked_array([pa.array(c) for c in chunks], type=pa.float64())

    scenarios: list[tuple[str, Callable[[], Any]]] = [
        ("reduction_sum_f64", lambda: sum_column(arrow_col)),
        ("reduction_mean_f64", lambda: mean_column(arrow_col)),
        ("reduction_min_max_f64", lambda: (min_column(arrow_col), max_column(arrow_col))),
        ("elementwise_add_f64", lambda: add_arrays(arrow_col, arrow_col)),
        ("elementwise_scalar_mul_f64", lambda: scalar_mul(arrow_col, 1.5)),
    ]

    rows_out: list[BenchmarkRow] = []
    for scenario, op in scenarios:
        with fx.config(kernel_backend="python"):
            python_sec = _median_time(op, repeats=repeats, warmups=warmups)

        with fx.config(kernel_backend="c"):
            c_sec = _median_time(op, repeats=repeats, warmups=warmups)

        rows_out.append(
            BenchmarkRow(
                category="c_backend",
                scenario=scenario,
                engine="python_backend",
                size=rows,
                workers=1,
                seconds=python_sec,
                speedup_vs_native=1.0,
            )
        )
        rows_out.append(
            BenchmarkRow(
                category="c_backend",
                scenario=scenario,
                engine="c_backend",
                size=rows,
                workers=1,
                seconds=c_sec,
                speedup_vs_native=_safe_speedup(python_sec, c_sec),
            )
        )

    return rows_out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")


def _plot_performance(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if plt is None:
        return []

    generated: list[Path] = []

    perf = df[df["category"] == "performance"].copy()
    if not perf.empty:
        pivot = perf.pivot_table(index="scenario", columns="engine", values="seconds", aggfunc="min")
        if "native" in pivot.columns and "framex" in pivot.columns:
            speedups = (pivot["native"] / pivot["framex"]).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            speedups.plot(kind="bar", ax=ax, color="#1f77b4")
            ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
            ax.set_title("Performance Speedup (Native / FrameX)")
            ax.set_ylabel("Speedup")
            ax.set_xlabel("Scenario")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            p = out_dir / "performance_speedup.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            generated.append(p)

    return generated


def _plot_parallel(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if plt is None:
        return []

    generated: list[Path] = []
    categories = ["parallel_processing", "multiprocessing"]
    for cat in categories:
        sub = df[df["category"] == cat].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for engine in sorted(sub["engine"].unique()):
            d = sub[sub["engine"] == engine].sort_values("workers")
            ax.plot(d["workers"], d["seconds"], marker="o", label=engine)

        ax.set_title(f"{cat.replace('_', ' ').title()} Scaling")
        ax.set_xlabel("Workers")
        ax.set_ylabel("Seconds (lower is better)")
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"{cat}_scaling.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        generated.append(p)

    return generated


def _plot_memory(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if plt is None:
        return []

    generated: list[Path] = []
    sub = df[df["category"] == "memory"].copy()
    if sub.empty:
        return generated

    pivot = sub.pivot_table(index="scenario", columns="engine", values="peak_rss_mb", aggfunc="max")
    if pivot.empty:
        return generated

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Peak RSS Memory (MB)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Peak RSS (MB)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    p = out_dir / "memory_peak_rss.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    generated.append(p)

    return generated


def _build_markdown_report(df: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> str:
    now = datetime.now().astimezone()
    ts = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    def _table(category: str) -> str:
        sub = df[df["category"] == category].copy()
        if sub.empty:
            return "No data"

        display = sub[
            ["scenario", "engine", "workers", "seconds", "speedup_vs_native", "peak_rss_mb"]
        ].copy()
        display["seconds"] = display["seconds"].map(lambda x: f"{x:.6f}")
        display["speedup_vs_native"] = display["speedup_vs_native"].map(
            lambda x: "" if pd.isna(x) else f"{x:.3f}"
        )
        display["peak_rss_mb"] = display["peak_rss_mb"].map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
        return "```text\n" + display.to_string(index=False) + "\n```"

    def _comparison_table(
        category: str,
        left_engine: str,
        right_engine: str,
        *,
        left_label: str,
        right_label: str,
    ) -> str:
        sub = df[df["category"] == category].copy()
        if sub.empty:
            return "No data"

        left = (
            sub[sub["engine"] == left_engine][["scenario", "workers", "seconds", "peak_rss_mb"]]
            .rename(
                columns={
                    "seconds": f"{left_label}_seconds",
                    "peak_rss_mb": f"{left_label}_peak_rss_mb",
                }
            )
        )
        right = (
            sub[sub["engine"] == right_engine][["scenario", "workers", "seconds", "peak_rss_mb"]]
            .rename(
                columns={
                    "seconds": f"{right_label}_seconds",
                    "peak_rss_mb": f"{right_label}_peak_rss_mb",
                }
            )
        )

        merged = left.merge(right, on=["scenario", "workers"], how="inner")
        if merged.empty:
            return "No comparable rows"

        merged["speedup"] = (
            merged[f"{left_label}_seconds"] / merged[f"{right_label}_seconds"]
        )
        merged["winner"] = merged["speedup"].map(
            lambda x: right_label if x > 1.0 else (left_label if x < 1.0 else "tie")
        )

        for col in merged.columns:
            if col.endswith("_seconds"):
                merged[col] = merged[col].map(lambda x: f"{x:.6f}")
            if col.endswith("_peak_rss_mb"):
                merged[col] = merged[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        merged["speedup"] = merged["speedup"].map(lambda x: f"{x:.3f}x")

        ordered_cols = [
            "scenario",
            "workers",
            f"{left_label}_seconds",
            f"{right_label}_seconds",
            "speedup",
            "winner",
        ]
        if f"{left_label}_peak_rss_mb" in merged.columns and merged[f"{left_label}_peak_rss_mb"].notna().any():
            ordered_cols.append(f"{left_label}_peak_rss_mb")
        if f"{right_label}_peak_rss_mb" in merged.columns and merged[f"{right_label}_peak_rss_mb"].notna().any():
            ordered_cols.append(f"{right_label}_peak_rss_mb")

        return "```text\n" + merged[ordered_cols].to_string(index=False) + "\n```"

    insights: list[str] = []

    perf = df[(df["category"] == "performance") & (df["engine"] == "framex")]
    if not perf.empty:
        best = perf.sort_values("speedup_vs_native", ascending=False).iloc[0]
        worst = perf.sort_values("speedup_vs_native", ascending=True).iloc[0]
        insights.append(
            f"Best FrameX speedup in performance bench: `{best['scenario']}` = {best['speedup_vs_native']:.2f}x vs native."
        )
        insights.append(
            f"Toughest performance case for FrameX: `{worst['scenario']}` = {worst['speedup_vs_native']:.2f}x vs native."
        )

    for cat in ("parallel_processing", "multiprocessing"):
        sub = df[(df["category"] == cat) & (df["engine"] == "framex")]
        if not sub.empty:
            best = sub.sort_values("seconds", ascending=True).iloc[0]
            insights.append(
                f"Fastest FrameX {cat.replace('_', ' ')} run used {int(best['workers'])} workers ({best['seconds']:.4f}s)."
            )

    mem = df[(df["category"] == "memory") & (df["engine"] == "framex")]
    if not mem.empty and mem["peak_rss_mb"].notna().any():
        worst_mem = mem.sort_values("peak_rss_mb", ascending=False).iloc[0]
        insights.append(
            f"Highest FrameX measured peak RSS: `{worst_mem['scenario']}` = {worst_mem['peak_rss_mb']:.2f} MB."
        )

    cbench = df[(df["category"] == "c_backend") & (df["engine"] == "c_backend")]
    if not cbench.empty:
        best_c = cbench.sort_values("speedup_vs_native", ascending=False).iloc[0]
        insights.append(
            f"Best C backend speedup: `{best_c['scenario']}` = {best_c['speedup_vs_native']:.2f}x vs python backend."
        )

    images = [
        out_dir / "performance_speedup.png",
        out_dir / "parallel_processing_scaling.png",
        out_dir / "multiprocessing_scaling.png",
        out_dir / "memory_peak_rss.png",
    ]
    image_lines = [f"- `{p.name}`" for p in images if p.exists()]

    c_backend_compare = (
        _comparison_table(
            "c_backend",
            "python_backend",
            "c_backend",
            left_label="framex_python",
            right_label="framex_c",
        )
        if args.include_c_backend
        else "Disabled via --no-c-backend."
    )

    report = f"""# FrameX Benchmark Report

Generated: {ts}

Command parameters:
- rows: {args.rows}
- repeats: {args.repeats}
- warmups: {args.warmups}
- workers: {','.join(str(w) for w in args.workers)}

## Compare: Native vs FrameX (Performance)

{_comparison_table('performance', 'native', 'framex', left_label='native', right_label='framex')}

## Compare: Native vs FrameX (Parallel processing)

{_comparison_table('parallel_processing', 'native', 'framex', left_label='native', right_label='framex')}

## Compare: Native vs FrameX (Single core)

{_comparison_table('single_core', 'native', 'framex', left_label='native', right_label='framex')}

## Compare: Native vs FrameX (Multiprocessing)

{_comparison_table('multiprocessing', 'native', 'framex', left_label='native', right_label='framex')}

## Compare: Native vs FrameX (Memory)

{_comparison_table('memory', 'native', 'framex', left_label='native', right_label='framex')}

## Compare: FrameX Python vs FrameX C backend

{c_backend_compare}

## Detailed rows: Performance

{_table('performance')}

## Detailed rows: Parallel processing

{_table('parallel_processing')}

## Detailed rows: Single core

{_table('single_core')}

## Detailed rows: Multiprocessing

{_table('multiprocessing')}

## Detailed rows: Memory

{_table('memory')}

## Detailed rows: C backend

{_table('c_backend') if args.include_c_backend else 'Disabled via --no-c-backend.'}

## Report benchmark and visualize

Generated visualizations:
{chr(10).join(image_lines) if image_lines else '- (No plots generated. Install matplotlib to enable.)'}

Key findings:
{chr(10).join(f'- {line}' for line in insights) if insights else '- Not enough data to compute findings.'}
"""
    return report


def _parse_workers(value: str) -> list[int]:
    parsed: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed.append(max(1, int(part)))
    if not parsed:
        return [1, 2, 4]
    return sorted(set(parsed))


def run_all(args: argparse.Namespace) -> tuple[pd.DataFrame, list[Path]]:
    rows_out: list[BenchmarkRow] = []

    rows_out.extend(run_performance_benchmark(args.rows, args.repeats, args.warmups))
    rows_out.extend(
        run_single_core_benchmark(
            elements=args.array_elements,
            repeats=args.repeats,
            warmups=args.warmups,
        )
    )
    rows_out.extend(
        run_parallel_benchmark(
            elements=args.array_elements,
            workers_list=args.workers,
            repeats=args.repeats,
            warmups=args.warmups,
        )
    )
    rows_out.extend(
        run_multiprocessing_benchmark(
            items=args.object_items,
            workers_list=args.workers,
            repeats=args.repeats,
            warmups=args.warmups,
        )
    )
    rows_out.extend(run_memory_benchmark(args.rows, args.repeats, args.warmups))
    if args.include_c_backend:
        rows_out.extend(run_c_backend_benchmark(args.rows, args.repeats, args.warmups))

    df = pd.DataFrame([asdict(r) for r in rows_out])

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(rows_out, out_dir / "benchmark_results.json")
    _write_csv(df, out_dir / "benchmark_results.csv")

    plots: list[Path] = []
    if not args.skip_plots:
        plots.extend(_plot_performance(df, out_dir))
        plots.extend(_plot_parallel(df, out_dir))
        plots.extend(_plot_memory(df, out_dir))

    md = _build_markdown_report(df, out_dir, args)
    (out_dir / "benchmark_report.md").write_text(md, encoding="utf-8")

    return df, plots


def parse_args() -> argparse.Namespace:
    cpu = os.cpu_count() or 4
    default_workers = [1, min(2, cpu), min(4, cpu), min(8, cpu)]
    default_workers = sorted(set(default_workers))

    parser = argparse.ArgumentParser(description="FrameX unified benchmark suite")
    parser.add_argument("--rows", type=int, default=300_000, help="Rows for dataframe benchmarks")
    parser.add_argument("--array-elements", type=int, default=2_000_000, help="Elements for numeric parallel benchmark")
    parser.add_argument("--object-items", type=int, default=400_000, help="Items for object multiprocessing benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per scenario")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs before timing")
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default=default_workers,
        help="Comma-separated worker counts, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory for JSON/CSV/Markdown/plots",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Disable plot generation")
    parser.add_argument(
        "--no-c-backend",
        action="store_true",
        help="Disable C backend benchmark section",
    )
    args = parser.parse_args()
    args.include_c_backend = not args.no_c_backend
    return args


def main() -> None:
    args = parse_args()
    df, plots = run_all(args)

    print("Benchmark suite completed")
    print(f"Rows captured: {len(df)}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    if plots:
        print("Generated plots:")
        for p in plots:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
