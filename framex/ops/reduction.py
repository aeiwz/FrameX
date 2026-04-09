"""Scalar reductions: sum, mean, count, min, max, std, var.

When ``kernel_backend="c"`` is active, float64 and int64 columns route
through the compiled C kernels (c_backend).  All other types, nullable
arrays, and unavailable C library fall back to pyarrow.compute.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from framex.config import get_config

_C_FUNC_CACHE: dict[str, Any] | None = None


def _c_funcs() -> dict[str, Any]:
    """Lazily load C backend symbols once to avoid per-call import overhead."""
    global _C_FUNC_CACHE
    if _C_FUNC_CACHE is None:
        try:
            from framex.backends import c_backend as cb
            _C_FUNC_CACHE = {
                "available": bool(getattr(cb, "C_AVAILABLE", False)),
                "sum_f64_chunked": getattr(cb, "sum_f64_chunked", None),
                "sum_i64_chunked": getattr(cb, "sum_i64_chunked", None),
                "mean_f64_chunked": getattr(cb, "mean_f64_chunked", None),
                "mean_i64_chunked": getattr(cb, "mean_i64_chunked", None),
                "min_f64_chunked": getattr(cb, "min_f64_chunked", None),
                "min_i64_chunked": getattr(cb, "min_i64_chunked", None),
                "max_f64_chunked": getattr(cb, "max_f64_chunked", None),
                "max_i64_chunked": getattr(cb, "max_i64_chunked", None),
                "std_f64_chunked": getattr(cb, "std_f64_chunked", None),
                "var_f64_chunked": getattr(cb, "var_f64_chunked", None),
            }
        except Exception:
            _C_FUNC_CACHE = {"available": False}
    return _C_FUNC_CACHE


def _use_c() -> bool:
    return get_config().kernel_backend == "c"


def _is_float(col: pa.ChunkedArray) -> bool:
    return pa.types.is_floating(col.type)


def _is_integer(col: pa.ChunkedArray) -> bool:
    return pa.types.is_integer(col.type)


def _has_nulls(col: pa.ChunkedArray) -> bool:
    return col.null_count != 0


def sum_column(column: pa.ChunkedArray) -> Any:
    if _use_c() and not _has_nulls(column):
        c = _c_funcs()
        try:
            if c.get("available", False):
                if _is_float(column):
                    fn = c.get("sum_f64_chunked")
                    if callable(fn):
                        return fn(column)
                if _is_integer(column):
                    fn = c.get("sum_i64_chunked")
                    if callable(fn):
                        return fn(column)
        except Exception:
            pass
    return pc.sum(column).as_py()


def mean_column(column: pa.ChunkedArray) -> Any:
    if _use_c() and not _has_nulls(column):
        c = _c_funcs()
        try:
            if c.get("available", False):
                if _is_float(column):
                    fn = c.get("mean_f64_chunked")
                    if callable(fn):
                        return fn(column)
                if _is_integer(column):
                    fn = c.get("mean_i64_chunked")
                    if callable(fn):
                        return fn(column)
        except Exception:
            pass
    return pc.mean(column).as_py()


def count_column(column: pa.ChunkedArray) -> int:
    # count is not type-specific — Arrow's is already C++.
    return pc.count(column).as_py()


def min_column(column: pa.ChunkedArray) -> Any:
    if _use_c() and not _has_nulls(column):
        c = _c_funcs()
        try:
            if c.get("available", False):
                if _is_float(column):
                    fn = c.get("min_f64_chunked")
                    if callable(fn):
                        return fn(column)
                if _is_integer(column):
                    fn = c.get("min_i64_chunked")
                    if callable(fn):
                        return fn(column)
        except Exception:
            pass
    return pc.min(column).as_py()


def max_column(column: pa.ChunkedArray) -> Any:
    if _use_c() and not _has_nulls(column):
        c = _c_funcs()
        try:
            if c.get("available", False):
                if _is_float(column):
                    fn = c.get("max_f64_chunked")
                    if callable(fn):
                        return fn(column)
                if _is_integer(column):
                    fn = c.get("max_i64_chunked")
                    if callable(fn):
                        return fn(column)
        except Exception:
            pass
    return pc.max(column).as_py()


def std_column(column: pa.ChunkedArray, ddof: int = 1) -> Any:
    if _use_c() and not _has_nulls(column) and _is_float(column):
        c = _c_funcs()
        try:
            fn = c.get("std_f64_chunked")
            if c.get("available", False) and callable(fn):
                return fn(column, ddof=ddof)
        except Exception:
            pass
    return pc.stddev(column, ddof=ddof).as_py()


def var_column(column: pa.ChunkedArray, ddof: int = 1) -> Any:
    if _use_c() and not _has_nulls(column) and _is_float(column):
        c = _c_funcs()
        try:
            fn = c.get("var_f64_chunked")
            if c.get("available", False) and callable(fn):
                return fn(column, ddof=ddof)
        except Exception:
            pass
    return pc.variance(column, ddof=ddof).as_py()
