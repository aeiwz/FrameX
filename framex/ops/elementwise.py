"""Element-wise operations: map, apply, binary/scalar array arithmetic.

Binary and scalar ops (add, sub, mul, div) dispatch to C kernels for
float64 arrays when ``kernel_backend="c"`` is active; otherwise they use
pyarrow.compute.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import pyarrow as pa
import pyarrow.compute as pc

from framex.config import get_config
from framex.runtime.partition import Partition

_C_FUNC_CACHE: dict[str, Any] | None = None
_C_MIN_ELEMENTS = int(os.getenv("FRAMEX_C_MIN_ELEMENTS", "1000000"))


def _c_funcs() -> dict[str, Any]:
    """Lazily load C backend elementwise symbols once."""
    global _C_FUNC_CACHE
    if _C_FUNC_CACHE is None:
        try:
            from framex.backends import c_backend as cb
            _C_FUNC_CACHE = {
                "available": bool(getattr(cb, "C_AVAILABLE", False)),
                "add_f64": getattr(cb, "add_f64", None),
                "sub_f64": getattr(cb, "sub_f64", None),
                "mul_f64": getattr(cb, "mul_f64", None),
                "div_f64": getattr(cb, "div_f64", None),
                "scalar_add_f64": getattr(cb, "scalar_add_f64", None),
                "scalar_sub_f64": getattr(cb, "scalar_sub_f64", None),
                "scalar_mul_f64": getattr(cb, "scalar_mul_f64", None),
                "scalar_div_f64": getattr(cb, "scalar_div_f64", None),
            }
        except Exception:
            _C_FUNC_CACHE = {"available": False}
    return _C_FUNC_CACHE


def map_series(chunked: pa.ChunkedArray, fn: Callable[[Any], Any]) -> pa.ChunkedArray:
    """Apply a Python callable element-wise to a ChunkedArray.

    This is inherently slow (Python loop).  Prefer Arrow compute when possible.
    """
    results: list[Any] = []
    for chunk in chunked.chunks:
        for val in chunk.to_pylist():
            results.append(fn(val))
    return pa.chunked_array([pa.array(results)])


def apply_partitions(
    partitions: list[Partition],
    fn: Callable[[pa.RecordBatch], pa.RecordBatch],
) -> list[Partition]:
    """Apply a function to each partition's RecordBatch."""
    return [
        Partition(record_batch=fn(p.record_batch), partition_id=p.partition_id)
        for p in partitions
    ]


# ── Binary array × array ─────────────────────────────────────────────────

def _use_c_for(col: pa.ChunkedArray) -> bool:
    return (
        get_config().kernel_backend == "c"
        and pa.types.is_floating(col.type)
        and col.null_count == 0
        and len(col) >= _C_MIN_ELEMENTS
    )


def add_arrays(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("add_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, b)
        except Exception:
            pass
    return pc.add(a, b)


def sub_arrays(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("sub_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, b)
        except Exception:
            pass
    return pc.subtract(a, b)


def mul_arrays(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("mul_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, b)
        except Exception:
            pass
    return pc.multiply(a, b)


def div_arrays(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("div_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, b)
        except Exception:
            pass
    return pc.divide(a, b)


# ── Scalar array × scalar ────────────────────────────────────────────────

def scalar_add(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("scalar_add_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, float(scalar))
        except Exception:
            pass
    return pc.add(a, scalar)


def scalar_sub(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("scalar_sub_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, float(scalar))
        except Exception:
            pass
    return pc.subtract(a, scalar)


def scalar_mul(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("scalar_mul_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, float(scalar))
        except Exception:
            pass
    return pc.multiply(a, scalar)


def scalar_div(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    if _use_c_for(a):
        c = _c_funcs()
        try:
            fn = c.get("scalar_div_f64")
            if c.get("available", False) and callable(fn):
                return fn(a, float(scalar))
        except Exception:
            pass
    return pc.divide(a, scalar)
