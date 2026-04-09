"""C backend for FrameX — compiles c_kernels.c at first import and loads
the shared library via ctypes.

Usage pattern (all public functions mirror the pyarrow.compute-based Python
equivalents so callers can swap transparently):

    from framex.backends.c_backend import C_AVAILABLE, sum_f64_chunked
    if C_AVAILABLE:
        result = sum_f64_chunked(my_chunked_array)

Null safety: every function checks ``chunk.null_count == 0`` and raises
``ValueError`` if nulls are present.  Callers in ops/reduction.py and
ops/elementwise.py catch this and fall back to pyarrow.compute.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

_HERE = Path(__file__).parent
_SRC = _HERE / "c_kernels.c"
_LIB = _HERE / "c_kernels.so"

logger = logging.getLogger(__name__)

# ── Compilation ──────────────────────────────────────────────────────────

_COMPILER_CANDIDATES = ["cc", "gcc", "clang"]


def _find_compiler() -> str | None:
    import shutil
    for candidate in _COMPILER_CANDIDATES:
        if shutil.which(candidate):
            return candidate
    return None


def _compile_kernels() -> bool:
    """Compile c_kernels.c → c_kernels.so.  Returns True on success."""
    compiler = _find_compiler()
    if compiler is None:
        logger.warning(
            "FrameX C backend: no C compiler found (%s). "
            "Falling back to Python backend.",
            ", ".join(_COMPILER_CANDIDATES),
        )
        return False
    cmd = [
        compiler, "-O2", "-shared", "-fPIC",
        "-o", str(_LIB),
        str(_SRC),
        "-lm",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "FrameX C backend: compilation failed.\n%s\n%s",
                result.stdout,
                result.stderr,
            )
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("FrameX C backend: compilation error: %s", exc)
        return False


def _load_lib() -> ctypes.CDLL | None:
    if not _LIB.exists():
        if not _compile_kernels():
            return None
    try:
        lib = ctypes.CDLL(str(_LIB))
    except OSError as exc:
        logger.warning("FrameX C backend: could not load shared lib: %s", exc)
        return None
    _annotate_signatures(lib)
    return lib


def _annotate_signatures(lib: ctypes.CDLL) -> None:
    """Attach argtypes and restype to each symbol for type safety."""
    _D = ctypes.c_double
    _I = ctypes.c_int64
    _U = ctypes.c_uint8
    _DP = ctypes.POINTER(_D)
    _IP = ctypes.POINTER(_I)
    _UP = ctypes.POINTER(_U)
    _INT = ctypes.c_int

    # Reductions → scalar
    for name in ("fx_sum_f64", "fx_mean_f64", "fx_min_f64", "fx_max_f64"):
        fn = getattr(lib, name)
        fn.argtypes = [_DP, _I]
        fn.restype = _D

    lib.fx_sum_i64.argtypes = [_IP, _I]
    lib.fx_sum_i64.restype = _I

    lib.fx_mean_i64.argtypes = [_IP, _I]
    lib.fx_mean_i64.restype = _D

    lib.fx_min_i64.argtypes = [_IP, _I]
    lib.fx_min_i64.restype = _I

    lib.fx_max_i64.argtypes = [_IP, _I]
    lib.fx_max_i64.restype = _I

    lib.fx_std_f64.argtypes = [_DP, _I, _INT]
    lib.fx_std_f64.restype = _D

    lib.fx_var_f64.argtypes = [_DP, _I, _INT]
    lib.fx_var_f64.restype = _D

    # Elementwise array × array → void
    for name in ("fx_add_f64", "fx_sub_f64", "fx_mul_f64", "fx_div_f64"):
        fn = getattr(lib, name)
        fn.argtypes = [_DP, _DP, _DP, _I]
        fn.restype = None

    # Elementwise array × scalar → void
    for name in ("fx_scalar_add_f64", "fx_scalar_sub_f64",
                 "fx_scalar_mul_f64", "fx_scalar_div_f64"):
        fn = getattr(lib, name)
        fn.argtypes = [_DP, _D, _DP, _I]
        fn.restype = None

    # Filter → int64 (number of written elements)
    lib.fx_filter_f64.argtypes = [_DP, _UP, _DP, _I]
    lib.fx_filter_f64.restype = _I

    lib.fx_filter_i64.argtypes = [_IP, _UP, _IP, _I]
    lib.fx_filter_i64.restype = _I


_lib: ctypes.CDLL | None = _load_lib()
C_AVAILABLE: bool = _lib is not None

# ── Buffer helpers ───────────────────────────────────────────────────────

def _require_no_nulls(chunk: pa.Array) -> None:
    if chunk.null_count != 0:
        raise ValueError(
            "C backend kernels do not support null values; "
            "use the Python backend or fill/drop nulls first."
        )


def _f64_np(chunk: pa.Array) -> np.ndarray:
    """Return a contiguous float64 numpy view of an Arrow array chunk."""
    _require_no_nulls(chunk)
    return chunk.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def _i64_np(chunk: pa.Array) -> np.ndarray:
    _require_no_nulls(chunk)
    return chunk.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _ptr_f64(arr: np.ndarray) -> ctypes.POINTER:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ptr_i64(arr: np.ndarray) -> ctypes.POINTER:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _ptr_u8(arr: np.ndarray) -> ctypes.POINTER:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

# ── Public reduction wrappers ────────────────────────────────────────────

def sum_f64_chunked(col: pa.ChunkedArray) -> float:
    total = 0.0
    for chunk in col.chunks:
        arr = _f64_np(chunk)
        total += float(_lib.fx_sum_f64(_ptr_f64(arr), len(arr)))  # type: ignore[union-attr]
    return total


def sum_i64_chunked(col: pa.ChunkedArray) -> int:
    total = 0
    for chunk in col.chunks:
        arr = _i64_np(chunk)
        total += int(_lib.fx_sum_i64(_ptr_i64(arr), len(arr)))  # type: ignore[union-attr]
    return total


def mean_f64_chunked(col: pa.ChunkedArray) -> float:
    """Welford-safe two-pass mean across chunks."""
    total = 0.0
    count = 0
    for chunk in col.chunks:
        arr = _f64_np(chunk)
        n = len(arr)
        total += float(_lib.fx_sum_f64(_ptr_f64(arr), n))  # type: ignore[union-attr]
        count += n
    return total / count if count > 0 else 0.0


def mean_i64_chunked(col: pa.ChunkedArray) -> float:
    total = 0
    count = 0
    for chunk in col.chunks:
        arr = _i64_np(chunk)
        total += int(_lib.fx_sum_i64(_ptr_i64(arr), len(arr)))  # type: ignore[union-attr]
        count += len(arr)
    return total / count if count > 0 else 0.0


def min_f64_chunked(col: pa.ChunkedArray) -> float:
    chunks_min = []
    for chunk in col.chunks:
        arr = _f64_np(chunk)
        chunks_min.append(float(_lib.fx_min_f64(_ptr_f64(arr), len(arr))))  # type: ignore[union-attr]
    return min(chunks_min)


def max_f64_chunked(col: pa.ChunkedArray) -> float:
    chunks_max = []
    for chunk in col.chunks:
        arr = _f64_np(chunk)
        chunks_max.append(float(_lib.fx_max_f64(_ptr_f64(arr), len(arr))))  # type: ignore[union-attr]
    return max(chunks_max)


def min_i64_chunked(col: pa.ChunkedArray) -> int:
    chunks_min = []
    for chunk in col.chunks:
        arr = _i64_np(chunk)
        chunks_min.append(int(_lib.fx_min_i64(_ptr_i64(arr), len(arr))))  # type: ignore[union-attr]
    return min(chunks_min)


def max_i64_chunked(col: pa.ChunkedArray) -> int:
    chunks_max = []
    for chunk in col.chunks:
        arr = _i64_np(chunk)
        chunks_max.append(int(_lib.fx_max_i64(_ptr_i64(arr), len(arr))))  # type: ignore[union-attr]
    return max(chunks_max)


def std_f64_chunked(col: pa.ChunkedArray, ddof: int = 1) -> float:
    """Population or sample std across all chunks (two-pass)."""
    all_np = np.concatenate([_f64_np(c) for c in col.chunks])
    return float(_lib.fx_std_f64(_ptr_f64(all_np), len(all_np), ddof))  # type: ignore[union-attr]


def var_f64_chunked(col: pa.ChunkedArray, ddof: int = 1) -> float:
    all_np = np.concatenate([_f64_np(c) for c in col.chunks])
    return float(_lib.fx_var_f64(_ptr_f64(all_np), len(all_np), ddof))  # type: ignore[union-attr]

# ── Public elementwise wrappers ──────────────────────────────────────────

def _apply_binary_f64(
    a: pa.ChunkedArray,
    b: pa.ChunkedArray,
    c_fn,
) -> pa.ChunkedArray:
    """Apply a C binary (arr, arr, out, n) kernel chunk-by-chunk."""
    # Combine to a single array first so shapes match trivially.
    a_np = np.concatenate([_f64_np(c) for c in a.chunks])
    b_np = np.concatenate([_f64_np(c) for c in b.chunks])
    if len(a_np) != len(b_np):
        raise ValueError(f"Array length mismatch: {len(a_np)} vs {len(b_np)}")
    out = np.empty(len(a_np), dtype=np.float64)
    c_fn(_ptr_f64(a_np), _ptr_f64(b_np), _ptr_f64(out), len(out))
    return pa.chunked_array([pa.array(out, type=pa.float64())])


def add_f64(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    return _apply_binary_f64(a, b, _lib.fx_add_f64)  # type: ignore[union-attr]


def sub_f64(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    return _apply_binary_f64(a, b, _lib.fx_sub_f64)  # type: ignore[union-attr]


def mul_f64(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    return _apply_binary_f64(a, b, _lib.fx_mul_f64)  # type: ignore[union-attr]


def div_f64(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    return _apply_binary_f64(a, b, _lib.fx_div_f64)  # type: ignore[union-attr]


def _apply_scalar_f64(
    a: pa.ChunkedArray,
    scalar: float,
    c_fn,
) -> pa.ChunkedArray:
    a_np = np.concatenate([_f64_np(c) for c in a.chunks])
    out = np.empty(len(a_np), dtype=np.float64)
    c_fn(_ptr_f64(a_np), ctypes.c_double(scalar), _ptr_f64(out), len(out))
    return pa.chunked_array([pa.array(out, type=pa.float64())])


def scalar_add_f64(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    return _apply_scalar_f64(a, scalar, _lib.fx_scalar_add_f64)  # type: ignore[union-attr]


def scalar_sub_f64(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    return _apply_scalar_f64(a, scalar, _lib.fx_scalar_sub_f64)  # type: ignore[union-attr]


def scalar_mul_f64(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    return _apply_scalar_f64(a, scalar, _lib.fx_scalar_mul_f64)  # type: ignore[union-attr]


def scalar_div_f64(a: pa.ChunkedArray, scalar: float) -> pa.ChunkedArray:
    return _apply_scalar_f64(a, scalar, _lib.fx_scalar_div_f64)  # type: ignore[union-attr]

# ── Column filter ────────────────────────────────────────────────────────

def filter_f64_column(
    col: pa.ChunkedArray,
    mask: pa.ChunkedArray,
) -> pa.ChunkedArray:
    """Filter a float64 column using the C kernel.

    ``mask`` must be a boolean ChunkedArray with the same total length.
    """
    src_np = np.concatenate([_f64_np(c) for c in col.chunks])
    # Arrow boolean arrays pack bits; convert to uint8 (1=keep, 0=drop).
    mask_np = np.concatenate(
        [c.to_numpy(zero_copy_only=False) for c in mask.chunks]
    ).astype(np.uint8)
    dst = np.empty(len(src_np), dtype=np.float64)
    k = int(_lib.fx_filter_f64(  # type: ignore[union-attr]
        _ptr_f64(src_np), _ptr_u8(mask_np), _ptr_f64(dst), len(src_np)
    ))
    return pa.chunked_array([pa.array(dst[:k], type=pa.float64())])


def filter_i64_column(
    col: pa.ChunkedArray,
    mask: pa.ChunkedArray,
) -> pa.ChunkedArray:
    src_np = np.concatenate([_i64_np(c) for c in col.chunks])
    mask_np = np.concatenate(
        [c.to_numpy(zero_copy_only=False) for c in mask.chunks]
    ).astype(np.uint8)
    dst = np.empty(len(src_np), dtype=np.int64)
    k = int(_lib.fx_filter_i64(  # type: ignore[union-attr]
        _ptr_i64(src_np), _ptr_u8(mask_np), _ptr_i64(dst), len(src_np)
    ))
    return pa.chunked_array([pa.array(dst[:k], type=pa.int64())])
