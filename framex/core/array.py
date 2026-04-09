"""Chunked NDArray with Arrow backing and NumPy protocol support."""

from __future__ import annotations

import numbers
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from framex.backends.array_accel import evaluate_ufunc, maybe_numba_jit
from framex.config import get_config
from framex.core.dtypes import DType, resolve_dtype
from framex.runtime.executor import WorkerExecutor


# Sentinel for __array_ufunc__ / __array_function__ to signal NotImplemented
_NOT_IMPLEMENTED = NotImplemented


def _implements(numpy_function: Any) -> Any:
    """Register an ``__array_function__`` implementation."""

    def decorator(func: Any) -> Any:
        _HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


_HANDLED_FUNCTIONS: dict[Any, Any] = {}


def _apply_block_fn(
    fn: Callable[[np.ndarray[Any, Any]], Any],
    block: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    result = fn(block)
    return np.asarray(result)


class NDArray:
    """Chunked numeric array with Arrow backing.

    Each chunk is a ``pyarrow.Array``.  Supports NumPy ufunc dispatch
    (``__array_ufunc__``) and ``__array_function__`` protocol (NEP 13/18).
    """

    def __init__(
        self,
        data: Sequence[Any] | np.ndarray[Any, Any] | pa.Array | pa.ChunkedArray | list[pa.Array] | None = None,
        *,
        dtype: str | pa.DataType | DType | None = None,
        chunks: int | None = None,
    ):
        resolved = resolve_dtype(dtype)
        arrow_type = resolved.to_arrow() if resolved else None

        if data is None:
            arrow_type = arrow_type or pa.float64()
            self._chunks: list[pa.Array] = []
        elif isinstance(data, list) and data and isinstance(data[0], pa.Array):
            self._chunks = data  # type: ignore[assignment]
        elif isinstance(data, pa.ChunkedArray):
            self._chunks = data.chunks
        elif isinstance(data, pa.Array):
            self._chunks = [data]
        elif isinstance(data, np.ndarray):
            arr = pa.array(data, type=arrow_type)
            self._chunks = self._split(arr, chunks)
        else:
            arr = pa.array(list(data), type=arrow_type or pa.float64())
            self._chunks = self._split(arr, chunks)

        if arrow_type and self._chunks:
            self._chunks = [c.cast(arrow_type) if c.type != arrow_type else c for c in self._chunks]
        self._chunked: pa.ChunkedArray | None = None

    @staticmethod
    def _split(arr: pa.Array, chunks: int | None) -> list[pa.Array]:
        if chunks is None or chunks <= 0 or len(arr) <= chunks:
            return [arr]
        result: list[pa.Array] = []
        for start in range(0, len(arr), chunks):
            result.append(arr.slice(start, min(chunks, len(arr) - start)))
        return result

    # -- Properties ----------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(c) for c in self._chunks)

    @property
    def dtype(self) -> DType:
        if not self._chunks:
            return DType.from_arrow(pa.float64())
        return DType.from_arrow(self._chunks[0].type)

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    # -- Conversion ----------------------------------------------------------

    def to_numpy(self) -> np.ndarray[Any, Any]:
        if not self._chunks:
            return np.array([], dtype="float64")
        arrays = [c.to_numpy(zero_copy_only=False) for c in self._chunks]
        return np.concatenate(arrays)

    def to_pyarrow(self) -> pa.ChunkedArray:
        if not self._chunks:
            return pa.chunked_array([], type=pa.float64())
        if self._chunked is None:
            self._chunked = pa.chunked_array(self._chunks)
        return self._chunked

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        result = self.to_numpy()
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def apply_blocks(
        self,
        fn: Callable[[np.ndarray[Any, Any]], Any],
        *,
        workers: int | None = None,
        backend: str = "auto",
    ) -> NDArray:
        """Apply a function to each physical chunk, optionally in parallel.

        ``fn`` receives a NumPy view/copy for each chunk and must return a
        1-D array-like with the same length as that input chunk.
        """
        if not self._chunks:
            return NDArray([], dtype=self.dtype.arrow_type)

        cfg = get_config()
        max_workers = workers or cfg.workers
        if max_workers < 1:
            raise ValueError("workers must be >= 1")

        if backend == "auto":
            if pa.types.is_integer(self.dtype.arrow_type) or pa.types.is_floating(self.dtype.arrow_type):
                resolved_backend = "threads"
            else:
                resolved_backend = "processes"
        else:
            resolved_backend = backend

        np_blocks = [chunk.to_numpy(zero_copy_only=False) for chunk in self._chunks]

        if max_workers == 1 or len(np_blocks) == 1:
            out_blocks = [_apply_block_fn(fn, block) for block in np_blocks]
        else:
            with WorkerExecutor(max_workers=max_workers, backend=resolved_backend) as executor:
                futures = [executor.submit(_apply_block_fn, fn, block) for block in np_blocks]
                out_blocks = [f.result() for f in futures]

        out_chunks: list[pa.Array] = []
        for i, (inp, out) in enumerate(zip(np_blocks, out_blocks)):
            if out.ndim != 1:
                raise ValueError(f"Block function must return a 1-D array, got ndim={out.ndim} for block {i}")
            if len(out) != len(inp):
                raise ValueError(
                    f"Block function must preserve block length: expected {len(inp)}, got {len(out)} for block {i}"
                )
            out_chunks.append(pa.array(out))

        return NDArray(out_chunks)

    def parallel_map(
        self,
        fn: Callable[[np.ndarray[Any, Any]], Any],
        *,
        workers: int | None = None,
        backend: str = "auto",
    ) -> NDArray:
        """Alias for :meth:`apply_blocks`."""
        return self.apply_blocks(fn, workers=workers, backend=backend)

    def jit_apply(
        self,
        fn: Callable[[np.ndarray[Any, Any]], Any],
        *,
        workers: int | None = None,
        backend: str = "threads",
    ) -> NDArray:
        """Apply a Numba-jitted function across blocks when Numba is available."""
        return self.apply_blocks(maybe_numba_jit(fn), workers=workers, backend=backend)

    # -- Reductions ----------------------------------------------------------

    def sum(self) -> Any:
        return pc.sum(self.to_pyarrow()).as_py()

    def mean(self) -> Any:
        return pc.mean(self.to_pyarrow()).as_py()

    def min(self) -> Any:
        return pc.min(self.to_pyarrow()).as_py()

    def max(self) -> Any:
        return pc.max(self.to_pyarrow()).as_py()

    def std(self) -> Any:
        return pc.stddev(self.to_pyarrow()).as_py()

    # -- Arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> NDArray:
        return self._binary_op(pc.add, other)

    def __radd__(self, other: Any) -> NDArray:
        return self._binary_op(pc.add, other, reverse=True)

    def __sub__(self, other: Any) -> NDArray:
        return self._binary_op(pc.subtract, other)

    def __rsub__(self, other: Any) -> NDArray:
        return self._binary_op(pc.subtract, other, reverse=True)

    def __mul__(self, other: Any) -> NDArray:
        return self._binary_op(pc.multiply, other)

    def __rmul__(self, other: Any) -> NDArray:
        return self._binary_op(pc.multiply, other, reverse=True)

    def __truediv__(self, other: Any) -> NDArray:
        return self._binary_op(pc.divide, other)

    def _binary_op(self, op: Any, other: Any, *, reverse: bool = False) -> NDArray:
        if isinstance(other, NDArray):
            left = self.to_pyarrow()
            right = other.to_pyarrow()
        elif isinstance(other, (int, float)):
            left = self.to_pyarrow()
            right = other
        else:
            return NotImplemented  # type: ignore[return-value]
        if reverse:
            result = op(right, left)
        else:
            result = op(left, right)
        if isinstance(result, pa.ChunkedArray):
            return NDArray(result.chunks)
        return NDArray(result)

    # -- NumPy interop: __array_ufunc__ (NEP 13) ----------------------------

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented

        binary_arrow_ops = {
            np.add: pc.add,
            np.subtract: pc.subtract,
            np.multiply: pc.multiply,
            np.divide: pc.divide,
        }
        arrow_binary = binary_arrow_ops.get(ufunc)
        if callable(arrow_binary) and len(inputs) == 2 and not kwargs:
            left, right = inputs
            if isinstance(left, NDArray) and isinstance(right, (NDArray, numbers.Real)):
                return left._binary_op(arrow_binary, right)
            if isinstance(right, NDArray) and isinstance(left, numbers.Real):
                return right._binary_op(arrow_binary, left, reverse=True)

        if len(inputs) == 1 and isinstance(inputs[0], NDArray):
            unary_arrow_ops = {
                np.sin: getattr(pc, "sin", None),
                np.cos: getattr(pc, "cos", None),
                np.tan: getattr(pc, "tan", None),
                np.exp: getattr(pc, "exp", None),
                np.log: getattr(pc, "ln", None),
                np.sqrt: getattr(pc, "sqrt", None),
                np.negative: getattr(pc, "negate", None),
                np.absolute: getattr(pc, "abs", None),
                np.floor: getattr(pc, "floor", None),
                np.ceil: getattr(pc, "ceil", None),
            }
            arrow_op = unary_arrow_ops.get(ufunc)
            if callable(arrow_op) and not kwargs:
                result = arrow_op(inputs[0].to_pyarrow())
                if isinstance(result, pa.ChunkedArray):
                    return NDArray(result.chunks)
                if isinstance(result, pa.Array):
                    return NDArray(result)

        # Convert all inputs to numpy, compute, wrap back.
        np_inputs: list[Any] = []
        for inp in inputs:
            if isinstance(inp, NDArray):
                np_inputs.append(inp.to_numpy())
            else:
                np_inputs.append(np.asarray(inp))

        result = evaluate_ufunc(ufunc, np_inputs, kwargs)
        if isinstance(result, np.ndarray):
            return NDArray(result, dtype=str(result.dtype))
        return result

    # -- NumPy interop: __array_function__ (NEP 18) -------------------------

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    # -- Indexing -------------------------------------------------------------

    def __getitem__(self, key: int | slice) -> Any:
        ca = self.to_pyarrow()
        if isinstance(key, int):
            return ca[key].as_py()
        if isinstance(key, slice):
            np_arr = self.to_numpy()[key]
            return NDArray(np_arr)
        raise TypeError(f"Invalid key type: {type(key)}")

    # -- Display -------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self)
        if n > 10:
            head = self.to_numpy()[:5].tolist()
            tail = self.to_numpy()[-5:].tolist()
            body = f"{head} ... {tail}"
        elif n > 0:
            body = str(self.to_numpy().tolist())
        else:
            body = "[]"
        return f"NDArray(len={n}, chunks={self.num_chunks}, dtype={self.dtype}, data={body})"

    def __getattr__(self, name: str) -> Any:
        """Fallback to NumPy ndarray methods for missing NDArray APIs."""
        if name.startswith("_"):
            raise AttributeError(name)

        arr = self.to_numpy()
        attr = getattr(arr, name, None)
        if attr is None:
            raise AttributeError(f"'NDArray' object has no attribute {name!r}")

        if callable(attr):
            def _call(*args: Any, **kwargs: Any) -> Any:
                out = attr(*args, **kwargs)
                if isinstance(out, np.ndarray):
                    if out.ndim == 1:
                        return NDArray(out, dtype=str(out.dtype))
                    return out
                return out

            return _call
        return attr


# ---------------------------------------------------------------------------
# __array_function__ implementations
# ---------------------------------------------------------------------------


@_implements(np.sum)
def _np_sum(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return a.sum()
    return np.sum(np.asarray(a), *args, **kwargs)


@_implements(np.mean)
def _np_mean(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return a.mean()
    return np.mean(np.asarray(a), *args, **kwargs)


@_implements(np.min)
def _np_min(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return a.min()
    return np.min(np.asarray(a), *args, **kwargs)


@_implements(np.max)
def _np_max(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return a.max()
    return np.max(np.asarray(a), *args, **kwargs)


@_implements(np.std)
def _np_std(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return a.std()
    return np.std(np.asarray(a), *args, **kwargs)


@_implements(np.concatenate)
def _np_concatenate(arrays: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
    chunks: list[pa.Array] = []
    for a in arrays:
        if isinstance(a, NDArray):
            chunks.extend(a._chunks)
        else:
            chunks.append(pa.array(np.asarray(a)))
    return NDArray(chunks)


@_implements(np.where)
def _np_where(condition: Any, x: Any = None, y: Any = None, **kwargs: Any) -> Any:
    cond_np = condition.to_numpy() if isinstance(condition, NDArray) else np.asarray(condition)
    if x is None and y is None:
        # np.where(cond) → return indices
        return (NDArray(np.where(cond_np)[0]),)
    x_np = x.to_numpy() if isinstance(x, NDArray) else np.asarray(x)
    y_np = y.to_numpy() if isinstance(y, NDArray) else np.asarray(y)
    return NDArray(np.where(cond_np, x_np, y_np))


@_implements(np.argmin)
def _np_argmin(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return int(pc.index(a.to_pyarrow(), pc.min(a.to_pyarrow())))
    return np.argmin(np.asarray(a), *args, **kwargs)


@_implements(np.argmax)
def _np_argmax(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return int(pc.index(a.to_pyarrow(), pc.max(a.to_pyarrow())))
    return np.argmax(np.asarray(a), *args, **kwargs)


@_implements(np.unique)
def _np_unique(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        unique_arr = pc.unique(a.to_pyarrow())
        sorted_arr = pc.sort_indices(unique_arr)
        return NDArray(unique_arr.take(sorted_arr))
    return np.unique(np.asarray(a), *args, **kwargs)


@_implements(np.sort)
def _np_sort(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        ca = a.to_pyarrow()
        indices = pc.sort_indices(ca)
        return NDArray(ca.take(indices).chunks)
    return np.sort(np.asarray(a), *args, **kwargs)


@_implements(np.zeros_like)
def _np_zeros_like(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return NDArray(np.zeros(len(a), dtype=a.to_numpy().dtype))
    return np.zeros_like(np.asarray(a), *args, **kwargs)


@_implements(np.ones_like)
def _np_ones_like(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return NDArray(np.ones(len(a), dtype=a.to_numpy().dtype))
    return np.ones_like(np.asarray(a), *args, **kwargs)


@_implements(np.abs)
def _np_abs(a: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(a, NDArray):
        return NDArray(pc.abs(a.to_pyarrow()).chunks)
    return np.abs(np.asarray(a), *args, **kwargs)
