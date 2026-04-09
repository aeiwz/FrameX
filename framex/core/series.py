"""Arrow-backed Series (single column)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from framex.core.dtypes import DType, resolve_dtype


class Series:
    """A single named column backed by ``pyarrow.ChunkedArray``."""

    def __init__(
        self,
        data: pa.ChunkedArray | pa.Array | Sequence[Any] | np.ndarray[Any, Any] | None = None,
        *,
        name: str | None = None,
        dtype: str | pa.DataType | DType | None = None,
    ):
        resolved = resolve_dtype(dtype)
        arrow_type = resolved.to_arrow() if resolved else None

        if data is None:
            arrow_type = arrow_type or pa.null()
            self._data = pa.chunked_array([], type=arrow_type)
        elif isinstance(data, pa.ChunkedArray):
            self._data = data.cast(arrow_type) if arrow_type else data
        elif isinstance(data, pa.Array):
            arr = data.cast(arrow_type) if arrow_type else data
            self._data = pa.chunked_array([arr])
        elif isinstance(data, np.ndarray):
            arr = pa.array(data, type=arrow_type)
            self._data = pa.chunked_array([arr])
        else:
            arr = pa.array(list(data), type=arrow_type)
            self._data = pa.chunked_array([arr])

        self.name = name

    # -- Properties ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    @property
    def dtype(self) -> DType:
        return DType.from_arrow(self._data.type)

    @property
    def arrow_type(self) -> pa.DataType:
        return self._data.type

    def to_pyarrow(self) -> pa.ChunkedArray:
        return self._data

    def to_numpy(self) -> np.ndarray[Any, Any]:
        return self._data.to_numpy()

    def to_pandas(self) -> Any:
        import pandas as pd

        return pd.Series(self._data.to_pylist(), name=self.name)

    def to_pylist(self) -> list[Any]:
        return self._data.to_pylist()

    # -- Scalar reductions ---------------------------------------------------

    def sum(self) -> Any:
        return pc.sum(self._data).as_py()

    def mean(self) -> Any:
        return pc.mean(self._data).as_py()

    def min(self) -> Any:
        return pc.min(self._data).as_py()

    def max(self) -> Any:
        return pc.max(self._data).as_py()

    def count(self) -> int:
        return pc.count(self._data).as_py()

    def std(self) -> Any:
        return pc.stddev(self._data).as_py()

    def var(self) -> Any:
        return pc.variance(self._data).as_py()

    def nunique(self) -> int:
        """Number of distinct non-null values."""
        return pc.count(pc.unique(self._data), count_mode="only_valid").as_py()

    def unique(self) -> Series:
        """Return distinct values (order not guaranteed)."""
        return Series(pc.unique(self._data), name=self.name)

    def value_counts(self) -> Any:
        """Return a DataFrame of (value, count) pairs sorted by count descending."""
        from framex.core.dataframe import DataFrame

        result = self._data.value_counts()
        # value_counts() returns a StructArray; convert to dict
        values = pc.struct_field(result, "values")
        counts = pc.struct_field(result, "counts")
        # Sort by count descending
        order = pc.sort_indices(counts, sort_keys=[("x", "descending")])
        col_name = self.name or "value"
        return DataFrame({col_name: values.take(order), "count": counts.take(order)})

    # -- Element-wise ops ----------------------------------------------------

    def map(self, fn: Callable[[Any], Any]) -> Series:
        values = [fn(v) for v in self._data.to_pylist()]
        return Series(values, name=self.name)

    def apply(self, fn: Callable[[Any], Any]) -> Series:
        """Alias for ``map`` for Pandas-like ergonomics."""
        return self.map(fn)

    def abs(self) -> Series:
        return Series(pc.abs(self._data), name=self.name)

    def round(self, decimals: int = 0) -> Series:
        return Series(pc.round(self._data, ndigits=decimals), name=self.name)

    def clip(self, lower: Any = None, upper: Any = None) -> Series:
        result = self._data
        if lower is not None:
            result = pc.max_element_wise(result, lower)
        if upper is not None:
            result = pc.min_element_wise(result, upper)
        return Series(result, name=self.name)

    def dropna(self) -> Series:
        """Remove null values."""
        mask = pc.invert(pc.is_null(self._data))
        return Series(pc.filter(self._data, mask), name=self.name)

    def is_null(self) -> Series:
        return Series(pc.is_null(self._data), name=self.name)

    def fill_null(self, value: Any) -> Series:
        return Series(pc.fill_null(self._data, value), name=self.name)

    def cast(self, dtype: str | pa.DataType | DType) -> Series:
        resolved = resolve_dtype(dtype)
        assert resolved is not None
        return Series(self._data.cast(resolved.to_arrow()), name=self.name)

    # -- Predicates ----------------------------------------------------------

    def isin(self, values: Sequence[Any]) -> Series:
        value_set = pa.array(values, type=self._data.type)
        return Series(pc.is_in(self._data, value_set=value_set), name=self.name)

    def __eq__(self, other: Any) -> Series:  # type: ignore[override]
        if isinstance(other, Series):
            return Series(pc.equal(self._data, other._data), name=self.name)
        return Series(pc.equal(self._data, other), name=self.name)

    def __ne__(self, other: Any) -> Series:  # type: ignore[override]
        if isinstance(other, Series):
            return Series(pc.not_equal(self._data, other._data), name=self.name)
        return Series(pc.not_equal(self._data, other), name=self.name)

    def __gt__(self, other: Any) -> Series:
        if isinstance(other, Series):
            return Series(pc.greater(self._data, other._data), name=self.name)
        return Series(pc.greater(self._data, other), name=self.name)

    def __ge__(self, other: Any) -> Series:
        if isinstance(other, Series):
            return Series(pc.greater_equal(self._data, other._data), name=self.name)
        return Series(pc.greater_equal(self._data, other), name=self.name)

    def __lt__(self, other: Any) -> Series:
        if isinstance(other, Series):
            return Series(pc.less(self._data, other._data), name=self.name)
        return Series(pc.less(self._data, other), name=self.name)

    def __le__(self, other: Any) -> Series:
        if isinstance(other, Series):
            return Series(pc.less_equal(self._data, other._data), name=self.name)
        return Series(pc.less_equal(self._data, other), name=self.name)

    # -- Arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> Series:
        rhs = other._data if isinstance(other, Series) else other
        return Series(pc.add(self._data, rhs), name=self.name)

    def __sub__(self, other: Any) -> Series:
        rhs = other._data if isinstance(other, Series) else other
        return Series(pc.subtract(self._data, rhs), name=self.name)

    def __mul__(self, other: Any) -> Series:
        rhs = other._data if isinstance(other, Series) else other
        return Series(pc.multiply(self._data, rhs), name=self.name)

    def __truediv__(self, other: Any) -> Series:
        rhs = other._data if isinstance(other, Series) else other
        return Series(pc.divide(self._data, rhs), name=self.name)

    # -- Logical ops ---------------------------------------------------------

    def __and__(self, other: Any) -> Series:
        rhs = other._data if isinstance(other, Series) else other
        return Series(pc.and_(self._data, rhs), name=self.name)

    def __or__(self, other: Any) -> Series:
        rhs = other._data if isinstance(other, Series) else other
        return Series(pc.or_(self._data, rhs), name=self.name)

    def __invert__(self) -> Series:
        return Series(pc.invert(self._data), name=self.name)

    # -- Display -------------------------------------------------------------

    def __repr__(self) -> str:
        values = self._data.to_pylist()
        if len(values) > 10:
            head = values[:5]
            tail = values[-5:]
            body = f"{head} ... {tail}"
        else:
            body = str(values)
        return f"Series(name={self.name!r}, dtype={self.arrow_type}, len={len(self)}, data={body})"

    def __getitem__(self, key: int | slice) -> Any:
        if isinstance(key, int):
            return self._data[key].as_py()
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            indices = range(start, stop, step)
            return Series(self._data.take(pa.array(list(indices), type=pa.int64())), name=self.name)
        raise TypeError(f"Invalid key type: {type(key)}")
