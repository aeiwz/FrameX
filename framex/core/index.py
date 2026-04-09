"""Arrow-backed Index for DataFrame row labels."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pyarrow as pa


class Index:
    """Immutable, Arrow-backed row index.

    Backed by a ``pyarrow.ChunkedArray`` internally.  Supports integer
    (RangeIndex-style) and label-based indices.
    """

    def __init__(self, data: pa.ChunkedArray | pa.Array | Sequence[Any] | None = None, *, name: str | None = None):
        if data is None:
            self._data = pa.chunked_array([], type=pa.int64())
        elif isinstance(data, pa.ChunkedArray):
            self._data = data
        elif isinstance(data, pa.Array):
            self._data = pa.chunked_array([data])
        else:
            self._data = pa.chunked_array([pa.array(list(data))])
        self.name = name

    # -- Factories -----------------------------------------------------------

    @classmethod
    def range(cls, n: int, *, name: str | None = None) -> Index:
        """Create a ``RangeIndex``-style monotonic integer index."""
        arr = pa.array(range(n), type=pa.int64())
        return cls(arr, name=name)

    # -- Properties ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    @property
    def dtype(self) -> pa.DataType:
        return self._data.type

    def to_pyarrow(self) -> pa.ChunkedArray:
        return self._data

    def to_pandas(self) -> Any:
        import pandas as pd

        return pd.Index(self._data.to_pylist(), name=self.name)

    def to_numpy(self) -> np.ndarray[Any, Any]:
        return self._data.to_numpy()

    def to_pylist(self) -> list[Any]:
        return self._data.to_pylist()

    # -- Slicing -------------------------------------------------------------

    def take(self, indices: Sequence[int] | pa.Array) -> Index:
        if not isinstance(indices, pa.Array):
            indices = pa.array(indices, type=pa.int64())
        return Index(self._data.take(indices), name=self.name)

    def slice(self, start: int, length: int | None = None) -> Index:
        return Index(self._data.slice(start, length), name=self.name)

    # -- Comparison / lookup -------------------------------------------------

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Index):
            return self._data.equals(other._data)
        return NotImplemented

    def __repr__(self) -> str:
        values = self._data.to_pylist()
        if len(values) > 6:
            head = values[:3]
            tail = values[-3:]
            body = f"{head} ... {tail}"
        else:
            body = str(values)
        return f"Index({body}, name={self.name!r}, dtype={self.dtype})"
