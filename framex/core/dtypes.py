"""DType system wrapping Arrow data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa


# Mapping from string shorthand to Arrow types.
_ARROW_TYPE_MAP: dict[str, pa.DataType] = {
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "string": pa.string(),
    "utf8": pa.string(),
    "binary": pa.binary(),
    "date32": pa.date32(),
    "date64": pa.date64(),
    "timestamp": pa.timestamp("us"),
}


@dataclass(frozen=True)
class DType:
    """Thin wrapper over ``pyarrow.DataType`` with convenience conversions."""

    arrow_type: pa.DataType

    # -- Construction helpers ------------------------------------------------

    @classmethod
    def from_string(cls, name: str) -> DType:
        arrow_t = _ARROW_TYPE_MAP.get(name)
        if arrow_t is None:
            raise ValueError(f"Unknown dtype string: {name!r}")
        return cls(arrow_type=arrow_t)

    @classmethod
    def from_numpy(cls, np_dtype: np.dtype[Any]) -> DType:
        return cls(arrow_type=pa.from_numpy_dtype(np_dtype))

    @classmethod
    def from_arrow(cls, arrow_type: pa.DataType) -> DType:
        return cls(arrow_type=arrow_type)

    # -- Conversions ---------------------------------------------------------

    def to_numpy(self) -> np.dtype[Any]:
        return self.arrow_type.to_pandas_dtype()  # type: ignore[return-value]

    def to_arrow(self) -> pa.DataType:
        return self.arrow_type

    # -- Display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DType({self.arrow_type})"

    def __str__(self) -> str:
        return str(self.arrow_type)


# Module-level convenience functions.


def from_arrow(arrow_type: pa.DataType) -> DType:
    return DType.from_arrow(arrow_type)


def to_arrow(dtype: DType) -> pa.DataType:
    return dtype.to_arrow()


def resolve_dtype(dtype: str | np.dtype[Any] | pa.DataType | DType | None) -> DType | None:
    """Resolve an arbitrary dtype specification to a ``DType`` or ``None``."""
    if dtype is None:
        return None
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, pa.DataType):
        return DType.from_arrow(dtype)
    if isinstance(dtype, np.dtype):
        return DType.from_numpy(dtype)
    if isinstance(dtype, str):
        return DType.from_string(dtype)
    raise TypeError(f"Cannot resolve dtype from {type(dtype)}")
