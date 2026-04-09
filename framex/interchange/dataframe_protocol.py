"""DataFrame interchange protocol (__dataframe__) and Pandas conversions."""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
import pyarrow.interchange as pai


def _is_pandas_dataframe(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "DataFrame" and cls.__module__.startswith(
        ("pandas.", "modin.pandas", "fireducks.pandas")
    )


def _is_dask_dataframe(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "DataFrame" and cls.__module__.startswith("dask.dataframe")


def _is_ray_dataset(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "Dataset" and cls.__module__.startswith("ray.data")


def _sanitize_pandas_attrs(pdf: Any) -> Any:
    attrs = getattr(pdf, "attrs", None)
    if not attrs:
        return pdf

    filtered: dict[str, Any] = {}
    changed = False
    for key, value in attrs.items():
        try:
            json.dumps(value)
            filtered[key] = value
        except (TypeError, ValueError):
            changed = True

    if not changed and len(filtered) == len(attrs):
        return pdf

    sanitized = pdf.copy(deep=False)
    sanitized.attrs = filtered
    return sanitized


def from_pandas(pdf: Any) -> Any:
    """Create a FrameX DataFrame from a Pandas DataFrame via Arrow."""
    from framex.core.dataframe import DataFrame

    return DataFrame(pa.Table.from_pandas(_sanitize_pandas_attrs(pdf), preserve_index=False))


def from_dask(ddf: Any) -> Any:
    """Create a FrameX DataFrame from a Dask DataFrame."""
    from framex.core.dataframe import DataFrame

    return DataFrame(ddf)


def from_ray(dataset: Any) -> Any:
    """Create a FrameX DataFrame from a Ray Dataset."""
    from framex.core.dataframe import DataFrame

    return DataFrame(dataset)


def from_dataframe(df_protocol: Any, *, allow_copy: bool = True) -> Any:
    """Create a FrameX DataFrame from any object implementing ``__dataframe__``.

    Supports:
    - Pandas DataFrames
    - Dask DataFrames
    - Ray Datasets
    - Any object with ``__dataframe__()`` method
    """
    from framex.core.dataframe import DataFrame

    if _is_pandas_dataframe(df_protocol):
        return from_pandas(df_protocol)
    if _is_dask_dataframe(df_protocol):
        return from_dask(df_protocol)
    if _is_ray_dataset(df_protocol):
        return from_ray(df_protocol)

    # Convert any interchange protocol producer to Arrow table without Pandas.
    if hasattr(df_protocol, "__dataframe__"):
        interchange_obj = df_protocol.__dataframe__(allow_copy=allow_copy)
        table = pai.from_dataframe(interchange_obj, allow_copy=allow_copy)
        return DataFrame(table)

    raise TypeError(
        f"Cannot create DataFrame from {type(df_protocol)}.  "
        f"Object must be pandas/dask/ray dataframe-like or implement __dataframe__()."
    )


def add_dataframe_protocol(cls: type) -> type:
    """Class decorator that adds ``__dataframe__`` to a DataFrame class.

    This is applied to ``framex.core.dataframe.DataFrame`` at import time.
    """

    def __dataframe__(self: Any, nan_as_null: bool = False, allow_copy: bool = True) -> Any:
        """Return an Arrow-backed dataframe interchange object."""
        # DataFrames and Tables expose __dataframe__ in pyarrow.
        return self.to_arrow().__dataframe__(allow_copy=allow_copy)

    cls.__dataframe__ = __dataframe__
    return cls
