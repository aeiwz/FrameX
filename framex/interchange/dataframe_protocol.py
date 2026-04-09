"""DataFrame interchange protocol (__dataframe__) and Pandas conversions."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _sanitize_pandas_attrs(pdf: pd.DataFrame) -> pd.DataFrame:
    """Return ``pdf`` with JSON-serializable ``attrs`` only.

    Arrow conversion serializes ``DataFrame.attrs`` into schema metadata.
    Non-serializable values (for example pandas internal buffers created by
    interchange consumers) trigger warnings; we drop those keys.
    """
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


def from_pandas(pdf: pd.DataFrame) -> Any:
    """Create a FrameX DataFrame from a Pandas DataFrame via Arrow."""
    from framex.core.dataframe import DataFrame

    return DataFrame(_sanitize_pandas_attrs(pdf))


def from_dataframe(df_protocol: Any, *, allow_copy: bool = True) -> Any:
    """Create a FrameX DataFrame from any object implementing ``__dataframe__``.

    Supports:
    - Pandas DataFrames (via ``pd.api.interchange.from_dataframe``)
    - Any object with ``__dataframe__()`` method
    """
    from framex.core.dataframe import DataFrame

    if isinstance(df_protocol, pd.DataFrame):
        return DataFrame(_sanitize_pandas_attrs(df_protocol))

    # If the object has __dataframe__, use Pandas interchange to get a pd.DataFrame
    # then convert to FrameX.
    if hasattr(df_protocol, "__dataframe__"):
        interchange_obj = df_protocol.__dataframe__(allow_copy=allow_copy)
        # Use pandas to consume the interchange protocol object.
        pdf = pd.api.interchange.from_dataframe(interchange_obj, allow_copy=allow_copy)
        return DataFrame(_sanitize_pandas_attrs(pdf))

    raise TypeError(
        f"Cannot create DataFrame from {type(df_protocol)}.  "
        f"Object must be a pandas DataFrame or implement __dataframe__()."
    )


def add_dataframe_protocol(cls: type) -> type:
    """Class decorator that adds ``__dataframe__`` to a DataFrame class.

    This is applied to ``framex.core.dataframe.DataFrame`` at import time.
    """

    def __dataframe__(self: Any, nan_as_null: bool = False, allow_copy: bool = True) -> Any:
        """Return a Pandas interchange-protocol object."""
        pdf = self.to_pandas()
        return pdf.__dataframe__(nan_as_null=nan_as_null, allow_copy=allow_copy)

    cls.__dataframe__ = __dataframe__
    return cls
