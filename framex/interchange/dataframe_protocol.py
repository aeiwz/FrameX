"""DataFrame interchange protocol (__dataframe__) and Pandas conversions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa


def from_pandas(pdf: pd.DataFrame) -> Any:
    """Create a FrameX DataFrame from a Pandas DataFrame via Arrow."""
    from framex.core.dataframe import DataFrame

    return DataFrame(pdf)


def from_dataframe(df_protocol: Any, *, allow_copy: bool = True) -> Any:
    """Create a FrameX DataFrame from any object implementing ``__dataframe__``.

    Supports:
    - Pandas DataFrames (via ``pd.api.interchange.from_dataframe``)
    - Any object with ``__dataframe__()`` method
    """
    from framex.core.dataframe import DataFrame

    if isinstance(df_protocol, pd.DataFrame):
        return DataFrame(df_protocol)

    # If the object has __dataframe__, use Pandas interchange to get a pd.DataFrame
    # then convert to FrameX.
    if hasattr(df_protocol, "__dataframe__"):
        interchange_obj = df_protocol.__dataframe__(allow_copy=allow_copy)
        # Use pandas to consume the interchange protocol object.
        pdf = pd.api.interchange.from_dataframe(interchange_obj, allow_copy=allow_copy)
        return DataFrame(pdf)

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
