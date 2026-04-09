"""FrameX: High-performance parallel dataframe and array processing."""

from framex._version import __version__
from framex.config import (
    Config,
    config,
    get_config,
    set_backend,
    set_kernel_backend,
    set_serializer,
    set_workers,
)
from framex.core.array import NDArray
from framex.core.dataframe import DataFrame, LazyFrame
from framex.core.dtypes import DType
from framex.core.index import Index
from framex.core.series import Series
from framex.interchange.dataframe_protocol import (
    add_dataframe_protocol,
    from_dataframe,
    from_pandas,
)
from framex.io.arrow_ipc import read_ipc, write_ipc
from framex.io.csv import read_csv
from framex.io.parquet import read_parquet, write_parquet
from framex.ops.window import rolling_mean, rolling_sum, rolling_std, rolling_min, rolling_max, top_k, rank
from framex.runtime.executor import detect_backend
from framex.compat import list_divergences, check_pandas_compat, DIVERGENCES

# Apply the __dataframe__ interchange protocol to DataFrame.
add_dataframe_protocol(DataFrame)


def array(
    data: list | None = None,
    *,
    dtype: str | None = None,
    chunks: int | None = None,
) -> NDArray:
    """Convenience constructor for ``NDArray``."""
    return NDArray(data, dtype=dtype, chunks=chunks)


__all__ = [
    "__version__",
    # Config
    "Config",
    "config",
    "get_config",
    "set_backend",
    "set_kernel_backend",
    "set_serializer",
    "set_workers",
    # Core types
    "DataFrame",
    "DType",
    "Index",
    "LazyFrame",
    "NDArray",
    "Series",
    # Interchange
    "from_dataframe",
    "from_pandas",
    # IO
    "read_csv",
    "read_ipc",
    "read_parquet",
    "write_ipc",
    "write_parquet",
    # Convenience
    "array",
    # Window ops
    "rolling_mean",
    "rolling_sum",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "top_k",
    "rank",
    # Runtime
    "detect_backend",
    # Compatibility
    "DIVERGENCES",
    "check_pandas_compat",
    "list_divergences",
]
