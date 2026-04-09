"""CSV reader via PyArrow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pcsv


def read_csv(
    path: str | Path,
    **kwargs: Any,
) -> Any:
    """Read a CSV file into a FrameX DataFrame."""
    from framex.core.dataframe import DataFrame

    table = pcsv.read_csv(str(path), **kwargs)
    return DataFrame(table)
