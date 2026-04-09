"""Window operations: rolling aggregations, top-k, and rank.

These are the most commonly needed window ops in analytics pipelines:
- ``rolling_*``: sliding window reductions on a Series
- ``top_k``: return the k rows with the largest/smallest values of a column
- ``rank``: assign row ranks within a Series

All functions operate on Arrow types and return Arrow / FrameX types so
they compose naturally with the rest of the library.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import numpy as np


# ── Rolling window helpers ────────────────────────────────────────────────

def _rolling_apply(
    col: pa.ChunkedArray,
    window: int,
    fn,  # numpy reduction callable
    *,
    min_periods: int | None = None,
) -> pa.ChunkedArray:
    """Apply a numpy reduction over a sliding window.

    Leading ``window-1`` values are null when ``min_periods`` is not set or
    equals ``window``.  Use ``min_periods=1`` to compute from the first row.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_periods is None:
        min_periods = window

    values = col.to_pylist()
    n = len(values)
    result: list[Any] = [None] * n

    for i in range(n):
        start = max(0, i - window + 1)
        window_vals = [v for v in values[start : i + 1] if v is not None]
        if len(window_vals) >= min_periods:
            result[i] = fn(window_vals)

    dtype = col.type
    # Numeric result types may differ; let Arrow infer.
    return pa.chunked_array([pa.array(result, type=pa.float64() if pa.types.is_integer(dtype) else dtype)])


def rolling_mean(
    series: Any,
    window: int,
    min_periods: int | None = None,
) -> Any:
    """Rolling mean with a given window size.

    Parameters
    ----------
    series : framex.Series
        Input series (numeric).
    window : int
        Number of rows in the rolling window.
    min_periods : int | None
        Minimum non-null observations required to produce a result.
        Defaults to ``window``.

    Returns
    -------
    framex.Series
        Series of rolling means (nulls where insufficient data).
    """
    from framex.core.series import Series

    result = _rolling_apply(
        series.to_pyarrow(), window, np.mean, min_periods=min_periods
    )
    return Series(result, name=series.name)


def rolling_sum(
    series: Any,
    window: int,
    min_periods: int | None = None,
) -> Any:
    """Rolling sum."""
    from framex.core.series import Series

    result = _rolling_apply(
        series.to_pyarrow(), window, sum, min_periods=min_periods
    )
    return Series(result, name=series.name)


def rolling_std(
    series: Any,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> Any:
    """Rolling standard deviation."""
    from framex.core.series import Series

    def _std(vals: list[float]) -> float:
        if len(vals) <= ddof:
            return float("nan")
        return float(np.std(vals, ddof=ddof))

    result = _rolling_apply(
        series.to_pyarrow(), window, _std, min_periods=min_periods
    )
    return Series(result, name=series.name)


def rolling_min(
    series: Any,
    window: int,
    min_periods: int | None = None,
) -> Any:
    """Rolling minimum."""
    from framex.core.series import Series

    result = _rolling_apply(
        series.to_pyarrow(), window, min, min_periods=min_periods
    )
    return Series(result, name=series.name)


def rolling_max(
    series: Any,
    window: int,
    min_periods: int | None = None,
) -> Any:
    """Rolling maximum."""
    from framex.core.series import Series

    result = _rolling_apply(
        series.to_pyarrow(), window, max, min_periods=min_periods
    )
    return Series(result, name=series.name)


# ── Top-K ─────────────────────────────────────────────────────────────────

def top_k(
    df: Any,
    k: int,
    by: str | list[str],
    ascending: bool = False,
) -> Any:
    """Return the top-k rows by one or more columns.

    Parameters
    ----------
    df : framex.DataFrame
        Input DataFrame.
    k : int
        Number of rows to return.
    by : str | list[str]
        Column(s) to sort by.
    ascending : bool
        ``False`` (default) returns the largest values; ``True`` returns
        the smallest (bottom-k).

    Returns
    -------
    framex.DataFrame
        k rows with extreme values, sorted.
    """
    from framex.core.dataframe import DataFrame

    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")

    table = df.to_arrow()
    if isinstance(by, str):
        by = [by]

    order = "ascending" if ascending else "descending"
    sort_keys = [(col, order) for col in by]
    indices = pc.sort_indices(table, sort_keys=sort_keys)
    top_indices = indices.slice(0, min(k, len(indices)))
    return DataFrame(table.take(top_indices))


# ── Rank ──────────────────────────────────────────────────────────────────

def rank(
    series: Any,
    method: str = "average",
    ascending: bool = True,
) -> Any:
    """Assign ranks to values in a Series.

    Parameters
    ----------
    series : framex.Series
        Input numeric or ordinal series.
    method : str
        How to handle ties:
        - ``"average"`` — mean rank of tied group
        - ``"min"`` — lowest rank in the tied group
        - ``"max"`` — highest rank in the tied group
        - ``"dense"`` — like min, but no gaps between ranks
        - ``"first"`` — ranks in order of first appearance
    ascending : bool
        ``True`` (default) ranks smallest first.

    Returns
    -------
    framex.Series
        Float64 Series of rank values.
    """
    from framex.core.series import Series

    values = series.to_pyarrow().to_pylist()
    n = len(values)

    if ascending:
        order = sorted(range(n), key=lambda i: (values[i] is None, values[i]))
    else:
        order = sorted(
            range(n),
            key=lambda i: (values[i] is None, values[i] if values[i] is None else -values[i]),  # type: ignore[operator]
        )

    ranks: list[float | None] = [None] * n

    if method == "first":
        for rank_val, original_idx in enumerate(order, start=1):
            if values[original_idx] is None:
                ranks[original_idx] = None
            else:
                ranks[original_idx] = float(rank_val)

    elif method in ("min", "max", "average", "dense"):
        # Group consecutive tied values.
        i = 0
        dense_rank = 0
        while i < n:
            idx = order[i]
            if values[idx] is None:
                i += 1
                continue
            dense_rank += 1
            j = i + 1
            while j < n and values[order[j]] == values[idx]:
                j += 1
            # Rows order[i:j] are tied.
            group_indices = [order[k] for k in range(i, j)]
            if method == "min":
                r: float = float(i + 1)
            elif method == "max":
                r = float(j)
            elif method == "average":
                r = (i + 1 + j) / 2.0
            else:  # dense
                r = float(dense_rank)
            for gi in group_indices:
                ranks[gi] = r
            i = j
    else:
        raise ValueError(
            f"Unknown rank method {method!r}. "
            "Use 'average', 'min', 'max', 'dense', or 'first'."
        )

    return Series(
        pa.chunked_array([pa.array(ranks, type=pa.float64())]),
        name=series.name,
    )
