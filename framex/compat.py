"""FrameX compatibility policy — explicit divergences from Pandas and NumPy.

This module documents every place where FrameX intentionally behaves
differently from Pandas or NumPy.  Knowing *why* avoids silent surprises
and lets callers write migration shims when they need Pandas semantics.

Usage::

    from framex.compat import DIVERGENCES, check_pandas_compat

    # Print all documented divergences
    for d in DIVERGENCES:
        print(d)

    # Raise if a feature is known-incompatible
    check_pandas_compat("copy-on-write")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Divergence:
    """A single documented difference from a reference library."""

    feature: str
    """Short name of the Pandas/NumPy feature."""

    reference: Literal["pandas", "numpy"]
    """Which library this divergence is from."""

    framex_behaviour: str
    """What FrameX does instead."""

    pandas_behaviour: str | None = None
    """What Pandas / NumPy does (for context)."""

    migration: str | None = None
    """How a caller can get the Pandas/NumPy behaviour if they need it."""

    def __str__(self) -> str:
        lines = [
            f"[{self.reference}] {self.feature}",
            f"  FrameX : {self.framex_behaviour}",
        ]
        if self.pandas_behaviour:
            lines.append(f"  Pandas : {self.pandas_behaviour}")
        if self.migration:
            lines.append(f"  Migrate: {self.migration}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry of all documented divergences
# ---------------------------------------------------------------------------

DIVERGENCES: list[Divergence] = [

    Divergence(
        feature="Copy-on-write semantics",
        reference="pandas",
        framex_behaviour=(
            "All operations return new DataFrames/Series; FrameX objects are "
            "immutable from the user's perspective.  There is no view/"
            "copy distinction — every transform is explicit."
        ),
        pandas_behaviour=(
            "Pandas 3.0+ uses Copy-on-Write by default, producing lazy copies "
            "that only materialise on mutation."
        ),
        migration="FrameX already behaves consistently — no migration needed.",
    ),

    Divergence(
        feature="In-place modification (df['col'] = value)",
        reference="pandas",
        framex_behaviour=(
            "Not supported.  Use ``df.with_column('col', series)`` or "
            "``df.assign(col=...)`` to produce a new DataFrame."
        ),
        pandas_behaviour="Mutates the DataFrame in place (deprecated under CoW).",
        migration="Replace ``df['col'] = val`` with ``df = df.with_column('col', val)``.",
    ),

    Divergence(
        feature="Index alignment on binary operations",
        reference="pandas",
        framex_behaviour=(
            "Series/DataFrame arithmetic does NOT align on index labels.  "
            "Operands must have the same length; positional alignment is used."
        ),
        pandas_behaviour=(
            "Pandas aligns on index before any binary op, inserting NaN where "
            "labels do not match."
        ),
        migration=(
            "Sort both operands by the same key column before operating, or "
            "use ``df.join()`` to align explicitly."
        ),
    ),

    Divergence(
        feature="MultiIndex / hierarchical index",
        reference="pandas",
        framex_behaviour=(
            "Not supported.  FrameX uses a flat integer row position as the "
            "implicit index.  Hierarchical grouping is done via groupby keys."
        ),
        pandas_behaviour="Pandas supports arbitrary MultiIndex on both rows and columns.",
        migration=(
            "Use ``df.groupby([level0_col, level1_col]).agg(...)`` instead of "
            "constructing a MultiIndex DataFrame."
        ),
    ),

    Divergence(
        feature="DataFrame.apply with axis=1 (row-wise)",
        reference="pandas",
        framex_behaviour=(
            "Not directly supported.  ``Series.apply`` works column-wise. "
            "Row-wise apply requires iterating with Python and is intentionally "
            "discouraged for performance."
        ),
        pandas_behaviour="``df.apply(fn, axis=1)`` applies fn to each row as a Series.",
        migration=(
            "Prefer vectorised column operations or ``df.assign``.  "
            "For unavoidable row-wise work: ``pd_df.apply(fn, axis=1)`` on the "
            "Pandas-converted frame."
        ),
    ),

    Divergence(
        feature="Nullable integer dtypes (Int8, Int16, …)",
        reference="pandas",
        framex_behaviour=(
            "Uses Arrow's native nullable integer representation internally.  "
            "Exposed to users as standard int8/int16/int32/int64 Arrow types."
        ),
        pandas_behaviour=(
            "Pandas has both numpy int (non-nullable) and extension Int (nullable) "
            "dtypes with different behaviour on NaN propagation."
        ),
        migration=(
            "FrameX integers are always nullable (null ≠ NaN).  "
            "Call ``.to_pandas()`` to recover Pandas nullable or numpy integer "
            "columns depending on null presence."
        ),
    ),

    Divergence(
        feature="NaN vs null distinction",
        reference="pandas",
        framex_behaviour=(
            "Uses Arrow's explicit null sentinel for all missing data.  "
            "There is no float NaN / None ambiguity — nulls are typed."
        ),
        pandas_behaviour=(
            "Pandas uses float NaN for numeric missing data, None for object "
            "columns, and pd.NA for extension types — three different sentinels."
        ),
        migration=(
            "Use ``series.is_null()`` / ``df.dropna()`` / ``df.fillna()`` "
            "which handle Arrow nulls correctly.  ``np.nan`` in user input is "
            "converted to Arrow null on construction."
        ),
    ),

    Divergence(
        feature="groupby sort order",
        reference="pandas",
        framex_behaviour=(
            "groupby result row order is not guaranteed (depends on Arrow's "
            "hash-groupby implementation).  Sort explicitly if needed."
        ),
        pandas_behaviour="Pandas groupby returns groups in sorted key order by default.",
        migration=(
            "Chain ``.sort('key_col')`` after groupby if a stable order is required."
        ),
    ),

    Divergence(
        feature="Rolling window with non-numeric columns",
        reference="pandas",
        framex_behaviour=(
            "Rolling ops (``rolling_mean`` etc.) only support numeric columns.  "
            "Passing a string column raises ValueError."
        ),
        pandas_behaviour="Pandas rolling skips non-numeric columns silently.",
        migration="Select numeric columns before rolling.",
    ),

    Divergence(
        feature="ndarray shape (multi-dimensional)",
        reference="numpy",
        framex_behaviour=(
            "``NDArray`` is always 1-D.  Multi-dimensional array operations "
            "fall back to NumPy via ``__array_ufunc__``."
        ),
        pandas_behaviour=None,
        migration=(
            "Convert to NumPy for 2-D+ work: ``arr.to_numpy().reshape(n, m)``."
        ),
    ),

    Divergence(
        feature="Serialization default (pickle vs Arrow IPC)",
        reference="pandas",
        framex_behaviour=(
            "Arrow IPC is the default serialization format for inter-process "
            "transport (``serializer='arrow'``).  Pickle is opt-in and must be "
            "explicitly configured with ``fx.set_serializer('pickle5')``."
        ),
        pandas_behaviour=(
            "Pandas pickles objects by default (e.g. ``pd.read_pickle``, "
            "joblib.dump)."
        ),
        migration=(
            "Use ``fx.set_serializer('pickle5')`` if your pipeline depends on "
            "pickle for Python-native objects.  Arrow IPC is safer across "
            "trust boundaries."
        ),
    ),
]

# Index by feature name for O(1) lookup.
_DIVERGENCE_INDEX: dict[str, Divergence] = {
    d.feature.lower().replace(" ", "-"): d for d in DIVERGENCES
}


def check_pandas_compat(feature_slug: str) -> None:
    """Raise ``NotImplementedError`` if ``feature_slug`` is a known incompatibility.

    Parameters
    ----------
    feature_slug : str
        A slug matching a registered divergence feature name
        (case-insensitive, spaces become hyphens).

    Example
    -------
    >>> check_pandas_compat("multiindex")
    NotImplementedError: ...
    """
    key = feature_slug.lower().replace(" ", "-")
    match = _DIVERGENCE_INDEX.get(key)
    if match is None:
        return  # No known divergence — assume compatible.
    msg = (
        f"FrameX diverges from Pandas on '{match.feature}'.\n"
        f"FrameX behaviour: {match.framex_behaviour}"
    )
    if match.migration:
        msg += f"\nMigration hint: {match.migration}"
    raise NotImplementedError(msg)


def list_divergences(reference: Literal["pandas", "numpy"] | None = None) -> list[Divergence]:
    """Return all documented divergences, optionally filtered by reference library."""
    if reference is None:
        return list(DIVERGENCES)
    return [d for d in DIVERGENCES if d.reference == reference]
