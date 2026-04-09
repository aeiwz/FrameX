"""GroupBy + aggregation operations using Arrow compute."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


# Map user-facing aggregation names to Arrow's names.
_AGG_FUNC_MAP: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "count": "count",
    "std": "stddev",
    "count_distinct": "count_distinct",
}


def groupby_agg(
    table: pa.Table,
    keys: list[str],
    aggregations: dict[str, str | list[str]],
) -> pa.Table:
    """Perform group-by aggregation on an Arrow Table.

    Parameters
    ----------
    table : pa.Table
        Input data.
    keys : list[str]
        Column names to group by.
    aggregations : dict[str, str | list[str]]
        Mapping of ``{column: func_or_list_of_funcs}``.

    Returns
    -------
    pa.Table
        Aggregated result with keys + aggregated columns.
    """
    agg_specs: list[tuple[str, str]] = []
    output_names: list[str] = []

    for col, funcs in aggregations.items():
        if isinstance(funcs, str):
            funcs = [funcs]
        for func in funcs:
            pa_func = _AGG_FUNC_MAP.get(func)
            if pa_func is None:
                raise ValueError(f"Unsupported aggregation: {func!r}")
            agg_specs.append((col, pa_func))
            output_names.append(f"{col}_{func}")

    result = table.group_by(keys).aggregate(agg_specs)

    # Rename auto-generated columns.
    current_names = result.schema.names
    new_names = list(keys)
    for i, name in enumerate(current_names):
        if name not in keys:
            idx = i - len(keys)
            if 0 <= idx < len(output_names):
                new_names.append(output_names[idx])
            else:
                new_names.append(name)
    result = result.rename_columns(new_names)
    return result
