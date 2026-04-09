"""Hash join implementation using Arrow."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def hash_join(
    left: pa.Table,
    right: pa.Table,
    on: str | list[str],
    how: str = "inner",
    right_suffix: str = "_right",
) -> pa.Table:
    """Perform a hash join between two Arrow Tables.

    Parameters
    ----------
    left, right : pa.Table
        Tables to join.
    on : str or list[str]
        Join key column name(s).
    how : str
        Join type: ``"inner"``, ``"left"``, ``"right"``, ``"outer"``.
    right_suffix : str
        Suffix for overlapping column names from the right table.
    """
    if isinstance(on, str):
        on = [on]

    join_type_map = {
        "inner": "inner",
        "left": "left outer",
        "right": "right outer",
        "outer": "full outer",
    }
    pa_join_type = join_type_map.get(how)
    if pa_join_type is None:
        raise ValueError(f"Unsupported join type: {how!r}")

    # Handle overlapping column names.
    right_names = set(right.schema.names) - set(on)
    left_names = set(left.schema.names) - set(on)
    overlapping = right_names & left_names
    rename_map: dict[str, str] = {}
    for name in overlapping:
        rename_map[name] = f"{name}{right_suffix}"
    if rename_map:
        new_names = [rename_map.get(n, n) for n in right.schema.names]
        right = right.rename_columns(new_names)

    # For multi-key joins, use the first key (Arrow limitation workaround).
    if len(on) == 1:
        join_key = on[0]
        right_key = rename_map.get(join_key, join_key)
        return left.join(right, keys=join_key, right_keys=right_key, join_type=pa_join_type)
    else:
        # Synthetic key approach for multi-key.
        import pyarrow.compute as pc

        synth = "__join_key__"

        def add_synth(table: pa.Table, keys: list[str], col: str) -> pa.Table:
            parts = [pc.cast(table.column(k), pa.string()) for k in keys]
            combined = parts[0]
            for p in parts[1:]:
                combined = pc.binary_join_element_wise(combined, p, "|")
            return table.append_column(col, combined)

        left = add_synth(left, on, synth)
        right = add_synth(right, on, synth)
        result = left.join(right, keys=synth, right_keys=synth, join_type=pa_join_type)

        # Drop synthetic columns.
        for name in list(result.schema.names):
            if name.startswith("__join_key__"):
                idx = result.schema.get_field_index(name)
                result = result.remove_column(idx)
        return result
