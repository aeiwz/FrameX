"""Pandas-compatible engine loader.

This module enables drop-in acceleration by preferring Modin/FireDucks when
available, while falling back to pandas for full compatibility.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_pd() -> tuple[Any, str]:
    preferred = os.getenv("FRAMEX_PANDAS_BACKEND", "auto").strip().lower()

    order: list[tuple[str, str]]
    if preferred == "auto":
        order = [
            ("modin.pandas", "modin"),
            ("fireducks.pandas", "fireducks"),
            ("pandas", "pandas"),
        ]
    elif preferred == "modin":
        order = [("modin.pandas", "modin"), ("pandas", "pandas")]
    elif preferred == "fireducks":
        order = [("fireducks.pandas", "fireducks"), ("pandas", "pandas")]
    else:
        order = [("pandas", "pandas")]

    last_exc: Exception | None = None
    for module_name, label in order:
        try:
            return importlib.import_module(module_name), label
        except Exception as exc:  # pragma: no cover - import depends on environment
            last_exc = exc
            continue

    raise ImportError("No pandas-compatible backend found") from last_exc


def get_pandas_module() -> Any:
    mod, _ = _load_pd()
    return mod


def get_pandas_backend_name() -> str:
    _, name = _load_pd()
    return name

