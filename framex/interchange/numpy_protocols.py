"""NumPy interop protocols: __array_ufunc__ (NEP 13) and __array_function__ (NEP 18).

These are implemented on the NDArray class itself.  This module provides the
supporting infrastructure and documents the protocol contract.
"""

from __future__ import annotations

from typing import Any


def implements_array_ufunc() -> bool:
    """Check that our NDArray implements ``__array_ufunc__``."""
    from framex.core.array import NDArray

    return hasattr(NDArray, "__array_ufunc__")


def implements_array_function() -> bool:
    """Check that our NDArray implements ``__array_function__``."""
    from framex.core.array import NDArray

    return hasattr(NDArray, "__array_function__")
