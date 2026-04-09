"""Optional accelerated NDArray execution backends.

All helpers return NumPy arrays and gracefully fall back to NumPy when an
optional dependency is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from framex.config import get_config

logger = logging.getLogger(__name__)

_NUMEXPR_BINARY: dict[Any, str] = {
    np.add: "a + b",
    np.subtract: "a - b",
    np.multiply: "a * b",
    np.divide: "a / b",
}

_NUMEXPR_UNARY: dict[Any, str] = {
    np.sin: "sin(a)",
    np.cos: "cos(a)",
    np.tan: "tan(a)",
    np.exp: "exp(a)",
    np.log: "log(a)",
    np.sqrt: "sqrt(a)",
    np.absolute: "abs(a)",
    np.floor: "floor(a)",
    np.ceil: "ceil(a)",
    np.negative: "-a",
}


def _as_numpy(x: Any) -> np.ndarray[Any, Any]:
    return np.asarray(x)


def evaluate_ufunc(
    ufunc: Any,
    inputs: list[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Evaluate a ufunc using the configured backend, with safe fallback."""
    backend = get_config().array_backend
    if backend == "auto":
        for evaluator in (_eval_numexpr, _eval_jax, _eval_torch, _eval_cupy):
            out = evaluator(ufunc, inputs, kwargs)
            if out is not None:
                return out
    elif backend == "numexpr":
        out = _eval_numexpr(ufunc, inputs, kwargs)
        if out is not None:
            return out
    elif backend == "jax":
        out = _eval_jax(ufunc, inputs, kwargs)
        if out is not None:
            return out
    elif backend == "torch":
        out = _eval_torch(ufunc, inputs, kwargs)
        if out is not None:
            return out
    elif backend == "cupy":
        out = _eval_cupy(ufunc, inputs, kwargs)
        if out is not None:
            return out
    # numba backend currently reuses NumPy ufunc calls; Numba is exposed via jit helpers.
    return ufunc(*inputs, **kwargs)


def maybe_numba_jit(
    fn: Callable[[np.ndarray[Any, Any]], Any],
) -> Callable[[np.ndarray[Any, Any]], Any]:
    """JIT-compile a function with Numba when available."""
    try:
        import numba as nb

        return nb.njit(cache=True, fastmath=True)(fn)  # type: ignore[return-value]
    except Exception:
        return fn


def _eval_numexpr(ufunc: Any, inputs: list[Any], kwargs: dict[str, Any]) -> np.ndarray[Any, Any] | None:
    if kwargs:
        return None
    try:
        import numexpr as ne
    except Exception:
        return None

    if len(inputs) == 2 and ufunc in _NUMEXPR_BINARY:
        a = _as_numpy(inputs[0])
        b = _as_numpy(inputs[1])
        expr = _NUMEXPR_BINARY[ufunc]
        return np.asarray(ne.evaluate(expr, local_dict={"a": a, "b": b}))

    if len(inputs) == 1 and ufunc in _NUMEXPR_UNARY:
        a = _as_numpy(inputs[0])
        expr = _NUMEXPR_UNARY[ufunc]
        return np.asarray(ne.evaluate(expr, local_dict={"a": a}))

    return None


def _eval_cupy(ufunc: Any, inputs: list[Any], kwargs: dict[str, Any]) -> np.ndarray[Any, Any] | None:
    try:
        import cupy as cp
    except Exception:
        return None

    try:
        cp_inputs = [cp.asarray(_as_numpy(v)) for v in inputs]
        result = ufunc(*cp_inputs, **kwargs)
        return cp.asnumpy(result)
    except Exception:
        logger.debug("CuPy backend evaluation failed, falling back to NumPy", exc_info=True)
        return None


def _eval_torch(ufunc: Any, inputs: list[Any], kwargs: dict[str, Any]) -> np.ndarray[Any, Any] | None:
    if kwargs:
        return None
    try:
        import torch
    except Exception:
        return None

    unary_map = {
        np.sin: torch.sin,
        np.cos: torch.cos,
        np.tan: torch.tan,
        np.exp: torch.exp,
        np.log: torch.log,
        np.sqrt: torch.sqrt,
        np.absolute: torch.abs,
        np.floor: torch.floor,
        np.ceil: torch.ceil,
        np.negative: torch.neg,
    }
    binary_map = {
        np.add: torch.add,
        np.subtract: torch.sub,
        np.multiply: torch.mul,
        np.divide: torch.div,
    }

    try:
        if len(inputs) == 1 and ufunc in unary_map:
            a = torch.as_tensor(_as_numpy(inputs[0]))
            return np.asarray(unary_map[ufunc](a).cpu().numpy())
        if len(inputs) == 2 and ufunc in binary_map:
            a = torch.as_tensor(_as_numpy(inputs[0]))
            b = torch.as_tensor(_as_numpy(inputs[1]))
            return np.asarray(binary_map[ufunc](a, b).cpu().numpy())
    except Exception:
        logger.debug("PyTorch backend evaluation failed, falling back to NumPy", exc_info=True)
    return None


def _eval_jax(ufunc: Any, inputs: list[Any], kwargs: dict[str, Any]) -> np.ndarray[Any, Any] | None:
    if kwargs:
        return None
    try:
        import jax.numpy as jnp
    except Exception:
        return None

    unary_map = {
        np.sin: jnp.sin,
        np.cos: jnp.cos,
        np.tan: jnp.tan,
        np.exp: jnp.exp,
        np.log: jnp.log,
        np.sqrt: jnp.sqrt,
        np.absolute: jnp.abs,
        np.floor: jnp.floor,
        np.ceil: jnp.ceil,
        np.negative: jnp.negative,
    }
    binary_map = {
        np.add: jnp.add,
        np.subtract: jnp.subtract,
        np.multiply: jnp.multiply,
        np.divide: jnp.divide,
    }

    try:
        if len(inputs) == 1 and ufunc in unary_map:
            a = jnp.asarray(_as_numpy(inputs[0]))
            return np.asarray(unary_map[ufunc](a))
        if len(inputs) == 2 and ufunc in binary_map:
            a = jnp.asarray(_as_numpy(inputs[0]))
            b = jnp.asarray(_as_numpy(inputs[1]))
            return np.asarray(binary_map[ufunc](a, b))
    except Exception:
        logger.debug("JAX backend evaluation failed, falling back to NumPy", exc_info=True)
    return None
