"""Tests for the C backend and kernel_backend config option."""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import framex as fx
from framex.backends.c_backend import C_AVAILABLE
from framex.ops.reduction import (
    sum_column, mean_column, min_column, max_column, std_column, var_column,
)
from framex.ops.elementwise import (
    add_arrays, sub_arrays, mul_arrays, div_arrays,
    scalar_add, scalar_sub, scalar_mul, scalar_div,
)


pytestmark = pytest.mark.skipif(
    not C_AVAILABLE,
    reason="C backend not available (no compiler found)",
)


class TestCBackendAvailability:
    def test_c_available_flag(self):
        assert C_AVAILABLE is True

    def test_set_kernel_backend_c(self):
        fx.set_kernel_backend("c")
        assert fx.get_config().kernel_backend == "c"
        fx.set_kernel_backend("python")
        assert fx.get_config().kernel_backend == "python"

    def test_invalid_kernel_backend(self):
        with pytest.raises(ValueError, match="kernel_backend"):
            fx.set_kernel_backend("gpu")  # type: ignore[arg-type]

    def test_context_manager(self):
        assert fx.get_config().kernel_backend == "python"
        with fx.config(kernel_backend="c") as cfg:
            assert cfg.kernel_backend == "c"
            assert fx.get_config().kernel_backend == "c"
        assert fx.get_config().kernel_backend == "python"


class TestCReductionsF64:
    """C reductions on float64 columns match pyarrow.compute results."""

    def setup_method(self):
        fx.set_kernel_backend("c")

    def teardown_method(self):
        fx.set_kernel_backend("python")

    def test_sum_f64(self):
        col = pa.chunked_array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        assert sum_column(col) == pytest.approx(15.0)

    def test_sum_f64_multi_chunk(self):
        col = pa.chunked_array([[1.0, 2.0], [3.0, 4.0], [5.0]])
        assert sum_column(col) == pytest.approx(15.0)

    def test_mean_f64(self):
        col = pa.chunked_array([[10.0, 20.0, 30.0]])
        assert mean_column(col) == pytest.approx(20.0)

    def test_min_f64(self):
        col = pa.chunked_array([[5.0, 1.0, 9.0, 3.0]])
        assert min_column(col) == pytest.approx(1.0)

    def test_max_f64(self):
        col = pa.chunked_array([[5.0, 1.0, 9.0, 3.0]])
        assert max_column(col) == pytest.approx(9.0)

    def test_std_f64(self):
        col = pa.chunked_array([[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]])
        # Population std (ddof=0) = 2.0
        assert std_column(col, ddof=0) == pytest.approx(2.0, rel=1e-6)

    def test_var_f64(self):
        col = pa.chunked_array([[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]])
        assert var_column(col, ddof=0) == pytest.approx(4.0, rel=1e-6)


class TestCReductionsI64:
    """C reductions on int64 columns."""

    def setup_method(self):
        fx.set_kernel_backend("c")

    def teardown_method(self):
        fx.set_kernel_backend("python")

    def test_sum_i64(self):
        col = pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int64())
        assert sum_column(col) == 15

    def test_mean_i64(self):
        col = pa.chunked_array([[10, 20, 30]], type=pa.int64())
        assert mean_column(col) == pytest.approx(20.0)

    def test_min_i64(self):
        col = pa.chunked_array([[-5, 0, 10, 3]], type=pa.int64())
        assert min_column(col) == -5

    def test_max_i64(self):
        col = pa.chunked_array([[-5, 0, 10, 3]], type=pa.int64())
        assert max_column(col) == 10


class TestCElementwise:
    """C elementwise ops on float64 ChunkedArrays."""

    def setup_method(self):
        fx.set_kernel_backend("c")

    def teardown_method(self):
        fx.set_kernel_backend("python")

    def test_add_arrays(self):
        a = pa.chunked_array([[1.0, 2.0, 3.0]])
        b = pa.chunked_array([[4.0, 5.0, 6.0]])
        result = add_arrays(a, b).to_pylist()
        assert result == pytest.approx([5.0, 7.0, 9.0])

    def test_sub_arrays(self):
        a = pa.chunked_array([[10.0, 20.0, 30.0]])
        b = pa.chunked_array([[1.0, 2.0, 3.0]])
        result = sub_arrays(a, b).to_pylist()
        assert result == pytest.approx([9.0, 18.0, 27.0])

    def test_mul_arrays(self):
        a = pa.chunked_array([[2.0, 3.0, 4.0]])
        b = pa.chunked_array([[5.0, 6.0, 7.0]])
        result = mul_arrays(a, b).to_pylist()
        assert result == pytest.approx([10.0, 18.0, 28.0])

    def test_div_arrays(self):
        a = pa.chunked_array([[10.0, 20.0, 30.0]])
        b = pa.chunked_array([[2.0, 4.0, 5.0]])
        result = div_arrays(a, b).to_pylist()
        assert result == pytest.approx([5.0, 5.0, 6.0])

    def test_scalar_add(self):
        a = pa.chunked_array([[1.0, 2.0, 3.0]])
        result = scalar_add(a, 10.0).to_pylist()
        assert result == pytest.approx([11.0, 12.0, 13.0])

    def test_scalar_sub(self):
        a = pa.chunked_array([[10.0, 20.0, 30.0]])
        result = scalar_sub(a, 5.0).to_pylist()
        assert result == pytest.approx([5.0, 15.0, 25.0])

    def test_scalar_mul(self):
        a = pa.chunked_array([[1.0, 2.0, 3.0]])
        result = scalar_mul(a, 3.0).to_pylist()
        assert result == pytest.approx([3.0, 6.0, 9.0])

    def test_scalar_div(self):
        a = pa.chunked_array([[10.0, 20.0, 30.0]])
        result = scalar_div(a, 10.0).to_pylist()
        assert result == pytest.approx([1.0, 2.0, 3.0])


class TestCFallback:
    """C backend falls back to Python for non-float64, nullable, or string data."""

    def setup_method(self):
        fx.set_kernel_backend("c")

    def teardown_method(self):
        fx.set_kernel_backend("python")

    def test_string_column_falls_back(self):
        # String columns have no C kernel; should still work via Arrow.
        col = pa.chunked_array([["a", "b", "c"]])
        assert count_column(col) == 3

    def test_nullable_float_falls_back(self):
        col = pa.chunked_array([[1.0, None, 3.0]])
        # Should fall back to Arrow and return 4.0 (ignoring null).
        result = sum_column(col)
        assert result == pytest.approx(4.0)

    def test_results_match_python_backend(self):
        """C and Python backends must produce identical results."""
        data = [float(i) for i in range(1, 101)]
        col = pa.chunked_array([data])

        with fx.config(kernel_backend="python"):
            py_sum = sum_column(col)
            py_mean = mean_column(col)
            py_min = min_column(col)
            py_max = max_column(col)

        with fx.config(kernel_backend="c"):
            c_sum = sum_column(col)
            c_mean = mean_column(col)
            c_min = min_column(col)
            c_max = max_column(col)

        assert c_sum == pytest.approx(py_sum)
        assert c_mean == pytest.approx(py_mean)
        assert c_min == pytest.approx(py_min)
        assert c_max == pytest.approx(py_max)


# Import count_column here so TestCFallback can use it.
from framex.ops.reduction import count_column
