"""Tests for NDArray: construction, chunking, NumPy dispatch, reductions."""

import numpy as np
import pytest

import framex as fx
from framex.core.array import NDArray


def _double_block(block: np.ndarray) -> np.ndarray:
    return block * 2


class TestConstruction:
    def test_from_list(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        assert len(x) == 3
        assert x.dtype.arrow_type.to_pandas_dtype() == np.float64

    def test_from_numpy(self):
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        x = NDArray(arr)
        assert len(x) == 4
        np.testing.assert_array_equal(x.to_numpy(), arr)

    def test_chunked(self):
        x = NDArray(list(range(100)), dtype="int64", chunks=25)
        assert x.num_chunks == 4
        assert len(x) == 100

    def test_single_chunk_when_small(self):
        x = NDArray([1, 2, 3], dtype="float64", chunks=1000)
        assert x.num_chunks == 1

    def test_empty(self):
        x = NDArray([], dtype="float64")
        assert len(x) == 0


class TestConversions:
    def test_to_numpy(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        result = x.to_numpy()
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_array_protocol(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        result = np.asarray(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestReductions:
    def test_sum(self):
        x = NDArray([1.0, 2.0, 3.0, 4.0], dtype="float64")
        assert x.sum() == 10.0

    def test_mean(self):
        x = NDArray([10.0, 20.0, 30.0], dtype="float64")
        assert x.mean() == 20.0

    def test_min_max(self):
        x = NDArray([3, 1, 4, 1, 5], dtype="int64")
        assert x.min() == 1
        assert x.max() == 5


class TestArithmetic:
    def test_add(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        y = NDArray([10.0, 20.0, 30.0], dtype="float64")
        result = x + y
        np.testing.assert_array_equal(result.to_numpy(), [11.0, 22.0, 33.0])

    def test_scalar_add(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        result = x + 10.0
        np.testing.assert_array_equal(result.to_numpy(), [11.0, 12.0, 13.0])

    def test_sub(self):
        x = NDArray([10.0, 20.0, 30.0], dtype="float64")
        y = NDArray([1.0, 2.0, 3.0], dtype="float64")
        result = x - y
        np.testing.assert_array_equal(result.to_numpy(), [9.0, 18.0, 27.0])

    def test_mul(self):
        x = NDArray([2.0, 3.0, 4.0], dtype="float64")
        result = x * 2.0
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0, 8.0])


class TestNumpyUfuncDispatch:
    def test_np_sin(self):
        x = NDArray([0.0, np.pi / 2, np.pi], dtype="float64")
        result = np.sin(x)
        assert isinstance(result, NDArray)
        np.testing.assert_allclose(result.to_numpy(), [0.0, 1.0, 0.0], atol=1e-10)

    def test_np_add(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        y = NDArray([10.0, 20.0, 30.0], dtype="float64")
        result = np.add(x, y)
        assert isinstance(result, NDArray)
        np.testing.assert_array_equal(result.to_numpy(), [11.0, 22.0, 33.0])

    def test_np_multiply_scalar(self):
        x = NDArray([1.0, 2.0, 3.0], dtype="float64")
        result = np.multiply(x, 5.0)
        assert isinstance(result, NDArray)
        np.testing.assert_array_equal(result.to_numpy(), [5.0, 10.0, 15.0])


class TestNumpyArrayFunction:
    def test_np_sum(self):
        x = NDArray([1.0, 2.0, 3.0, 4.0], dtype="float64")
        assert np.sum(x) == 10.0

    def test_np_mean(self):
        x = NDArray([10.0, 20.0, 30.0], dtype="float64")
        assert np.mean(x) == 20.0

    def test_np_concatenate(self):
        x = NDArray([1.0, 2.0], dtype="float64")
        y = NDArray([3.0, 4.0], dtype="float64")
        result = np.concatenate([x, y])
        assert isinstance(result, NDArray)
        np.testing.assert_array_equal(result.to_numpy(), [1.0, 2.0, 3.0, 4.0])


class TestIndexing:
    def test_scalar_index(self):
        x = NDArray([10, 20, 30], dtype="int64")
        assert x[0] == 10
        assert x[2] == 30

    def test_slice(self):
        x = NDArray([10, 20, 30, 40, 50], dtype="int64")
        result = x[1:4]
        assert isinstance(result, NDArray)
        np.testing.assert_array_equal(result.to_numpy(), [20, 30, 40])


class TestConvenienceConstructor:
    def test_fx_array(self):
        x = fx.array([1.0, 2.0, 3.0], dtype="float64", chunks=2)
        assert isinstance(x, NDArray)
        assert len(x) == 3
        assert x.num_chunks == 2


class TestParallelBlocks:
    def test_apply_blocks_threads(self):
        x = NDArray(np.arange(32, dtype=np.float64), chunks=8)
        result = x.apply_blocks(_double_block, workers=4, backend="threads")
        np.testing.assert_array_equal(result.to_numpy(), np.arange(32, dtype=np.float64) * 2)

    def test_apply_blocks_processes(self):
        x = NDArray(np.arange(24, dtype=np.float64), chunks=6)
        result = x.apply_blocks(_double_block, workers=2, backend="processes")
        np.testing.assert_array_equal(result.to_numpy(), np.arange(24, dtype=np.float64) * 2)
