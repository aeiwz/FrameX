"""Tests for memory: BufferPool lifecycle, zero-copy SharedMemory, pickle5 fallback."""

import pickle

import pyarrow as pa
import pytest

from framex.memory.buffer import Buffer, BufferBackend
from framex.memory.pool import BufferPool
from framex.memory.transport import recv_zero_copy, send_zero_copy, unlink_shm


class TestBuffer:
    def test_arrow_buffer_from_data(self):
        buf = Buffer(backend=BufferBackend.ARROW, data=b"hello world")
        assert buf.size == 11
        assert buf.as_bytes()[:11] == b"hello world"
        buf.close()

    def test_arrow_buffer_allocate(self):
        buf = Buffer(backend=BufferBackend.ARROW, size=1024)
        assert buf.size == 1024
        buf.close()

    def test_shared_memory_buffer(self):
        data = b"test data for shm"
        buf = Buffer(backend=BufferBackend.SHARED_MEMORY, data=data)
        try:
            assert buf.name is not None
            assert buf.size >= len(data)
            assert buf.as_bytes()[:len(data)] == data
        finally:
            buf.close()
            buf.unlink()

    def test_shared_memory_attach(self):
        data = b"attach test"
        buf1 = Buffer(backend=BufferBackend.SHARED_MEMORY, data=data)
        try:
            buf2 = Buffer(backend=BufferBackend.SHARED_MEMORY, name=buf1.name)
            assert buf2.as_bytes()[:len(data)] == data
            buf2.close()
        finally:
            buf1.close()
            buf1.unlink()

    def test_context_manager(self):
        with Buffer(backend=BufferBackend.ARROW, data=b"ctx") as buf:
            assert buf.size == 3


class TestBufferPool:
    def test_lifecycle(self):
        pool = BufferPool()
        pool.start()
        try:
            shm = pool.allocate(256)
            assert shm.size >= 256
            assert pool.active_segments == 1

            shm2 = pool.allocate(512)
            assert pool.active_segments == 2

            pool.release(shm.name)
            assert pool.active_segments == 1
        finally:
            pool.shutdown()

        assert pool.active_segments == 0

    def test_context_manager(self):
        with BufferPool() as pool:
            shm = pool.allocate(128)
            assert pool.active_segments == 1
        # After context exit, pool is shut down.
        assert pool.active_segments == 0

    def test_no_leaks(self):
        """Allocate and shutdown without manual release; pool should clean up."""
        with BufferPool() as pool:
            for _ in range(10):
                pool.allocate(64)
            assert pool.active_segments == 10
        assert pool.active_segments == 0


class TestZeroCopyTransport:
    def test_roundtrip(self):
        batch = pa.record_batch(
            {"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]},
        )
        shm_name, schema = send_zero_copy(batch)
        try:
            recovered = recv_zero_copy(shm_name, schema)
            assert recovered.equals(batch)
        finally:
            unlink_shm(shm_name)

    def test_roundtrip_with_strings(self):
        batch = pa.record_batch(
            {"name": ["alice", "bob", "charlie"], "score": [90, 85, 92]},
        )
        shm_name, schema = send_zero_copy(batch)
        try:
            recovered = recv_zero_copy(shm_name, schema)
            assert recovered.equals(batch)
        finally:
            unlink_shm(shm_name)

    def test_empty_batch(self):
        schema = pa.schema([("a", pa.int64())])
        batch = pa.record_batch({"a": pa.array([], type=pa.int64())})
        shm_name, s = send_zero_copy(batch)
        try:
            recovered = recv_zero_copy(shm_name, s)
            assert recovered.num_rows == 0
        finally:
            unlink_shm(shm_name)


class TestPickle5Fallback:
    def test_record_batch_pickle_roundtrip(self):
        batch = pa.record_batch({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        data = pickle.dumps(batch, protocol=5)
        recovered = pickle.loads(data)
        assert recovered.equals(batch)
