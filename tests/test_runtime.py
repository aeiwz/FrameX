"""Tests for runtime: TaskGraph, LocalScheduler, Partition."""

import pyarrow as pa
import pytest

from framex.runtime.partition import Partition, partition_table
from framex.runtime.task import Task, TaskGraph
from framex.runtime.scheduler import LocalScheduler
from framex.runtime.streaming import StreamProcessor


def _const_val(v: int) -> int:
    return v


class TestPartition:
    def test_basic(self):
        batch = pa.record_batch({"a": [1, 2, 3]})
        p = Partition(record_batch=batch, partition_id=0)
        assert p.num_rows == 3
        assert p.num_columns == 1
        assert p.partition_id == 0

    def test_partition_table(self):
        table = pa.table({"x": list(range(100))})
        partitions = partition_table(table, partition_size=25)
        total_rows = sum(p.num_rows for p in partitions)
        assert total_rows == 100
        # IDs should be sequential.
        assert [p.partition_id for p in partitions] == list(range(len(partitions)))

    def test_partition_small_table(self):
        table = pa.table({"a": [1, 2, 3]})
        partitions = partition_table(table)
        # Small table should have 1 partition.
        assert len(partitions) == 1
        assert partitions[0].num_rows == 3


class TestTaskGraph:
    def test_simple_graph(self):
        graph = TaskGraph()
        t1 = Task(fn=lambda: 1)
        t2 = Task(fn=lambda: 2)
        t3 = Task(fn=lambda: 3)

        id1 = graph.add_task(t1)
        id2 = graph.add_task(t2)
        id3 = graph.add_task(t3)

        graph.add_dependency(id3, id1)
        graph.add_dependency(id3, id2)

        order = graph.topological_order()
        # t3 must come after t1 and t2.
        assert order.index(id3) > order.index(id1)
        assert order.index(id3) > order.index(id2)

    def test_linear_chain(self):
        graph = TaskGraph()
        ids: list[str] = []
        for i in range(5):
            t = Task(fn=lambda: i)
            tid = graph.add_task(t)
            if ids:
                graph.add_dependency(tid, ids[-1])
            ids.append(tid)

        order = graph.topological_order()
        assert order == ids

    def test_cycle_detection(self):
        graph = TaskGraph()
        t1 = Task(fn=lambda: 1)
        t2 = Task(fn=lambda: 2)
        id1 = graph.add_task(t1)
        id2 = graph.add_task(t2)
        graph.add_dependency(id1, id2)
        graph.add_dependency(id2, id1)
        with pytest.raises(RuntimeError, match="cycle"):
            graph.topological_order()


class TestLocalScheduler:
    def test_execute_simple(self):
        graph = TaskGraph()
        t1 = Task(fn=lambda: 10)
        t2 = Task(fn=lambda: 20)
        t3 = Task(fn=lambda: 30)
        id1 = graph.add_task(t1)
        id2 = graph.add_task(t2)
        id3 = graph.add_task(t3)

        scheduler = LocalScheduler(max_workers=2, backend="threads")
        results = scheduler.execute(graph)

        assert results[id1] == 10
        assert results[id2] == 20
        assert results[id3] == 30

    def test_execute_with_dependencies(self):
        graph = TaskGraph()
        t1 = Task(fn=lambda: 5)
        t2 = Task(fn=lambda: 7)
        # t3 doesn't directly use dep results in this simple model,
        # but depends on them for ordering.
        t3 = Task(fn=lambda: 12)

        id1 = graph.add_task(t1)
        id2 = graph.add_task(t2)
        id3 = graph.add_task(t3)
        graph.add_dependency(id3, id1)
        graph.add_dependency(id3, id2)

        scheduler = LocalScheduler(max_workers=2, backend="threads")
        results = scheduler.execute(graph)

        assert results[id1] == 5
        assert results[id2] == 7
        assert results[id3] == 12

    def test_partition_roundtrip(self):
        """Create partitions, process in tasks, and verify results."""
        table = pa.table({"val": [1, 2, 3, 4, 5, 6]})
        parts = partition_table(table, partition_size=3)

        def process_partition(batch_bytes: bytes, schema: pa.Schema) -> int:
            reader = pa.ipc.open_stream(batch_bytes)
            batch = reader.read_next_batch()
            import pyarrow.compute as pc
            return pc.sum(batch.column(0)).as_py()

        graph = TaskGraph()
        for p in parts:
            # Serialize partition for the task.
            sink = pa.BufferOutputStream()
            writer = pa.ipc.new_stream(sink, p.schema)
            writer.write_batch(p.record_batch)
            writer.close()
            batch_bytes = sink.getvalue().to_pybytes()

            t = Task(fn=process_partition, args=(batch_bytes, p.schema))
            graph.add_task(t)

        scheduler = LocalScheduler(max_workers=2, backend="threads")
        results = scheduler.execute(graph)
        total = sum(results.values())
        assert total == 21  # 1+2+3+4+5+6


class TestStreaming:
    def test_stream_processor_runs_batches(self):
        source = [
            {"value": [1, 2, 3], "keep": [True, False, True]},
            {"value": [4, 5], "keep": [True, True]},
        ]

        def transform(df):
            return df.filter(df["keep"]).drop(["keep"])

        collected = []
        processor = StreamProcessor(transform, sink=lambda out: collected.append(out.num_rows))
        stats = processor.run(source)

        assert stats.batches_in == 2
        assert stats.batches_out == 2
        assert stats.rows_in == 5
        assert stats.rows_out == 4
        assert collected == [2, 2]


class TestRayScheduler:
    def test_execute_simple_ray_backend(self):
        pytest.importorskip("ray")
        graph = TaskGraph()
        id1 = graph.add_task(Task(fn=_const_val, args=(10,)))
        id2 = graph.add_task(Task(fn=_const_val, args=(20,)))

        scheduler = LocalScheduler(max_workers=2, backend="ray")
        results = scheduler.execute(graph)
        assert results[id1] == 10
        assert results[id2] == 20


class TestDaskScheduler:
    def test_execute_simple_dask_backend(self):
        pytest.importorskip("dask.distributed")
        graph = TaskGraph()
        id1 = graph.add_task(Task(fn=_const_val, args=(10,)))
        id2 = graph.add_task(Task(fn=_const_val, args=(20,)))

        scheduler = LocalScheduler(max_workers=2, backend="dask")
        results = scheduler.execute(graph)
        assert results[id1] == 10
        assert results[id2] == 20


class TestHpcScheduler:
    def test_execute_simple_hpc_backend_with_dask_engine(self, monkeypatch):
        pytest.importorskip("dask.distributed")
        monkeypatch.setenv("FRAMEX_HPC_ENGINE", "dask")

        graph = TaskGraph()
        id1 = graph.add_task(Task(fn=_const_val, args=(10,)))
        id2 = graph.add_task(Task(fn=_const_val, args=(20,)))

        scheduler = LocalScheduler(max_workers=2, backend="hpc")
        results = scheduler.execute(graph)
        assert results[id1] == 10
        assert results[id2] == 20
