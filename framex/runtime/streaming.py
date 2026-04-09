"""Lightweight micro-batch streaming runtime for FrameX."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass
class StreamStats:
    batches_in: int = 0
    batches_out: int = 0
    rows_in: int = 0
    rows_out: int = 0
    elapsed_seconds: float = 0.0


class StreamProcessor:
    """Process micro-batches through a transform and optional sink.

    This is intentionally simple and single-node. It provides a stable
    production-oriented API for batch-like streaming ingestion.
    """

    def __init__(
        self,
        transform: Callable[[Any], Any],
        *,
        sink: Callable[[Any], Any] | None = None,
    ) -> None:
        self._transform = transform
        self._sink = sink

    def _to_dataframe(self, batch: Any) -> Any:
        from framex.core.dataframe import DataFrame

        if isinstance(batch, DataFrame):
            return batch
        return DataFrame(batch)

    def process_batch(self, batch: Any) -> Any:
        df = self._to_dataframe(batch)
        out = self._transform(df)
        out_df = self._to_dataframe(out)
        if self._sink is not None:
            self._sink(out_df)
        return out_df

    def run(self, source: Iterable[Any]) -> StreamStats:
        stats = StreamStats()
        t0 = time.perf_counter()
        for batch in source:
            in_df = self._to_dataframe(batch)
            stats.batches_in += 1
            stats.rows_in += in_df.num_rows

            out_df = self.process_batch(in_df)
            stats.batches_out += 1
            stats.rows_out += out_df.num_rows

        stats.elapsed_seconds = time.perf_counter() - t0
        return stats

