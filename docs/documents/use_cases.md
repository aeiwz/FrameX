---
title: Use Cases
description: Practical scenarios where FrameX fits today.
order: 6
section: Guides
---

# FrameX Use Cases

## 1. Local ETL for Analytics Teams

Use FrameX to process Parquet datasets on a single workstation before loading into BI systems.

Why it fits:

- Arrow-native IO
- familiar dataframe API
- efficient local execution for medium-large files

## 2. Feature Engineering Before Model Training

Use FrameX for cleaning, grouping, and joining tabular data before exporting arrays/dataframes for ML tooling.

Why it fits:

- fast tabular transforms
- direct conversion to Pandas/NumPy
- chunked arrays for large vectors

## 3. Replacing Slow Single-Threaded Pandas Scripts

If a script is CPU-bound and spending time in repetitive filters/groupbys/sorts, FrameX can be a migration path that keeps dataframe ergonomics.

Migration pattern:

1. replace `import pandas as pd` with `import framex as fx` for target segments
2. keep logic shape similar
3. compare outputs via `.to_pandas()` during rollout

## 4. Interop Bridge Between Ecosystems

FrameX can sit between Arrow-heavy and Pandas/NumPy-heavy parts of a pipeline.

Useful boundaries:

- `from_pandas(...)`
- `from_dask(...)`
- `from_ray(...)`
- `from_dataframe(...)`
- `.to_arrow()` for Arrow-first systems
- `.to_dask()` / `.to_ray()` for pipeline handoff

## 5. Incremental Adoption in Existing Codebases

Adopt only the heaviest stages first, then expand.

Suggested order:

1. IO and wide aggregations
2. joins and sorts
3. optional lazy pipelines for long transformations

## When Not to Use FrameX (Yet)

- strict requirement for complete Pandas API parity
- workloads that require full multi-node distributed orchestration
- pipelines dominated by custom Python object logic where Arrow typing is minimal
