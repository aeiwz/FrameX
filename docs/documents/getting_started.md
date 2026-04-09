# Getting Started with FrameX

FrameX is an Arrow-backed parallel dataframe and array library that mirrors Pandas and NumPy semantics while delivering multi-processing concurrency natively to your local workstation.

## Installation

```bash
pip install framex
```

## Creating a DataFrame

FrameX eagerly accepts data from Native Python, Pandas, and PyArrow.

```python
import framex as fx
import pandas as pd

# From raw python dict
df1 = fx.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

# From Pandas
pdf = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
df2 = fx.from_pandas(pdf)
```

## Eager vs Lazy Execution

FrameX `DataFrame` objects are eager by default. Modifications to the frame run sequentially and immediately. 
If you have a large cascade of operations (filters, joins, assignments), you can use `.lazy()` to build an execution plan and `.collect()` to finalize it. 

FrameX uses an intelligent runtime heuristic: **Threads** are used for numeric structures where the Global Interpreter Lock (GIL) is released, and **Processes** are utilized for object/string heavy operations!

### Eager Filtering

```python
# Create a dummy parquet file
# fx.write_parquet(df, "data.parquet")

df = fx.read_parquet("data.parquet")

# Filter data
filtered = df.filter(df["A"] > 5)

# Calculate aggregations
summary = filtered.groupby(["B"]).agg({"A": "mean"})

print(summary)
```

### Lazy Execution Workflow

```python
df = fx.read_parquet("large_data.parquet")

# Build query logic
query = (
    df.lazy()
    .filter(lambda d: d["A"] >= 100)
    .with_column("A_Squared", lambda d: d["A"] * d["A"])
    .select(["B", "A_Squared"])
    .groupby("B")
    .agg({"A_Squared": "sum"})
)

# Execute query operations parallelized and optimized
result_df = query.collect()
```

## Parallel NumPy Semantics

FrameX isn't just about DataFrames. It features an `NDArray` that natively dispatches to NumPy semantics while allowing underlying multiprocessing capabilities via chunks.

```python
import numpy as np
import framex as fx

# Build a chunked 10-million row array
x = fx.array(data=np.random.rand(10_000_000), chunks=1_000_000)

# Native numpy functions delegate correctly!
y = np.sin(x) + np.log(x)

# Triggers parallel reduction compute
total = y.sum().compute()
```

## Seamless Interchange

Need to return to standard tools after doing your heavy lifting? FrameX relies natively on the Apache Arrow format, making export extremely cheap and often zero-copy.

```python
# Move back to pandas easily.
pandas_df = result_df.to_pandas()

# Or move directly to pyarrow representations.
arrow_table = result_df.to_arrow()
```
