---
title: SQLite Guide
description: Read and write FrameX DataFrames to SQLite tables using table and query workflows.
order: 11
section: Guides
---

# SQLite Guide

Use SQLite when you want a portable local database file with SQL query support.

## Write a DataFrame to SQLite

```python
import framex as fx

df = fx.DataFrame(
    {
        "order_id": [101, 102, 103],
        "region": ["APAC", "US", "APAC"],
        "amount": [120.0, 80.5, 99.0],
    }
)

fx.write_file(df, "analytics.sqlite", table="orders")
```

Default behavior is `if_exists="replace"` and `index=False`.

## Append Incremental Data

```python
delta = fx.DataFrame({"order_id": [104], "region": ["EU"], "amount": [150.0]})
fx.write_file(delta, "analytics.sqlite", table="orders", if_exists="append")
```

## Read a Table

```python
orders = fx.read_file("analytics.sqlite", table="orders")
print(orders)
```

## Read with SQL Query

```python
top_regions = fx.read_file(
    "analytics.sqlite",
    query="""
        SELECT region, SUM(amount) AS total
        FROM orders
        GROUP BY region
        ORDER BY total DESC
    """,
)
```

## Useful Parameters

- `write_file(..., table="name")`
- `write_file(..., if_exists="replace"|"append"|"fail")`
- `write_file(..., index=False)` (default)
- `read_file(..., table="name")`
- `read_file(..., query="SELECT ...")`

If both `table` and `query` are omitted when reading, FrameX loads the first non-system table in the SQLite file.
