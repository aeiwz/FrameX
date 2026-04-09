---
title: Getting Started
description: Install FrameX and run your first end-to-end dataframe pipeline.
order: 2
section: Introduction
---

# Getting Started

This guide takes you from install to a complete mini pipeline: load data, transform it, aggregate it, and export results.

## 1. Install

```bash
pip install pyframe-xpy
```

Optional extras:

```bash
pip install pyframe-xpy[distributed]  # Dask + Ray runtime integrations
pip install pyframe-xpy[accel]        # numexpr + numba
```

## 2. Import and Create a DataFrame

```python
import framex as fx

df = fx.DataFrame(
    {
        "customer_id": [101, 102, 101, 103, 102],
        "country": ["TH", "US", "TH", "JP", "US"],
        "amount": [120.0, 80.5, 45.0, 220.0, 99.5],
        "is_refund": [False, False, True, False, False],
    }
)

print(df.shape)        # (5, 4)
print(df.columns)      # ['customer_id', 'country', 'amount', 'is_refund']
```

## 3. Filter and Add a Derived Column

```python
clean = df.filter(~df["is_refund"])

enriched = clean.assign(
    amount_with_tax=lambda d: d["amount"] * 1.07,
)
```

## 4. Group and Aggregate

```python
summary = (
    enriched
    .groupby("country")
    .agg({"amount": ["sum", "mean", "count"]})
    .sort("amount_sum", ascending=False)
)

print(summary.to_pandas())
```

## 5. Write and Read Parquet

```python
fx.write_parquet(summary, "country_summary.parquet")
roundtrip = fx.read_parquet("country_summary.parquet")
```

## 6. Convert to Pandas or Arrow

```python
pdf = roundtrip.to_pandas()
table = roundtrip.to_arrow()
```

## 7. Optional Lazy Mode

For longer transformation chains:

```python
lazy_result = (
    df.lazy()
    .filter(lambda d: ~d["is_refund"])
    .with_column("amount_with_tax", lambda d: d["amount"] * 1.07)
    .groupby("country")
    .agg({"amount_with_tax": "sum"})
    .collect()
)
```

Move on to [Tutorial: ETL Pipeline](/docs/tutorial_etl_pipeline) for a realistic scenario.
