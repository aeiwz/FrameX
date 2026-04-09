---
title: "Tutorial: ETL Pipeline"
description: Build a practical Parquet ETL flow with filtering, enrichment, and grouped outputs.
order: 4
section: Tutorials
---

# Tutorial: ETL Pipeline

This tutorial shows a standard analytics ETL shape:

1. Read raw records
2. Filter invalid rows
3. Derive columns
4. Aggregate by business key
5. Write Parquet outputs

## Scenario

You receive transaction records and need a daily country-level revenue table.

## Step 1: Load Data

```python
import framex as fx

raw = fx.read_parquet("transactions.parquet")
```

## Step 2: Filter Invalid and Refund Rows

```python
filtered = raw.filter((raw["amount"] > 0) & (~raw["is_refund"]))
```

## Step 3: Add Computed Columns

```python
enriched = filtered.assign(
    gross_amount=lambda d: d["amount"] * 1.07,
)
```

## Step 4: Aggregate KPI Table

```python
kpi = (
    enriched
    .groupby(["event_date", "country"])
    .agg({"gross_amount": ["sum", "mean", "count"]})
    .sort(["event_date", "gross_amount_sum"], ascending=[True, False])
)
```

## Step 5: Export

```python
fx.write_parquet(kpi, "outputs/country_daily_kpi.parquet")
```

## Optional: Lazy Plan for Long Pipelines

```python
kpi_lazy = (
    raw.lazy()
    .filter(lambda d: (d["amount"] > 0) & (~d["is_refund"]))
    .with_column("gross_amount", lambda d: d["amount"] * 1.07)
    .groupby(["event_date", "country"])
    .agg({"gross_amount": ["sum", "mean", "count"]})
    .collect()
)
```

## Validation Tip

At integration boundaries, compare against Pandas for confidence:

```python
pandas_result = kpi.to_pandas()
```
