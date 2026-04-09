# FrameX

FrameX is a high-performance Python library for parallel dataframe and array processing on a single machine.

It combines:

- Arrow-native storage and IO
- Pandas-like dataframe semantics
- NumPy-compatible chunked arrays
- Eager execution with optional lazy pipelines

## Install

```bash
pip install framex
```

## Quick Example

```python
import framex as fx

df = fx.DataFrame({"group": ["a", "a", "b"], "value": [10, 20, 30]})

result = (
    df.groupby("group")
      .agg({"value": ["sum", "mean", "count"]})
      .sort("value_sum", ascending=False)
)

print(result.to_pandas())
```

## Documentation

- Overview: `docs/documents/overview.md`
- Getting Started: `docs/documents/getting_started.md`
- Tutorials: `docs/documents/tutorial_etl_pipeline.md`, `docs/documents/tutorial_numpy_array.md`
- Use Cases: `docs/documents/use_cases.md`
- Architecture: `docs/documents/architecture.md`
- API Reference: `docs/documents/api_reference.md`

Website docs are served from the `website` Next.js app and read directly from `docs/documents`.

## Project Status

Pre-1.0 and actively evolving. APIs are usable now, but expect changes as performance and compatibility work continues.
