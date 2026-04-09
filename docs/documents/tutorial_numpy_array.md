---
title: "Tutorial: NumPy NDArray Interop"
description: Use FrameX NDArray with NumPy ufuncs and reductions.
order: 5
section: Tutorials
---

# Tutorial: NumPy NDArray Interop

`framex.NDArray` supports NumPy protocols, so many NumPy operations work directly while retaining chunked storage.

## Step 1: Create Chunked Arrays

```python
import numpy as np
import framex as fx

x = fx.array(np.random.rand(2_000_000), chunks=250_000)
y = fx.array(np.random.rand(2_000_000), chunks=250_000)
```

## Step 2: Run Ufuncs

```python
z = np.sin(x) + np.log(y + 1.0)
```

## Step 3: Use Reductions

```python
print(np.mean(z))
print(np.max(z))
print(np.std(z))
```

## Step 4: Use `np.where`

```python
mask = z > 0.5
clipped = np.where(mask, z, 0.5)
```

## Step 5: Convert Out

```python
as_numpy = clipped.to_numpy()
```

## When to Use NDArray

Use `NDArray` when your workflow is mostly numeric vector transforms and you want chunk-aware behavior with an easy NumPy-facing API.
