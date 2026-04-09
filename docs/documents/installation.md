---
title: Installation
description: Python requirements, optional dependencies, and local development setup.
order: 3
section: Introduction
---

# Installation

## Requirements

- Python 3.11+
- macOS, Linux, or Windows

## Install from PyPI

```bash
pip install framex
```

## Development Install

```bash
git clone https://github.com/aeiwz/FrameX.git
cd FrameX
pip install -e .
```

## Verify Installation

```python
import framex as fx

print(fx.__version__)
print(fx.get_config())
```

## Optional Performance Tooling

For benchmarking and profiling:

```bash
pip install asv pyperf
```

## Build and Test Locally

```bash
pytest
```

```bash
cd website
npm install
npm run dev
```
