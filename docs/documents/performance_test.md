---
title: Performance Test
description: How to run, interpret, and compare FrameX benchmark results.
order: 10
section: Guides
---

# Performance Test

Use FrameX benchmark tooling to measure throughput, scaling, and memory behavior.

## Run Full Benchmark Suite

```bash
python3 -m benchmarks.benchmark_suite
```

## Fast Smoke Benchmark

```bash
python3 -m benchmarks.benchmark_suite \
  --rows 20000 \
  --array-elements 120000 \
  --object-items 30000 \
  --workers 1,2 \
  --repeats 1 \
  --warmups 0 \
  --skip-plots
```

## Run Workload Capability Check

```bash
python3 -m benchmarks.check_framex_workloads
```

## Output Artifacts

- `benchmarks/results/benchmark_results.json`
- `benchmarks/results/benchmark_results.csv`
- `benchmarks/results/benchmark_report.md`
- `benchmarks/results/framex_workload_check.json`
- plot images in `benchmarks/results/*.png`

## Interpreting Key Metrics

- `speedup > 1.0`: FrameX faster than baseline
- `seconds`: lower is better
- `peak_rss_mb`: lower memory footprint is better

## Optimization Workflow

1. Run benchmark suite and capture baseline artifacts.
2. Apply one optimization at a time.
3. Re-run with identical parameters.
4. Compare scenario-level speedup and memory impact.
