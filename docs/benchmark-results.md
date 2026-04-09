# FrameX Benchmarking

This repository now includes a unified benchmark suite that compares **FrameX** with native libraries.

Runner:
- `python3 -m benchmarks.benchmark_suite`

Output directory:
- `benchmarks/results`

Generated artifacts:
- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_report.md`
- `performance_speedup.png` (if matplotlib installed)
- `parallel_processing_scaling.png` (if matplotlib installed)
- `multiprocessing_scaling.png` (if matplotlib installed)
- `memory_peak_rss.png` (if matplotlib installed)

Covered benchmark categories:
1. Performance benchmark
2. Parallel processing benchmark
3. Single-core benchmark
4. Multiprocessing benchmark
5. Memory benchmark
6. Report + visualization
7. C backend benchmark (`python_backend` vs `c_backend`, auto-skipped if C backend unavailable)

## Quick start

```bash
python3 -m pip install -e '.[bench]'
python3 -m benchmarks.benchmark_suite
```

For a fast smoke run:

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

Disable C backend benchmark section:

```bash
python3 -m benchmarks.benchmark_suite --no-c-backend
```
