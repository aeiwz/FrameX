# FrameX Benchmark Suite

Unified benchmark runner for FrameX vs native libraries (Pandas/NumPy + Python stdlib executors).

## Covers

1. Performance benchmark
2. Parallel processing benchmark
3. Single core benchmark
4. Multiprocessing benchmark
5. Memory benchmark
6. Report benchmark + visualization
7. C backend benchmark (`kernel_backend=python` vs `kernel_backend=c`, when available)
8. Workload capability matrix check (`benchmarks.check_framex_workloads`)

## Install benchmark dependencies

```bash
python3 -m pip install -e '.[bench]'
```

## Run

```bash
python3 -m benchmarks.benchmark_suite
```

Disable C backend benchmarks:

```bash
python3 -m benchmarks.benchmark_suite --no-c-backend
```

Run workload capability matrix check:

```bash
python3 -m benchmarks.check_framex_workloads
```

Example with custom sizes:

```bash
python3 -m benchmarks.benchmark_suite \
  --rows 500000 \
  --array-elements 4000000 \
  --object-items 600000 \
  --workers 1,2,4,8 \
  --repeats 5 \
  --warmups 1
```

## Outputs

Default output directory: `benchmarks/results`

- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_report.md`
- `framex_workload_check.json`
- `performance_speedup.png` (if matplotlib installed)
- `parallel_processing_scaling.png` (if matplotlib installed)
- `multiprocessing_scaling.png` (if matplotlib installed)
- `memory_peak_rss.png` (if matplotlib installed)
