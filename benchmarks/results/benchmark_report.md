# FrameX Benchmark Report

Generated: 2026-04-09 12:52:46 +07

Command parameters:
- rows: 300000
- repeats: 3
- warmups: 1
- workers: 1,2,4,8

## Compare: Native vs FrameX (Performance)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
     filter_val2_gt_500        1       0.000526       0.000601  0.875x native                                      
   groupby_key_sum_mean        1       0.001536       0.000993  1.547x framex                                      
              sort_val1        1       0.017539       0.017863  0.982x native                                      
             join_inner        1       0.001303       0.001883  0.692x native                                      
            ndarray_sum        1       0.000036       0.000040  0.883x native                                      
ndarray_np_sin_dispatch        1       0.001252       0.001217  1.029x framex                                      
```

## Compare: Native vs FrameX (Parallel processing)

```text
              scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
numeric_kernel_threads        1       0.000974       0.000984  0.990x native                                      
numeric_kernel_threads        2       0.000599       0.000624  0.960x native                                      
numeric_kernel_threads        4       0.000490       0.000505  0.970x native                                      
numeric_kernel_threads        8       0.000603       0.000625  0.964x native                                      
```

## Compare: Native vs FrameX (Single core)

```text
      scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
numeric_kernel        1       0.000908       0.001002  0.906x native                                      
```

## Compare: Native vs FrameX (Multiprocessing)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
object_kernel_processes        1       0.498882       0.499265  0.999x native                                      
object_kernel_processes        2       0.487430       0.481395  1.013x framex                                      
object_kernel_processes        4       0.501971       0.496042  1.012x framex                                      
object_kernel_processes        8       0.595120       0.660479  0.901x native                                      
```

## Compare: Native vs FrameX (Memory)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
     filter_val2_gt_500        1       0.000912       0.000691  1.320x framex             141.12             141.12
        groupby_key_sum        1       0.001900       0.000931  2.040x framex             154.72             154.72
ndarray_np_sin_dispatch        1       0.001567       0.001294  1.212x framex             157.19             157.19
```

## Compare: FrameX Python vs FrameX C backend

```text
                  scenario  workers framex_python_seconds framex_c_seconds speedup        winner framex_python_peak_rss_mb framex_c_peak_rss_mb
         reduction_sum_f64        1              0.000042         0.000042  0.996x framex_python                                               
        reduction_mean_f64        1              0.000042         0.000045  0.935x framex_python                                               
     reduction_min_max_f64        1              0.000308         0.000289  1.066x      framex_c                                               
       elementwise_add_f64        1              0.000043         0.000038  1.120x      framex_c                                               
elementwise_scalar_mul_f64        1              0.000037         0.000037  0.985x framex_python                                               
```

## Detailed rows: Performance

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
     filter_val2_gt_500 native        1 0.000526             1.000            
     filter_val2_gt_500 framex        1 0.000601             0.875            
   groupby_key_sum_mean native        1 0.001536             1.000            
   groupby_key_sum_mean framex        1 0.000993             1.547            
              sort_val1 native        1 0.017539             1.000            
              sort_val1 framex        1 0.017863             0.982            
             join_inner native        1 0.001303             1.000            
             join_inner framex        1 0.001883             0.692            
            ndarray_sum native        1 0.000036             1.000            
            ndarray_sum framex        1 0.000040             0.883            
ndarray_np_sin_dispatch native        1 0.001252             1.000            
ndarray_np_sin_dispatch framex        1 0.001217             1.029            
```

## Detailed rows: Parallel processing

```text
              scenario engine  workers  seconds speedup_vs_native peak_rss_mb
numeric_kernel_threads native        1 0.000974             1.000            
numeric_kernel_threads framex        1 0.000984             0.990            
numeric_kernel_threads native        2 0.000599             1.000            
numeric_kernel_threads framex        2 0.000624             0.960            
numeric_kernel_threads native        4 0.000490             1.000            
numeric_kernel_threads framex        4 0.000505             0.970            
numeric_kernel_threads native        8 0.000603             1.000            
numeric_kernel_threads framex        8 0.000625             0.964            
```

## Detailed rows: Single core

```text
      scenario engine  workers  seconds speedup_vs_native peak_rss_mb
numeric_kernel native        1 0.000908             1.000            
numeric_kernel framex        1 0.001002             0.906            
```

## Detailed rows: Multiprocessing

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
object_kernel_processes native        1 0.498882             1.000            
object_kernel_processes framex        1 0.499265             0.999            
object_kernel_processes native        2 0.487430             1.000            
object_kernel_processes framex        2 0.481395             1.013            
object_kernel_processes native        4 0.501971             1.000            
object_kernel_processes framex        4 0.496042             1.012            
object_kernel_processes native        8 0.595120             1.000            
object_kernel_processes framex        8 0.660479             0.901            
```

## Detailed rows: Memory

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
     filter_val2_gt_500 native        1 0.000912             1.000      141.12
     filter_val2_gt_500 framex        1 0.000691             1.320      141.12
        groupby_key_sum native        1 0.001900             1.000      154.72
        groupby_key_sum framex        1 0.000931             2.040      154.72
ndarray_np_sin_dispatch native        1 0.001567             1.000      157.19
ndarray_np_sin_dispatch framex        1 0.001294             1.212      157.19
```

## Detailed rows: C backend

```text
                  scenario         engine  workers  seconds speedup_vs_native peak_rss_mb
         reduction_sum_f64 python_backend        1 0.000042             1.000            
         reduction_sum_f64      c_backend        1 0.000042             0.996            
        reduction_mean_f64 python_backend        1 0.000042             1.000            
        reduction_mean_f64      c_backend        1 0.000045             0.935            
     reduction_min_max_f64 python_backend        1 0.000308             1.000            
     reduction_min_max_f64      c_backend        1 0.000289             1.066            
       elementwise_add_f64 python_backend        1 0.000043             1.000            
       elementwise_add_f64      c_backend        1 0.000038             1.120            
elementwise_scalar_mul_f64 python_backend        1 0.000037             1.000            
elementwise_scalar_mul_f64      c_backend        1 0.000037             0.985            
```

## Report benchmark and visualize

Generated visualizations:
- `performance_speedup.png`
- `parallel_processing_scaling.png`
- `multiprocessing_scaling.png`
- `memory_peak_rss.png`

Key findings:
- Best FrameX speedup in performance bench: `groupby_key_sum_mean` = 1.55x vs native.
- Toughest performance case for FrameX: `join_inner` = 0.69x vs native.
- Fastest FrameX parallel processing run used 4 workers (0.0005s).
- Fastest FrameX multiprocessing run used 2 workers (0.4814s).
- Highest FrameX measured peak RSS: `ndarray_np_sin_dispatch` = 157.19 MB.
- Best C backend speedup: `elementwise_add_f64` = 1.12x vs python backend.
