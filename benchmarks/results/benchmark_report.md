# FrameX Benchmark Report

Generated: 2026-04-09 12:01:30 +07

Command parameters:
- rows: 300000
- repeats: 3
- warmups: 1
- workers: 1,2,4,8

## Compare: Native vs FrameX (Performance)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
     filter_val2_gt_500        1       0.000608       0.000610  0.998x native                                      
   groupby_key_sum_mean        1       0.001498       0.000976  1.535x framex                                      
              sort_val1        1       0.017562       0.017765  0.989x native                                      
             join_inner        1       0.001302       0.001799  0.723x native                                      
            ndarray_sum        1       0.000033       0.000039  0.849x native                                      
ndarray_np_sin_dispatch        1       0.001248       0.001386  0.901x native                                      
```

## Compare: Native vs FrameX (Parallel processing)

```text
              scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
numeric_kernel_threads        1       0.001034       0.000987  1.048x framex                                      
numeric_kernel_threads        2       0.000611       0.000665  0.919x native                                      
numeric_kernel_threads        4       0.000495       0.000520  0.952x native                                      
numeric_kernel_threads        8       0.000599       0.000639  0.937x native                                      
```

## Compare: Native vs FrameX (Single core)

```text
      scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
numeric_kernel        1       0.000962       0.001016  0.947x native                                      
```

## Compare: Native vs FrameX (Multiprocessing)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
object_kernel_processes        1       0.512135       0.521872  0.981x native                                      
object_kernel_processes        2       0.494854       0.495764  0.998x native                                      
object_kernel_processes        4       0.495986       0.491521  1.009x framex                                      
object_kernel_processes        8       0.573142       0.574908  0.997x native                                      
```

## Compare: Native vs FrameX (Memory)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
     filter_val2_gt_500        1       0.000672       0.000638  1.053x framex             346.28             346.28
        groupby_key_sum        1       0.001265       0.000871  1.453x framex             346.30             346.30
ndarray_np_sin_dispatch        1       0.001257       0.001682  0.748x native             346.64             346.64
```

## Compare: FrameX Python vs FrameX C backend

```text
                  scenario  workers framex_python_seconds framex_c_seconds speedup        winner framex_python_peak_rss_mb framex_c_peak_rss_mb
         reduction_sum_f64        1              0.000040         0.000152  0.266x framex_python                                               
        reduction_mean_f64        1              0.000040         0.000157  0.255x framex_python                                               
     reduction_min_max_f64        1              0.000272         0.000291  0.935x framex_python                                               
       elementwise_add_f64        1              0.000041         0.000130  0.317x framex_python                                               
elementwise_scalar_mul_f64        1              0.000038         0.000076  0.500x framex_python                                               
```

## Detailed rows: Performance

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
     filter_val2_gt_500 native        1 0.000608             1.000            
     filter_val2_gt_500 framex        1 0.000610             0.998            
   groupby_key_sum_mean native        1 0.001498             1.000            
   groupby_key_sum_mean framex        1 0.000976             1.535            
              sort_val1 native        1 0.017562             1.000            
              sort_val1 framex        1 0.017765             0.989            
             join_inner native        1 0.001302             1.000            
             join_inner framex        1 0.001799             0.723            
            ndarray_sum native        1 0.000033             1.000            
            ndarray_sum framex        1 0.000039             0.849            
ndarray_np_sin_dispatch native        1 0.001248             1.000            
ndarray_np_sin_dispatch framex        1 0.001386             0.901            
```

## Detailed rows: Parallel processing

```text
              scenario engine  workers  seconds speedup_vs_native peak_rss_mb
numeric_kernel_threads native        1 0.001034             1.000            
numeric_kernel_threads framex        1 0.000987             1.048            
numeric_kernel_threads native        2 0.000611             1.000            
numeric_kernel_threads framex        2 0.000665             0.919            
numeric_kernel_threads native        4 0.000495             1.000            
numeric_kernel_threads framex        4 0.000520             0.952            
numeric_kernel_threads native        8 0.000599             1.000            
numeric_kernel_threads framex        8 0.000639             0.937            
```

## Detailed rows: Single core

```text
      scenario engine  workers  seconds speedup_vs_native peak_rss_mb
numeric_kernel native        1 0.000962             1.000            
numeric_kernel framex        1 0.001016             0.947            
```

## Detailed rows: Multiprocessing

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
object_kernel_processes native        1 0.512135             1.000            
object_kernel_processes framex        1 0.521872             0.981            
object_kernel_processes native        2 0.494854             1.000            
object_kernel_processes framex        2 0.495764             0.998            
object_kernel_processes native        4 0.495986             1.000            
object_kernel_processes framex        4 0.491521             1.009            
object_kernel_processes native        8 0.573142             1.000            
object_kernel_processes framex        8 0.574908             0.997            
```

## Detailed rows: Memory

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
     filter_val2_gt_500 native        1 0.000672             1.000      346.28
     filter_val2_gt_500 framex        1 0.000638             1.053      346.28
        groupby_key_sum native        1 0.001265             1.000      346.30
        groupby_key_sum framex        1 0.000871             1.453      346.30
ndarray_np_sin_dispatch native        1 0.001257             1.000      346.64
ndarray_np_sin_dispatch framex        1 0.001682             0.748      346.64
```

## Detailed rows: C backend

```text
                  scenario         engine  workers  seconds speedup_vs_native peak_rss_mb
         reduction_sum_f64 python_backend        1 0.000040             1.000            
         reduction_sum_f64      c_backend        1 0.000152             0.266            
        reduction_mean_f64 python_backend        1 0.000040             1.000            
        reduction_mean_f64      c_backend        1 0.000157             0.255            
     reduction_min_max_f64 python_backend        1 0.000272             1.000            
     reduction_min_max_f64      c_backend        1 0.000291             0.935            
       elementwise_add_f64 python_backend        1 0.000041             1.000            
       elementwise_add_f64      c_backend        1 0.000130             0.317            
elementwise_scalar_mul_f64 python_backend        1 0.000038             1.000            
elementwise_scalar_mul_f64      c_backend        1 0.000076             0.500            
```

## Report benchmark and visualize

Generated visualizations:
- `performance_speedup.png`
- `parallel_processing_scaling.png`
- `multiprocessing_scaling.png`
- `memory_peak_rss.png`

Key findings:
- Best FrameX speedup in performance bench: `groupby_key_sum_mean` = 1.54x vs native.
- Toughest performance case for FrameX: `join_inner` = 0.72x vs native.
- Fastest FrameX parallel processing run used 4 workers (0.0005s).
- Fastest FrameX multiprocessing run used 4 workers (0.4915s).
- Highest FrameX measured peak RSS: `ndarray_np_sin_dispatch` = 346.64 MB.
- Best C backend speedup: `reduction_min_max_f64` = 0.94x vs python backend.
