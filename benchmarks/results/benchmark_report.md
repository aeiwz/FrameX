# FrameX Benchmark Report

Generated: 2026-04-09 14:14:36 +07

Command parameters:
- rows: 50000
- repeats: 1
- warmups: 0
- workers: 1,2,4

## Compare: Native vs FrameX (Performance)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
     filter_val2_gt_500        1       0.000119       0.000077  1.545x framex                                      
   groupby_key_sum_mean        1       0.000454       0.000207  2.192x framex                                      
              sort_val1        1       0.002371       0.002422  0.979x native                                      
             join_inner        1       0.000380       0.000600  0.633x native                                      
            ndarray_sum        1       0.000006       0.000008  0.800x native                                      
ndarray_np_sin_dispatch        1       0.000197       0.000165  1.192x framex                                      
```

## Compare: Native vs FrameX (Parallel processing)

```text
              scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
numeric_kernel_threads        1       0.000135       0.000140  0.964x native                                      
numeric_kernel_threads        2       0.000148       0.000179  0.828x native                                      
numeric_kernel_threads        4       0.000257       0.000265  0.971x native                                      
```

## Compare: Native vs FrameX (Single core)

```text
      scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
numeric_kernel        1       0.000090       0.000142  0.635x native                                      
```

## Compare: Native vs FrameX (Multiprocessing)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
object_kernel_processes        1       0.429862       0.422515  1.017x framex                                      
object_kernel_processes        2       0.466965       0.421897  1.107x framex                                      
object_kernel_processes        4       0.488280       0.472171  1.034x framex                                      
```

## Compare: Native vs FrameX (Memory)

```text
               scenario  workers native_seconds framex_seconds speedup winner native_peak_rss_mb framex_peak_rss_mb
     filter_val2_gt_500        1       0.000152       0.000094  1.617x framex             168.06             168.06
        groupby_key_sum        1       0.000434       0.000188  2.307x framex             168.56             168.56
ndarray_np_sin_dispatch        1       0.000198       0.000188  1.053x framex             168.61             168.59
```

## Compare: FrameX Python vs FrameX C backend

```text
                  scenario  workers framex_python_seconds framex_c_seconds speedup        winner framex_python_peak_rss_mb framex_c_peak_rss_mb
         reduction_sum_f64        1              0.000007         0.000008  0.969x framex_python                                               
        reduction_mean_f64        1              0.000007         0.000008  0.986x framex_python                                               
     reduction_min_max_f64        1              0.000047         0.000048  0.991x framex_python                                               
       elementwise_add_f64        1              0.000006         0.000007  0.954x framex_python                                               
elementwise_scalar_mul_f64        1              0.000008         0.000009  0.901x framex_python                                               
```

## Detailed rows: Performance

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
     filter_val2_gt_500 native        1 0.000119             1.000            
     filter_val2_gt_500 framex        1 0.000077             1.545            
   groupby_key_sum_mean native        1 0.000454             1.000            
   groupby_key_sum_mean framex        1 0.000207             2.192            
              sort_val1 native        1 0.002371             1.000            
              sort_val1 framex        1 0.002422             0.979            
             join_inner native        1 0.000380             1.000            
             join_inner framex        1 0.000600             0.633            
            ndarray_sum native        1 0.000006             1.000            
            ndarray_sum framex        1 0.000008             0.800            
ndarray_np_sin_dispatch native        1 0.000197             1.000            
ndarray_np_sin_dispatch framex        1 0.000165             1.192            
```

## Detailed rows: Parallel processing

```text
              scenario engine  workers  seconds speedup_vs_native peak_rss_mb
numeric_kernel_threads native        1 0.000135             1.000            
numeric_kernel_threads framex        1 0.000140             0.964            
numeric_kernel_threads native        2 0.000148             1.000            
numeric_kernel_threads framex        2 0.000179             0.828            
numeric_kernel_threads native        4 0.000257             1.000            
numeric_kernel_threads framex        4 0.000265             0.971            
```

## Detailed rows: Single core

```text
      scenario engine  workers  seconds speedup_vs_native peak_rss_mb
numeric_kernel native        1 0.000090             1.000            
numeric_kernel framex        1 0.000142             0.635            
```

## Detailed rows: Multiprocessing

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
object_kernel_processes native        1 0.429862             1.000            
object_kernel_processes framex        1 0.422515             1.017            
object_kernel_processes native        2 0.466965             1.000            
object_kernel_processes framex        2 0.421897             1.107            
object_kernel_processes native        4 0.488280             1.000            
object_kernel_processes framex        4 0.472171             1.034            
```

## Detailed rows: Memory

```text
               scenario engine  workers  seconds speedup_vs_native peak_rss_mb
     filter_val2_gt_500 native        1 0.000152             1.000      168.06
     filter_val2_gt_500 framex        1 0.000094             1.617      168.06
        groupby_key_sum native        1 0.000434             1.000      168.56
        groupby_key_sum framex        1 0.000188             2.307      168.56
ndarray_np_sin_dispatch native        1 0.000198             1.000      168.61
ndarray_np_sin_dispatch framex        1 0.000188             1.053      168.59
```

## Detailed rows: C backend

```text
                  scenario         engine  workers  seconds speedup_vs_native peak_rss_mb
         reduction_sum_f64 python_backend        1 0.000007             1.000            
         reduction_sum_f64      c_backend        1 0.000008             0.969            
        reduction_mean_f64 python_backend        1 0.000007             1.000            
        reduction_mean_f64      c_backend        1 0.000008             0.986            
     reduction_min_max_f64 python_backend        1 0.000047             1.000            
     reduction_min_max_f64      c_backend        1 0.000048             0.991            
       elementwise_add_f64 python_backend        1 0.000006             1.000            
       elementwise_add_f64      c_backend        1 0.000007             0.954            
elementwise_scalar_mul_f64 python_backend        1 0.000008             1.000            
elementwise_scalar_mul_f64      c_backend        1 0.000009             0.901            
```

## Report benchmark and visualize

Generated visualizations:
- `performance_speedup.png`
- `parallel_processing_scaling.png`
- `multiprocessing_scaling.png`
- `memory_peak_rss.png`

Key findings:
- Best FrameX speedup in performance bench: `groupby_key_sum_mean` = 2.19x vs native.
- Toughest performance case for FrameX: `join_inner` = 0.63x vs native.
- Fastest FrameX parallel processing run used 1 workers (0.0001s).
- Fastest FrameX multiprocessing run used 2 workers (0.4219s).
- Highest FrameX measured peak RSS: `ndarray_np_sin_dispatch` = 168.59 MB.
- Best C backend speedup: `reduction_min_max_f64` = 0.99x vs python backend.
