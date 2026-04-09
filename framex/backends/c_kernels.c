/*
 * FrameX C kernels — pure, GIL-free compute kernels for hot paths.
 *
 * Compile flags expected: -O2 -shared -fPIC -lm
 * Supports: float64 (double) and int64 reductions, elementwise ops,
 *           and column filter.
 *
 * Null handling: callers must verify null_count == 0 before using these
 * kernels; Arrow arrays with nulls must fall back to pyarrow.compute.
 */

#include <stdint.h>
#include <math.h>

/* ── Reductions ────────────────────────────────────────────────────────── */

double fx_sum_f64(const double *arr, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += arr[i];
    return s;
}

int64_t fx_sum_i64(const int64_t *arr, int64_t n) {
    int64_t s = 0;
    for (int64_t i = 0; i < n; i++) s += arr[i];
    return s;
}

double fx_mean_f64(const double *arr, int64_t n) {
    if (n == 0) return 0.0;
    return fx_sum_f64(arr, n) / (double)n;
}

double fx_mean_i64(const int64_t *arr, int64_t n) {
    if (n == 0) return 0.0;
    return (double)fx_sum_i64(arr, n) / (double)n;
}

/* ddof: degrees-of-freedom correction (0 = population, 1 = sample) */
double fx_std_f64(const double *arr, int64_t n, int ddof) {
    if (n <= ddof) return 0.0;
    double mean = fx_mean_f64(arr, n);
    double sq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = arr[i] - mean;
        sq += d * d;
    }
    return sqrt(sq / (double)(n - ddof));
}

double fx_var_f64(const double *arr, int64_t n, int ddof) {
    if (n <= ddof) return 0.0;
    double mean = fx_mean_f64(arr, n);
    double sq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = arr[i] - mean;
        sq += d * d;
    }
    return sq / (double)(n - ddof);
}

double fx_min_f64(const double *arr, int64_t n) {
    if (n == 0) return 0.0;
    double m = arr[0];
    for (int64_t i = 1; i < n; i++) if (arr[i] < m) m = arr[i];
    return m;
}

double fx_max_f64(const double *arr, int64_t n) {
    if (n == 0) return 0.0;
    double m = arr[0];
    for (int64_t i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
    return m;
}

int64_t fx_min_i64(const int64_t *arr, int64_t n) {
    if (n == 0) return 0;
    int64_t m = arr[0];
    for (int64_t i = 1; i < n; i++) if (arr[i] < m) m = arr[i];
    return m;
}

int64_t fx_max_i64(const int64_t *arr, int64_t n) {
    if (n == 0) return 0;
    int64_t m = arr[0];
    for (int64_t i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
    return m;
}

/* ── Elementwise: array × array ────────────────────────────────────────── */

void fx_add_f64(const double *a, const double *b, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void fx_sub_f64(const double *a, const double *b, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void fx_mul_f64(const double *a, const double *b, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}

void fx_div_f64(const double *a, const double *b, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] / b[i];
}

/* ── Elementwise: array × scalar ───────────────────────────────────────── */

void fx_scalar_add_f64(const double *a, double scalar, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + scalar;
}

void fx_scalar_sub_f64(const double *a, double scalar, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] - scalar;
}

void fx_scalar_mul_f64(const double *a, double scalar, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * scalar;
}

void fx_scalar_div_f64(const double *a, double scalar, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] / scalar;
}

/* ── Column filter ─────────────────────────────────────────────────────── */

/*
 * Filter a float64 column by a boolean mask (1 = keep, 0 = drop).
 * `dst` must be pre-allocated with at least `n` elements.
 * Returns the number of elements written to `dst`.
 */
int64_t fx_filter_f64(const double *src, const uint8_t *mask,
                      double *dst, int64_t n) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        if (mask[i]) dst[k++] = src[i];
    }
    return k;
}

int64_t fx_filter_i64(const int64_t *src, const uint8_t *mask,
                      int64_t *dst, int64_t n) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        if (mask[i]) dst[k++] = src[i];
    }
    return k;
}
