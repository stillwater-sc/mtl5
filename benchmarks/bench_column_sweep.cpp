// MTL5 Benchmark -- the column sweep against the row kernel, by shape (#884)
//
// A contiguous column-major A takes the column sweep in gemv, trmv and the
// forward substitution (#777). It is much faster than the row kernels on large
// matrices, and was measured slower on small and on short, wide ones (#884).
// This tool times both on the same column-major A for each shape, so the
// crossover can be read off on each machine:
//
//   sweep  detail::mult_rows_column_sweep / trmv_column_sweep /
//          lower_trisolve_column_sweep, S = column_strip_rows<T>, C =
//          column_sweep_columns columns per pass;
//   sweep1 the same with C = 1, the sweep as #777 wrote it;
//   rows   detail::mult_rows_interleaved / trmv_lower_rows /
//          lower_trisolve_rows at this build's default width
//          (interface::interleave_rows_v<T, kernel>);
//   public mtl::mult / mtl::trmv / mtl::lower_trisolve, with the dispatch
//          #884 settled on (the row kernels for the smallest shapes, the
//          sweep above them, one column per pass at wide column strides).
//
// All give the one-row loop's bits, so only the speed differs. A ratio
// rows / sweep above 1 means the sweep is faster.
//
// Each sample runs the kernel enough times to take at least --min-us
// microseconds, and an entry is the median of --reps samples per call.
// Output is CSV on stdout:
//
//   kernel,type,m,n,sweep_ns,sweep1_ns,rows_ns,public_ns,rows_over_sweep,rows_over_sweep1,sweep1_over_public
//
// The public calls thread over row bands when the pool has more than one
// thread; run with MTL5_NUM_THREADS=1 (run_column_sweep.sh/.ps1 do) to time
// one core.
//
// Usage:
//   bench_column_sweep [--reps 11] [--min-us 2000] [--quick]

#include <mtl/interface/interleave_rows.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

using cm = mtl::mat::parameters<mtl::tag::col_major>;
template <typename T>
using cmat = mtl::mat::dense2D<T, cm>;

volatile double g_sink = 0;

template <typename V>
void consume(const V& v) {
    double s = 0;
    for (std::size_t i = 0; i < v.size(); ++i) s += static_cast<double>(v(i));
    g_sink = g_sink + s;
}

/// Every kernel of an entry runs once before any is timed (warm_up): the
/// first one timed otherwise paid for faulting in and warming A, which read
/// as the sweep 25-35% slower than one column per pass at n = 1024-4096 when
/// the two run the same loop (#884).
template <typename... F>
void warm_up(F&&... f) {
    (f(), ...);
}

/// Nanoseconds per call: the median of reps samples, each a batch of calls
/// long enough to take min_us, with reset() before every call.
template <typename Reset, typename Run>
double ns_per_call(Reset&& reset, Run&& run, int reps, double min_us) {
    using clk = std::chrono::steady_clock;
    // Calibrate the batch size.
    std::size_t batch = 1;
    for (;;) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < batch; ++i) { reset(); run(); }
        const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
        if (us >= min_us || batch >= (std::size_t(1) << 30)) break;
        batch = us > 0 ? std::max(batch * 2, static_cast<std::size_t>(batch * min_us / us * 1.1)) : batch * 8;
    }
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < batch; ++i) { reset(); run(); }
        t.push_back(std::chrono::duration<double, std::nano>(clk::now() - t0).count() / double(batch));
    }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return t[static_cast<std::size_t>(reps / 2)];
}

template <typename T>
const char* type_name() { return sizeof(T) == 4 ? "f32" : "f64"; }

template <typename T>
cmat<T> make_matrix(std::size_t m, std::size_t n) {
    cmat<T> A(m, n);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i)
            A(i, j) = static_cast<T>(static_cast<double>((i * 7 + j * 13) % 101) / 404.0 - 0.1);
    for (std::size_t i = 0; i < std::min(m, n); ++i) A(i, i) = static_cast<T>(2.0);
    return A;
}

void print(const char* kernel, const char* type, std::size_t m, std::size_t n, double sweep, double sweep1,
           double rows, double pub) {
    std::printf("%s,%s,%zu,%zu,%.1f,%.1f,%.1f,%.1f,%.3f,%.3f,%.3f\n", kernel, type, m, n, sweep, sweep1, rows, pub,
                rows / sweep, rows / sweep1, sweep1 / pub);
    std::fflush(stdout);
}

template <typename T>
void gemv(std::size_t m, std::size_t n, int reps, double min_us) {
    namespace iface = mtl::interface;
    constexpr std::size_t W = iface::interleave_rows_v<T, iface::kernel::gemv>;
    constexpr std::size_t S = mtl::detail::column_strip_rows<T>;
    const auto A = make_matrix<T>(m, n);
    mtl::vec::dense_vector<T> x(n), y(m);
    for (std::size_t i = 0; i < n; ++i) x(i) = static_cast<T>(static_cast<double>(i % 17) / 17.0);
    const auto none = [] {};
    warm_up([&] { mtl::detail::mult_rows_column_sweep<S>(A, x, y, 0, m); },
            [&] { mtl::detail::mult_rows_column_sweep<S, 1>(A, x, y, 0, m); },
            [&] { mtl::detail::mult_rows_interleaved<W>(A, x, y, 0, m); }, [&] { mtl::mult(A, x, y); });
    const double s = ns_per_call(none, [&] { mtl::detail::mult_rows_column_sweep<S>(A, x, y, 0, m); }, reps, min_us);
    consume(y);
    const double s1 = ns_per_call(none, [&] { mtl::detail::mult_rows_column_sweep<S, 1>(A, x, y, 0, m); }, reps, min_us);
    consume(y);
    const double r = ns_per_call(none, [&] { mtl::detail::mult_rows_interleaved<W>(A, x, y, 0, m); }, reps, min_us);
    consume(y);
    const double p = ns_per_call(none, [&] { mtl::mult(A, x, y); }, reps, min_us);
    consume(y);
    print("gemv", type_name<T>(), m, n, s, s1, r, p);
}

template <typename T>
void trmv(std::size_t n, int reps, double min_us) {
    namespace iface = mtl::interface;
    constexpr std::size_t W = iface::interleave_rows_v<T, iface::kernel::trmv>;
    constexpr std::size_t S = mtl::detail::column_strip_rows<T>;
    const auto A = make_matrix<T>(n, n);
    mtl::vec::dense_vector<T> x0(n), x(n);
    for (std::size_t i = 0; i < n; ++i) x0(i) = static_cast<T>(static_cast<double>(i % 17) / 17.0);
    const auto restore = [&] { for (std::size_t i = 0; i < n; ++i) x(i) = x0(i); };
    // The restore is in both timings; it is O(n) against the kernels' O(n^2).
    warm_up([&] { restore(); mtl::detail::trmv_column_sweep<S>(A, x, false, false); },
            [&] { restore(); mtl::detail::trmv_column_sweep<S, 1>(A, x, false, false); },
            [&] { restore(); mtl::detail::trmv_lower_rows<W>(A, x, n, false); },
            [&] { restore(); mtl::trmv(A, x, false, false); });
    const double s = ns_per_call(restore, [&] { mtl::detail::trmv_column_sweep<S>(A, x, false, false); }, reps, min_us);
    consume(x);
    const double s1 = ns_per_call(restore, [&] { mtl::detail::trmv_column_sweep<S, 1>(A, x, false, false); }, reps, min_us);
    consume(x);
    const double r = ns_per_call(restore, [&] { mtl::detail::trmv_lower_rows<W>(A, x, n, false); }, reps, min_us);
    consume(x);
    const double p = ns_per_call(restore, [&] { mtl::trmv(A, x, false, false); }, reps, min_us);
    consume(x);
    print("trmv_lower", type_name<T>(), n, n, s, s1, r, p);
}

template <typename T>
void trisolve(std::size_t n, int reps, double min_us) {
    namespace iface = mtl::interface;
    constexpr std::size_t W = iface::interleave_rows_v<T, iface::kernel::lower_trisolve>;
    constexpr std::size_t S = mtl::detail::column_strip_rows<T>;
    const auto A = make_matrix<T>(n, n);
    mtl::vec::dense_vector<T> b(n), x(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = static_cast<T>(static_cast<double>(i % 17) / 17.0);
    const auto none = [] {};
    warm_up([&] { mtl::detail::lower_trisolve_column_sweep<S>(A, x, b, false); },
            [&] { mtl::detail::lower_trisolve_column_sweep<S, 1>(A, x, b, false); },
            [&] { mtl::detail::lower_trisolve_rows<W>(A, x, b, false); },
            [&] { mtl::lower_trisolve(A, x, b, false); });
    const double s = ns_per_call(none, [&] { mtl::detail::lower_trisolve_column_sweep<S>(A, x, b, false); }, reps, min_us);
    consume(x);
    const double s1 = ns_per_call(none, [&] { mtl::detail::lower_trisolve_column_sweep<S, 1>(A, x, b, false); }, reps, min_us);
    consume(x);
    const double r = ns_per_call(none, [&] { mtl::detail::lower_trisolve_rows<W>(A, x, b, false); }, reps, min_us);
    consume(x);
    const double p = ns_per_call(none, [&] { mtl::lower_trisolve(A, x, b, false); }, reps, min_us);
    consume(x);
    print("lower_trisolve", type_name<T>(), n, n, s, s1, r, p);
}

template <typename T>
void run_type(bool quick, int reps, double min_us) {
    const std::vector<std::size_t> square = quick ? std::vector<std::size_t>{4, 64, 1024}
        : std::vector<std::size_t>{2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096};
    const std::vector<std::size_t> short_m = quick ? std::vector<std::size_t>{4}
        : std::vector<std::size_t>{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    const std::vector<std::size_t> wide_n = quick ? std::vector<std::size_t>{1000}
        : std::vector<std::size_t>{256, 1000, 4096};
    for (const auto n : square) gemv<T>(n, n, reps, min_us);
    for (const auto m : short_m)
        for (const auto n : wide_n) gemv<T>(m, n, reps, min_us);
    // Tall, narrow: few columns, many rows.
    for (const auto mn : quick ? std::vector<std::pair<std::size_t, std::size_t>>{{1000, 4}}
                               : std::vector<std::pair<std::size_t, std::size_t>>{{1000, 4}, {4096, 8}, {256, 16}})
        gemv<T>(mn.first, mn.second, reps, min_us);
    for (const auto n : square) trmv<T>(n, reps, min_us);
    for (const auto n : square) trisolve<T>(n, reps, min_us);
}

} // namespace

int main(int argc, char** argv) {
    int reps = 11;
    double min_us = 2000;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--min-us") && i + 1 < argc) min_us = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--quick")) quick = true;
        else {
            std::printf("usage: bench_column_sweep [--reps 11] [--min-us 2000] [--quick]\n");
            return argv[i][0] == '-' && argv[i][1] == 'h' ? 0 : 1;
        }
    }
    namespace iface = mtl::interface;
    std::printf("# column sweep vs row kernel on a column-major A; median of %d samples of >= %.0f us\n",
                reps, min_us);
    std::printf("# strip rows f64 %zu f32 %zu; row widths gemv/trmv/trisolve f64 %zu/%zu/%zu f32 %zu/%zu/%zu\n",
                mtl::detail::column_strip_rows<double>, mtl::detail::column_strip_rows<float>,
                iface::interleave_rows_v<double, iface::kernel::gemv>,
                iface::interleave_rows_v<double, iface::kernel::trmv>,
                iface::interleave_rows_v<double, iface::kernel::lower_trisolve>,
                iface::interleave_rows_v<float, iface::kernel::gemv>,
                iface::interleave_rows_v<float, iface::kernel::trmv>,
                iface::interleave_rows_v<float, iface::kernel::lower_trisolve>);
    std::printf("# columns per pass %zu\n", mtl::detail::column_sweep_columns);
    std::printf("kernel,type,m,n,sweep_ns,sweep1_ns,rows_ns,public_ns,rows_over_sweep,rows_over_sweep1,"
                "sweep1_over_public\n");
    run_type<double>(quick, reps, min_us);
    run_type<float>(quick, reps, min_us);
    return g_sink == 12345.678 ? 1 : 0;
}
