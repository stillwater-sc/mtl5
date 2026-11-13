// MTL5 Benchmark -- the interleave width of the generic row kernels (#772)
//
// The generic mat*vec, trmv and forward substitution sum W rows per pass, one
// accumulator each, and give the same bits for every W (#749). Which W is
// fastest depends on the add latency, the register file, the element size and
// the memory bandwidth of the machine, so this tool measures it: it times each
// kernel at every width in {1, 2, 4, 6, 8, 12, 16} for float and double, and
// prints the time and GB/s of each and the -D defines that select the fastest.
// The width this build uses by default (interface::interleave_rows) is marked
// with '*'.
//
// The kernels are called directly (detail::mult_rows_interleaved,
// detail::trmv_lower_rows, detail::lower_trisolve_rows), so BLAS and the
// native-fast paths never take over, and every width runs whatever the
// configuration. Bytes are the compulsory traffic, as in mtl5_bench: all of A
// for gemv, one triangle for trmv and the solve.
//
// It is built with MTL5_INTERLEAVE_RUNTIME (#778), which changes none of the
// timings above: after each type it prints the width run-time selection
// picks on this CPU and how far that is from the fastest, and at the end it
// times what the dispatch costs, on a gemv small enough for it to show.
//
// Usage:
//   bench_interleave_rows [--n 4096] [--reps 15]
// Each entry is the median of --reps timed runs after two warm-up runs.

#include <mtl/interface/interleave_rows.hpp>
#include <mtl/interface/interleave_runtime.hpp>
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
#include <string>
#include <utility>
#include <vector>

namespace {

using widths = std::index_sequence<1, 2, 4, 6, 8, 12, 16>;

/// The median of reps timed calls of run(), each after reset(), in seconds.
template <typename Reset, typename Run>
double median_seconds(Reset&& reset, Run&& run, int reps) {
    for (int i = 0; i < 2; ++i) { reset(); run(); }
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        reset();
        const auto t0 = std::chrono::steady_clock::now();
        run();
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return t[static_cast<std::size_t>(reps / 2)];
}

struct result { std::size_t width; double seconds; };

/// Every kernel's output is summed into this after its timed runs, so the
/// optimizer cannot treat the timed work as dead.
volatile double g_sink = 0;

template <typename V>
void consume(const V& v) {
    double s = 0;
    for (std::size_t i = 0; i < v.size(); ++i) s += static_cast<double>(v(i));
    g_sink = g_sink + s;
}

template <typename T>
const char* type_name() { return sizeof(T) == 4 ? "f32" : "f64"; }

template <typename T>
const char* define_suffix() { return sizeof(T) == 4 ? "F32" : "F64"; }

/// One kernel at every width: prints a row per width, returns them all.
template <typename T, typename Kernel, typename Call, typename Reset, typename Out>
std::vector<result> sweep(const char* kernel, double bytes, Reset&& reset, Call&& call, const Out& out, int reps) {
    constexpr std::size_t dflt = mtl::interface::interleave_rows_v<T, Kernel>;
    std::vector<result> rs;
    [&]<std::size_t... W>(std::index_sequence<W...>) {
        ((rs.push_back({W, median_seconds(reset, [&] { call(std::integral_constant<std::size_t, W>{}); }, reps)}),
          consume(out)),
         ...);
    }(widths{});
    const auto best = std::min_element(rs.begin(), rs.end(),
                                       [](const result& a, const result& b) { return a.seconds < b.seconds; });
    for (const auto& r : rs)
        std::printf("%-15s %-4s W=%-3zu%c %9.3f ms %8.2f GB/s%s\n", kernel, type_name<T>(), r.width,
                    r.width == dflt ? '*' : ' ', r.seconds * 1e3, bytes / r.seconds * 1e-9,
                    &r == &*best ? "  <- fastest" : "");
    return rs;
}

std::size_t fastest(const std::vector<result>& rs) {
    return std::min_element(rs.begin(), rs.end(),
                            [](const result& a, const result& b) { return a.seconds < b.seconds; })->width;
}

/// "W (time at W / time at the fastest)" for the width run-time selection picks.
template <typename T, typename Kernel>
void print_choice(const char* kernel, const std::vector<result>& rs) {
    const std::size_t w = mtl::interface::selected_interleave_rows<T, Kernel>();
    const double best = std::min_element(rs.begin(), rs.end(), [](const result& a, const result& b) {
                            return a.seconds < b.seconds; })->seconds;
    for (const auto& r : rs)
        if (r.width == w) { std::printf(" %s %zu (%.3f)", kernel, w, r.seconds / best); return; }
    std::printf(" %s %zu (not swept)", kernel, w);
}

template <typename T>
void run_type(std::size_t n, int reps) {
    namespace iface = mtl::interface;
    mtl::mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(static_cast<double>((i * 7 + j * 13) % 101) / 404.0 - 0.1);
    for (std::size_t i = 0; i < n; ++i) A(i, i) = static_cast<T>(2.0);
    mtl::vec::dense_vector<T> x0(n), x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) x0(i) = static_cast<T>(static_cast<double>(i % 17) / 17.0);
    const double full = (1.0 * n * n + 2.0 * n) * sizeof(T);
    const double tri  = (0.5 * n * n + 2.0 * n) * sizeof(T);
    const auto none = [] {};
    const auto restore = [&] { x = x0; };

    const auto g = sweep<T, iface::kernel::gemv>("gemv", full, none, [&](auto w) {
        mtl::detail::mult_rows_interleaved<decltype(w)::value>(A, x0, y, 0, n);
    }, y, reps);
    const auto t = sweep<T, iface::kernel::trmv>("trmv lower", tri, restore, [&](auto w) {
        mtl::detail::trmv_lower_rows<decltype(w)::value>(A, x, n, false);
    }, x, reps);
    const auto s = sweep<T, iface::kernel::lower_trisolve>("lower_trisolve", tri, none, [&](auto w) {
        mtl::detail::lower_trisolve_rows<decltype(w)::value>(A, x, x0, false);
    }, x, reps);
    const std::size_t fg = fastest(g), ft = fastest(t), fs = fastest(s);
    std::printf("fastest %s: gemv %zu, trmv %zu, lower_trisolve %zu", type_name<T>(), fg, ft, fs);
    if (fg == ft && ft == fs) std::printf("  -> -DMTL5_INTERLEAVE_ROWS_%s=%zu", define_suffix<T>(), fg);
    std::printf("\nrun-time choice %s (cpu %s; time / fastest):", type_name<T>(),
                to_string(iface::runtime_interleave_cpu()));
    print_choice<T, iface::kernel::gemv>("gemv", g);
    print_choice<T, iface::kernel::trmv>("trmv", t);
    print_choice<T, iface::kernel::lower_trisolve>("lower_trisolve", s);
    std::printf("\n\n");
}

#if defined(_MSC_VER) && !defined(__clang__)
#  define BENCH_NOINLINE __declspec(noinline)
#else
#  define BENCH_NOINLINE __attribute__((noinline))
#endif

/// gemv through detail::mult_generic_rows, which dispatches on the run-time
/// width every call. Out of line, as a caller's call would be: inlined into the
/// timing loop, the compiler can work across the 2000 identical calls, and
/// that (Clang, AArch64: 12% at n = 64) is not what a caller gets.
template <typename T>
BENCH_NOINLINE void gemv_dispatched(const mtl::mat::dense2D<T>& A, const mtl::vec::dense_vector<T>& x,
                                    mtl::vec::dense_vector<T>& y) {
    mtl::detail::mult_generic_rows(A, x, y, 0, A.num_rows());
}

/// The same width's kernel, called directly and just as far out of line.
template <std::size_t W, typename T>
BENCH_NOINLINE void gemv_direct(const mtl::mat::dense2D<T>& A, const mtl::vec::dense_vector<T>& x,
                                mtl::vec::dense_vector<T>& y) {
    mtl::detail::mult_rows_interleaved<W>(A, x, y, 0, A.num_rows());
}

/// The cost of choosing the width per call, on a gemv small enough to show it.
template <typename T>
void dispatch_cost(std::size_t n, int reps) {
    namespace iface = mtl::interface;
    mtl::mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<T>(static_cast<double>((i + 2 * j) % 7) / 7.0);
    mtl::vec::dense_vector<T> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) x(i) = static_cast<T>(static_cast<double>(i % 5) / 5.0);
    constexpr int calls = 2000;
    const auto none = [] {};
    const double dispatched = median_seconds(none, [&] {
        for (int c = 0; c < calls; ++c) gemv_dispatched(A, x, y);
    }, reps) / calls;
    consume(y);
    double direct = 0;
    iface::with_interleave_rows<T, iface::kernel::gemv>([&](auto w) {
        direct = median_seconds(none, [&] {
            for (int c = 0; c < calls; ++c) gemv_direct<decltype(w)::value>(A, x, y);
        }, reps) / calls;
    });
    consume(y);
    std::printf("dispatch %s gemv n=%zu W=%zu: %8.1f ns dispatched, %8.1f ns direct (%+.1f%%)\n", type_name<T>(), n,
                iface::selected_interleave_rows<T, iface::kernel::gemv>(), dispatched * 1e9, direct * 1e9,
                (dispatched / direct - 1.0) * 100.0);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t n = 4096;
    int reps = 15;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else {
            std::printf("usage: bench_interleave_rows [--n 4096] [--reps 15]\n");
            return argv[i][0] == '-' && argv[i][1] == 'h' ? 0 : 1;
        }
    }
    std::printf("interleave width sweep, n = %zu, median of %d; vector registers = %zu; * = this build's default\n\n",
                n, reps, mtl::interface::vector_registers);
    run_type<double>(n, reps);
    run_type<float>(n, reps);
    for (std::size_t m : {16, 64}) {
        dispatch_cost<double>(m, reps);
        dispatch_cost<float>(m, reps);
    }
    std::printf("\n");
    std::printf("A width differing per kernel is set by specializing "
                "mtl::interface::interleave_rows<T, Kernel>.\n");
}
