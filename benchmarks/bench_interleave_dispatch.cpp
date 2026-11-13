// MTL5 Benchmark -- what run-time selection of the interleave width costs per call (#778)
//
// With MTL5_INTERLEAVE_RUNTIME the generic gemv picks its width with a switch
// on every call. This times the generic gemv (detail::mult_generic_rows) at
// small n, where that would show. CMake builds it twice from this file:
// bench_interleave_dispatch with the compile-time width, and
// bench_interleave_dispatch_runtime with run-time selection. Compare the two
// at the same width: the second prints the width it picked, and --force W
// makes it use W (interface::force_interleave_rows).
//
// The call is kept out of line, as a caller's would be. Inlined into the
// timing loop, the compiler can work across the identical calls, and timing a
// kernel that way against one called through mult_generic_rows measured a
// 12% gap (Clang 18, f64 at n = 64, Ventuno Q) that the off and on builds
// share.
//
// Usage:
//   bench_interleave_dispatch[_runtime] [--reps 31] [--force W]
// Each entry is the median of --reps timed batches of 2000 calls, per call.

// The compile-time build stays one when the tree is configured with
// MTL5_INTERLEAVE_RUNTIME=ON.
#if defined(BENCH_COMPILE_TIME_WIDTH)
#  undef MTL5_INTERLEAVE_RUNTIME
#endif

#include <mtl/interface/interleave_rows.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#  define BENCH_NOINLINE __declspec(noinline)
#else
#  define BENCH_NOINLINE __attribute__((noinline))
#endif

namespace {

volatile double g_sink = 0;

template <typename T>
BENCH_NOINLINE void gemv(const mtl::mat::dense2D<T>& A, const mtl::vec::dense_vector<T>& x,
                         mtl::vec::dense_vector<T>& y) {
    mtl::detail::mult_generic_rows(A, x, y, 0, A.num_rows());
}

template <typename T>
void run(std::size_t n, int reps) {
    mtl::mat::dense2D<T> A(n, n);
    mtl::vec::dense_vector<T> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = static_cast<T>(static_cast<double>(i % 5) / 5.0);
        for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<T>(static_cast<double>((i + 2 * j) % 7) / 7.0);
    }
    constexpr int calls = 2000;
    for (int c = 0; c < calls; ++c) gemv(A, x, y);   // warm-up
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < calls; ++c) gemv(A, x, y);
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / calls);
    }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    g_sink = g_sink + static_cast<double>(y(0));
    std::printf("gemv %s n=%-4zu W=%-3zu %9.1f ns\n", sizeof(T) == 4 ? "f32" : "f64", n,
                mtl::interface::selected_interleave_rows<T, mtl::interface::kernel::gemv>(),
                t[static_cast<std::size_t>(reps / 2)] * 1e9);
}

} // namespace

int main(int argc, char** argv) {
    int reps = 31;
    std::size_t force = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--force") && i + 1 < argc) force = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::printf("usage: bench_interleave_dispatch[_runtime] [--reps 31] [--force W]\n");
            return argv[i][0] == '-' && argv[i][1] == 'h' ? 0 : 1;
        }
    }
#if defined(MTL5_INTERLEAVE_RUNTIME)
    if (force) {
        if (!mtl::interface::is_runtime_interleave_width(force)) {
            std::printf("--force: W must be 2, 4, 6, 8 or 12\n");
            return 1;
        }
        mtl::interface::force_interleave_rows(force);
    }
    std::printf("generic gemv, run-time width selection, median of %d\n", reps);
#else
    if (force) {
        std::printf("--force needs the run-time build, bench_interleave_dispatch_runtime\n");
        return 1;
    }
    std::printf("generic gemv, compile-time width, median of %d\n", reps);
#endif
    for (std::size_t n : {16, 64, 256}) {
        run<double>(n, reps);
        run<float>(n, reps);
    }
}
