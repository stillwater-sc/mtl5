// Native KLU: the wall-clock scaling claim, in the tier built for it (#454).
//
// The unit suite asserts that FILL grows sub-quadratically, which is the
// deterministic observable and needs no clock. This is the complementary claim
// -- that the *time* follows it, i.e. the implementation is not doing extra
// asymptotic work the fill count cannot see -- and it belongs here rather than
// in the fast tier for reasons that are measurement, not taste:
//
//   * this tier is gated behind MTL5_BUILD_REGRESSION_TESTS and runs Tier 2 on
//     Linux GCC only, not on the shared macOS runners where the old assertion
//     failed on a PR that changed no C++
//   * 300 s timeouts leave room to take REPEATED samples, so the statistic can
//     be a minimum rather than a single draw
//
// That second point is the substance. The retired assertion divided one timing
// by another, doubling the variance of an already noisy quantity: sampled three
// times on an idle machine it moved 5.24, 5.91, 6.87 -- 31% -- and CI produced
// 15.0 against a threshold of 13.0. A minimum over several reps is the standard
// fix (the benchmark harness in benchmarks/ uses it throughout: the minimum is
// the least contaminated sample, where a mean folds in every interruption).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/generators/poisson.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>

using namespace mtl;

namespace {

/// Minimum wall-clock over `reps` factor+solve rounds, in milliseconds.
double min_factor_solve_ms(const mat::compressed2D<double>& A, int reps) {
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> ones(n, 1.0), b(n, 0.0);

    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t r = 0; r < n; ++r) {
        double s = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            s += dat[k] * ones(static_cast<int>(ci[k]));
        b(static_cast<int>(r)) = s;
    }

    double best = 0.0;
    for (int i = 0; i < reps; ++i) {
        vec::dense_vector<double> x(n, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        auto fac = sparse::factorization::native_klu_factor(A);
        fac.solve(x, b);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (i == 0 || ms < best) best = ms;
    }
    return best;
}

} // namespace

TEST_CASE("native KLU factor+solve time grows sub-quadratically",
          "[sparse][klu][native][scaling][regression]") {
    constexpr int REPS = 3;
    struct Row { std::size_t N, n; double ms; };
    std::vector<Row> rows;

    for (std::size_t N : {64u, 128u, 256u}) {
        auto A = generators::poisson2d_dirichlet<double>(N, N);
        const double ms = min_factor_solve_ms(A, REPS);
        UNSCOPED_INFO("Poisson " << N << "x" << N << "  n=" << A.num_rows()
                      << "  min of " << REPS << ": " << ms << " ms");
        rows.push_back({N, A.num_rows(), ms});
    }

    // n quadruples per refinement. An O(n^2) factorization would grow ~16x per
    // step; O(n^1.5) is ~8x, and the measured figure is 5-7x. The threshold sits
    // between the two hypotheses rather than close to the measurement, so it
    // discriminates the thing it names -- an asymptotic regression -- and not the
    // machine's mood.
    for (std::size_t i = 1; i < rows.size(); ++i) {
        const double ratio = rows[i].ms / rows[i - 1].ms;
        INFO("n " << rows[i - 1].n << " -> " << rows[i].n
             << " (4x): time ratio " << ratio);
        REQUIRE(ratio < 13.0);
    }

    // Sanity: the run must be long enough for the ratio to mean anything. A
    // sub-millisecond baseline would make the quotient a timer-resolution
    // artefact -- the failure mode that a threshold alone does not catch.
    REQUIRE(rows.front().ms > 1.0);
}
