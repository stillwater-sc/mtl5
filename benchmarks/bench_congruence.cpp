// bench_congruence -- congruence C = A*P*A^T through gemmt (#656) against the
// two-product form a caller writes without it.
//
// A is the shape of a filter transition, F = Phi (+) I: a dense 15x15 block,
// the identity elsewhere. At d = 15 it is fully dense; at d = 200 it holds
// 225 + 185 stored entries. P is a random symmetric d x d covariance. Values
// are random: the time a product takes does not depend on them.
//
// Arms, all row-major double:
//   two_step       T = mult(A, P); C = mult(T, trans(A)); C = (C + C^T)/2
//                  -- what branes-ai/cortex does today (cortex#448)
//   two_step_copy  as two_step, but trans(A) copied to a dense matrix first,
//                  so the second product is a plain mult that BLAS can take
//   congruence     congruence(A, P, C), A dense
//   congruence_crs congruence(A, P, C), A compressed2D
//
// Output: `d,arm,nnz,us_per_call`, the minimum over rounds of the mean time
// per call within a round.
//
// Build the generic configuration (no BLAS, no MTL5_NATIVE_FAST_GEMM):
//   g++ -std=c++20 -O3 -DNDEBUG -I<mtl5>/include bench_congruence.cpp
// and the BLAS one, against a library that exports dgemmt_:
//   g++ -std=c++20 -O3 -DNDEBUG -DMTL5_HAS_BLAS -DMTL5_BLAS_HAS_GEMMT \
//       -I<mtl5>/include bench_congruence.cpp -lopenblas
// Run single-threaded and pinned:
//   MTL5_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 4 ./a.out
// docs/algorithms/gemmt.md reports the results.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <random>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/gemmt.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>

using dense = mtl::mat::dense2D<double>;
using crs   = mtl::mat::compressed2D<double>;

namespace {

constexpr std::size_t kBlock = 15;

std::mt19937_64 rng(20261028);

/// F = Phi (+) I.
dense transition(std::size_t d) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    dense F(d, d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
            F(i, j) = (i < kBlock && j < kBlock) ? u(rng) : (i == j ? 1.0 : 0.0);
    return F;
}

dense covariance(std::size_t d) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    dense P(d, d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            P(i, j) = P(j, i) = (i == j ? 2.0 : 0.0) + 0.1 * u(rng);
    return P;
}

crs to_crs(const dense& A) {
    crs S(A.num_rows(), A.num_cols());
    {
        mtl::mat::inserter<crs> ins(S);
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            for (std::size_t j = 0; j < A.num_cols(); ++j)
                if (A(i, j) != 0.0)
                    ins[i][j] << A(i, j);
    }
    return S;
}

void symmetrize(dense& C) {
    const std::size_t n = C.num_rows();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            C(i, j) = C(j, i) = (C(i, j) + C(j, i)) / 2.0;
}

volatile double sink = 0.0;

/// Mean microseconds per call within a round of at least `min_ns`, minimum
/// over `rounds`.
double time_us(const std::function<void()>& f, int rounds = 15, double min_ns = 2e6) {
    using clk = std::chrono::steady_clock;
    std::size_t reps = 1;
    for (;;) {
        auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i) f();
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - t0).count();
        if (ns >= min_ns) break;
        reps *= 2;
    }
    double best = 1e300;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i) f();
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - t0).count();
        best = std::min(best, ns / static_cast<double>(reps));
    }
    return best / 1000.0;
}

} // namespace

int main() {
    std::printf("d,arm,nnz,us_per_call\n");
    for (std::size_t d : {15u, 30u, 60u, 100u, 150u, 200u}) {
        const dense A = transition(d);
        const dense P = covariance(d);
        const crs S = to_crs(A);
        const std::size_t nnz = S.nnz();
        dense T(d, d), C(d, d), At(d, d);

        auto report = [&](const char* arm, const std::function<void()>& f) {
            const double us = time_us(f);
            sink = sink + C(d - 1, 0);
            std::printf("%zu,%s,%zu,%.3f\n", d, arm, nnz, us);
            std::fflush(stdout);
        };
        report("two_step", [&] {
            mtl::mult(A, P, T);
            mtl::mult(T, mtl::trans(A), C);
            symmetrize(C);
        });
        report("two_step_copy", [&] {
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t j = 0; j < d; ++j)
                    At(i, j) = A(j, i);
            mtl::mult(A, P, T);
            mtl::mult(T, At, C);
            symmetrize(C);
        });
        report("congruence", [&] { mtl::congruence(A, P, C); });
        report("congruence_crs", [&] { mtl::congruence(S, P, C); });
    }
    return 0;
}
