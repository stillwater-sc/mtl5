// MTL5 -- generic iterative refinement core (issue #119).
// A low-precision (float) factorization refined with a double-precision residual
// recovers accuracy a float direct solve cannot. Universal-free: the same core
// is reused by the mp-spice mixed-precision study for posit/cfloat factors.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/iterative_refine.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {

// Dense, diagonally-dominant matrix with deterministic mixed-sign off-diagonals:
// long, cancellation-prone inner products in the LU, but well-conditioned, so a
// float factorization is meaningful and a double-residual refinement converges.
template <typename T>
mat::compressed2D<T> dense_mixed(std::size_t n) {
    mat::compressed2D<T> A(n, n);
    mat::inserter<mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            ins[i][j] << static_cast<T>(i == j ? static_cast<double>(n)
                                               : std::sin(static_cast<double>(i * 7 + j * 13)));
    return A;
}

double forward_error(const vec::dense_vector<double>& x, double exact = 1.0) {
    double m = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::abs(x(static_cast<int>(i)) - exact));
    return m;
}

// b = A * ones in double, so the exact solution is all-ones.
vec::dense_vector<double> rhs_ones(const mat::compressed2D<double>& A) {
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n, 0.0);
    const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& dat = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) s += dat[k];
        b(static_cast<int>(i)) = s;
    }
    return b;
}

} // namespace

TEST_CASE("iterative_refine recovers accuracy from a float factorization",
          "[sparse][iterative_refine]") {
    std::size_t n = 50;
    auto Ad = dense_mixed<double>(n);
    auto Af = dense_mixed<float>(n);                     // low-precision copy to factor
    auto b  = rhs_ones(Ad);

    auto fac = sparse::factorization::native_klu_factor(Af);   // factor in float

    // Direct float solve (no refinement): cast b to float, solve, back to double.
    vec::dense_vector<float> bf(n), xf(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) bf(static_cast<int>(i)) = static_cast<float>(b(static_cast<int>(i)));
    fac.solve(xf, bf);
    vec::dense_vector<double> x_direct(n);
    for (std::size_t i = 0; i < n; ++i) x_direct(static_cast<int>(i)) = static_cast<double>(xf(static_cast<int>(i)));
    double err_direct = forward_error(x_direct);

    // Iterative refinement with a double residual (x starts at zero).
    vec::dense_vector<double> x(n, 0.0);
    auto res = sparse::iterative_refine<double>(Ad, fac, b, x);
    double err_ir = forward_error(x);

    INFO("direct float err = " << err_direct << ", IR err = " << err_ir
         << ", iters = " << res.iters << ", rel_resid = " << res.rel_residual);
    REQUIRE(err_direct > 1e-8);          // float direct solve is float-limited
    REQUIRE(err_ir < 1e-12);             // refinement recovers ~double accuracy
    REQUIRE(err_ir < err_direct);
    REQUIRE(res.iters >= 1);
}

TEST_CASE("iterative_refine rel_tol stops early and reports convergence",
          "[sparse][iterative_refine]") {
    std::size_t n = 40;
    auto Ad = dense_mixed<double>(n);
    auto fac = sparse::factorization::native_klu_factor(dense_mixed<float>(n));
    auto b  = rhs_ones(Ad);

    vec::dense_vector<double> x(n, 0.0);
    sparse::refine_options opt; opt.rel_tol = 1e-10; opt.max_iter = 50;
    auto res = sparse::iterative_refine<double>(Ad, fac, b, x, opt);
    REQUIRE(res.converged);
    REQUIRE(res.rel_residual <= 1e-10);
}

TEST_CASE("scaled iterative_refine matches unscaled for a wide-range factor type",
          "[sparse][iterative_refine]") {
    // For float (wide exponent range) scaling is a no-op on convergence; it must
    // still produce a correct solution. (The scaled variant's payoff is on
    // narrow-range types, exercised in the mp-spice study with posit/cfloat.)
    std::size_t n = 40;
    auto Ad = dense_mixed<double>(n);
    auto fac = sparse::factorization::native_klu_factor(dense_mixed<float>(n));
    auto b  = rhs_ones(Ad);

    vec::dense_vector<double> xu(n, 0.0), xs(n, 0.0);
    sparse::refine_options uo; uo.max_iter = 50;
    sparse::refine_options so; so.max_iter = 50; so.scaled = true;
    sparse::iterative_refine<double>(Ad, fac, b, xu, uo);
    sparse::iterative_refine<double>(Ad, fac, b, xs, so);

    REQUIRE(forward_error(xu) < 1e-10);
    REQUIRE(forward_error(xs) < 1e-10);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(xs(static_cast<int>(i)), WithinAbs(xu(static_cast<int>(i)), 1e-9));
}
