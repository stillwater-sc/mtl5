#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/ordering/colamd.hpp>

using namespace mtl;
using namespace mtl::sparse;

// ---- generators -----------------------------------------------------------
static mat::compressed2D<double> random_unsym(std::size_t n, double density, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0), val(-1.0, 1.0);
    mat::compressed2D<double> A(n, n);
    { mat::inserter<mat::compressed2D<double>> ins(A);
      for (std::size_t i = 0; i < n; ++i) {
          ins[i][i] << static_cast<double>(n) + 1.0;
          for (std::size_t j = 0; j < n; ++j)
              if (i != j && u(rng) < density) ins[i][j] << val(rng);
      } }
    return A;
}
static mat::compressed2D<double> dense_unsym(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    { mat::inserter<mat::compressed2D<double>> ins(A);
      for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
              ins[i][j] << (i == j ? static_cast<double>(2 * n)
                                   : 0.5 + 0.25 * std::sin(0.6 * i + 0.4 * j)); }
    return A;
}
static mat::compressed2D<double> convdiff(std::size_t N, double bx = 1.0, double by = 1.0) {
    std::size_t n = N * N; mat::compressed2D<double> A(n, n);
    { mat::inserter<mat::compressed2D<double>> ins(A);
      auto id = [N](std::size_t r, std::size_t c) { return r * N + c; };
      for (std::size_t r = 0; r < N; ++r) for (std::size_t c = 0; c < N; ++c) {
          std::size_t i = id(r, c); ins[i][i] << (4.0 + bx + by);
          if (c > 0) ins[i][id(r, c - 1)] << -(1.0 + bx);
          if (c + 1 < N) ins[i][id(r, c + 1)] << -1.0;
          if (r > 0) ins[i][id(r - 1, c)] << -(1.0 + by);
          if (r + 1 < N) ins[i][id(r + 1, c)] << -1.0;
      } }
    return A;
}
template <typename VecX>
static double rel_residual(const mat::compressed2D<double>& A, const VecX& x,
                           const vec::dense_vector<double>& b) {
    std::size_t n = b.size();
    const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& dat = A.ref_data();
    double rn = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += dat[k] * static_cast<double>(x(static_cast<int>(ci[k])));
        double d = ax - b(static_cast<int>(i)); rn += d * d; bn += b(static_cast<int>(i)) * b(static_cast<int>(i));
    }
    return bn == 0.0 ? std::sqrt(rn) : std::sqrt(rn) / std::sqrt(bn);
}

// ---- correctness vs scalar sparse_lu --------------------------------------
TEST_CASE("Supernodal LU matches scalar LU", "[sparse][lu][supernodal]") {
    std::vector<mat::compressed2D<double>> mats;
    mats.push_back(random_unsym(30, 0.10, 1));
    mats.push_back(random_unsym(20, 0.30, 2));
    mats.push_back(dense_unsym(12));
    mats.push_back(convdiff(6));   // 36x36

    for (auto& A : mats) {
        std::size_t n = A.num_rows();
        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.3 * static_cast<double>(i);

        vec::dense_vector<double> xs(n, 0.0), xn(n, 0.0);
        factorization::sparse_lu_solve(A, xs, b, ordering::colamd{});
        factorization::supernodal_lu_solve(A, xn, b, ordering::colamd{});

        REQUIRE(rel_residual(A, xn, b) < 1e-12);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE_THAT(xn(static_cast<int>(i)),
                         Catch::Matchers::WithinAbs(xs(static_cast<int>(i)), 1e-9));
    }
}

// ---- analyze/factor/refactor split (#184) ---------------------------------
TEST_CASE("Supernodal LU refactor reuses the pattern", "[sparse][lu][supernodal][refactor]") {
    // Same sparsity pattern, different values (the transient-SPICE scenario):
    // factor once, then refactor (numeric only, no reach/pivot search).
    auto A1 = random_unsym(40, 0.10, 3);
    auto sym = factorization::supernodal_lu_symbolic_analyze(A1, ordering::colamd{});
    auto fac = factorization::supernodal_lu_numeric(A1, sym);

    // A2: identical pattern, perturbed values (reuse A1's structure).
    auto A2 = A1;
    {
        auto& d = const_cast<std::vector<double>&>(A2.ref_data());
        for (std::size_t k = 0; k < d.size(); ++k) d[k] *= (1.0 + 0.1 * std::sin(0.3 * k));
    }
    std::size_t n = A2.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.2 * static_cast<double>(i);

    auto re = factorization::supernodal_lu_refactor(A2, fac);
    vec::dense_vector<double> xr(n, 0.0); re.solve(xr, b);
    REQUIRE(rel_residual(A2, xr, b) < 1e-11);

    // Pattern preserved and result matches a fresh factor of A2.
    REQUIRE(re.L.col_ptr == fac.L.col_ptr);
    REQUIRE(re.U.col_ptr == fac.U.col_ptr);
    auto fresh = factorization::supernodal_lu_numeric(A2, sym);
    vec::dense_vector<double> xf(n, 0.0); fresh.solve(xf, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(xr(static_cast<int>(i)),
                     Catch::Matchers::WithinAbs(xf(static_cast<int>(i)), 1e-9));

    // Several refactors in a row (the SPICE loop) stay correct.
    for (int it = 0; it < 3; ++it) {
        re = factorization::supernodal_lu_refactor(A2, re);
        vec::dense_vector<double> x(n, 0.0); re.solve(x, b);
        REQUIRE(rel_residual(A2, x, b) < 1e-11);
    }
}

// ---- row equilibration / scaling (#185) -----------------------------------
TEST_CASE("Supernodal LU row scaling", "[sparse][lu][supernodal][scale]") {
    // Badly row-scaled SPD-ish matrix: rows multiplied by a large dynamic range.
    auto make_badscaled = [](std::size_t n) {
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> v(-1.0, 1.0);
        mat::compressed2D<double> A(n, n);
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            double s = std::pow(10.0, static_cast<double>(static_cast<int>(i % 9) - 4)); // 1e-4..1e4
            ins[i][i] << s * (static_cast<double>(n) + 1.0);
            for (std::size_t j = 0; j < n; ++j)
                if (i != j && (i + 3 * j) % 5 == 0) ins[i][j] << s * v(rng);
        }
        return A;
    };
    auto A = make_badscaled(60);
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.1 * static_cast<double>(i);

    auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});

    // Scaled (factor R*A) is correct and mathematically equivalent to unscaled;
    // row_scale = 1/max|row| is populated and positive. (Equilibration improves
    // worst-case pivot stability; it is not a per-instance monotonic win, so we
    // assert correctness + equivalence rather than "scaled is always smaller".)
    auto fs = factorization::supernodal_lu_numeric(A, sym, 1.0, 64, /*scale=*/true);
    auto fu = factorization::supernodal_lu_numeric(A, sym, 1.0, 64, /*scale=*/false);
    REQUIRE(fs.row_scale.size() == n);
    REQUIRE(fu.row_scale.empty());
    for (double s : fs.row_scale) REQUIRE(s > 0.0);

    vec::dense_vector<double> xs(n, 0.0), xu(n, 0.0);
    fs.solve(xs, b); fu.solve(xu, b);
    REQUIRE(rel_residual(A, xs, b) < 1e-10);   // scaled solve correct
    REQUIRE(rel_residual(A, xu, b) < 1e-10);   // unscaled solve correct
    for (std::size_t i = 0; i < n; ++i)        // mathematically equivalent
        REQUIRE_THAT(xs(static_cast<int>(i)),
                     Catch::Matchers::WithinAbs(xu(static_cast<int>(i)), 1e-7));

    // Scaling survives refactor (recomputed for the new values).
    auto re = factorization::supernodal_lu_refactor(A, fs);
    REQUIRE(re.row_scale.size() == n);
    vec::dense_vector<double> xr(n, 0.0); re.solve(xr, b);
    REQUIRE(rel_residual(A, xr, b) < 1e-10);
}

// ---- pivoting -------------------------------------------------------------
TEST_CASE("Supernodal LU pivots on a zero diagonal", "[sparse][lu][supernodal]") {
    // [[0 1],[1 1]] requires a row swap.
    mat::compressed2D<double> A(2, 2);
    { mat::inserter<mat::compressed2D<double>> ins(A);
      ins[0][1] << 1.0; ins[1][0] << 1.0; ins[1][1] << 1.0; }
    vec::dense_vector<double> b = {1.0, 2.0}, x(2, 0.0);
    factorization::supernodal_lu_solve(A, x, b, ordering::colamd{});
    REQUIRE(rel_residual(A, x, b) < 1e-12);
}

// ---- supernode formation --------------------------------------------------
TEST_CASE("Supernodal LU supernode structure", "[sparse][lu][supernodal]") {
    SECTION("diagonal => all singletons, valid partition") {
        std::size_t n = 6; mat::compressed2D<double> A(n, n);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          for (std::size_t i = 0; i < n; ++i) ins[i][i] << static_cast<double>(i + 2); }
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
        auto fac = factorization::supernodal_lu_numeric(A, sym);
        REQUIRE(fac.nsuper() == n);
        REQUIRE(fac.lsuper_first.front() == 0);
        REQUIRE(fac.lsuper_first.back() == n);
        for (std::size_t s = 0; s + 1 < fac.lsuper_first.size(); ++s)
            REQUIRE(fac.lsuper_first[s] < fac.lsuper_first[s + 1]);
    }
    SECTION("dense => merges into few supernodes") {
        std::size_t n = 16; auto A = dense_unsym(n);
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
        auto fac = factorization::supernodal_lu_numeric(A, sym);
        REQUIRE(fac.nsuper() < n);                 // real merging happened
        REQUIRE(fac.lsuper_first.back() == n);
    }
}

// ---- mixed precision: wider accumulator => more accurate factor ------------
TEST_CASE("Supernodal LU accumulator precision drives accuracy",
          "[sparse][lu][supernodal][accumulator]") {
    auto A = dense_unsym(24);                       // big dense supernode => lots of accumulation
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.1 * static_cast<double>(i);

    auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
    // Storage double (accurate solve) so the residual reflects FACTOR quality.
    auto fac_f = factorization::supernodal_lu_numeric<double, decltype(A)::param_type, float>(A, sym);
    auto fac_d = factorization::supernodal_lu_numeric<double, decltype(A)::param_type, double>(A, sym);
    vec::dense_vector<double> xf(n, 0.0), xd(n, 0.0);
    fac_f.solve(xf, b); fac_d.solve(xd, b);
    double res_f = rel_residual(A, xf, b), res_d = rel_residual(A, xd, b);

    REQUIRE(std::isfinite(res_f));
    REQUIRE(res_d < 5e-11);                        // double accumulation -> accurate factor
    REQUIRE(res_d < res_f);                        // ... decisively better than float
    REQUIRE(res_f > 1e3 * res_d);                  // lock in a clear separation
}

// ---- mixed-precision iterative refinement ---------------------------------
TEST_CASE("Supernodal LU mixed-precision iterative refinement",
          "[sparse][lu][supernodal][iterative_refine]") {
    auto A = convdiff(7);                           // 49x49 unsymmetric
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.05 * static_cast<double>(i);

    // float-only baseline
    using st = mat::compressed2D<double>::size_type;
    std::vector<st> starts(A.ref_major().begin(), A.ref_major().end());
    std::vector<st> idx(A.ref_minor().begin(), A.ref_minor().end());
    std::vector<float> dat(A.nnz());
    for (std::size_t k = 0; k < A.nnz(); ++k) dat[k] = static_cast<float>(A.ref_data()[k]);
    mat::compressed2D<float> Af(n, n, A.nnz(), starts.data(), idx.data(), dat.data());
    vec::dense_vector<float> bf(n), xbase(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) bf(static_cast<int>(i)) = static_cast<float>(b(static_cast<int>(i)));
    factorization::supernodal_lu_solve(Af, xbase, bf, ordering::colamd{});
    double res_base = rel_residual(A, xbase, b);

    refine_options opt; opt.max_iter = 50; opt.rel_tol = 1e-12;
    vec::dense_vector<double> x(n, 0.0);
    auto rr = factorization::supernodal_lu_solve_refined<float, double>(A, x, b, ordering::colamd{}, opt);

    REQUIRE(rr.rel_residual < 1e-9);
    REQUIRE(rr.rel_residual < res_base);          // refinement beats the float-only solve
    REQUIRE(rel_residual(A, x, b) < 1e-9);
}

// ---- edge cases -----------------------------------------------------------
TEST_CASE("Supernodal LU edge cases", "[sparse][lu][supernodal]") {
    SECTION("1x1") {
        mat::compressed2D<double> A(1, 1);
        { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 3.0; }
        vec::dense_vector<double> b = {6.0}, x(1, 0.0);
        factorization::supernodal_lu_solve(A, x, b, ordering::colamd{});
        REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(2.0, 1e-12));
    }
    SECTION("singular throws") {
        mat::compressed2D<double> A(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 1.0; ins[1][0] << 1.0; }
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
        REQUIRE_THROWS_AS(factorization::supernodal_lu_numeric(A, sym), std::runtime_error);
    }
    SECTION("empty matrix => zero supernodes") {
        mat::compressed2D<double> A(0, 0);
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
        auto fac = factorization::supernodal_lu_numeric(A, sym);
        REQUIRE(fac.nsuper() == 0);
        REQUIRE(fac.num_rows() == 0);
    }
    SECTION("invalid threshold rejected") {
        auto A = dense_unsym(4);
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
        REQUIRE_THROWS_AS(factorization::supernodal_lu_numeric(A, sym, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(factorization::supernodal_lu_numeric(A, sym, 1.5), std::invalid_argument);
    }
}

// Issue #185 (pivot robustness): opt-in zero-pivot perturbation, mirroring #123
// for sparse_lu / native_klu. Default off => byte-identical hard-throw behavior.
TEST_CASE("Supernodal LU zero-pivot perturbation (#185)", "[sparse][lu][supernodal][perturb]") {
    SECTION("exactly singular: default throws; perturbation completes") {
        mat::compressed2D<double> A(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          ins[0][0] << 1.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 1.0; }
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});

        REQUIRE_THROWS_AS(factorization::supernodal_lu_numeric(A, sym), std::runtime_error);

        auto num = factorization::supernodal_lu_numeric(
            A, sym, /*threshold=*/1.0, /*max_super=*/64, /*scale=*/false, /*pivot_perturb=*/1e-8);
        REQUIRE(num.num_perturbed >= 1);
        vec::dense_vector<double> b = {1.0, 2.0}, x(2, 0.0);
        num.solve(x, b);
        REQUIRE(std::isfinite(x(0)));
        REQUIRE(std::isfinite(x(1)));
    }

    SECTION("nonsingular: perturbation never fires (clean factor, same accuracy)") {
        auto A = dense_unsym(12);
        std::size_t n = A.num_rows();
        auto sym = factorization::supernodal_lu_symbolic_analyze(A, ordering::colamd{});
        auto num = factorization::supernodal_lu_numeric(A, sym, 1.0, 64, false, 1e-8);
        REQUIRE(num.num_perturbed == 0);
        vec::dense_vector<double> b(n), x(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.3 * static_cast<double>(i);
        num.solve(x, b);
        REQUIRE(rel_residual(A, x, b) < 1e-12);
    }

    SECTION("refactor reports a clean perturbation count (no stale carry-over)") {
        mat::compressed2D<double> A1(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A1);
          ins[0][0] << 1.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 1.0; }
        auto sym = factorization::supernodal_lu_symbolic_analyze(A1, ordering::colamd{});
        auto prev = factorization::supernodal_lu_numeric(A1, sym, 1.0, 64, false, 1e-8);
        REQUIRE(prev.num_perturbed >= 1);

        mat::compressed2D<double> A2(2, 2);                  // same pattern, well-conditioned
        { mat::inserter<mat::compressed2D<double>> ins(A2);
          ins[0][0] << 2.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 3.0; }
        auto re = factorization::supernodal_lu_refactor(A2, prev);
        REQUIRE(re.num_perturbed == 0);
        vec::dense_vector<double> b = {1.0, 2.0}, x(2, 0.0);
        re.solve(x, b);
        REQUIRE(rel_residual(A2, x, b) < 1e-12);
    }
}
