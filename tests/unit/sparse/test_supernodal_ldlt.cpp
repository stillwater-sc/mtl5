#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/ordering/amd.hpp>

using namespace mtl;
using namespace mtl::sparse;

// ---------------------------------------------------------------------------
// Matrix generators (deterministic)
// ---------------------------------------------------------------------------

// SPD tridiagonal: A(i,i)=4, A(i,i+-1)=-1.
static mat::compressed2D<double> make_spd_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 4.0;
        if (i + 1 < n) { ins[i][i + 1] << -1.0; ins[i + 1][i] << -1.0; }
    }
    return A;
}

// 5-point 2D Laplacian on a g x g grid (n = g*g): nontrivial fill / supernodes.
static mat::compressed2D<double> make_laplacian2d(std::size_t g) {
    std::size_t n = g * g;
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            std::size_t i = id(r, c);
            ins[i][i] << 4.0;
            if (r + 1 < g) { ins[i][id(r + 1, c)] << -1.0; ins[id(r + 1, c)][i] << -1.0; }
            if (c + 1 < g) { ins[i][id(r, c + 1)] << -1.0; ins[id(r, c + 1)][i] << -1.0; }
        }
    return A;
}

// Dense SPD: A = M^T M + n I with a deterministic, mildly varying M.
static mat::compressed2D<double> make_dense_spd(std::size_t n) {
    std::vector<std::vector<double>> M(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            M[i][j] = 0.5 + std::sin(0.7 * static_cast<double>(i) + 0.3 * static_cast<double>(j));
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double a = (i == j) ? static_cast<double>(n) : 0.0;
            for (std::size_t k = 0; k < n; ++k) a += M[k][i] * M[k][j];
            ins[i][j] << a;
        }
    return A;
}

// Diagonal SPD.
static mat::compressed2D<double> make_diagonal(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) ins[i][i] << static_cast<double>(i + 2);
    return A;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ||Ax - b|| / ||b|| with x cast to double (works for float-typed solutions).
template <typename VecX>
static double rel_residual(const mat::compressed2D<double>& A,
                           const VecX& x,
                           const vec::dense_vector<double>& b) {
    std::size_t n = b.size();
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    double rn = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            ax += data[k] * static_cast<double>(x(static_cast<int>(indices[k])));
        double ri = ax - b(static_cast<int>(i));
        rn += ri * ri;
        bn += b(static_cast<int>(i)) * b(static_cast<int>(i));
    }
    return bn == 0.0 ? std::sqrt(rn) : std::sqrt(rn) / std::sqrt(bn);
}

static mat::compressed2D<float> to_float(const mat::compressed2D<double>& A) {
    using st = mat::compressed2D<double>::size_type;
    std::vector<st> starts(A.ref_major().begin(), A.ref_major().end());
    std::vector<st> idx(A.ref_minor().begin(), A.ref_minor().end());
    std::vector<float> dat(A.nnz());
    for (std::size_t k = 0; k < A.nnz(); ++k) dat[k] = static_cast<float>(A.ref_data()[k]);
    return mat::compressed2D<float>(A.num_rows(), A.num_cols(), A.nnz(),
                                    starts.data(), idx.data(), dat.data());
}

// ---------------------------------------------------------------------------
// Symbolic
// ---------------------------------------------------------------------------

TEST_CASE("Supernodal LDL^T symbolic: diagonal => all singleton supernodes",
          "[sparse][ldlt][supernodal]") {
    auto A = make_diagonal(6);
    auto sym = factorization::supernodal_ldlt_symbolic(A);

    REQUIRE(sym.n == 6);
    REQUIRE(sym.snodes.nsuper == 6);                 // no merging possible
    REQUIRE(util::is_valid_permutation(sym.sperm));
    REQUIRE(util::is_valid_permutation(sym.spinv));
    REQUIRE(sym.snodes.sn_first.size() == 7);
}

TEST_CASE("Supernodal LDL^T symbolic: dense SPD => single supernode",
          "[sparse][ldlt][supernodal]") {
    auto A = make_dense_spd(8);
    auto sym = factorization::supernodal_ldlt_symbolic(A);

    REQUIRE(sym.n == 8);
    REQUIRE(sym.snodes.nsuper == 1);                 // one dense panel
    REQUIRE(sym.snodes.sn_first.front() == 0);
    REQUIRE(sym.snodes.sn_first.back() == 8);
}

TEST_CASE("Supernodal LDL^T symbolic rejects rectangular matrix",
          "[sparse][ldlt][supernodal]") {
    mat::compressed2D<double> A(3, 4);
    { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 1.0; }
    REQUIRE_THROWS_AS(factorization::supernodal_ldlt_symbolic(A), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Numeric correctness (vs scalar LDL^T oracle)
// ---------------------------------------------------------------------------

TEST_CASE("Supernodal LDL^T numeric matches scalar oracle",
          "[sparse][ldlt][supernodal]") {
    std::vector<mat::compressed2D<double>> mats;
    mats.push_back(make_spd_tridiag(7));
    mats.push_back(make_laplacian2d(5));     // 25x25, real fill
    mats.push_back(make_dense_spd(10));

    for (auto& A : mats) {
        std::size_t n = A.num_rows();
        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.3 * static_cast<double>(i);

        vec::dense_vector<double> xs(n, 0.0), xn(n, 0.0);
        factorization::sparse_ldlt_solve(A, xs, b, ordering::amd{});      // scalar oracle
        factorization::supernodal_ldlt_solve(A, xn, b, ordering::amd{});  // supernodal

        REQUIRE(rel_residual(A, xn, b) < 1e-12);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE_THAT(xn(static_cast<int>(i)),
                         Catch::Matchers::WithinAbs(xs(static_cast<int>(i)), 1e-9));
    }
}

TEST_CASE("Supernodal LDL^T solve: natural ordering and reuse",
          "[sparse][ldlt][supernodal]") {
    auto A = make_laplacian2d(4);   // 16x16
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n, 1.0);

    auto sym = factorization::supernodal_ldlt_symbolic(A);
    auto num = factorization::supernodal_ldlt_numeric(A, sym);
    vec::dense_vector<double> x(n, 0.0);
    num.solve(x, b);
    REQUIRE(rel_residual(A, x, b) < 1e-12);

    // Reuse the symbolic factorization for the same pattern, different values.
    auto A2 = make_laplacian2d(4);
    auto num2 = factorization::supernodal_ldlt_numeric(A2, sym);
    vec::dense_vector<double> x2(n, 0.0);
    num2.solve(x2, b);
    REQUIRE(rel_residual(A2, x2, b) < 1e-12);
}

TEST_CASE("Supernodal LDL^T edge cases", "[sparse][ldlt][supernodal]") {
    SECTION("1x1") {
        mat::compressed2D<double> A(1, 1);
        { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 4.0; }
        vec::dense_vector<double> b = {8.0}, x(1, 0.0);
        factorization::supernodal_ldlt_solve(A, x, b);
        REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(2.0, 1e-12));
    }
    SECTION("diagonal") {
        auto A = make_diagonal(5);
        std::size_t n = A.num_rows();
        vec::dense_vector<double> b(n, 2.0), x(n, 0.0);
        factorization::supernodal_ldlt_solve(A, x, b);
        REQUIRE(rel_residual(A, x, b) < 1e-12);
    }
    SECTION("zero pivot throws") {
        mat::compressed2D<double> A(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          ins[0][0] << 0.0; ins[0][1] << 1.0; ins[1][0] << 1.0; ins[1][1] << 0.0; }
        auto sym = factorization::supernodal_ldlt_symbolic(A);
        REQUIRE_THROWS_AS(factorization::supernodal_ldlt_numeric(A, sym), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// Mixed precision: wider accumulator is no worse (usually better)
// ---------------------------------------------------------------------------

TEST_CASE("Supernodal LDL^T: double accumulator >= float accumulator accuracy",
          "[sparse][ldlt][supernodal][accumulator]") {
    auto A = make_dense_spd(24);              // one big panel => lots of accumulation
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.1 * static_cast<double>(i);

    auto Af = to_float(A);
    vec::dense_vector<float> bf(n);
    for (std::size_t i = 0; i < n; ++i) bf(static_cast<int>(i)) = static_cast<float>(b(static_cast<int>(i)));

    auto symf = factorization::supernodal_ldlt_symbolic(Af, ordering::amd{});

    // Storage = float; accumulate in float vs double.
    auto fac_f = factorization::supernodal_ldlt_numeric<float, decltype(Af)::param_type, float>(Af, symf);
    auto fac_d = factorization::supernodal_ldlt_numeric<float, decltype(Af)::param_type, double>(Af, symf);

    vec::dense_vector<float> xf(n, 0.0f), xd(n, 0.0f);
    fac_f.solve(xf, bf);
    fac_d.solve(xd, bf);

    double res_f = rel_residual(A, xf, b);
    double res_d = rel_residual(A, xd, b);

    REQUIRE(std::isfinite(res_f));
    REQUIRE(std::isfinite(res_d));
    REQUIRE(res_d <= res_f);                  // wider accumulator never less accurate
}

// ---------------------------------------------------------------------------
// Mixed-precision iterative refinement
// ---------------------------------------------------------------------------

TEST_CASE("Supernodal LDL^T mixed-precision iterative refinement",
          "[sparse][ldlt][supernodal][iterative_refine]") {
    auto A = make_laplacian2d(7);             // 49x49 SPD
    std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = 1.0 + 0.05 * static_cast<double>(i);

    // Plain float factorization, no refinement (baseline accuracy).
    auto Af = to_float(A);
    vec::dense_vector<float> bf(n);
    for (std::size_t i = 0; i < n; ++i) bf(static_cast<int>(i)) = static_cast<float>(b(static_cast<int>(i)));
    auto symf = factorization::supernodal_ldlt_symbolic(Af, ordering::amd{});
    auto facf = factorization::supernodal_ldlt_numeric(Af, symf);
    vec::dense_vector<float> xbase(n, 0.0f);
    facf.solve(xbase, bf);
    double res_base = rel_residual(A, xbase, b);

    // Float factor + double-residual iterative refinement.
    refine_options opt;
    opt.max_iter = 50;
    opt.rel_tol = 1e-12;
    vec::dense_vector<double> x(n, 0.0);
    auto rr = factorization::supernodal_ldlt_solve_refined<float, double>(A, x, b, ordering::amd{}, opt);

    REQUIRE(rr.rel_residual < 1e-10);         // recovers near-double accuracy
    REQUIRE(rr.rel_residual < res_base);      // strictly better than float-only
    REQUIRE(rel_residual(A, x, b) < 1e-10);
}
