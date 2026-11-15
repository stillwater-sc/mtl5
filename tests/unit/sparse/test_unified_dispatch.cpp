#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/sparse_solve.hpp>

using namespace mtl;

static double sparse_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res = 0.0, bnorm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bnorm += b(i) * b(i);
    }
    if (bnorm == 0.0) return std::sqrt(res);
    return std::sqrt(res / bnorm);
}

static double dense_residual(
    const mat::dense2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res = 0.0, bnorm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            Ax_i += A(i, j) * x(j);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bnorm += b(i) * b(i);
    }
    if (bnorm == 0.0) return std::sqrt(res);
    return std::sqrt(res / bnorm);
}

// ---- solve() dispatches to LU ----

TEST_CASE("solve() dispatches sparse LU", "[dispatch][lu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0;
    }
    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    solve(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("solve() dispatches dense LU", "[dispatch][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 3.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 4.0; A(1, 2) = 2.0;
    A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 5.0;

    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    solve(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- cholesky_solve_dispatch ----

TEST_CASE("cholesky_solve_dispatch on sparse SPD", "[dispatch][cholesky]") {
    std::size_t n = 10;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    cholesky_solve_dispatch(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("cholesky_solve_dispatch on dense SPD", "[dispatch][cholesky]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 4.0;  A(0, 1) = -1.0; A(0, 2) = 0.0;
    A(1, 0) = -1.0; A(1, 1) = 4.0;  A(1, 2) = -1.0;
    A(2, 0) = 0.0;  A(2, 1) = -1.0; A(2, 2) = 4.0;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    cholesky_solve_dispatch(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- lu_solve_dispatch ----

TEST_CASE("lu_solve_dispatch on sparse unsymmetric", "[dispatch][lu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;  ins[0][1] << 3.0;  ins[0][2] << 1.0;
        ins[1][0] << 4.0;  ins[1][1] << 7.0;  ins[1][2] << 5.0;
        ins[2][0] << 6.0;  ins[2][1] << 18.0; ins[2][2] << 22.0;
    }
    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    lu_solve_dispatch(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("lu_solve_dispatch on dense unsymmetric", "[dispatch][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 2.0;  A(0, 1) = 3.0;  A(0, 2) = 1.0;
    A(1, 0) = 4.0;  A(1, 1) = 7.0;  A(1, 2) = 5.0;
    A(2, 0) = 6.0;  A(2, 1) = 18.0; A(2, 2) = 22.0;

    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    lu_solve_dispatch(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- qr_solve_dispatch ----

TEST_CASE("qr_solve_dispatch on sparse square", "[dispatch][qr]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;  ins[0][1] << -1.0;
        ins[1][0] << -1.0; ins[1][1] << 4.0;  ins[1][2] << -1.0;
        ins[2][1] << -1.0; ins[2][2] << 4.0;
    }
    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    qr_solve_dispatch(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("qr_solve_dispatch on dense square", "[dispatch][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 4.0;  A(0, 1) = -1.0; A(0, 2) = 0.0;
    A(1, 0) = -1.0; A(1, 1) = 4.0;  A(1, 2) = -1.0;
    A(2, 0) = 0.0;  A(2, 1) = -1.0; A(2, 2) = 4.0;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    qr_solve_dispatch(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- Consistency: dispatch produces same answer as explicit call ----

TEST_CASE("Dispatch matches explicit sparse Cholesky", "[dispatch][consistency]") {
    std::size_t n = 8;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }
    vec::dense_vector<double> b(n, 1.0);

    vec::dense_vector<double> x_dispatch(n, 0.0);
    cholesky_solve_dispatch(A, x_dispatch, b);

    vec::dense_vector<double> x_explicit(n, 0.0);
    sparse::factorization::sparse_cholesky_solve(A, x_explicit, b,
        sparse::ordering::amd{});

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x_dispatch(i), Catch::Matchers::WithinAbs(x_explicit(i), 1e-12));
}

TEST_CASE("Dispatch matches explicit sparse LU", "[dispatch][consistency]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0; ins[2][3] << 1.0;
        ins[3][1] << 1.0; ins[3][3] << 3.0;
    }
    vec::dense_vector<double> b = {4.0, 10.0, 12.0, 7.0};

    vec::dense_vector<double> x_dispatch(4, 0.0);
    lu_solve_dispatch(A, x_dispatch, b);

    vec::dense_vector<double> x_explicit(4, 0.0);
    sparse::factorization::sparse_lu_solve(A, x_explicit, b,
        sparse::ordering::colamd{});

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(x_dispatch(i), Catch::Matchers::WithinAbs(x_explicit(i), 1e-10));
}

// ---- Larger dispatch tests ----

TEST_CASE("solve() on 20x20 sparse system", "[dispatch][scale]") {
    std::size_t n = 20;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -0.5;
            }
        }
    }
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    solve(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

// ---- operands the SuiteSparse bindings do not take (#850) ----
//
// umfpack_solve, cholmod_solve and spqr_solve take compressed2D<double> and
// dense_vector<double> only. The dispatch used to gate on the matrix being
// float or double, so with the bindings on (MTL5_HAS_UMFPACK, ...CHOLMOD,
// ...SPQR) a compressed2D<float> did not compile. These cases must take the
// native solvers; CI builds this file with all three bindings on.

static_assert(interface::suitesparse_solve_dispatches_v<
    mat::compressed2D<double>, vec::dense_vector<double>, vec::dense_vector<double>>);
static_assert(!interface::suitesparse_solve_dispatches_v<
    mat::compressed2D<float>, vec::dense_vector<float>, vec::dense_vector<float>>);
static_assert(!interface::suitesparse_solve_dispatches_v<
    mat::compressed2D<double>, vec::dense_vector<float>, vec::dense_vector<double>>);
static_assert(!interface::suitesparse_solve_dispatches_v<
    mat::compressed2D<double>, vec::dense_vector<double>, vec::dense_vector<float>>);
static_assert(!interface::suitesparse_solve_dispatches_v<
    mat::dense2D<double>, vec::dense_vector<double>, vec::dense_vector<double>>);

namespace {

// SPD tridiagonal [4 -1; -1 4 -1; -1 4], usable by all three dispatches.
template <typename T>
mat::compressed2D<T> spd3() {
    mat::compressed2D<T> A(3, 3);
    {
        mat::inserter<mat::compressed2D<T>> ins(A);
        ins[0][0] << T(4);  ins[0][1] << T(-1);
        ins[1][0] << T(-1); ins[1][1] << T(4);  ins[1][2] << T(-1);
        ins[2][1] << T(-1); ins[2][2] << T(4);
    }
    return A;
}

// x = [1 2 3] for spd3: b = [2 4 10].
template <typename V>
void require_x123(const V& x, double tol) {
    REQUIRE(std::abs(double(x(0)) - 1.0) < tol);
    REQUIRE(std::abs(double(x(1)) - 2.0) < tol);
    REQUIRE(std::abs(double(x(2)) - 3.0) < tol);
}

} // namespace

TEST_CASE("compressed2D<float> takes the native solvers", "[dispatch][850]") {
    auto A = spd3<float>();
    vec::dense_vector<float> b = {2.0f, 4.0f, 10.0f};

    vec::dense_vector<float> x_lu(3, 0.0f), x_ch(3, 0.0f), x_qr(3, 0.0f);
    lu_solve_dispatch(A, x_lu, b);
    cholesky_solve_dispatch(A, x_ch, b);
    qr_solve_dispatch(A, x_qr, b);
    require_x123(x_lu, 1e-5);
    require_x123(x_ch, 1e-5);
    require_x123(x_qr, 1e-5);
}

TEST_CASE("compressed2D<double> with dense_vector<float> x and b takes the native solvers",
          "[dispatch][850]") {
    auto A = spd3<double>();
    vec::dense_vector<float> b = {2.0f, 4.0f, 10.0f};

    vec::dense_vector<float> x_lu(3, 0.0f), x_ch(3, 0.0f), x_qr(3, 0.0f);
    lu_solve_dispatch(A, x_lu, b);
    cholesky_solve_dispatch(A, x_ch, b);
    qr_solve_dispatch(A, x_qr, b);
    require_x123(x_lu, 1e-5);
    require_x123(x_ch, 1e-5);
    require_x123(x_qr, 1e-5);
}

TEST_CASE("qr_solve_dispatch rejects an underdetermined system on every path",
          "[dispatch][qr][850]") {
    // SPQR and the native sparse QR both take m >= n only, and both throw
    // std::invalid_argument for m < n, so the result does not depend on which
    // one the build dispatches to.
    mat::compressed2D<double> W(2, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(W);
        ins[0][0] << 1.0; ins[0][2] << 1.0;
        ins[1][1] << 1.0;
    }
    vec::dense_vector<double> b(2, 1.0), x(3, 0.0);
    REQUIRE_THROWS_AS(qr_solve_dispatch(W, x, b), std::invalid_argument);

    mat::compressed2D<float> Wf(2, 3);
    {
        mat::inserter<mat::compressed2D<float>> ins(Wf);
        ins[0][0] << 1.0f; ins[0][2] << 1.0f;
        ins[1][1] << 1.0f;
    }
    vec::dense_vector<float> bf(2, 1.0f), xf(3, 0.0f);
    REQUIRE_THROWS_AS(qr_solve_dispatch(Wf, xf, bf), std::invalid_argument);
}

TEST_CASE("sparse dispatch solves an unsymmetric system with the matrix the right way round",
          "[dispatch][lu][850]") {
    // A = [4 1 0; 2 5 1; 0 3 6], b = [6 15 24] => x = [1 2 3]; with UMFPACK on
    // this goes through its CRS-to-CCS conversion, which a symmetric matrix
    // cannot check.
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 2.0; ins[1][1] << 5.0; ins[1][2] << 1.0;
        ins[2][1] << 3.0; ins[2][2] << 6.0;
    }
    vec::dense_vector<double> b = {6.0, 15.0, 24.0}, x(3, 0.0);
    lu_solve_dispatch(A, x, b);
    require_x123(x, 1e-12);
}
