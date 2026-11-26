#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/gmres.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using namespace mtl;

TEST_CASE("GMRES on 3x3 non-symmetric system", "[itl][gmres]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 3;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::gmres(A, x, b, pc, iter, 30);

    REQUIRE(err == 0);

    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("GMRES with restart on 10x10 system", "[itl][gmres][restart]") {
    const std::size_t n = 10;
    mat::dense2D<double> A(n, n);
    // Zero out
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    // Diagonally dominant non-symmetric
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 5.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -2.0;  // non-symmetric
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    // Use restart=3 to force multiple restarts
    int err = itl::gmres(A, x, b, pc, iter, 3);

    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-7));
    }
}

TEST_CASE("GMRES with diagonal preconditioner", "[itl][gmres][pc]") {
    const std::size_t n = 5;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -2.0;
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::gmres(A, x, b, pc, iter);

    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

// --- Regression: lucky breakdown must not hang or leave x/iter unchanged.
// A matrix with only 2 distinct eigenvalues converges GMRES in <=2 Krylov
// steps; if x(1)^2 spans exactly that subspace, H(k+1,k) hits an exact zero. ---
TEST_CASE("GMRES lucky breakdown converges (does not hang)", "[itl][gmres][breakdown]") {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i,j) = 0.0;
    // Two distinct eigenvalues (2 and 5), each with multiplicity 2.
    A(0,0) = 2; A(1,1) = 2; A(2,2) = 5; A(3,3) = 5;

    vec::dense_vector<double> b = {1.0, 1.0, 1.0, 1.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, /*max_iter=*/3, 1e-10);

    int err = itl::gmres(A, x, b, pc, iter, /*restart=*/10);
    REQUIRE(err == 0);
    REQUIRE(iter.iterations() == 2);   // old buggy code cannot reach exact convergence in this budget

    auto r = A * x;
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("GMRES on a complex non-Hermitian system", "[itl][gmres][853]") {
    // The Givens rotations were not unitary for complex entries, so the
    // least-squares problem GMRES solved was the wrong one.
    using C = std::complex<double>;
    const std::size_t n = 6;
    mat::dense2D<C> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? C(4.0 + double(i), 1.0)
                               : C(double((i + 2 * j) % 3) - 1.0, double((2 * i + j) % 5) / 4.0 - 0.5);
    vec::dense_vector<C> b(n), x(n, C(0));
    for (std::size_t i = 0; i < n; ++i) b(i) = C(1.0 + double(i), -double(i % 2));
    itl::pc::identity<mat::dense2D<C>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-12);
    const int err = itl::gmres(A, x, b, pc, iter, int(n));
    CHECK(err == 0);
    double r2 = 0, b2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        C ax(0);
        for (std::size_t j = 0; j < n; ++j) ax += A(i, j) * x(j);
        r2 += std::norm(b(i) - ax);
        b2 += std::norm(b(i));
    }
    CHECK(std::sqrt(r2 / b2) < 1e-10);
}

namespace {
// 100 x (2-D Laplacian on a k x k grid, plus 0.3 on the superdiagonal):
// non-symmetric, with 400 on the diagonal (issue 936).
mat::compressed2D<double> scaled_convective_laplacian(std::size_t k) {
    const std::size_t n = k * k;
    mat::compressed2D<double> A(n, n);
    {   // the inserter writes A when it is destroyed: before the return
        mat::inserter<mat::compressed2D<double>> ins(A, 6);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 400.0;
                if (i > 0)     ins[r][r - k] << -100.0;
                if (i + 1 < k) ins[r][r + k] << -100.0;
                if (j > 0)     ins[r][r - 1] << -100.0;
                if (j + 1 < k)      ins[r][r + 1] << -70.0;
                else if (r + 1 < n) ins[r][r + 1] << 30.0;
            }
    }
    return A;
}

template <typename Matrix, typename Vector>
double true_relative_residual(const Matrix& A, const Vector& x, const Vector& b) {
    Vector Ax(b.size()), r(b.size());
    mult(A, x, Ax);
    for (std::size_t i = 0; i < b.size(); ++i) r(i) = b(i) - Ax(i);
    return two_norm(r) / two_norm(b);
}

// out = in / d, elementwise: a diagonal preconditioner given its diagonal,
// for a matrix whose own diagonal has zeros.
struct given_diagonal_pc {
    std::vector<double> d;
    template <typename VecOut, typename VecIn>
    void solve(VecOut& out, const VecIn& in) const {
        for (std::size_t i = 0; i < out.size(); ++i) out(i) = in(i) / d[i];
    }
};

// A preconditioner that returns NaN.
struct nan_pc {
    template <typename VecOut, typename VecIn>
    void solve(VecOut& out, const VecIn&) const {
        for (std::size_t i = 0; i < out.size(); ++i) out(i) = std::numeric_limits<double>::quiet_NaN();
    }
};

// A preconditioner that maps every vector to zero: a singular M.
struct zero_pc {
    template <typename VecOut, typename VecIn>
    void solve(VecOut& out, const VecIn&) const {
        for (std::size_t i = 0; i < out.size(); ++i) out(i) = 0.0;
    }
};
} // namespace

TEST_CASE("GMRES with a preconditioner converges on the true residual", "[itl][gmres][936]") {
    // GMRES is left-preconditioned and tested ||M^{-1}(b - A x)|| against
    // rtol * ||b||. With the diagonal preconditioner (400 I here) it returned
    // 0 at a true relative residual of 3.5e-6 for rtol 1e-8, and relresid()
    // reported the preconditioned 8.7e-9.
    using Mat = mat::compressed2D<double>;
    const auto A = scaled_convective_laplacian(12);
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n, 1.0);
    const double rtol = 1e-8;

    auto check = [&](const auto& M) {
        vec::dense_vector<double> x(n, 0.0);
        itl::basic_iteration<double> iter(b, 1000, rtol);
        const int info = itl::gmres(A, x, b, M, iter, 30);
        const double rel = true_relative_residual(A, x, b);
        INFO("iterations " << iter.iterations() << ", relresid " << iter.relresid() << ", true " << rel);
        CHECK(info == 0);
        CHECK(rel <= rtol);
        // relresid() reports the residual the decision was made on.
        CHECK_THAT(iter.relresid(), Catch::Matchers::WithinRel(rel, 1e-6));
    };
    SECTION("identity") { check(itl::pc::identity<Mat>(A)); }
    SECTION("diagonal") { check(itl::pc::diagonal<Mat>(A)); }
}

TEST_CASE("GMRES with a diagonal preconditioner and short restarts", "[itl][gmres][936]") {
    // A restart after a false convergence by the preconditioned estimate
    // must still make progress: the estimate is scaled to the true residual,
    // so a restarted cycle does not open below the tolerance and stall at one
    // step per cycle. The diagonal is not constant here, so the scale is
    // only an estimate and the true residual decides.
    using Mat = mat::compressed2D<double>;
    const std::size_t n = 200;
    Mat A(n, n);
    {
        mat::inserter<Mat> ins(A, 3);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0 + 1000.0 * double(i % 7);
            if (i > 0)     ins[i][i - 1] << -1.5;
            if (i + 1 < n) ins[i][i + 1] << -0.5;
        }
    }
    vec::dense_vector<double> b(n, 1.0), x(n, 0.0);
    itl::pc::diagonal<Mat> M(A);
    itl::basic_iteration<double> iter(b, 2000, 1e-10);
    const int info = itl::gmres(A, x, b, M, iter, 5);
    const double rel = true_relative_residual(A, x, b);
    INFO("iterations " << iter.iterations() << ", true " << rel);
    CHECK(info == 0);
    CHECK(rel <= 1e-10);
    CHECK(iter.iterations() < 200);
}

TEST_CASE("GMRES with a preconditioner that maps the residual to zero breaks down", "[itl][gmres][936]") {
    // An unconverged b - A x with M^{-1}(b - A x) == 0 (a singular M) made the
    // first basis vector 0 / 0. It is a breakdown now, error 2.
    mat::dense2D<double> A(2, 2);
    A(0, 0) = 2; A(0, 1) = 1; A(1, 0) = 0; A(1, 1) = 3;
    vec::dense_vector<double> b = {1.0, 1.0}, x(2, 0.0);
    itl::basic_iteration<double> iter(b, 50, 1e-10);
    const int info = itl::gmres(A, x, b, zero_pc{}, iter, 2);
    CHECK(info == 2);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
}

TEST_CASE("GMRES with a preconditioner that returns NaN fails with x unchanged", "[itl][gmres][936]") {
    // A non-finite M^{-1}(b - A x) was divided into the first basis vector
    // and reached x. It is error 6, the non-finite residual, as before 936.
    mat::dense2D<double> A(2, 2);
    A(0, 0) = 2; A(0, 1) = 1; A(1, 0) = 0; A(1, 1) = 3;
    vec::dense_vector<double> b = {1.0, 1.0}, x(2, 0.0);
    itl::basic_iteration<double> iter(b, 50, 1e-10);
    const int info = itl::gmres(A, x, b, nan_pc{}, iter, 2);
    CHECK(info == 6);
    CHECK(x(0) == 0.0);
    CHECK(x(1) == 0.0);
}

TEST_CASE("GMRES: a singular breakdown is reported only if the true residual has not converged", "[itl][gmres][936]") {
    // A singular on the Krylov space: Arnoldi's second column is exactly zero
    // and is dropped. The update gives x = (1/4, 1/4, 1/4, 1/4), whose true
    // residual 0.707 is below rtol ||b|| = 0.907, but the preconditioned
    // estimate (8.0) reported breakdown, error 2, without testing it.
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) A(i, j) = i < 2 ? 8.0 : 0.0;
    vec::dense_vector<double> b = {8.0, 8.0, 0.5, 0.5}, x(4, 0.0);
    given_diagonal_pc M{{16.0, 16.0, 1.0, 1.0}};
    const double rtol = 0.08;
    itl::basic_iteration<double> iter(b, 50, rtol);
    const int info = itl::gmres(A, x, b, M, iter, 2);
    const double rel = true_relative_residual(A, x, b);
    INFO("info " << info << ", true " << rel << ", relresid " << iter.relresid());
    CHECK(info == 0);
    CHECK(rel <= rtol);
    CHECK_THAT(iter.relresid(), Catch::Matchers::WithinRel(rel, 1e-12));
}

namespace {
// out = 1e-310 * in: a preconditioner whose output is tiny but not zero.
struct tiny_pc {
    template <typename VecOut, typename VecIn>
    void solve(VecOut& out, const VecIn& in) const {
        for (std::size_t i = 0; i < out.size(); ++i) out(i) = 1e-310 * in(i);
    }
};
} // namespace

TEST_CASE("GMRES with a preconditioner that scales by 1e-310 still converges", "[itl][gmres][936]") {
    // ||r0|| / ||M^{-1} r0|| = 1e310 overflowed: every scaled estimate was
    // inf, each cycle kept one column, and gmres ran to its iteration limit.
    mat::dense2D<double> A(2, 2);
    A(0, 0) = 1; A(0, 1) = 0; A(1, 0) = 0; A(1, 1) = 2;
    vec::dense_vector<double> b = {1.0, 1.0}, x(2, 0.0);
    itl::basic_iteration<double> iter(b, 100, 1e-10);
    const int info = itl::gmres(A, x, b, tiny_pc{}, iter, 2);
    const double rel = true_relative_residual(A, x, b);
    INFO("info " << info << ", iterations " << iter.iterations() << ", true " << rel);
    CHECK(info == 0);
    CHECK(rel <= 1e-10);
    CHECK(iter.iterations() <= 10);
}
