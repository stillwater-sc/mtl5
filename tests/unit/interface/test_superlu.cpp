// MTL5 -- Tests for the SuperLU sparse direct solver binding (#850)
//
// superlu_solver factors A once in its constructor (COLAMD, then dgstrf) and
// solves each right-hand side with dgstrs. It used to run dgssv -- a full
// factorization -- on every solve(), so factor_nnz() was 0 until the first
// solve and a singular A was only reported by solve().
//
// These tests only do real work when MTL5_HAS_SUPERLU is defined (CMake option
// MTL5_WITH_SUPERLU=ON). Otherwise a placeholder confirms the file compiles.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

#ifdef MTL5_HAS_SUPERLU

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include <mtl/interface/superlu.hpp>

namespace {

using csr = mat::compressed2D<double>;

// Unsymmetric: A = [4 1 0; 2 5 1; 0 3 6], x = [1 2 3] gives b = [6 15 24].
// A transposed CRS-to-CCS conversion solves A^T x = b instead, a different x.
csr unsymmetric3() {
    csr A(3, 3);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 2.0; ins[1][1] << 5.0; ins[1][2] << 1.0;
        ins[2][1] << 3.0; ins[2][2] << 6.0;
    }
    return A;
}

// Unsymmetric tridiagonal: A(i,i) = 4, A(i,i-1) = -1, A(i,i+1) = -2.
csr unsym_tridiag(std::size_t n) {
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     ins[i][i - 1] << -1.0;
            ins[i][i] << 4.0;
            if (i + 1 < n) ins[i][i + 1] << -2.0;
        }
    }
    return A;
}

double residual_inf(const csr& A, const vec::dense_vector<double>& x,
                    const vec::dense_vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v  = A.ref_data();
    double r = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ax += v[k] * x(ci[k]);
        const double ri = std::abs(ax - b(i));
        if (!std::isfinite(ri)) return std::numeric_limits<double>::infinity();
        r = std::max(r, ri);
    }
    return r;
}

void require_x123(const vec::dense_vector<double>& x) {
    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(x(1), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(x(2), WithinAbs(3.0, 1e-12));
}

} // namespace

TEST_CASE("superlu_solver on an unsymmetric matrix", "[interface][superlu][850]") {
    vec::dense_vector<double> b = {6.0, 15.0, 24.0}, x(3, 0.0);
    interface::superlu_solve(unsymmetric3(), x, b);
    require_x123(x);
}

TEST_CASE("superlu_solver factors once and reuses the factors", "[interface][superlu][850]") {
    const std::size_t n = 30;
    auto A = unsym_tridiag(n);
    interface::superlu_solver solver(A);

    // The fill is known as soon as the constructor returns: L and U each hold
    // at least the n diagonal entries.
    const std::size_t fill = solver.factor_nnz();
    REQUIRE(fill >= 2 * n);

    for (int rhs = 0; rhs < 4; ++rhs) {
        vec::dense_vector<double> b(n), x(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) b(i) = double((i * (rhs + 3)) % 7) - 2.0;
        solver.solve(x, b);
        REQUIRE(residual_inf(A, x, b) < 1e-12);
        REQUIRE(solver.factor_nnz() == fill);
    }
}

TEST_CASE("superlu_solver throws on a singular matrix in the constructor",
          "[interface][superlu][850]") {
    csr A(3, 3);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][0] << 2.0; ins[1][1] << 4.0;
        ins[2][2] << 1.0;
    }
    REQUIRE_THROWS_AS(interface::superlu_solver(A), std::runtime_error);

    // Structurally singular: column 1 is empty.
    csr S(3, 3);
    {
        mat::inserter<csr> ins(S);
        ins[0][0] << 1.0; ins[1][0] << 1.0; ins[2][2] << 1.0;
    }
    REQUIRE_THROWS_AS(interface::superlu_solver(S), std::runtime_error);
}

TEST_CASE("superlu_solver rejects a non-square matrix and wrong vector sizes",
          "[interface][superlu][850]") {
    csr R(2, 3);
    {
        mat::inserter<csr> ins(R);
        ins[0][0] << 1.0; ins[1][1] << 1.0;
    }
    REQUIRE_THROWS_AS(interface::superlu_solver(R), std::invalid_argument);

    interface::superlu_solver solver(unsymmetric3());
    vec::dense_vector<double> b3(3, 1.0), x2(2), b4(4, 1.0), x3(3);
    REQUIRE_THROWS_AS(solver.solve(x2, b3), std::invalid_argument);
    REQUIRE_THROWS_AS(solver.solve(x3, b4), std::invalid_argument);
}

TEST_CASE("superlu_solver is movable", "[interface][superlu][850]") {
    vec::dense_vector<double> b = {6.0, 15.0, 24.0}, x(3, 0.0);

    interface::superlu_solver solver(unsymmetric3());
    const std::size_t fill = solver.factor_nnz();
    interface::superlu_solver moved(std::move(solver));
    REQUIRE(solver.num_rows() == 0);
    REQUIRE(solver.factor_nnz() == 0);
    REQUIRE_THROWS_AS(solver.solve(x, b), std::logic_error);
    REQUIRE(moved.factor_nnz() == fill);
    moved.solve(x, b);
    require_x123(x);

    // Move assignment frees the target's factors and takes the source's.
    csr I(1, 1);
    { mat::inserter<csr> ins(I); ins[0][0] << 2.0; }
    interface::superlu_solver target(I);
    target = std::move(moved);
    REQUIRE(target.num_rows() == 3);
    x = vec::dense_vector<double>(3, 0.0);
    target.solve(x, b);
    require_x123(x);
}

TEST_CASE("superlu_solver on 0x0 and 1x1 matrices", "[interface][superlu][850]") {
    csr E(0, 0);
    interface::superlu_solver empty(E);
    REQUIRE(empty.num_rows() == 0);
    REQUIRE(empty.factor_nnz() == 0);
    vec::dense_vector<double> e0(0), x0(0);
    empty.solve(x0, e0);

    csr A(1, 1);
    { mat::inserter<csr> ins(A); ins[0][0] << 4.0; }
    interface::superlu_solver one(A);
    REQUIRE(one.factor_nnz() == 2);   // L(0,0) = 1 and U(0,0) = 4
    vec::dense_vector<double> b(1, 2.0), x(1, 0.0);
    one.solve(x, b);
    REQUIRE(x(0) == 0.5);
}

#else // !MTL5_HAS_SUPERLU

TEST_CASE("SuperLU not available -- placeholder", "[interface][superlu]") {
    SUCCEED("MTL5_HAS_SUPERLU not defined -- the SuperLU binding tests are skipped");
}

#endif // MTL5_HAS_SUPERLU
