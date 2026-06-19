// MTL5 -- Tests for the KLU sparse direct solver binding
//
// KLU (Davis & Palamadai Natarajan, ACM TOMS Algorithm 907) is a sparse LU
// solver tuned for circuit-simulation matrices: it permutes to block
// triangular form (BTF) via Dulmage-Mendelsohn, then applies Gilbert-Peierls
// left-looking LU to each diagonal block.
//
// These tests only do real work when MTL5_HAS_KLU is defined (CMake option
// MTL5_WITH_SUITESPARSE_KLU=ON with SuiteSparse installed). Otherwise a
// placeholder confirms the test infrastructure compiles and links.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

#ifdef MTL5_HAS_KLU

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include <mtl/interface/klu.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

namespace {

// Residual infinity-norm ||A*x - b|| for a CSR (row-major) compressed2D.
double residual_inf_norm(const mat::compressed2D<double>& A,
                         const vec::dense_vector<double>& x,
                         const vec::dense_vector<double>& b) {
    const auto& row_ptr = A.ref_major();
    const auto& col_idx = A.ref_minor();
    const auto& data    = A.ref_data();
    double max_r = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
            ax += data[k] * x(static_cast<int>(col_idx[k]));
        max_r = std::max(max_r, std::abs(ax - b(static_cast<int>(r))));
    }
    return max_r;
}

// Unsymmetric tridiagonal: A(i,i)=4, A(i,i-1)=-1, A(i,i+1)=-2.
mat::compressed2D<double> make_unsym_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     ins[i][i - 1] << -1.0;
            ins[i][i] << 4.0;
            if (i + 1 < n) ins[i][i + 1] << -2.0;
        }
    }
    return A;
}

} // namespace

TEST_CASE("klu_solve on small unsymmetric system", "[interface][klu]") {
    // A = [2 1; 1 3], b = [5; 10] => x = [1; 3]
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;
        ins[0][1] << 1.0;
        ins[1][0] << 1.0;
        ins[1][1] << 3.0;
    }

    vec::dense_vector<double> b = {5.0, 10.0};
    vec::dense_vector<double> x(2, 0.0);

    interface::klu_solve(A, x, b);

    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(x(1), WithinAbs(3.0, 1e-10));
}

TEST_CASE("klu_solver RAII class on identity", "[interface][klu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
        ins[2][2] << 1.0;
    }

    vec::dense_vector<double> b = {7.0, 8.0, 9.0};
    vec::dense_vector<double> x(3, 0.0);

    interface::klu_solver solver(A);
    solver.solve(x, b);

    for (int i = 0; i < 3; ++i)
        REQUIRE_THAT(x(i), WithinAbs(b(i), 1e-10));
}

TEST_CASE("klu_solver reuses factorization across right-hand sides",
          "[interface][klu]") {
    auto A = make_unsym_tridiag(8);
    interface::klu_solver solver(A);

    // Factor once, solve twice with different RHS; check each residual.
    vec::dense_vector<double> b1(8), b2(8), x1(8, 0.0), x2(8, 0.0);
    for (int i = 0; i < 8; ++i) {
        b1(i) = 1.0;
        b2(i) = static_cast<double>(i + 1);
    }

    solver.solve(x1, b1);
    solver.solve(x2, b2);

    REQUIRE(residual_inf_norm(A, x1, b1) < 1e-10);
    REQUIRE(residual_inf_norm(A, x2, b2) < 1e-10);
}

TEST_CASE("klu_solver handles block triangular structure (BTF path)",
          "[interface][klu]") {
    // Two decoupled 2x2 blocks linked by an upper off-diagonal coupling, so
    // the matrix is reducible -- exactly the structure KLU's Dulmage-Mendelsohn
    // BTF permutation is designed to exploit.
    //   [ 2  1 | 0  5 ]
    //   [ 1  2 | 0  0 ]
    //   [ 0  0 | 3  1 ]
    //   [ 0  0 | 1  3 ]
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0; ins[0][3] << 5.0;
        ins[1][0] << 1.0; ins[1][1] << 2.0;
        ins[2][2] << 3.0; ins[2][3] << 1.0;
        ins[3][2] << 1.0; ins[3][3] << 3.0;
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    interface::klu_solver solver(A);
    solver.solve(x, b);

    REQUIRE(residual_inf_norm(A, x, b) < 1e-10);
}

TEST_CASE("klu_solver agrees with native sparse LU", "[interface][klu]") {
    for (std::size_t n : {5u, 16u, 40u}) {
        auto A = make_unsym_tridiag(n);

        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b(static_cast<int>(i)) = static_cast<double>((i % 7) + 1);

        vec::dense_vector<double> x_klu(n, 0.0);
        interface::klu_solver(A).solve(x_klu, b);

        vec::dense_vector<double> x_lu(n, 0.0);
        sparse::factorization::sparse_lu_solve(A, x_lu, b);

        for (std::size_t i = 0; i < n; ++i)
            REQUIRE_THAT(x_klu(static_cast<int>(i)),
                         WithinAbs(x_lu(static_cast<int>(i)), 1e-8));
    }
}

TEST_CASE("klu_solver is movable", "[interface][klu]") {
    auto A = make_unsym_tridiag(6);
    interface::klu_solver solver(A);
    interface::klu_solver moved(std::move(solver));

    vec::dense_vector<double> b(6, 1.0), x(6, 0.0);
    moved.solve(x, b);
    REQUIRE(residual_inf_norm(A, x, b) < 1e-10);
}

TEST_CASE("klu_solver rejects non-square matrix", "[interface][klu]") {
    mat::compressed2D<double> A(2, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
    }
    REQUIRE_THROWS_AS(interface::klu_solver(A), std::invalid_argument);
}

#else // !MTL5_HAS_KLU

TEST_CASE("KLU not available -- placeholder", "[interface][klu]") {
    // KLU support not compiled in; verify test infrastructure works.
    REQUIRE(true);
}

#endif // MTL5_HAS_KLU
