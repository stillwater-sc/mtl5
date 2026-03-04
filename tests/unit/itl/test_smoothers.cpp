#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/sor.hpp>

using namespace mtl;

// Helper: build a 4x4 diagonally dominant SPD dense matrix
static mat::dense2D<double> make_dense_spd() {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0;
    A(0,0) = 4; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 4; A(1,2) = 1;
    A(2,1) = 1; A(2,2) = 4; A(2,3) = 1;
    A(3,2) = 1; A(3,3) = 4;
    return A;
}

// Helper: build same matrix as sparse
static mat::compressed2D<double> make_sparse_spd() {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 4.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 4.0; ins[2][3] << 1.0;
        ins[3][2] << 1.0; ins[3][3] << 4.0;
    }
    return A;
}

TEST_CASE("Gauss-Seidel reduces residual (dense)", "[itl][smoother][gauss_seidel]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::smoother::gauss_seidel<mat::dense2D<double>> gs(A);

    auto r0 = A * x;
    for (std::size_t i = 0; i < 4; ++i) r0(i) = b(i) - r0(i);
    double norm0 = two_norm(r0);

    // Apply several sweeps
    for (int sweep = 0; sweep < 20; ++sweep)
        gs(x, b);

    auto r1 = A * x;
    for (std::size_t i = 0; i < 4; ++i) r1(i) = b(i) - r1(i);
    double norm1 = two_norm(r1);

    REQUIRE(norm1 < norm0 * 1e-6);
}

TEST_CASE("Gauss-Seidel reduces residual (sparse)", "[itl][smoother][gauss_seidel][sparse]") {
    auto A = make_sparse_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A);

    for (int sweep = 0; sweep < 20; ++sweep)
        gs(x, b);

    auto r = A * x;
    for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
    REQUIRE(two_norm(r) < 1e-6);
}

TEST_CASE("Jacobi reduces residual", "[itl][smoother][jacobi]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::smoother::jacobi<mat::dense2D<double>> jac(A);

    for (int sweep = 0; sweep < 50; ++sweep)
        jac(x, b);

    auto r = A * x;
    for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
    REQUIRE(two_norm(r) < 1e-6);
}

TEST_CASE("SOR with omega=1.0 matches Gauss-Seidel", "[itl][smoother][sor]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x_gs(4, 0.0);
    vec::dense_vector<double> x_sor(4, 0.0);

    itl::smoother::gauss_seidel<mat::dense2D<double>> gs(A);
    itl::smoother::sor<mat::dense2D<double>> sor_1(A, 1.0);

    for (int sweep = 0; sweep < 10; ++sweep) {
        gs(x_gs, b);
        sor_1(x_sor, b);
    }

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(x_sor(i), Catch::Matchers::WithinAbs(x_gs(i), 1e-12));
    }
}

TEST_CASE("Sparse specialization matches generic version", "[itl][smoother][sparse]") {
    auto Ad = make_dense_spd();
    auto As = make_sparse_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x_dense(4, 0.0);
    vec::dense_vector<double> x_sparse(4, 0.0);

    itl::smoother::gauss_seidel<mat::dense2D<double>> gs_d(Ad);
    itl::smoother::gauss_seidel<mat::compressed2D<double>> gs_s(As);

    for (int sweep = 0; sweep < 10; ++sweep) {
        gs_d(x_dense, b);
        gs_s(x_sparse, b);
    }

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(x_sparse(i), Catch::Matchers::WithinAbs(x_dense(i), 1e-12));
    }
}
