// Tests for the BLAS Level-3 symmetric operators: symm, syrk, syr2k
// (#229, batch 3). Exercises the generic (row-major) path and the column-major
// (BLAS-dispatch) path against a plain reference; syrk/syr2k results are checked
// to be full symmetric matrices.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/syr2k.hpp>

#include <cmath>
#include <vector>

using namespace mtl;
template <typename Params = mat::parameters<>>
using dmat = mat::dense2D<double, Params>;
using colmat = mat::dense2D<double, mat::parameters<tag::col_major>>;

namespace {

// C(i,j) = alpha * sum_k A(i,k) B(k,j) + beta * C(i,j), A treated symmetric/full.
double sym_symm_ref(const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    double alpha, double beta, double c0,
                    std::size_t i, std::size_t j, std::size_t m) {
    double acc = 0.0;
    for (std::size_t k = 0; k < m; ++k) acc += A[i][k] * B[k][j];
    return alpha * acc + beta * c0;
}

template <typename M>
bool is_symmetric(const M& C, std::size_t n, double tol) {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (std::abs(C(i, j) - C(j, i)) > tol) return false;
    return true;
}

} // namespace

TEST_CASE("symm: C = alpha A B + beta C, A symmetric", "[operation][blas][symm]") {
    // A symmetric 3x3, B 3x2.
    std::vector<std::vector<double>> Av = {{2,1,0},{1,3,1},{0,1,2}};
    std::vector<std::vector<double>> Bv = {{1,4},{2,5},{3,6}};
    for (auto orient : {0, 1}) {   // 0 = row-major (generic), 1 = col-major (BLAS)
        auto run = [&](auto& A, auto& B, auto& C) {
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j) A(i, j) = Av[i][j];
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 2; ++j) { B(i, j) = Bv[i][j]; }
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 2; ++j) C(i, j) = 1.0;
            symm(2.0, A, B, 0.5, C);
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 2; ++j)
                    REQUIRE_THAT(C(i, j),
                        Catch::Matchers::WithinAbs(sym_symm_ref(Av, Bv, 2.0, 0.5, 1.0, i, j, 3), 1e-12));
        };
        if (orient == 0) { dmat<> A(3,3), B(3,2), C(3,2); run(A, B, C); }
        else             { colmat A(3,3), B(3,2), C(3,2); run(A, B, C); }
    }
}

TEST_CASE("syrk: C = alpha A A^T + beta C is full symmetric", "[operation][blas][syrk]") {
    // A is 3x2.
    std::vector<std::vector<double>> Av = {{1,2},{3,4},{5,6}};
    auto run = [&](auto& A, auto& C) {
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 2; ++j) A(i, j) = Av[i][j];
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) C(i, j) = 0.0;
        syrk(1.0, A, 0.0, C);
        // reference (A A^T)
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (std::size_t l = 0; l < 2; ++l) acc += Av[i][l] * Av[j][l];
                REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(acc, 1e-12));
            }
        REQUIRE(is_symmetric(C, 3, 1e-12));
    };
    { dmat<> A(3,2), C(3,3); run(A, C); }        // generic
    { colmat A(3,2), C(3,3); run(A, C); }        // BLAS
}

TEST_CASE("syrk: alpha/beta accumulation", "[operation][blas][syrk]") {
    colmat A(2, 2), C(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    C(0,0)=10; C(0,1)=20; C(1,0)=20; C(1,1)=30;   // symmetric input
    syrk(2.0, A, 0.5, C);
    // A A^T = [[5,11],[11,25]]; 2*that + 0.5*C = [[10+5, 22+10],[22+10, 50+15]] = [[15,32],[32,65]]
    REQUIRE_THAT(C(0,0), Catch::Matchers::WithinAbs(15.0, 1e-12));
    REQUIRE_THAT(C(0,1), Catch::Matchers::WithinAbs(32.0, 1e-12));
    REQUIRE_THAT(C(1,0), Catch::Matchers::WithinAbs(32.0, 1e-12));
    REQUIRE_THAT(C(1,1), Catch::Matchers::WithinAbs(65.0, 1e-12));
}

TEST_CASE("syr2k: C = alpha(A B^T + B A^T) + beta C is full symmetric", "[operation][blas][syr2k]") {
    std::vector<std::vector<double>> Av = {{1,0},{2,1},{0,3}};
    std::vector<std::vector<double>> Bv = {{2,1},{1,0},{1,1}};
    auto run = [&](auto& A, auto& B, auto& C) {
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 2; ++j) { A(i,j)=Av[i][j]; B(i,j)=Bv[i][j]; }
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) C(i, j) = 0.0;
        syr2k(1.0, A, B, 0.0, C);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (std::size_t l = 0; l < 2; ++l) acc += Av[i][l]*Bv[j][l] + Bv[i][l]*Av[j][l];
                REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(acc, 1e-12));
            }
        REQUIRE(is_symmetric(C, 3, 1e-12));
    };
    { dmat<> A(3,2), B(3,2), C(3,3); run(A, B, C); }   // generic
    { colmat A(3,2), B(3,2), C(3,3); run(A, B, C); }   // BLAS
}
