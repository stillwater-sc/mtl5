#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>

using namespace mtl;
using namespace mtl::sparse;

TEST_CASE("Dense lower triangular solve in CSC", "[sparse][trisolve]") {
    // L = [[2 0 0]
    //      [1 3 0]
    //      [0 1 4]]
    // Solve Lx = b with b = [2, 7, 14]
    // x = [1, 2, 3]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0;
        ins[2][1] << 1.0; ins[2][2] << 4.0;
    }

    auto L = util::crs_to_csc(A);
    std::vector<double> x = {2.0, 7.0, 14.0};

    factorization::dense_lower_solve(L, x);

    REQUIRE_THAT(x[0], Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(x[1], Catch::Matchers::WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(x[2], Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("Dense upper triangular solve in CSC", "[sparse][trisolve]") {
    // U = [[2 1 0]
    //      [0 3 1]
    //      [0 0 4]]
    // Solve Ux = b with b = [4, 7, 12]
    // 4x[2] = 12 -> x[2] = 3
    // 3x[1] + 3 = 7 -> x[1] = 4/3
    // 2x[0] + 4/3 = 4 -> x[0] = 4/3
    mat::compressed2D<double> U_crs(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(U_crs);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][2] << 4.0;
    }

    auto U = util::crs_to_csc(U_crs);
    std::vector<double> x = {4.0, 7.0, 12.0};

    factorization::dense_upper_solve(U, x);

    REQUIRE_THAT(x[2], Catch::Matchers::WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(x[1], Catch::Matchers::WithinAbs(4.0 / 3.0, 1e-12));
    REQUIRE_THAT(x[0], Catch::Matchers::WithinAbs(4.0 / 3.0, 1e-12));
}

TEST_CASE("Dense lower transpose solve in CSC", "[sparse][trisolve]") {
    // L = [[2 0 0]
    //      [1 3 0]
    //      [0 1 4]]
    // Solve L^T x = b with b = [4, 3, 12]
    // L^T = [[2 1 0]
    //        [0 3 1]
    //        [0 0 4]]
    // Same as upper solve: x[2] = 3, x[1] = 0, x[0] = 2
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0;
        ins[2][1] << 1.0; ins[2][2] << 4.0;
    }

    auto L = util::crs_to_csc(A);
    std::vector<double> x = {4.0, 3.0, 12.0};

    factorization::dense_lower_transpose_solve(L, x);

    REQUIRE_THAT(x[2], Catch::Matchers::WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(x[1], Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(x[0], Catch::Matchers::WithinAbs(2.0, 1e-12));
}

TEST_CASE("Lower solve then transpose solve recovers original", "[sparse][trisolve]") {
    // L = [[3 0 0]
    //      [2 5 0]
    //      [1 4 7]]
    // b = [6, 17, 37]
    // Solve Lx = b -> x = [2, 13/5, 37/7 - ...]
    // Then solve L^T y = x should give back some vector,
    // and L * L^T * y = b (i.e., y = (L L^T)^{-1} b)

    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0;
        ins[1][0] << 2.0; ins[1][1] << 5.0;
        ins[2][0] << 1.0; ins[2][1] << 4.0; ins[2][2] << 7.0;
    }

    auto L = util::crs_to_csc(A);

    // Forward solve: Ly = b
    std::vector<double> b = {6.0, 17.0, 37.0};
    std::vector<double> y = b;
    factorization::dense_lower_solve(L, y);

    // Back solve: L^T x = y
    std::vector<double> x = y;
    factorization::dense_lower_transpose_solve(L, x);

    // Verify: L * (L^T * x) should equal b
    // First compute L^T * x
    std::vector<double> Ltx(3, 0.0);
    // L^T = [[3 2 1], [0 5 4], [0 0 7]]
    Ltx[0] = 3.0 * x[0] + 2.0 * x[1] + 1.0 * x[2];
    Ltx[1] = 5.0 * x[1] + 4.0 * x[2];
    Ltx[2] = 7.0 * x[2];

    // Then compute L * Ltx
    std::vector<double> result(3, 0.0);
    result[0] = 3.0 * Ltx[0];
    result[1] = 2.0 * Ltx[0] + 5.0 * Ltx[1];
    result[2] = 1.0 * Ltx[0] + 4.0 * Ltx[1] + 7.0 * Ltx[2];

    REQUIRE_THAT(result[0], Catch::Matchers::WithinAbs(b[0], 1e-10));
    REQUIRE_THAT(result[1], Catch::Matchers::WithinAbs(b[1], 1e-10));
    REQUIRE_THAT(result[2], Catch::Matchers::WithinAbs(b[2], 1e-10));
}
