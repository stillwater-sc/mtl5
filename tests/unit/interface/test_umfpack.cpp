// MTL5 Phase 14 -- Tests for UMFPACK sparse direct solver
// These tests are only compiled when MTL5_HAS_UMFPACK is defined.
// When UMFPACK is not available, a placeholder test confirms the test compiles.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

#ifdef MTL5_HAS_UMFPACK

#include <mtl/interface/umfpack.hpp>

TEST_CASE("umfpack_solve on small system", "[interface][umfpack]") {
    // A = [2 1; 1 3] (sparse), b = [5; 10] => x = [1; 3]
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

    interface::umfpack_solve(A, x, b);

    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(x(1), WithinAbs(3.0, 1e-10));
}

TEST_CASE("umfpack_solver RAII class", "[interface][umfpack]") {
    // 3x3 identity: x = b
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
        ins[2][2] << 1.0;
    }

    vec::dense_vector<double> b = {7.0, 8.0, 9.0};
    vec::dense_vector<double> x(3, 0.0);

    interface::umfpack_solver solver(A);
    solver.solve(x, b);

    for (int i = 0; i < 3; ++i)
        REQUIRE_THAT(x(i), WithinAbs(b(i), 1e-10));
}

#else // !MTL5_HAS_UMFPACK

TEST_CASE("UMFPACK not available -- placeholder", "[interface][umfpack]") {
    // UMFPACK support not compiled in; verify test infrastructure works.
    REQUIRE(true);
}

#endif // MTL5_HAS_UMFPACK
