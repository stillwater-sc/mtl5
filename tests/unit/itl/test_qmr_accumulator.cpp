// MTL5 -- accumulator policy for QMR (itl::qmr), mirroring cg/gmres (#237, #268).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/qmr.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("QMR default (unspecified Accumulator) behavior is unchanged",
          "[itl][qmr][accumulator]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 4; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 4;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};

    vec::dense_vector<double> x1(3, 0.0);
    itl::pc::identity<mat::dense2D<double>> pc1(A);
    itl::basic_iteration<double> iter1(b, 200, 1e-10);
    itl::qmr(A, x1, b, pc1, iter1);

    vec::dense_vector<double> x2(3, 0.0);
    itl::pc::identity<mat::dense2D<double>> pc2(A);
    itl::basic_iteration<double> iter2(b, 200, 1e-10);
    itl::qmr<mat::dense2D<double>, vec::dense_vector<double>, vec::dense_vector<double>,
             itl::pc::identity<mat::dense2D<double>>, itl::basic_iteration<double>, void>
             (A, x2, b, pc2, iter2);

    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(x1(i) == x2(i));   // bit-for-bit identical
}

TEST_CASE("QMR sparse tridiagonal system with explicit double Accumulator",
          "[itl][qmr][accumulator]") {
    const std::size_t n = 20;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i-1] << -1.0;
            if (i < n - 1) ins[i][i+1] << -1.0;
        }
    }
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 500, 1e-10);

    int err = itl::qmr<mat::compressed2D<double>, vec::dense_vector<double>,
                        vec::dense_vector<double>, itl::pc::identity<mat::compressed2D<double>>,
                        itl::basic_iteration<double>, double>
                        (A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), WithinAbs(b(i), 1e-8));
}
