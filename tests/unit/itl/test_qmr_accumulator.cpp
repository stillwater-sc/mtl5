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
    int err1 = itl::qmr(A, x1, b, pc1, iter1);

    vec::dense_vector<double> x2(3, 0.0);
    itl::pc::identity<mat::dense2D<double>> pc2(A);
    itl::basic_iteration<double> iter2(b, 200, 1e-10);
    int err2 = itl::qmr<mat::dense2D<double>, vec::dense_vector<double>, vec::dense_vector<double>,
             itl::pc::identity<mat::dense2D<double>>, itl::basic_iteration<double>, void>
             (A, x2, b, pc2, iter2);

    REQUIRE(err1 == 0);
    REQUIRE(err2 == 0);
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(x1(i) == x2(i));   // bit-for-bit identical

    // Also confirm convergence is real, not just agreement between two failures.
    auto r1 = A * x1;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r1(i), WithinAbs(b(i), 1e-8));
}

TEST_CASE("QMR nonsymmetric sparse system with explicit float->double Accumulator",
          "[itl][qmr][accumulator]") {
    // Nonsymmetric on purpose: A != A^T here, so a broken trans(A) routing
    // (e.g. accidentally reusing A instead of A^T) would fail to converge
    // or converge to the wrong solution, unlike a symmetric test matrix
    // where A and A^T are indistinguishable. float operands with a double
    // Accumulator also exercises genuine mixed-precision accumulation
    // instead of a same-type no-op.
    const std::size_t n = 20;
    mat::compressed2D<float> A(n, n);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0f;
            if (i > 0)     ins[i][i-1] << -1.5f;   // asymmetric off-diagonals
            if (i < n - 1) ins[i][i+1] << -0.5f;
        }
    }
    vec::dense_vector<float> b(n, 1.0f);
    vec::dense_vector<float> x(n, 0.0f);

    itl::pc::identity<mat::compressed2D<float>> pc(A);
    itl::basic_iteration<float> iter(b, 500, 1e-5f);

    int err = itl::qmr<mat::compressed2D<float>, vec::dense_vector<float>,
                        vec::dense_vector<float>, itl::pc::identity<mat::compressed2D<float>>,
                        itl::basic_iteration<float>, double>
                        (A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), WithinAbs(b(i), 1e-3));
}

TEST_CASE("QMR transposed sparse matvec: A^T*x direct check",
          "[itl][qmr][accumulator]") {
    // Direct, isolated coverage for the mult<Accumulator>(trans(A), x, y)
    // path qmr.hpp relies on: build a small nonsymmetric sparse matrix,
    // compute A^T*x via mult(), and check it against a hand-computed
    // reference -- independent of whether QMR itself converges.
    mat::compressed2D<double> A(2, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][1] << 3.0; ins[1][2] << 4.0;
    }
    // A = [1 2 0]
    //     [0 3 4]
    // A^T = [1 0]
    //       [2 3]
    //       [0 4]
    vec::dense_vector<double> x = {1.0, 1.0};
    vec::dense_vector<double> y(3, 0.0);
    mtl::mult(trans(A), x, y);

    REQUIRE_THAT(y(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(y(1), WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(y(2), WithinAbs(4.0, 1e-12));
}
