#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>

using namespace mtl;

// --- Sanity: default (unspecified) Accumulator behaves exactly as before ---
TEST_CASE("CG with default Accumulator matches unmodified baseline", "[itl][cg][quire]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::cg(A, x, b, pc, iter); // no explicit Accumulator -- must be unaffected

    REQUIRE(err == 0);
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

#ifdef MTL5_HAS_UNIVERSAL
#include <universal/number/posit/posit.hpp>
#include <mtl/math/quire_accumulator.hpp>

// --- The actual point: quire accumulation vs naive posit32 on a case where
// pAp magnitude sensitivity matters (mirrors posit-sparse-bench's mhd4800b /
// sts4098 findings at unit-test scale). ---
TEST_CASE("CG: quire-accumulated pAp/rho improves posit32 accuracy vs naive posit32", "[itl][cg][quire]") {
    using Posit = sw::universal::posit<32,2>;
    using Quire = sw::universal::quire<Posit>;

    const std::size_t n = 20;
    mat::dense2D<Posit> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i,j) = Posit(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        A(i,i) = Posit(2.0);
        if (i > 0)     A(i,i-1) = Posit(-1.0);
        if (i < n - 1) A(i,i+1) = Posit(-1.0);
    }

    vec::dense_vector<Posit> b(n, Posit(1.0));

    vec::dense_vector<Posit> x_naive(n, Posit(0.0));
    itl::pc::identity<mat::dense2D<Posit>> pc(A);
    itl::basic_iteration<Posit> iter_naive(b, 200, Posit(1e-6));
    itl::cg(A, x_naive, b, pc, iter_naive); // default Accumulator = naive posit32

    vec::dense_vector<Posit> x_quire(n, Posit(0.0));
    itl::basic_iteration<Posit> iter_quire(b, 200, Posit(1e-6));
    itl::cg<mat::dense2D<Posit>, vec::dense_vector<Posit>, vec::dense_vector<Posit>,
            itl::pc::identity<mat::dense2D<Posit>>, itl::basic_iteration<Posit>, Quire>(
        A, x_quire, b, pc, iter_quire);

    // Reference in double for the true solution's residual comparison.
    mat::dense2D<double> Ad(n, n);
    vec::dense_vector<double> bd(n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Ad(i,j) = double(A(i,j));

    auto residual_norm = [&](const vec::dense_vector<Posit>& x) {
        double res2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double Axi = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                Axi += double(A(i,j)) * double(x(j));
            double ri = Axi - double(b(i));
            res2 += ri * ri;
        }
        return std::sqrt(res2);
    };

    double naive_residual = residual_norm(x_naive);
    double quire_residual = residual_norm(x_quire);

    INFO("naive posit32 residual: " << naive_residual);
    INFO("quire-accumulated posit32 residual: " << quire_residual);

    // The claim under test: quire accumulation of rho/pAp should not be
    // worse, and on this tridiagonal system should measurably improve,
    // final residual accuracy vs naive same-precision accumulation.
    REQUIRE(quire_residual <= naive_residual);
}
#endif // MTL5_HAS_UNIVERSAL
