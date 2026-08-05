// Self-containment of mtl/itl/mg/multigrid.hpp (#401).
//
// vcycle and wcycle form the residual with `levels_[l] * x`, whose operator*
// lives in mtl/mat/operators.hpp. That header was not included, so multigrid.hpp
// compiled only when the CALLER had already pulled it in -- order-dependent, and
// a translation unit that included multigrid.hpp alone failed.
//
// Two things make this easy to miss, and both are the reason this file exists
// separately rather than as another case in test_multigrid.cpp:
//
//   1. Merely INCLUDING the header succeeds. `A*x` is a dependent expression
//      inside a class template, so the lookup is deferred to instantiation. A
//      naive "does it compile on its own" check passes. The cycles have to
//      actually be called.
//
//   2. test_multigrid.cpp includes mtl/operation/operators.hpp (which pulls in
//      mat/operators.hpp) at line 7, BEFORE multigrid.hpp at line 16. It
//      satisfies the missing dependency itself, so it could never have caught
//      this no matter how much multigrid behaviour it exercised.
//
// THE INCLUDE LIST BELOW IS LOAD-BEARING. Adding mtl/operation/operators.hpp,
// mtl/mat/operators.hpp, or anything that pulls them in silently disables this
// test -- it will still pass, but it will have stopped checking anything.
// mat/inserter.hpp is verified not to pull them.
#include <catch2/catch_test_macros.hpp>

#include <mtl/itl/mg/multigrid.hpp>
#include <mtl/mat/inserter.hpp>

namespace {

using SMat = mtl::mat::compressed2D<double>;
using Vec  = mtl::vec::dense_vector<double>;

SMat tridiag(std::size_t n) {
    SMat A(n, n);
    mtl::mat::inserter<SMat> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 2.0;
        if (i)         ins[i][i - 1] << -1.0;
        if (i + 1 < n) ins[i][i + 1] << -1.0;
    }
    return A;
}

}  // namespace

TEST_CASE("mg/multigrid.hpp is self-contained (#401)", "[itl][mg][regression]") {
    // The assertions are almost beside the point: this test's value is that the
    // translation unit COMPILES while instantiating both cycles.
    const std::size_t nf = 7, nc = 3;
    SMat Af = tridiag(nf), Ac = tridiag(nc);

    SMat R(nc, nf), P(nf, nc);
    { mtl::mat::inserter<SMat> ins(R); for (std::size_t i = 0; i < nc; ++i) ins[i][2 * i + 1] << 1.0; }
    { mtl::mat::inserter<SMat> ins(P); for (std::size_t i = 0; i < nc; ++i) ins[2 * i + 1][i] << 1.0; }

    auto smoother_factory = [](const SMat&) {
        return [](Vec& x, const Vec& b) { for (std::size_t i = 0; i < x.size(); ++i) x(i) = 0.5 * b(i); };
    };
    auto coarse_solver = [](Vec& x, const Vec& b) {
        for (std::size_t i = 0; i < x.size(); ++i) x(i) = 0.5 * b(i);
    };

    mtl::itl::mg::multigrid<double> mg({Af, Ac}, {R}, {P}, smoother_factory, coarse_solver, 1, 1);

    Vec b(nf, 1.0);

    SECTION("vcycle instantiates and runs") {
        Vec x(nf, 0.0);
        REQUIRE_NOTHROW(mg.vcycle(x, b));
        REQUIRE(x.size() == nf);
    }

    SECTION("wcycle instantiates and runs") {
        Vec x(nf, 0.0);
        REQUIRE_NOTHROW(mg.wcycle(x, b));
        REQUIRE(x.size() == nf);
    }
}
