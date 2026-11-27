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
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <cmath>

using namespace mtl;

TEST_CASE("BiCGSTAB converges on non-symmetric 3x3 with identity PC", "[itl][bicgstab]") {
    // Non-symmetric matrix
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 3;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~= b
    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("BiCGSTAB converges with diagonal PC", "[itl][bicgstab]") {
    // Diagonally dominant non-symmetric system
    const std::size_t n = 8;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -0.5; // asymmetric off-diagonals
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~= b
    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("BiCGSTAB converges on SPD system too", "[itl][bicgstab]") {
    // BiCGSTAB should also work on SPD (though CG is preferred)
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    vec::dense_vector<double> b = {5.0, 5.0, 5.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);

    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

// Issue 958: an SPD preconditioner scaled by s leaves BiCGSTAB's iterates
// unchanged in exact arithmetic (alpha and omega scale as s), but t^T t scales
// as 1/s^2 and left double's range beyond 2^+-512: 0 at 2^600 (a false
// breakdown, code 4), inf at 2^-600 (omega = 0, code 5).
namespace {
struct given_diagonal_958 {
    vec::dense_vector<double> d;
    template <typename VecX, typename VecB>
    void solve(VecX& z, const VecB& v) const {
        for (std::size_t i = 0; i < d.size(); ++i) z(i) = v(i) / d(i);
    }
};
} // namespace

TEST_CASE("bicgstab: a preconditioner scaled beyond 2^+-512 (issue 958)", "[itl][bicgstab][958]") {
    const std::size_t n = 50;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i - 1] << -1.2;
            if (i + 1 < n) ins[i][i + 1] << -0.8;
        }
    }
    vec::dense_vector<double> b(n, 1.0);
    auto solve = [&](double s, vec::dense_vector<double>& x) {
        given_diagonal_958 M{vec::dense_vector<double>(n, 4.0 * s)};
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        const int code = itl::bicgstab(A, x, b, M, iter);
        return std::pair{code, iter.iterations()};
    };
    vec::dense_vector<double> x1(n, 0.0);
    const auto [c1, it1] = solve(1.0, x1);
    REQUIRE(c1 == 0);
    for (const double s : {0x1p300, 0x1p-300, 0x1p600, 0x1p-600, 0x1p900, 0x1p-900}) {
        CAPTURE(std::ilogb(s));
        vec::dense_vector<double> x(n, 0.0);
        const auto [c, it] = solve(s, x);
        REQUIRE(c == 0);
        CHECK(it == it1);
        // A power of two scales without rounding, and omega's rescaling undoes
        // it exactly: the iterates are those of s = 1, bit for bit.
        bool same = true;
        for (std::size_t i = 0; i < n; ++i) same = same && x(i) == x1(i);
        CHECK(same);
    }
}

TEST_CASE("bicgstab: a true omega = 0 is still a breakdown after the rescaling (issue 958)", "[itl][bicgstab][958]") {
    // [[1,1],[1,0]], b = e1 (test_krylov_edges.cpp, issue 866): s = (0,-1) and
    // t = A M^-1 s is orthogonal to s, so t^T s = 0 exactly at every scale of M.
    // At 2^+-600 t^T t leaves the range, so omega comes from the rescaled
    // branch, and must still be exactly 0: code 5, with x the half step, as at
    // s = 1.
    mat::dense2D<double> A = {{1.0, 1.0}, {1.0, 0.0}};
    vec::dense_vector<double> b(2, 0.0);
    b(0) = 1.0;
    vec::dense_vector<double> x1(2, 0.0);
    {
        given_diagonal_958 M{vec::dense_vector<double>(2, 1.0)};
        itl::basic_iteration<double> iter(b, 50, 1e-12);
        REQUIRE(itl::bicgstab(A, x1, b, M, iter) == 5);
    }
    for (const double s : {0x1p600, 0x1p-600}) {
        CAPTURE(std::ilogb(s));
        vec::dense_vector<double> x(2, 0.0);
        given_diagonal_958 M{vec::dense_vector<double>(2, s)};
        itl::basic_iteration<double> iter(b, 50, 1e-12);
        CHECK(itl::bicgstab(A, x, b, M, iter) == 5);
        CHECK(x(0) == x1(0));
        CHECK(x(1) == x1(1));
    }
}
