#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/minres.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

using namespace mtl;

static mat::compressed2D<double> make_tridiagonal(std::size_t n, double diag, double off) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << diag;
            if (i > 0)     ins[i][i-1] << off;
            if (i < n - 1) ins[i][i+1] << off;
        }
    }
    return A;
}

TEST_CASE("MINRES converges on SPD tridiagonal system", "[itl][minres]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 300, 1e-10);

    int err = itl::minres(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-6));
}

TEST_CASE("MINRES on symmetric indefinite system", "[itl][minres]") {
    // Symmetric indefinite: diag has both positive and negative entries
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0;
    A(0,0) =  4.0; A(0,1) = -1.0;
    A(1,0) = -1.0; A(1,1) = -3.0; A(1,2) = 1.0;
    A(2,1) =  1.0; A(2,2) =  5.0; A(2,3) = -1.0;
    A(3,2) = -1.0; A(3,3) = -2.0;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::minres(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-6));
}

TEST_CASE("MINRES with diagonal PC", "[itl][minres]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 300, 1e-10);

    int err = itl::minres(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-6));
}

// Issue 949: an M that is not positive definite stops MINRES with code 7.
namespace {
// diag(d) as a preconditioner: M.solve(z, v) is z = v ./ d.
struct given_diagonal_949 {
    vec::dense_vector<double> d;
    template <typename VecX, typename VecB>
    void solve(VecX& z, const VecB& v) const {
        for (std::size_t i = 0; i < d.size(); ++i) z(i) = v(i) / d(i);
    }
};

// tridiag(1, [4, -3, 4, -3, ...], 1), the issue's indefinite matrix.
mat::compressed2D<double> alternating_949(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << (i % 2 == 0 ? 4.0 : -3.0);
            if (i > 0)     ins[i][i - 1] << 1.0;
            if (i + 1 < n) ins[i][i + 1] << 1.0;
        }
    }
    return A;
}

double relres_949(const mat::compressed2D<double>& A, const vec::dense_vector<double>& x,
                  const vec::dense_vector<double>& b) {
    vec::dense_vector<double> r = A * x;
    for (std::size_t i = 0; i < r.size(); ++i) r(i) = b(i) - r(i);
    return mtl::two_norm(r) / mtl::two_norm(b);
}
} // namespace

TEST_CASE("minres: a preconditioner that is not positive definite returns 7 (issue 949)",
          "[itl][minres][949]") {
    const std::size_t n = 50;
    const auto A = alternating_949(n);
    vec::dense_vector<double> b(n, 1.0);
    using csr = mat::compressed2D<double>;
    STATIC_REQUIRE(itl::basic_iteration<double>::not_positive_definite == 7);

    SECTION("M = diag(A), indefinite: 7 at once, not 2 after 149 iterations") {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::diagonal<csr> M(A);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        REQUIRE(itl::minres(A, x, b, M, iter) == 7);
        CHECK(iter.iterations() == 0);
        for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == 0.0);
    }

    SECTION("M = |diag(A)|, SPD: unchanged, 0 in 18") {
        vec::dense_vector<double> x(n, 0.0), d(n);
        for (std::size_t i = 0; i < n; ++i) d(i) = (i % 2 == 0 ? 4.0 : 3.0);
        given_diagonal_949 M{d};
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        REQUIRE(itl::minres(A, x, b, M, iter) == 0);
        CHECK(iter.iterations() == 18);
        CHECK(relres_949(A, x, b) < 1e-10);
    }

    SECTION("found at a later Lanczos step, x as the last valid step left it") {
        // r0 = e0 and a tridiagonal A: the Lanczos vectors reach index k at
        // step k, and M is negative only at index 5.
        const auto T = make_tridiagonal(n, 4.0, 1.0);
        vec::dense_vector<double> e0(n, 0.0), d(n, 4.0);
        e0(0) = 1.0;
        d(5) = -4.0;
        given_diagonal_949 M{d};
        vec::dense_vector<double> x(n, 0.0);
        itl::basic_iteration<double> iter(e0, 500, 1e-12);
        REQUIRE(itl::minres(T, x, e0, M, iter) == 7);
        const int k = iter.iterations();
        REQUIRE(k >= 2);
        // The same solve stopped by the limit just before that step.
        vec::dense_vector<double> y(n, 0.0);
        itl::basic_iteration<double> limit(e0, k - 1, 1e-12);
        REQUIRE(itl::minres(T, y, e0, M, limit) == 1);
        for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == y(i));
        CHECK(mtl::two_norm(x) > 0.0);
    }

    SECTION("the plain and the scaled M-norm agree (issue 963)") {
        // v^T M^-1 v is formed plainly when it lies in [min / eps, max], and
        // scaled otherwise: at 2^+-300 it is about 2^-+300, plain; at 2^+-600
        // it would underflow or overflow, scaled. Beyond about 2^+-700 the
        // new Lanczos vector's solve itself left the range (code 2 above,
        // code 6 below), and is redone on a power-of-two scaled copy (issue
        // 971). All take the iterations of s = 1, to x within rounding.
        const auto T = make_tridiagonal(n, 4.0, -1.0);
        auto solve = [&](double s, vec::dense_vector<double>& x) {
            vec::dense_vector<double> d(n, 4.0 * s);
            given_diagonal_949 M{d};
            itl::basic_iteration<double> iter(b, 500, 1e-10);
            const int code = itl::minres(T, x, b, M, iter);
            return std::pair{code, iter.iterations()};
        };
        vec::dense_vector<double> x1(n, 0.0);
        const auto [c1, it1] = solve(1.0, x1);
        REQUIRE(c1 == 0);
        for (const double s : {0x1p300, 0x1p-300, 0x1p600, 0x1p-600, 0x1p700, 0x1p-700,
                               0x1p750, 0x1p-750, 0x1p900, 0x1p-900, 0x1p1000, 0x1p-1000}) {
            CAPTURE(std::ilogb(s));
            vec::dense_vector<double> x(n, 0.0);
            const auto [c, it] = solve(s, x);
            REQUIRE(c == 0);
            CHECK(it == it1);
            double err = 0.0;
            for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(x(i) - x1(i)));
            CHECK(err < 1e-12 * mtl::two_norm(x1));
        }
    }

    SECTION("a negative v^T M^-1 v whose norms overflow still returns 7 (issue 963)") {
        // #968 review. A = I, b = 2^-8 ones (n = 16), M = diag(-2^-1031):
        // z = M^-1 r0 = -2^1023 ones, r0^T z = -2^1019 is in range, but
        // ||z||_2 = 2^1025 overflows. Taken plainly, the code-7 tolerance
        // n eps ||r0|| ||z|| was infinite and MINRES went on with |r0^T z|.
        const std::size_t m = 16;
        mat::dense2D<double> I(m, m);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) I(i, j) = (i == j ? 1.0 : 0.0);
        vec::dense_vector<double> b16(m, 0x1p-8), x(m, 0.0), d(m, -0x1p-1031);
        given_diagonal_949 M{d};
        itl::basic_iteration<double> iter(b16, 50, 1e-10);
        CHECK(itl::minres(I, x, b16, M, iter) == 7);
        CHECK(iter.iterations() == 0);
        for (std::size_t i = 0; i < m; ++i) CHECK(x(i) == 0.0);
    }

    SECTION("code 7 at a later Lanczos step, with M at 2^+-900 (issue 971)") {
        // As "found at a later Lanczos step", scaled: the step that reaches
        // index 5 runs on the rescaled solve, whose M-norm must still see the
        // negative entry. The out-of-range first M-norm must not mask it, nor
        // invent one.
        const auto T = make_tridiagonal(n, 4.0, 1.0);
        vec::dense_vector<double> e0(n, 0.0);
        e0(0) = 1.0;
        int k1 = 0;
        {
            vec::dense_vector<double> d(n, 4.0), x(n, 0.0);
            d(5) = -4.0;
            given_diagonal_949 M{d};
            itl::basic_iteration<double> iter(e0, 500, 1e-12);
            REQUIRE(itl::minres(T, x, e0, M, iter) == 7);
            k1 = iter.iterations();
        }
        for (const double s : {0x1p900, 0x1p-900}) {
            CAPTURE(std::ilogb(s));
            vec::dense_vector<double> d(n, 4.0 * s), x(n, 0.0);
            d(5) = -4.0 * s;
            given_diagonal_949 M{d};
            itl::basic_iteration<double> iter(e0, 500, 1e-12);
            CHECK(itl::minres(T, x, e0, M, iter) == 7);
            CHECK(iter.iterations() == k1);
            // An SPD M at the same scale converges: no false code 7.
            vec::dense_vector<double> dp(n, 4.0 * s), y(n, 0.0);
            given_diagonal_949 P{dp};
            itl::basic_iteration<double> it2(e0, 500, 1e-12);
            CHECK(itl::minres(T, y, e0, P, it2) == 0);
        }
    }

    SECTION("an SPD M at 2^+-600 is unaffected") {
        const auto T = make_tridiagonal(n, 4.0, -1.0);
        for (const double s : {0x1p600, 0x1p-600}) {
            vec::dense_vector<double> x(n, 0.0), d(n, 4.0 * s);
            given_diagonal_949 M{d};
            itl::basic_iteration<double> iter(b, 500, 1e-10);
            REQUIRE(itl::minres(T, x, b, M, iter) == 0);
            CHECK(relres_949(T, x, b) < 1e-10);
        }
    }
}
