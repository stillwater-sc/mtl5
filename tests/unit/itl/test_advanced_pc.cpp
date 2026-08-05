#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/cg.hpp>

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

// --- ILUT tests ---

TEST_CASE("ILUT preconditioned BiCGSTAB converges", "[itl][pc][ilut]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ilut<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("ILUT converges in fewer iterations than identity PC", "[itl][pc][ilut]") {
    const std::size_t n = 30;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // Identity PC
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> id_pc(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab(A, x1, b, id_pc, iter1);
    int iters_id = iter1.iterations();

    // ILUT PC
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::ilut<double> ilut_pc(A);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab(A, x2, b, ilut_pc, iter2);
    int iters_ilut = iter2.iterations();

    REQUIRE(iters_ilut <= iters_id);
}

TEST_CASE("ILUT with fill-in on arrowhead matrix", "[itl][pc][ilut]") {
    // Arrowhead matrix: row 0 couples to all columns, forcing fill-in at
    // low column indices when processing later rows.  This is the exact
    // pattern that triggers out-of-order column processing if fill-in
    // entries are simply appended rather than visited in ascending order.
    const std::size_t n = 30;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 10.0;
            if (i > 0) {
                ins[0][i] << 1.0;
                ins[i][0] << 1.0;
            }
        }
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    // Generous fill allowance to ensure fill-in actually happens
    itl::pc::ilut<double> pc(A, /*fill=*/20, /*threshold=*/1e-6);
    itl::basic_iteration<double> iter(b, 300, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    // Verify the actual residual is small (not just the preconditioned one)
    auto Ax = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = Ax(i) - b(i);
        res += diff * diff;
    }
    res = std::sqrt(res);
    REQUIRE(res < 1e-8);
}

// --- ILDL tests ---

TEST_CASE("ILDL preconditioned CG on SPD system converges", "[itl][pc][ildl]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ildl<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// --- Block diagonal tests ---

TEST_CASE("Block diagonal preconditioned BiCGSTAB converges", "[itl][pc][block_diagonal]") {
    const std::size_t n = 20;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -1.0;
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::block_diagonal<mat::dense2D<double>> pc(A, 5);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// --- SSOR tests ---

namespace {
int g_ssor_addproduct_calls = 0;
struct counting_wide_acc { double v = 0.0; };
}  // namespace

namespace mtl::math {
template <>
struct accumulator_traits<counting_wide_acc, float> {
    static void clear(counting_wide_acc& a) { a.v = 0.0; }
    static void assign(counting_wide_acc& a, const float& x) { a.v = static_cast<double>(x); }
    template <typename Result = float>
    static Result value(const counting_wide_acc& a) { return static_cast<Result>(a.v); }
    static void add_product(counting_wide_acc& a, const float& m, const float& x) {
        ++g_ssor_addproduct_calls;
        a.v += static_cast<double>(m) * static_cast<double>(x);
    }
};
}  // namespace mtl::math

TEST_CASE("SSOR preconditioner routes through accumulator_traits (#405)",
          "[itl][pc][ssor][accumulator]") {
    // The capability the #405 refactor buys. Before it, pc::ssor reimplemented
    // the sweeps and had no Accumulator parameter at all: you could SMOOTH in
    // mixed precision but not PRECONDITION in it, and nothing in the API said
    // so. Now the parameter is forwarded to smoother::sor, which already
    // routes the off-diagonal row sum through accumulator_traits.
    const std::size_t n = 16;
    mat::compressed2D<float> A(n, n);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0f;
            if (i)         ins[i][i - 1] << -1.0f;
            if (i + 1 < n) ins[i][i + 1] << -1.0f;
        }
    }
    vec::dense_vector<float> b(n, 1.0f), x(n, 0.0f);

    itl::pc::ssor<mat::compressed2D<float>, counting_wide_acc> pc(A, 1.0f);
    g_ssor_addproduct_calls = 0;
    pc.solve(x, b);

    // The accumulator was actually used, and BOTH sweeps ran. An exact count,
    // not just > 0: solve() now DELEGATES rather than running two visible
    // loops, so losing one of the two sweeps is exactly the regression this
    // refactor makes possible, and > 0 would sail straight past it.
    //
    // add_product fires once per STORED off-diagonal entry (the diagonal is
    // skipped by index, not by value). This tridiagonal A holds 1 off-diagonal
    // in each of rows 0 and n-1 and 2 in each of the n-2 interior rows, so
    // 2*(n-1) per sweep and 4*(n-1) for the forward+backward pair.
    INFO("add_product calls = " << g_ssor_addproduct_calls);
    REQUIRE(g_ssor_addproduct_calls == static_cast<int>(4 * (n - 1)));

    // And it still computes M^-1 b: applying M to the result returns b.
    // M = (D + L) D^-1 (D + U) at omega = 1, so M x is three triangular
    // products; forming it densely is clearer than reasoning about the sweeps.
    vec::dense_vector<float> t(n, 0.0f), Mx(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {            // t = D^-1 (D + U) x
        float s = 0.0f;
        for (std::size_t j = i; j < n; ++j) s += A(i, j) * x(j);
        t(i) = s / A(i, i);
    }
    for (std::size_t i = 0; i < n; ++i) {            // Mx = (D + L) t
        float s = 0.0f;
        for (std::size_t j = 0; j <= i; ++j) s += A(i, j) * t(j);
        Mx(i) = s;
    }
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Mx(i), Catch::Matchers::WithinAbs(b(i), 1e-5f));

    // The default (void) accumulator must not touch it -- same object, no calls.
    itl::pc::ssor<mat::compressed2D<float>> pc_default(A, 1.0f);
    vec::dense_vector<float> x2(n, 0.0f);
    g_ssor_addproduct_calls = 0;
    pc_default.solve(x2, b);
    REQUIRE(g_ssor_addproduct_calls == 0);
}

TEST_CASE("SSOR preconditioner on a 1x1 matrix (#405)",
          "[itl][pc][ssor][accumulator][edge]") {
    // The one case where the accumulator branch yields a result without ever
    // accumulating: with no off-diagonal entries the row sum is empty, so
    // add_product is never called and sigma comes from AT::value on a freshly
    // cleared accumulator. If clear() and value() disagreed about what an empty
    // sum means, every other size would hide it behind real terms.
    mat::compressed2D<float> A(1, 1);
    { mat::inserter<mat::compressed2D<float>> ins(A); ins[0][0] << 4.0f; }
    vec::dense_vector<float> b(1, 1.0f), x(1, 0.0f);

    itl::pc::ssor<mat::compressed2D<float>, counting_wide_acc> pc(A, 1.0f);
    g_ssor_addproduct_calls = 0;
    pc.solve(x, b);

    REQUIRE(g_ssor_addproduct_calls == 0);       // nothing to accumulate
    // M = (D + L) D^-1 (D + U) collapses to D = 4, so M^-1 b = 1/4 exactly.
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(0.25f, 1e-6f));

    // Same answer without the accumulator, so the empty-sum path agrees with
    // the naive one rather than merely being self-consistent.
    itl::pc::ssor<mat::compressed2D<float>> pc_default(A, 1.0f);
    vec::dense_vector<float> x2(1, 0.0f);
    pc_default.solve(x2, b);
    REQUIRE_THAT(x2(0), Catch::Matchers::WithinAbs(0.25f, 1e-6f));
}

TEST_CASE("SSOR applies exactly the classical SSOR operator (#398)",
          "[itl][pc][ssor][regression]") {
    // The sharpest statement of the fix, and the reason it is a fix rather than
    // a change of preconditioner: run from x = 0, the forward + backward sweeps
    // ARE the classical operator
    //
    //     M = w/(2-w) (D/w + L) D^-1 (D/w + U)
    //
    // to the last bit, scalar factor included. So this compares pc::ssor's
    // output against a dense M solved by LU. Before #398 the sweeps started
    // from x = b, adding a G_B G_F b term, and this test fails outright
    // (measured relative error 3.7e-01 at w = 1.0).
    //
    // Two omegas: w = 1 collapses the scalar factor to 1 and would not catch a
    // wrong one, so w = 1.3 carries that part of the claim.
    for (double omega : {1.0, 1.3}) {
        const std::size_t n = 24;
        const auto A = make_tridiagonal(n, 4.0, -1.0);

        // Dense M = w/(2-w) (D/w + L) D^-1 (D/w + U), from A's own entries.
        mat::dense2D<double> F(n, n), B(n, n), M(n, n);
        vec::dense_vector<double> d(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) { F(i, j) = 0.0; B(i, j) = 0.0; }
            d(i) = A(i, i);
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const double a = A(i, j);
                if (i == j)     { F(i, j) = a / omega; B(i, j) = a / omega; }
                else if (j < i) { F(i, j) = a; }        // L
                else            { B(i, j) = a; }        // U
            }
        const double c = omega / (2.0 - omega);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < n; ++k) s += F(i, k) * B(k, j) / d(k);
                M(i, j) = c * s;
            }

        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b(i) = 1.0 + 0.5 * std::sin(static_cast<double>(3 * i));

        // Reference: solve M y = b densely.
        vec::dense_vector<double> y(n, 0.0);
        std::vector<mat::dense2D<double>::size_type> pivot;
        mat::dense2D<double> Mf(M);
        REQUIRE(lu_factor(Mf, pivot) == 0);
        lu_solve(Mf, pivot, y, b);

        // The preconditioner must produce the same vector.
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::ssor<mat::compressed2D<double>> pc(A, omega);
        pc.solve(x, b);

        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            num += (x(i) - y(i)) * (x(i) - y(i));
            den += y(i) * y(i);
        }
        const double rel = std::sqrt(num) / std::sqrt(den);
        INFO("omega = " << omega << ", ||x - M^-1 b|| / ||M^-1 b|| = " << rel);
        REQUIRE(rel < 1e-12);
    }
}

TEST_CASE("SSOR preconditioned BiCGSTAB converges", "[itl][pc][ssor]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ssor<mat::compressed2D<double>> pc(A, 1.0);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("SSOR converges in fewer iterations than identity PC", "[itl][pc][ssor]") {
    const std::size_t n = 30;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // Identity PC
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> id_pc(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab(A, x1, b, id_pc, iter1);
    int iters_id = iter1.iterations();

    // SSOR PC
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::ssor<mat::compressed2D<double>> ssor_pc(A, 1.2);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab(A, x2, b, ssor_pc, iter2);
    int iters_ssor = iter2.iterations();

    REQUIRE(iters_ssor <= iters_id);
}
