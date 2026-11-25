// Preconditioners, smoothers, eigensolvers and multigrid: the edge cases of
// issue 867 (QA sweep #718).
//
//   - ssor::solve(b, b) gives M^-1 b;
//   - a zero or missing diagonal throws std::runtime_error at construction in
//     ilu_0, ilut, pc::diagonal and the Jacobi, Gauss-Seidel and SOR smoothers
//     (the decision recorded on the issue);
//   - lanczos and arnoldi stop at an invariant subspace without reading past
//     it, judge breakdown relative to A's scale, and order non-trivial spectra;
//   - power_iteration on a null operator converges to 0; a negative dominant
//     eigenvalue;
//   - multigrid is move-only, solves through a const object, and the 1D
//     transfer builders reject sizes they cannot pair;
//   - the smoothers on coordinate2D and ell_matrix match compressed2D and
//     sweep in O(nnz).
//
// And those of issue 925 (QA sweep #913):
//
//   - ic_0 and ildl reject a zero, missing or (ic_0, real) negative pivot;
//   - block_diagonal rejects a singular block, bs = 0 and a non-square A, and
//     solves with a partial last block, bs > n and a compressed2D;
//   - ilu_0, ilut, ssor, ic_0, ildl and multigrid reject bad shapes in every
//     build; ilut keeps no U fill for orig_u = 0 and fill = 0;
//   - power_iteration with n = 0; arnoldi's smallest selectors, subspace < k,
//     a zero v0 and k > n;
//   - the W-cycle reduces the residual, with and without smoothing steps.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <string>
#include <utility>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/ell_matrix.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/itl.hpp>
#include <mtl/operation/spgemm.hpp>

using namespace mtl;

namespace {

using csr     = mat::compressed2D<double>;
using vec_t   = vec::dense_vector<double>;
using entries = std::vector<std::tuple<std::size_t, std::size_t, double>>;

csr make(std::size_t n, const entries& e) {
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (const auto& [i, j, v] : e) ins[i][j] << v;
    }
    return A;
}

entries tridiagonal_entries(std::size_t n, double d = 4.0, double off = -1.0, double scale = 1.0) {
    entries e;
    for (std::size_t i = 0; i < n; ++i) {
        e.emplace_back(i, i, d * scale);
        if (i + 1 < n) { e.emplace_back(i, i + 1, off * scale); e.emplace_back(i + 1, i, off * scale); }
    }
    return e;
}

vec_t ramp(std::size_t n) {
    vec_t b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + double(i % 3);
    return b;
}

} // namespace

// --- ssor ----------------------------------------------------------------------------

TEST_CASE("ssor: solve(b, b) is M^-1 b", "[itl][pc][ssor][867]") {
    const csr A = make(6, tridiagonal_entries(6));
    const itl::pc::ssor<csr> M(A, 1.2);
    const vec_t b = ramp(6);
    vec_t x(6);
    M.solve(x, b);
    vec_t y = b;
    // Zeroing x zeroed b too: y(0) came back 0.
    M.solve(y, y);
    for (std::size_t i = 0; i < 6; ++i) CHECK(y(i) == x(i));
}

// --- zero or missing diagonal ----------------------------------------------------------

TEST_CASE("a zero or missing diagonal throws at construction", "[itl][pc][smoother][867]") {
    // No (1,1) entry: ilu_0's solve gave (-inf, inf, 0.33), Jacobi x = (0.5, 0).
    const csr missing = make(3, {{0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 4.0}});
    // A stored zero at (1,1).
    const csr stored_zero = make(3, {{0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 0.0}, {1, 2, 1.0},
                                     {2, 1, 1.0}, {2, 2, 4.0}});
    // ILU(0) keeps no fill, so a missing U(1,1) stays missing. A stored zero
    // is updated by the elimination (0 - 1/4 * 1), a valid pivot.
    CHECK_THROWS_AS(itl::pc::ilu_0<double>(missing), std::runtime_error);
    CHECK_NOTHROW(itl::pc::ilu_0<double>(stored_zero));
    // The others divide by A's own diagonal: both throw.
    for (const csr* A : {&missing, &stored_zero}) {
        CHECK_THROWS_AS(itl::pc::diagonal<csr>(*A), std::runtime_error);
        CHECK_THROWS_AS(itl::smoother::jacobi<csr>(*A), std::runtime_error);
        CHECK_THROWS_AS(itl::smoother::gauss_seidel<csr>(*A), std::runtime_error);
        CHECK_THROWS_AS(itl::smoother::sor<csr>(*A, 1.2), std::runtime_error);
        CHECK_THROWS_AS(itl::pc::ssor<csr>(*A), std::runtime_error);
    }
    // The message names the row.
    try {
        itl::pc::ilu_0<double> M(missing);
        FAIL("ilu_0 accepted a missing diagonal");
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("row 1") != std::string::npos);
    }
    // ILUT: no (1,1) and no fill reaching it (row 1 eliminates column 0,
    // whose U row has no (0,1) entry).
    const csr no_fill = make(3, {{0, 0, 4.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 4.0}});
    CHECK_THROWS_AS(itl::pc::ilut<double>(no_fill, 5, 0.0), std::runtime_error);

    // Dense: the generic smoothers.
    mat::dense2D<double> D(2, 2);
    D(0, 0) = 2.0; D(0, 1) = 1.0; D(1, 0) = 1.0; D(1, 1) = 0.0;
    CHECK_THROWS_AS(itl::smoother::jacobi<mat::dense2D<double>>(D), std::runtime_error);
    CHECK_THROWS_AS(itl::smoother::gauss_seidel<mat::dense2D<double>>(D), std::runtime_error);
    CHECK_THROWS_AS(itl::smoother::sor<mat::dense2D<double>>(D), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::diagonal<mat::dense2D<double>>(D), std::runtime_error);
}

TEST_CASE("ILU: a pivot that eliminates to zero, and one fill makes", "[itl][pc][867]") {
    // [[1,1],[1,1]]: U(1,1) = 1 - 1*1 = 0 in both ILU(0) and ILUT.
    const csr singular = make(2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    CHECK_THROWS_AS(itl::pc::ilu_0<double>(singular), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::ilut<double>(singular, 5, 0.0), std::runtime_error);
    // [[1,1],[1,.]]: no (1,1) in A, but ILUT's fill forms U(1,1) = -1, and its
    // solve is exact. ILU(0) keeps no fill, so its U(1,1) is missing.
    const csr filled = make(2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}});
    CHECK_THROWS_AS(itl::pc::ilu_0<double>(filled), std::runtime_error);
    const itl::pc::ilut<double> M(filled, 5, 0.0);
    vec_t x(2), b(2);
    b(0) = 3.0; b(1) = 1.0;   // A x = b for x = (1, 2)
    M.solve(x, b);
    CHECK(x(0) == 1.0);
    CHECK(x(1) == 2.0);
}

// --- Krylov eigensolvers ---------------------------------------------------------------

TEST_CASE("lanczos and arnoldi stop at an invariant subspace", "[itl][eigen][867]") {
    // The identity with v0 = ones: built = 1. k = 3 read S(m-1, c) past the
    // one Ritz pair (dense_vector: index out of range).
    const csr I = make(10, [] { entries e; for (std::size_t i = 0; i < 10; ++i) e.emplace_back(i, i, 1.0); return e; }());
    const vec_t v0(10, 1.0);
    const auto rl = itl::lanczos(I, v0, 3);
    REQUIRE(rl.values.size() == 1);
    CHECK(std::abs(rl.values(0) - 1.0) < 1e-14);
    CHECK(rl.subspace == 1);
    const auto ra = itl::arnoldi(I, v0, 3);
    REQUIRE(ra.values.size() == 1);
    CHECK(std::abs(ra.values(0).real() - 1.0) < 1e-14);
}

TEST_CASE("lanczos and arnoldi: breakdown is relative to A's scale", "[itl][eigen][867]") {
    // A = 1e-10 * tridiag(-1, 4, -1), n = 40: an absolute sqrt(eps) stopped
    // the subspace at 1, largest eigenvalue 2.06e-10 for 5.99e-10. (The start
    // is ramp, not ones: the top eigenvector alternates in sign and is
    // antisymmetric about the middle, orthogonal to ones, whose largest
    // reachable eigenvalue is 4 + 2 cos(2 pi / 41).)
    const std::size_t n = 40;
    const csr A = make(n, tridiagonal_entries(n, 4.0, -1.0, 1e-10));
    const double exact = 1e-10 * (4.0 + 2.0 * std::cos(std::numbers::pi / double(n + 1)));
    const auto rl = itl::lanczos(A, ramp(n), 1, itl::eigen_which::largest_algebraic, n);
    CHECK(rl.subspace > 1);
    CHECK(std::abs(rl.values(0) / exact - 1.0) < 1e-8);
    vec_t v0 = ramp(n);
    const auto ra = itl::arnoldi(A, v0, 1, itl::eigen_which::largest_magnitude, n);
    CHECK(ra.subspace > 1);
    CHECK(std::abs(ra.values(0).real() / exact - 1.0) < 1e-8);
}

TEST_CASE("lanczos: smallest and magnitude orderings on a non-trivial spectrum", "[itl][eigen][867]") {
    // diag(-5, 1, 2, 3, 4): largest magnitude -5, smallest magnitude 1,
    // smallest algebraic -5, largest algebraic 4.
    const csr A = make(5, {{0, 0, -5.0}, {1, 1, 1.0}, {2, 2, 2.0}, {3, 3, 3.0}, {4, 4, 4.0}});
    const vec_t v0 = ramp(5);
    auto top = [&](itl::eigen_which w) { return itl::lanczos(A, v0, 1, w, 5).values(0); };
    CHECK(std::abs(top(itl::eigen_which::largest_magnitude) + 5.0) < 1e-12);
    CHECK(std::abs(top(itl::eigen_which::smallest_magnitude) - 1.0) < 1e-12);
    CHECK(std::abs(top(itl::eigen_which::smallest_algebraic) + 5.0) < 1e-12);
    CHECK(std::abs(top(itl::eigen_which::largest_algebraic) - 4.0) < 1e-12);
}

TEST_CASE("power_iteration: a null operator and a negative dominant eigenvalue", "[itl][eigen][867]") {
    // A zero 4x4 reported converged = false with residual exactly 0.
    const csr Z(4, 4);
    const auto r0 = itl::power_iteration(Z, vec_t(4, 1.0));
    CHECK(r0.converged);
    CHECK(r0.value == 0.0);
    const csr A = make(3, {{0, 0, -3.0}, {1, 1, 1.0}, {2, 2, 2.0}});
    const auto r = itl::power_iteration(A, vec_t(3, 1.0), 2000, 1e-12);
    CHECK(r.converged);
    CHECK(std::abs(r.value + 3.0) < 1e-9);
}

// --- multigrid ---------------------------------------------------------------------------

namespace {

itl::mg::multigrid<double> make_mg(std::size_t n_fine) {
    // Two levels of the 1D Poisson problem, Galerkin coarse operator R A P.
    const csr Af = make(n_fine, tridiagonal_entries(n_fine, 2.0, -1.0));
    const csr R = itl::mg::make_restriction_1d(n_fine);
    const csr P = itl::mg::make_prolongation_1d((n_fine - 1) / 2);
    const csr Ac = mtl::spgemm(R, mtl::spgemm(Af, P));
    auto sf = [](const csr& L) { return itl::smoother::gauss_seidel<csr>(L); };
    auto cs = [Ac](vec_t& x, const vec_t& b) {
        itl::pc::identity<csr> I(Ac);
        itl::basic_iteration<double> it(b, 1000, 1e-14);
        itl::cg(Ac, x, b, I, it);
    };
    return itl::mg::multigrid<double>({Af, Ac}, {R}, {P}, sf, cs, 2, 2);
}

} // namespace

TEST_CASE("multigrid: move-only, and solve through a const object", "[itl][mg][867]") {
    // A copy's smoothers referred to the source's levels and dangled.
    static_assert(!std::is_copy_constructible_v<itl::mg::multigrid<double>>);
    static_assert(!std::is_copy_assignable_v<itl::mg::multigrid<double>>);
    static_assert(std::is_move_constructible_v<itl::mg::multigrid<double>>);

    const std::size_t n = 31;
    itl::mg::multigrid<double> moved = make_mg(n);   // moved out of make_mg
    const itl::mg::multigrid<double> mg = std::move(moved);
    const csr A = make(n, tridiagonal_entries(n, 2.0, -1.0));
    const vec_t b = ramp(n);
    // As a preconditioner: solve() const, no const_cast.
    vec_t x(n, 0.0);
    itl::basic_iteration<double> iter(b, 100, 1e-10);
    CHECK(itl::cg(A, x, b, mg, iter) == 0);
    CHECK(iter.iterations() < 20);
}

TEST_CASE("multigrid: the 1D transfer builders reject sizes they cannot pair", "[itl][mg][867]") {
    // n_fine = 0 underflowed (n_fine - 1) / 2; an even n_fine did not match
    // any prolongation.
    for (std::size_t nf : {0u, 1u, 2u, 4u, 16u})
        CHECK_THROWS_AS(itl::mg::make_restriction_1d(nf), std::invalid_argument);
    CHECK_THROWS_AS(itl::mg::make_prolongation_1d(0), std::invalid_argument);
    for (std::size_t nc : {1u, 3u, 7u}) {
        const auto R = itl::mg::make_restriction_1d(2 * nc + 1);
        const auto P = itl::mg::make_prolongation_1d(nc);
        CHECK(R.num_rows() == P.num_cols());
        CHECK(R.num_cols() == P.num_rows());
    }
}

// --- smoothers on coordinate2D and ell_matrix ------------------------------------------------

namespace {

template <template <typename, typename> class Smoother, typename... Args>
void check_formats(std::size_t n, Args... args) {
    const entries e = tridiagonal_entries(n, 4.0, -1.0);
    const csr A = make(n, e);
    mat::coordinate2D<double> C(n, n);
    for (const auto& [i, j, v] : e) C.insert(i, j, v);
    const mat::ell_matrix<double> E(A);
    const vec_t b = ramp(n);
    vec_t xa(n, 0.0), xc(n, 0.0), xe(n, 0.0);
    Smoother<csr, void> sa(A, args...);
    Smoother<mat::coordinate2D<double>, void> sc(C, args...);
    Smoother<mat::ell_matrix<double>, void> se(E, args...);
    auto sc2 = sc;   // a copy shares the converted matrix
    for (int s = 0; s < 3; ++s) { sa(xa, b); sc2(xc, b); se(xe, b); }
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(xc(i) == xa(i));
        CHECK(xe(i) == xa(i));
    }
}

} // namespace

TEST_CASE("smoothers on coordinate2D and ell_matrix match compressed2D", "[itl][smoother][867]") {
    check_formats<itl::smoother::jacobi>(30);
    check_formats<itl::smoother::gauss_seidel>(30);
    check_formats<itl::smoother::sor>(30, 1.3);
}

TEST_CASE("smoothers on a column-major coordinate2D", "[itl][smoother][867]") {
    // compressed2D rejects column-major parameters, so the copy is a
    // row-major compressed2D<Value> whatever the coordinate2D's (#870 review).
    using cm_coo = mat::coordinate2D<double, mat::parameters<tag::col_major>>;
    const std::size_t n = 12;
    const entries e = tridiagonal_entries(n, 4.0, -1.0);
    const csr A = make(n, e);
    cm_coo C(n, n);
    for (const auto& [i, j, v] : e) C.insert(i, j, v);
    const vec_t b = ramp(n);
    vec_t xa(n, 0.0), xj(n, 0.0), xg(n, 0.0), xs(n, 0.0);
    vec_t ya(n, 0.0), za(n, 0.0);
    itl::smoother::jacobi<cm_coo> sj(C);
    itl::smoother::gauss_seidel<cm_coo> sg(C);
    itl::smoother::sor<cm_coo> ss(C, 1.3);
    itl::smoother::jacobi<csr> rj(A);
    itl::smoother::gauss_seidel<csr> rg(A);
    itl::smoother::sor<csr> rs(A, 1.3);
    for (int s = 0; s < 2; ++s) { sj(xj, b); sg(xg, b); ss(xs, b); rj(xa, b); rg(ya, b); rs(za, b); }
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(xj(i) == xa(i));
        CHECK(xg(i) == ya(i));
        CHECK(xs(i) == za(i));
    }
}

TEST_CASE("smoothers on coordinate2D sweep in O(nnz)", "[itl][smoother][cost][867]") {
    // The generic sweep looked up every A(i, j), and coordinate2D's lookup
    // scans every entry: O(n^2 nnz) per sweep. 4x in n is ~4x in time now;
    // the generic sweep took ~64x. The bound is 16x.
    auto t = [](std::size_t n) {
        const entries e = tridiagonal_entries(n);
        mat::coordinate2D<double> C(n, n);
        for (const auto& [i, j, v] : e) C.insert(i, j, v);
        const itl::smoother::gauss_seidel<mat::coordinate2D<double>> gs(C);
        const vec_t b = ramp(n);
        vec_t x(n, 0.0);
        double best = std::numeric_limits<double>::infinity();
        for (int r = 0; r < 3; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < 10; ++s) gs(x, b);
            const auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        return best;
    };
    const double small = t(2000), large = t(8000);
    INFO("t(2000) = " << small << " s, t(8000) = " << large << " s");
    CHECK(large < 16.0 * std::max(small, 1e-5));
}

// --- issue 925 ---------------------------------------------------------------------------

TEST_CASE("ic_0 and ildl reject a zero, missing or negative pivot (issue 925)", "[itl][pc][925]") {
    // IC(0) on [[1,2],[2,1]]: the second pivot is 1 - 4 = -3 and its square
    // root NaN. On [[0,1],[1,1]] the first pivot is 0. ILDL on [[0,1],[1,1]]:
    // d_0 = 0 divides L(1,0). All three were constructed without complaint and
    // solve() returned NaN.
    const csr negative = make(2, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 2.0}, {1, 1, 1.0}});
    const csr zero     = make(2, {{0, 0, 0.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    const csr missing  = make(2, {{0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    CHECK_THROWS_AS(itl::pc::ic_0<double>(negative), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::ic_0<double>(zero), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::ic_0<double>(missing), std::runtime_error);
    CHECK_THROWS_AS((itl::pc::ic_0<double, mat::parameters<>, double>(negative)), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::ildl<double>(zero), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::ildl<double>(missing), std::runtime_error);
    CHECK_THROWS_AS((itl::pc::ildl<double, mat::parameters<>, double>(zero)), std::runtime_error);
    // ILDL is for indefinite matrices: d = (1, -3) is fine, and exact here.
    const itl::pc::ildl<double> L(negative);
    vec_t x(2), b(2);
    b(0) = 5.0; b(1) = 4.0;   // A (1, 2) = (5, 4)
    L.solve(x, b);
    CHECK(std::abs(x(0) - 1.0) < 1e-14);
    CHECK(std::abs(x(1) - 2.0) < 1e-14);
    // The message names the row.
    try {
        itl::pc::ic_0<double> M(negative);
        FAIL("ic_0 accepted a negative pivot");
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("row 1") != std::string::npos);
    }
    // A complex symmetric A has no sign: IC(0) of [[1,2],[2,1]] in complex is
    // L = [[1,0],[2,sqrt(-3)]], and its solve is exact.
    using C    = std::complex<double>;
    using ccsr = mat::compressed2D<C>;
    ccsr Ac(2, 2);
    {
        mat::inserter<ccsr> ins(Ac);
        ins[0][0] << C(1.0); ins[0][1] << C(2.0); ins[1][0] << C(2.0); ins[1][1] << C(1.0);
    }
    const itl::pc::ic_0<C> Mc(Ac);
    vec::dense_vector<C> xc(2), bc(2);
    bc(0) = C(5.0); bc(1) = C(4.0);
    Mc.solve(xc, bc);
    CHECK(std::abs(xc(0) - C(1.0)) < 1e-14);
    CHECK(std::abs(xc(1) - C(2.0)) < 1e-14);
}

TEST_CASE("block_diagonal: a singular block, bad arguments and block layouts (issue 925)",
          "[itl][pc][block_diagonal][925]") {
    using D = mat::dense2D<double>;
    // [[1,1],[1,1]] with bs = 2 gave -inf and inf, silently.
    D S(2, 2);
    S(0, 0) = S(0, 1) = S(1, 0) = S(1, 1) = 1.0;
    CHECK_THROWS_AS(itl::pc::block_diagonal<D>(S, 2), std::runtime_error);
    CHECK_THROWS_AS(itl::pc::block_diagonal<D>(S, 0), std::invalid_argument);
    D R(2, 3);
    CHECK_THROWS_AS(itl::pc::block_diagonal<D>(R, 1), std::invalid_argument);

    // A block-diagonal A (blocks of 3 and a last block of 2, n = 8) is its own
    // preconditioner: M^-1 b solves A x = b, for bs = 3, and for bs > n.
    const std::size_t n = 8;
    D A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = 0.0;
    for (std::size_t blk = 0; blk * 3 < n; ++blk)
        for (std::size_t i = blk * 3; i < std::min(n, blk * 3 + 3); ++i)
            for (std::size_t j = blk * 3; j < std::min(n, blk * 3 + 3); ++j)
                A(i, j) = (i == j) ? 5.0 + double(i) : 1.0 / double(1 + i + 2 * j);
    const vec_t b = ramp(n);
    auto residual = [&](const vec_t& x) {
        double r = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < n; ++j) s += A(i, j) * x(j);
            r = std::max(r, std::abs(s - b(i)));
        }
        return r;
    };
    for (const std::size_t bs : {3u, 100u}) {
        INFO("bs = " << bs);
        const itl::pc::block_diagonal<D> M(A, bs);
        vec_t x(n, 0.0);
        M.solve(x, b);
        CHECK(residual(x) < 1e-13);
        vec_t y(n, 0.0);
        M.adjoint_solve(y, b);   // A^T y = b
        double r = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < n; ++j) s += A(j, i) * y(j);
            r = std::max(r, std::abs(s - b(i)));
        }
        CHECK(r < 1e-13);
    }
    // compressed2D, the same matrix.
    entries e;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (A(i, j) != 0.0) e.emplace_back(i, j, A(i, j));
    const csr As = make(n, e);
    const itl::pc::block_diagonal<csr> Ms(As, 3);
    vec_t xs(n, 0.0);
    Ms.solve(xs, b);
    CHECK(residual(xs) < 1e-13);
}

TEST_CASE("the preconditioners and multigrid reject bad shapes in every build (issue 925)",
          "[itl][pc][mg][925]") {
    // The checks were assert-only: a non-square A indexed ilu_0's work arrays,
    // sized by rows, by column, in a release build.
    csr R(2, 3);
    {
        mat::inserter<csr> ins(R);
        ins[0][0] << 1.0; ins[1][1] << 1.0; ins[1][2] << 1.0;
    }
    CHECK_THROWS_AS(itl::pc::ilu_0<double>(R), std::invalid_argument);
    CHECK_THROWS_AS(itl::pc::ilut<double>(R), std::invalid_argument);
    CHECK_THROWS_AS(itl::pc::ic_0<double>(R), std::invalid_argument);
    CHECK_THROWS_AS(itl::pc::ildl<double>(R), std::invalid_argument);
    CHECK_THROWS_AS(itl::pc::ssor<csr>(R), std::invalid_argument);

    const std::size_t nf = 15, nc = 7;
    const csr Af = make(nf, tridiagonal_entries(nf, 2.0, -1.0));
    const csr Rt = itl::mg::make_restriction_1d(nf);
    const csr Pt = itl::mg::make_prolongation_1d(nc);
    const csr Ac = mtl::spgemm(Rt, mtl::spgemm(Af, Pt));
    auto sf = [](const csr& L) { return itl::smoother::gauss_seidel<csr>(L); };
    auto cs = [](vec_t&, const vec_t&) {};
    using MG = itl::mg::multigrid<double>;
    CHECK_THROWS_AS(MG({Af}, {}, {}, sf, cs), std::invalid_argument);              // one level
    CHECK_THROWS_AS(MG({Af, Ac}, {}, {Pt}, sf, cs), std::invalid_argument);        // no restrictor
    CHECK_THROWS_AS(MG({Af, Ac}, {Pt}, {Pt}, sf, cs), std::invalid_argument);      // R is P's shape
    CHECK_THROWS_AS(MG({Af, Ac}, {Rt}, {Rt}, sf, cs), std::invalid_argument);      // P is R's shape
    CHECK_NOTHROW(MG({Af, Ac}, {Rt}, {Pt}, sf, cs));
}

TEST_CASE("ilut keeps no U fill for a row with no upper part and fill = 0 (issue 925)",
          "[itl][pc][ilut][925]") {
    // Row 1 stores only A(1,0): orig_u = 0. Eliminating column 0 fills U(1,1)
    // = -1 and U(1,2) = -1. With fill = 0 at most orig_u + fill - 1 = -1, so
    // no strictly upper entry, survives; the count wrapped to SIZE_MAX and
    // U(1,2) was kept. With b = (0, 0, 1): x(2) = 1, x(1) = -U(1,2) x(2) / U(1,1),
    // so x(1) is 0 without U(1,2) and -1 with it.
    const csr A = make(3, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {2, 2, 1.0}});
    const itl::pc::ilut<double> M(A, 0, 0.0);
    vec_t x(3), b(3, 0.0);
    b(2) = 1.0;
    M.solve(x, b);
    CHECK(x(2) == 1.0);
    CHECK(x(1) == 0.0);
    // fill = 1 keeps one U entry, the diagonal, so x(1) is still 0. fill = 2
    // keeps U(1,2): the factorization is the exact LU.
    const itl::pc::ilut<double> F1(A, 1, 0.0);
    F1.solve(x, b);
    CHECK(x(1) == 0.0);
    const itl::pc::ilut<double> F(A, 2, 0.0);
    F.solve(x, b);
    CHECK(x(1) == -1.0);
}

TEST_CASE("power_iteration on an empty operator (issue 925)", "[itl][eigen][925]") {
    // n = 0 returned with `value` never written; reading it was undefined.
    const csr Z(0, 0);
    const auto r = itl::power_iteration(Z, vec_t(0));
    CHECK(r.value == 0.0);
    CHECK_FALSE(r.converged);
    CHECK(r.iterations == 0);
    CHECK(r.vector.size() == 0);
}

TEST_CASE("arnoldi: smallest selectors and argument edges (issue 925)", "[itl][eigen][925]") {
    // diag(-5, 1, 2, 3, 4) as a general operator: smallest magnitude 1,
    // smallest algebraic (real part) -5.
    const csr A = make(5, {{0, 0, -5.0}, {1, 1, 1.0}, {2, 2, 2.0}, {3, 3, 3.0}, {4, 4, 4.0}});
    const vec_t v0 = ramp(5);
    const auto sm = itl::arnoldi(A, v0, 1, itl::eigen_which::smallest_magnitude, 5);
    REQUIRE(sm.values.size() == 1);
    CHECK(std::abs(sm.values(0) - std::complex<double>(1.0)) < 1e-10);
    const auto sa = itl::arnoldi(A, v0, 1, itl::eigen_which::smallest_algebraic, 5);
    REQUIRE(sa.values.size() == 1);
    CHECK(std::abs(sa.values(0) - std::complex<double>(-5.0)) < 1e-10);
    // k > n: at most n pairs.
    const auto all = itl::arnoldi(A, v0, 9, itl::eigen_which::largest_magnitude, 5);
    CHECK(all.values.size() <= 5);
    CHECK(all.values.size() >= 1);
    // subspace < k: k is cut to the subspace.
    const auto cut = itl::arnoldi(A, v0, 4, itl::eigen_which::largest_magnitude, 2);
    CHECK(cut.values.size() <= 2);
    // A zero v0 is replaced by a nonzero start, and the run is finite.
    const auto z = itl::arnoldi(A, vec_t(5, 0.0), 1, itl::eigen_which::largest_magnitude, 5);
    for (std::size_t i = 0; i < z.values.size(); ++i) {
        CHECK(std::isfinite(z.values(i).real()));
        CHECK(std::isfinite(z.values(i).imag()));
    }
    // Too small a subspace to converge on the dominant pair: converged false.
    const csr T = make(40, tridiagonal_entries(40, 4.0, -1.0));
    const auto nc = itl::arnoldi(T, ramp(40), 1, itl::eigen_which::largest_magnitude, 3, 1e-14);
    CHECK_FALSE(nc.converged);
}

TEST_CASE("multigrid: the W-cycle reduces the residual (issue 925)", "[itl][mg][925]") {
    // wcycle was only checked not to throw. Three levels of 1D Poisson, so the
    // W-cycle recurses twice at level 1 and once, now, at the coarsest.
    const std::size_t n0 = 31, n1 = 15, n2 = 7;
    const csr A0 = make(n0, tridiagonal_entries(n0, 2.0, -1.0));
    const csr R0 = itl::mg::make_restriction_1d(n0), P0 = itl::mg::make_prolongation_1d(n1);
    const csr A1 = mtl::spgemm(R0, mtl::spgemm(A0, P0));
    const csr R1 = itl::mg::make_restriction_1d(n1), P1 = itl::mg::make_prolongation_1d(n2);
    const csr A2 = mtl::spgemm(R1, mtl::spgemm(A1, P1));
    auto sf = [](const csr& L) { return itl::smoother::gauss_seidel<csr>(L); };
    int coarse_solves = 0;
    auto cs = [A2, &coarse_solves](vec_t& x, const vec_t& b) {
        ++coarse_solves;
        itl::pc::identity<csr> I(A2);
        itl::basic_iteration<double> it(b, 1000, 1e-14);
        itl::cg(A2, x, b, I, it);
    };
    const vec_t b = ramp(n0);
    auto resid = [&](const vec_t& x) {
        const vec_t Ax = A0 * x;
        double r = 0;
        for (std::size_t i = 0; i < n0; ++i) r += (b(i) - Ax(i)) * (b(i) - Ax(i));
        return std::sqrt(r);
    };
    for (const auto& [pre, post] : {std::pair{2, 2}, std::pair{0, 2}, std::pair{2, 0}}) {
        INFO("nu_pre = " << pre << ", nu_post = " << post);
        const itl::mg::multigrid<double> mg({A0, A1, A2}, {R0, R1}, {P0, P1}, sf, cs, pre, post);
        vec_t x(n0, 0.0);
        const double r0 = resid(x);
        coarse_solves = 0;
        mg.wcycle(x, b);
        // Level 1 visits the coarsest level twice, each visit one solve.
        CHECK(coarse_solves == 2);
        for (int c = 1; c < 4; ++c) mg.wcycle(x, b);
        INFO("residual after 4 W-cycles: " << resid(x) / r0);   // 2.5e-7, 1.5e-4, 1.2e-3
        CHECK(resid(x) < 1e-2 * r0);
    }
}
