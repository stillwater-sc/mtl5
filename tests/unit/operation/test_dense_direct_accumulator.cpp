// The accumulator policy in the dense direct factorizations (#585): lu_factor,
// lu_solve, lu_adjoint_solve, lu_apply, inv, cholesky_factor / _solve,
// cholesky_h_factor / _solve, ldlt_factor / _solve, ldlt_h_factor / _solve,
// ldlt_bk_factor / _solve.
//
// What is checked:
//   - REACH. A counting ADL accumulator sees exactly the products each routine
//     forms, so the policy reaches every reduction and nothing is formed outside
//     it. The counts are the closed forms of the left-looking loops.
//   - IDENTITY. A plain accumulator in the element type (configuration 1) does
//     the same arithmetic as the void loops in the same order, so it is
//     compared BIT for bit: lu_factor against void, cholesky_factor
//     against its term-by-term overload (s = a; s = s - p_k), which is what an
//     accumulator assigned the matrix entry computes. Built with
//     -ffp-contract=off (tests/unit/CMakeLists.txt): a contracted loop is
//     configuration 2. The void comparison is skipped under MTL5_HAS_LAPACK,
//     where void dispatches to getrf. The solves form b(i) - sum term by term
//     (the sparse solves' form), so lu_solve<float> is compared with a
//     reference written that way.
//   - COMPLEX. The Hermitian forms take a complex accumulator.
//   - BUNCH-KAUFMAN. The accumulator ldlt_bk_factor is a different, left-looking
//     algorithm that must leave the factor in the void one's per-step layout, so
//     a random indefinite sweep with interchanges and 2x2 blocks (#335) mixes
//     the accumulator factor with the void solve and vice versa.
//
// The accuracy ordering of configurations 1/2/3 is measured in
// test_accumulation_solvers_mt.cpp ([586:direct]).
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/tag/summation_order.hpp>
#include <mtl/vec/dense_vector.hpp>

#include "exact_accumulator.hpp"

using namespace mtl;

namespace counting {

// Sums in double and counts every product it is handed. No accumulator_merge.
struct count_acc {
    double s;
};

inline std::atomic<std::size_t> products{0};

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }

template <typename V>
    requires std::is_floating_point_v<V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }

template <typename V>
    requires std::is_floating_point_v<V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}

template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

using counting::count_acc;
using dmat = mat::dense2D<double>;
using dvec = vec::dense_vector<double>;

/// Symmetric positive definite, with distinct pivots: 1/(i+j+1) + n on the diagonal.
template <typename T>
mat::dense2D<T> spd(std::size_t n) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(1.0 / static_cast<double>(i + j + 1) + (i == j ? double(n) : 0.0));
    return A;
}

/// Nonsymmetric and well conditioned, needing row interchanges: column 0 grows
/// down the rows. (The sine term alone has rank 2; the diagonal makes it full.)
template <typename T>
mat::dense2D<T> pivoting(std::size_t n) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(std::sin(static_cast<double>(3 * i + 7 * j + 1)) +
                                     (i == j ? 2.0 : 0.0) +
                                     (j == 0 ? static_cast<double>(i) : 0.0));
    return A;
}

template <typename F>
std::size_t products_of(F f) {
    counting::products = 0;
    f();
    return counting::products;
}

// Closed forms of the left-looking loops.
std::size_t lu_products(std::size_t n) {          // column k: (n-k) rows, row k: (n-k-1) cols, k terms each
    std::size_t c = 0;
    for (std::size_t k = 0; k < n; ++k) c += k * (n - k) + k * (n - k - 1);
    return c;
}
std::size_t cholesky_products(std::size_t n) {    // pivot j: j terms; rows i > j: j terms each
    std::size_t c = 0;
    for (std::size_t j = 0; j < n; ++j) c += j + (n - j - 1) * j;
    return c;
}
std::size_t bk_products(std::size_t n) {          // 1x1 steps, no interchange: column k only
    std::size_t c = 0;
    for (std::size_t k = 0; k < n; ++k) c += k * (n - k);
    return c;
}
std::size_t solve_products(std::size_t n) { return n * (n - 1); }   // two triangles

template <typename M>
bool same_bits(const M& A, const M& B) {
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            if (!(A(i, j) == B(i, j))) return false;
    return true;
}

} // namespace

TEST_CASE("dense direct: the Accumulator reaches every reduction", "[operation][accumulator][585]") {
    constexpr std::size_t n = 9;
    const dvec b = [] { dvec v(n); for (std::size_t i = 0; i < n; ++i) v(i) = 1.0 + double(i % 4); return v; }();
    dvec x(n);

    SECTION("lu_factor, lu_solve, lu_adjoint_solve, lu_apply, inv") {
        dmat A = pivoting<double>(n);
        std::vector<std::size_t> piv;
        CHECK(products_of([&] { REQUIRE(lu_factor<count_acc>(A, piv) == 0); }) == lu_products(n));
        CHECK(products_of([&] { lu_solve<count_acc>(A, piv, x, b); }) == solve_products(n));
        CHECK(products_of([&] { lu_adjoint_solve<count_acc>(A, piv, x, b); }) == solve_products(n));
        dmat B = pivoting<double>(n);
        CHECK(products_of([&] { REQUIRE(lu_apply<count_acc>(B, x, b) == 0); }) ==
              lu_products(n) + solve_products(n));
        CHECK(products_of([&] { (void)inv<count_acc>(pivoting<double>(n)); }) ==
              lu_products(n) + n * solve_products(n));
    }
    SECTION("cholesky_factor / _solve and cholesky_h_factor / _solve") {
        dmat A = spd<double>(n);
        CHECK(products_of([&] { REQUIRE(cholesky_factor<count_acc>(A) == 0); }) == cholesky_products(n));
        CHECK(products_of([&] { cholesky_solve<count_acc>(A, x, b); }) == solve_products(n));
        dmat H = spd<double>(n);
        CHECK(products_of([&] { REQUIRE(cholesky_h_factor<count_acc>(H) == 0); }) == cholesky_products(n));
        CHECK(products_of([&] { cholesky_h_solve<count_acc>(H, x, b); }) == solve_products(n));
    }
    SECTION("ldlt_factor / _solve and ldlt_h_factor / _solve") {
        // count_acc is an ADL accumulator over a real IEEE type, so each
        // three-factor term arrives as hi plus, when its rounding error is
        // nonzero, lo (#602): between one and two products per term.
        const std::size_t terms = cholesky_products(n);
        dmat A = spd<double>(n);
        const std::size_t f = products_of([&] { REQUIRE(ldlt_factor<count_acc>(A) == 0); });
        CHECK(f > terms);
        CHECK(f <= 2 * terms);
        CHECK(products_of([&] { ldlt_solve<count_acc>(A, x, b); }) == solve_products(n));
        dmat H = spd<double>(n);
        const std::size_t fh = products_of([&] { REQUIRE(ldlt_h_factor<count_acc>(H) == 0); });
        CHECK(fh > terms);
        CHECK(fh <= 2 * terms);
        CHECK(products_of([&] { ldlt_h_solve<count_acc>(H, x, b); }) == solve_products(n));
    }
    SECTION("ldlt_bk_factor / _solve, no interchanges") {
        // Diagonally dominant: every step takes test 1, a 1x1 pivot in place,
        // so column k is n - k entries of k products.
        dmat A = spd<double>(n);
        bk_pivot_info piv;
        CHECK(products_of([&] { REQUIRE(ldlt_bk_factor<count_acc>(A, piv) == 0); }) ==
              bk_products(n));
        for (std::size_t k = 0; k < n; ++k) CHECK(piv.ipiv[k] == static_cast<int>(k + 1));
        CHECK(products_of([&] { ldlt_bk_solve<count_acc>(A, piv, x, b); }) == solve_products(n));
    }
    SECTION("void hands the accumulator nothing") {
        dmat A = pivoting<double>(n);
        std::vector<std::size_t> piv;
        CHECK(products_of([&] { lu_factor(A, piv); lu_solve(A, piv, x, b); }) == 0);
    }
}

TEST_CASE("dense direct: a plain accumulator is the void arithmetic, bit for bit",
          "[operation][accumulator][585]") {
    constexpr std::size_t n = 23;

    SECTION("cholesky_factor<float> == cholesky_factor(A, subtract_term_by_term)") {
        mat::dense2D<float> A = spd<float>(n), B = spd<float>(n);
        REQUIRE(cholesky_factor<float>(A) == 0);
        REQUIRE(cholesky_factor(B, tag::subtract_term_by_term{}) == 0);
        CHECK(same_bits(A, B));
    }
#ifndef MTL5_HAS_LAPACK
    SECTION("lu_factor<float> == void") {
        // Left-looking with the matrix entry assigned first subtracts the same
        // products in the same order as right-looking does.
        mat::dense2D<float> A = pivoting<float>(n), B = pivoting<float>(n);
        std::vector<std::size_t> pa, pb;
        REQUIRE(lu_factor<float>(A, pa) == 0);
        REQUIRE(lu_factor(B, pb) == 0);
        CHECK(pa == pb);
        CHECK(same_bits(A, B));
    }
#endif
    SECTION("lu_solve<float> == the term-by-term substitutions") {
        // The accumulator solve is s = b(i); s = s - T(i,j) x(j); then the
        // division -- the sparse solves' form (#535), not the void loop's
        // b(i) - (sum), so it is compared with a reference written that way.
        mat::dense2D<float> A = pivoting<float>(n);
        std::vector<std::size_t> piv;
        REQUIRE(lu_factor<float>(A, piv) == 0);
        vec::dense_vector<float> b(n), x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = static_cast<float>(i % 5) - 2.0f;
        lu_solve<float>(A, piv, x, b);
        for (std::size_t i = 0; i < n; ++i) y(i) = b(i);
        for (std::size_t i = 0; i < n; ++i) std::swap(y(i), y(piv[i]));
        for (std::size_t i = 0; i < n; ++i) {
            float s = y(i);
            for (std::size_t j = 0; j < i; ++j) s = s - A(i, j) * y(j);
            y(i) = s;
        }
        for (std::size_t ii = n; ii-- > 0;) {
            float s = y(ii);
            for (std::size_t j = ii + 1; j < n; ++j) s = s - A(ii, j) * y(j);
            y(ii) = s / A(ii, ii);
        }
        for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == y(i));
    }
}

TEST_CASE("dense direct: wider accumulators solve, real and complex", "[operation][accumulator][585]") {
    constexpr std::size_t n = 12;

    SECTION("lu_apply<double> on float") {
        const auto A0 = pivoting<float>(n);
        auto A = A0;
        vec::dense_vector<float> x(n), b(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = static_cast<float>(i % 3) - 1.0f;
        REQUIRE(lu_apply<double>(A, x, b) == 0);
        for (std::size_t i = 0; i < n; ++i) {
            double r = b(i);
            for (std::size_t j = 0; j < n; ++j) r -= double(A0(i, j)) * double(x(j));
            CHECK(std::fabs(r) < 1e-4);
        }
    }
    SECTION("Hermitian cholesky_h and ldlt_h with a complex<double> accumulator") {
        using cf = std::complex<float>;
        using cd = std::complex<double>;
        mat::dense2D<cf> H(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const float re = 1.0f / static_cast<float>(i + j + 1) + (i == j ? float(n) : 0.0f);
                const float im = i == j ? 0.0f : 0.1f * (i > j ? 1.0f : -1.0f) * static_cast<float>((i + j) % 3);
                H(i, j) = cf(re, im);
            }
        vec::dense_vector<cf> b(n), x(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = cf(1.0f, static_cast<float>(i % 2));
        auto residual = [&] {
            double worst = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                cd r = cd(b(i));
                for (std::size_t j = 0; j < n; ++j) r -= cd(H(i, j)) * cd(x(j));
                worst = std::max(worst, std::abs(r));
            }
            return worst;
        };
        auto L = H;
        REQUIRE(cholesky_h_factor<cd>(L) == 0);
        cholesky_h_solve<cd>(L, x, b);
        CHECK(residual() < 1e-5);
        auto D = H;
        REQUIRE(ldlt_h_factor<cd>(D) == 0);
        ldlt_h_solve<cd>(D, x, b);
        CHECK(residual() < 1e-5);
    }
}

namespace {

double bk_backward_error(const dmat& A, const dvec& x, const dvec& b) {
    double r2 = 0.0, a2 = 0.0, x2 = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double r = b(i);
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            r -= A(i, j) * x(j);
            a2 += A(i, j) * A(i, j);
        }
        r2 += r * r;
        x2 += x(i) * x(i);
    }
    return std::sqrt(r2) / (std::sqrt(a2) * std::sqrt(x2));
}

} // namespace

TEST_CASE("dense direct: accumulator Bunch-Kaufman through interchanges and 2x2 blocks",
          "[operation][accumulator][585][ldlt_bk]") {
    // The #335 sweep (test_ldlt_bk.cpp): random symmetric indefinite n = 4..12.
    // Each draw is solved four ways. Mixing a factor from one path with the
    // solve from the other checks that both leave and read the same per-step
    // layout, which a residual of matched pairs alone would not.
    using exact_test::exact_acc;
    std::uint64_t seed = 20260802u;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((seed >> 11) % 2000001) / 1000000.0 - 1.0;
    };
    std::size_t saw_2x2 = 0, saw_swap = 0, checked = 0;
    for (std::size_t n = 4; n <= 12; ++n) {
        for (int trial = 0; trial < 20; ++trial) {
            dmat A0(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j <= i; ++j) A0(i, j) = A0(j, i) = next();
            dvec b(n), x(n);
            for (std::size_t i = 0; i < n; ++i) b(i) = next();

            dmat Fv = A0, Fa = A0, Fe = A0;
            bk_pivot_info pv, pa, pe;
            if (ldlt_bk_factor(Fv, pv) != 0) continue;   // singular draw
            REQUIRE(ldlt_bk_factor<double>(Fa, pa) == 0);
            REQUIRE(ldlt_bk_factor<exact_acc<double>>(Fe, pe) == 0);
            ++checked;
            for (std::size_t k = 0; k < n; ++k) {
                if (pa.ipiv[k] < 0) { ++saw_2x2; break; }
                if (static_cast<std::size_t>(pa.ipiv[k] - 1) != k) { ++saw_swap; break; }
            }
            INFO("n = " << n << ", trial = " << trial);
            ldlt_bk_solve<double>(Fa, pa, x, b);
            CHECK(bk_backward_error(A0, x, b) < 1e-10);
            ldlt_bk_solve(Fa, pa, x, b);                 // accumulator factor, void solve
            CHECK(bk_backward_error(A0, x, b) < 1e-10);
            ldlt_bk_solve<double>(Fv, pv, x, b);         // void factor, accumulator solve
            CHECK(bk_backward_error(A0, x, b) < 1e-10);
            ldlt_bk_solve<exact_acc<double>>(Fe, pe, x, b);
            CHECK(bk_backward_error(A0, x, b) < 1e-10);
        }
    }
    REQUIRE(checked > 0);
    REQUIRE(saw_2x2 > 0);
    REQUIRE(saw_swap > 0);
}
