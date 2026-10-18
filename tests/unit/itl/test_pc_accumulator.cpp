// MTL5 -- Accumulator policy for the incomplete-factorization preconditioners
// (#585).
//
// Each preconditioner takes a trailing `Accumulator` (class templates keep it
// trailing, like pc::ssor). Three claims per preconditioner:
//
//   reach    -- a counting accumulator sees exactly the products of the
//               factorization and of each solve, computed here from the
//               sparsity pattern alone, and the void default sees none;
//   identity -- a plain accumulator in the element type reproduces the void
//               path bit for bit (the same products in the same order), which
//               is why this target is built with -ffp-contract=off;
//   use      -- a wider or exact accumulator still applies M^-1.
//
// No accuracy ordering is asserted, because none exists to assert. ILU(0)'s
// reductions are as long as the stencil -- at most four terms on a 5-point grid,
// one on a tridiagonal -- so there is almost nothing for an exact accumulator
// to keep. Measured in float with exact residuals (plain / fma / exact):
//
//   tridiagonal (ILU(0) = LU), n = 1024:   1.7e-5 / 1.4e-5 / 1.4e-5
//   5-point, ||A M^-1 b - b||, n = 256:    6.2e-1 / 6.2e-1 / 6.2e-1
//
// On the tridiagonal FMA and exact agree: a one-term reduction rounds once
// either way. On the 5-point grid the dropped fill is the whole error.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <set>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/pc/ilu_0.hpp>

#include "../operation/exact_accumulator.hpp"

using namespace mtl;

namespace counting {

/// Configuration-3 shape (the ADL hook) that counts its products.
struct count_acc { double s; };
inline std::size_t products = 0;

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }
template <typename V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }
template <typename V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}
template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

/// Non-symmetric 5-point convection-diffusion on an m x m grid; distinct
/// values off the diagonal so a misplaced product shows.
template <typename T>
mat::compressed2D<T> convection_diffusion(std::size_t m) {
    const std::size_t n = m * m;
    mat::compressed2D<T> A(n, n);
    mat::inserter<mat::compressed2D<T>> ins(A);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < m; ++c) {
            const std::size_t i = r * m + c;
            const double w = 0.1 * static_cast<double>((7 * i) % 5);
            ins[i][i] << T(4.0 + w);
            if (c > 0)     ins[i][i - 1] << T(-1.3 + 0.05 * w);
            if (c + 1 < m) ins[i][i + 1] << T(-0.7);
            if (r > 0)     ins[i][i - m] << T(-1.1);
            if (r + 1 < m) ins[i][i + m] << T(-0.9 - 0.05 * w);
        }
    return A;
}

template <typename T>
std::vector<std::set<std::size_t>> pattern(const mat::compressed2D<T>& A) {
    std::vector<std::set<std::size_t>> p(A.num_rows());
    const auto& s = A.ref_major();
    const auto& c = A.ref_minor();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (auto k = s[i]; k < s[i + 1]; ++k) p[i].insert(c[k]);
    return p;
}

/// ILU(0) factorization products: for each row i, each stored k < i, and each
/// stored U(k,j) (j > k) that row i also stores.
inline std::size_t ilu0_products(const std::vector<std::set<std::size_t>>& p) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        for (auto k : p[i]) {
            if (k >= i) break;
            for (auto j : p[k])
                if (j > k && p[i].count(j)) ++count;
        }
    return count;
}

/// One product per stored off-diagonal entry: L in the forward, U in the back.
inline std::size_t offdiag(const std::vector<std::set<std::size_t>>& p) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < p.size(); ++i) count += p[i].size() - (p[i].count(i) ? 1 : 0);
    return count;
}

template <typename T>
vec::dense_vector<T> rhs(std::size_t n) {
    vec::dense_vector<T> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = T(std::sin(0.37 * static_cast<double>(i) + 0.2));
    return b;
}

template <typename T>
bool same_bits(const vec::dense_vector<T>& a, const vec::dense_vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a(i), &b(i), sizeof(T)) != 0) return false;
    return true;
}

template <typename T>
mat::compressed2D<std::complex<T>> complexify(const mat::compressed2D<T>& A) {
    const std::size_t n = A.num_rows();
    mat::compressed2D<std::complex<T>> C(n, n);
    mat::inserter<mat::compressed2D<std::complex<T>>> ins(C);
    const auto& s = A.ref_major();
    const auto& c = A.ref_minor();
    const auto& d = A.ref_data();
    for (std::size_t i = 0; i < n; ++i)
        for (auto k = s[i]; k < s[i + 1]; ++k)
            ins[i][c[k]] << std::complex<T>(d[k], c[k] == i ? T(0.5) : T(0.1) * T((i + c[k]) % 3));
    return C;
}

/// ||A x - b||_inf in double.
template <typename T, typename VX, typename VB>
double residual(const mat::compressed2D<T>& A, const VX& x, const VB& b) {
    const auto& s = A.ref_major();
    const auto& c = A.ref_minor();
    const auto& d = A.ref_data();
    double worst = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double r = -static_cast<double>(b(i));
        for (auto k = s[i]; k < s[i + 1]; ++k)
            r += static_cast<double>(d[k]) * static_cast<double>(x(c[k]));
        worst = std::max(worst, std::fabs(r));
    }
    return worst;
}

} // namespace

TEST_CASE("ilu_0: the accumulator reaches the factorization and both solves (#585)",
          "[itl][pc][ilu_0][accumulator]") {
    const auto A = convection_diffusion<double>(6);
    const std::size_t n = A.num_rows();
    const auto p = pattern(A);
    const auto b = rhs<double>(n);

    counting::products = 0;
    itl::pc::ilu_0<double, mat::parameters<>, counting::count_acc> M(A);
    REQUIRE(counting::products == ilu0_products(p));
    REQUIRE(ilu0_products(p) > 0);

    vec::dense_vector<double> x(n);
    counting::products = 0;
    M.solve(x, b);
    REQUIRE(counting::products == offdiag(p));

    counting::products = 0;
    M.adjoint_solve(x, b);
    REQUIRE(counting::products == offdiag(p));

    // The default never touches an accumulator.
    counting::products = 0;
    itl::pc::ilu_0<double> V(A);
    V.solve(x, b);
    V.adjoint_solve(x, b);
    REQUIRE(counting::products == 0);
}

TEST_CASE("ilu_0: a plain element-type accumulator is the void path bit for bit (#585)",
          "[itl][pc][ilu_0][accumulator]") {
    // The factorization and adjoint_solve() update one entry at a time in both
    // paths, so they agree on any pattern. solve() differs by design once a row
    // has two or more off-diagonal terms: the void path rounds the row sum and
    // then subtracts it from b(i), the accumulator path forms b(i) - sum inside
    // the accumulator. With one term per row (a tridiagonal) the two coincide.
    //
    // adjoint_solve() reads every stored L and U value, so agreeing bit for bit
    // on it is the check that the two factorizations agree.
    SECTION("float, 5-point: factorization and adjoint_solve") {
        const auto A = convection_diffusion<float>(7);
        const auto b = rhs<float>(A.num_rows());
        itl::pc::ilu_0<float> V(A);
        itl::pc::ilu_0<float, mat::parameters<>, float> P(A);
        vec::dense_vector<float> xv(A.num_rows()), xp(A.num_rows());
        V.adjoint_solve(xv, b);
        P.adjoint_solve(xp, b);
        REQUIRE(same_bits(xv, xp));
    }
    SECTION("float, tridiagonal: solve") {
        const std::size_t n = 50;
        mat::compressed2D<float> T(n, n);
        {
            mat::inserter<mat::compressed2D<float>> ins(T);
            for (std::size_t i = 0; i < n; ++i) {
                ins[i][i] << 3.0f + 0.1f * float(i % 7);
                if (i)         ins[i][i - 1] << -1.0f - 0.01f * float(i % 5);
                if (i + 1 < n) ins[i][i + 1] << -0.9f;
            }
        }
        const auto b = rhs<float>(n);
        itl::pc::ilu_0<float> V(T);
        itl::pc::ilu_0<float, mat::parameters<>, float> P(T);
        vec::dense_vector<float> xv(n), xp(n);
        V.solve(xv, b);
        P.solve(xp, b);
        REQUIRE(same_bits(xv, xp));
        V.adjoint_solve(xv, b);
        P.adjoint_solve(xp, b);
        REQUIRE(same_bits(xv, xp));
    }
    SECTION("complex<double>, 5-point: the conjugated scatter of adjoint_solve") {
        const auto A = complexify(convection_diffusion<double>(5));
        const std::size_t n = A.num_rows();
        vec::dense_vector<std::complex<double>> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b(i) = {std::sin(0.3 * double(i)), std::cos(0.7 * double(i))};
        using C = std::complex<double>;
        itl::pc::ilu_0<C> V(A);
        itl::pc::ilu_0<C, mat::parameters<>, C> P(A);
        vec::dense_vector<C> xv(n), xp(n);
        V.adjoint_solve(xv, b);
        P.adjoint_solve(xp, b);
        REQUIRE(same_bits(xv, xp));
    }
}

TEST_CASE("ilu_0: a wider or exact accumulator still applies M^-1 (#585)",
          "[itl][pc][ilu_0][accumulator]") {
    // A tridiagonal pattern has no fill, so ILU(0) IS the LU factorization and
    // solve() is an exact solve of A x = b; its residual measures the factor
    // and the substitutions together.
    const std::size_t n = 40;
    mat::compressed2D<float> A(n, n);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0f + 0.01f * float(i % 3);
            if (i)         ins[i][i - 1] << -1.0f;
            if (i + 1 < n) ins[i][i + 1] << -0.97f;
        }
    }
    const auto b = rhs<float>(n);
    vec::dense_vector<float> x(n);

    itl::pc::ilu_0<float, mat::parameters<>, double> W(A);
    W.solve(x, b);
    CHECK(residual(A, x, b) < 1e-4);

    itl::pc::ilu_0<float, mat::parameters<>, exact_test::exact_acc<float>> E(A);
    E.solve(x, b);
    CHECK(residual(A, x, b) < 1e-4);
}
