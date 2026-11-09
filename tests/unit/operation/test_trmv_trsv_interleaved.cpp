// The generic trmv and the generic forward substitution (lower_trisolve, and
// trsv with a lower A) take four rows at a time, and the result is
// BIT-IDENTICAL to the one-row loops they replace (#749).
//
// The one-row loops are a single serial reduction per row: each add waits on
// the last, so they ran at one add per FMA latency, 41% of the bandwidth
// roofline. Four rows give four independent chains. Every row still sums the
// same terms in the same order with the same expressions, so the bits do not
// change; this test holds them to that, against verbatim copies of the old
// loops below, for n below, at and past multiples of 4.
//
// In a BLAS build, float and double trmv and trsv go to BLAS, so those cases
// are only checked to be close; long double never reaches BLAS, and
// lower_trisolve is called directly as well, so every build exercises the new
// loops bit for bit. With MTL5_NATIVE_FAST_GEMM, trmv and trsv block only above
// detail::triangular_block rows, and every n here is at or below it.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt): with contraction
// on, a compiler may fuse a product into its add in one form and not the other.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

template <typename T> using rm = mat::dense2D<T, mat::parameters<tag::row_major>>;
template <typename T> using cm = mat::dense2D<T, mat::parameters<tag::col_major>>;

constexpr std::size_t sizes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 31, 33, 64};
static_assert(64 <= detail::triangular_block, "every n stays on the row loop in a native-fast build");

// -- the pre-#749 generic loops, verbatim ---------------------------------------

template <typename M, typename VX>
void old_trmv(const M& A, VX& x, bool upper, bool unit_diag) {
    using size_type = typename M::size_type;
    const size_type n = A.num_rows();
    if (upper) {
        for (size_type i = 0; i < n; ++i) {
            auto acc = unit_diag ? x(i) : A(i, i) * x(i);
            for (size_type j = i + 1; j < n; ++j)
                acc += A(i, j) * x(j);
            x(i) = acc;
        }
    } else {
        for (size_type ii = n; ii-- > 0; ) {
            auto acc = unit_diag ? x(ii) : A(ii, ii) * x(ii);
            for (size_type j = 0; j < ii; ++j)
                acc += A(ii, j) * x(j);
            x(ii) = acc;
        }
    }
}

template <typename M, typename VX, typename VB>
void old_lower_trisolve(const M& L, VX& x, const VB& b, bool unit_diag) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        if (unit_diag)
            x(i) = b(i) - sum;
        else
            x(i) = (b(i) - sum) / L(i, i);
    }
}

template <typename M, typename VX>
void old_lower_trisolve(const M& L, VX& x, bool unit_diag) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        if (unit_diag)
            x(i) = x(i) - sum;
        else
            x(i) = (x(i) - sum) / L(i, i);
    }
}

template <typename M, typename VX>
void old_upper_trisolve(const M& U, VX& x, bool unit_diag) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = U.num_rows();
    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += U(i, j) * x(j);
        if (unit_diag)
            x(i) = x(i) - sum;
        else
            x(i) = (x(i) - sum) / U(i, i);
    }
}

// -- inputs and comparison --------------------------------------------------------

double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::size_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 29; h *= 0xD6E8FEB86659FD93ull; h ^= h >> 32;
    return static_cast<double>(h % 1000003) / 500001.5 - 1.0;   // many distinct mantissas
}

/// Full storage, both triangles filled (the routines must ignore the other
/// one), with a diagonal that keeps the solves well conditioned.
template <typename M>
M triangular(std::size_t n, std::size_t seed) {
    using T = typename M::value_type;
    M A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<T>(val(seed, i, j) / 4);
    for (std::size_t i = 0; i < n; ++i) A(i, i) = static_cast<T>(2.0 + val(seed, i, i) / 2);
    return A;
}

template <typename T>
vec::dense_vector<T> filled(std::size_t n, std::size_t seed) {
    vec::dense_vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v(i) = static_cast<T>(val(seed, i, 7));
    return v;
}

/// The same value, and the same sign of zero. Not memcmp: x87 long double has
/// six bytes of padding whose contents are unspecified.
template <typename T>
bool identical(T x, T y) {
    if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
    return x == y && std::signbit(x) == std::signbit(y);
}

template <typename V>
void require_identical(const V& u, const V& w) {
    REQUIRE(u.size() == w.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
        INFO("i = " << i << ": " << static_cast<double>(u(i)) << " vs " << static_cast<double>(w(i)));
        REQUIRE(identical(u(i), w(i)));
    }
}

/// Close, for the cases a BLAS build sends to BLAS instead of the row loop.
template <typename V>
void require_close(const V& u, const V& w) {
    using T = typename V::value_type;
    const double tol = 256 * static_cast<double>(std::numeric_limits<T>::epsilon());
    REQUIRE(u.size() == w.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
        INFO("i = " << i << ": " << static_cast<double>(u(i)) << " vs " << static_cast<double>(w(i)));
        REQUIRE(std::fabs(static_cast<double>(u(i)) - static_cast<double>(w(i))) <=
                tol * (1 + std::fabs(static_cast<double>(w(i)))));
    }
}

} // namespace

TEMPLATE_TEST_CASE("generic trmv is bit-identical to the one-row loop", "[operation][749]",
                   rm<float>, rm<double>, rm<long double>, cm<float>, cm<double>, cm<long double>) {
    using M = TestType;
    using T = typename M::value_type;
    using V = vec::dense_vector<T>;
    constexpr bool via_blas = detail::trmv_dispatches_blas_v<void, M, V>;
    for (std::size_t n : sizes)
        for (bool upper : {false, true})
            for (bool unit : {false, true}) {
                INFO("n = " << n << (upper ? " upper" : " lower") << (unit ? " unit" : ""));
                const M A = triangular<M>(n, 1 + n);
                const V x0 = filled<T>(n, 2 + n);
                V x = x0, y = x0;
                trmv(A, x, upper, unit);
                old_trmv(A, y, upper, unit);
                if constexpr (via_blas) require_close(x, y); else require_identical(x, y);
            }
}

TEMPLATE_TEST_CASE("generic trsv and lower_trisolve are bit-identical to the one-row loops", "[operation][749]",
                   rm<float>, rm<double>, rm<long double>, cm<float>, cm<double>, cm<long double>) {
    using M = TestType;
    using T = typename M::value_type;
    using V = vec::dense_vector<T>;
    constexpr bool via_blas = detail::trsv_dispatches_blas_v<void, M, V>;
    for (std::size_t n : sizes)
        for (bool unit : {false, true}) {
            INFO("n = " << n << (unit ? " unit" : ""));
            const M A = triangular<M>(n, 3 + n);
            const V b = filled<T>(n, 4 + n);

            // lower_trisolve directly: the new loop in every build.
            V x = b, y = b;
            lower_trisolve(A, x, unit);
            old_lower_trisolve(A, y, unit);
            require_identical(x, y);
            V u(n), w(n);
            lower_trisolve(A, u, b, unit);
            old_lower_trisolve(A, w, b, unit);
            require_identical(u, w);
            require_identical(u, x);

            // trsv, both triangles; the upper one is the unchanged one-row loop.
            for (bool upper : {false, true}) {
                INFO((upper ? "upper" : "lower"));
                V p = b, q = b;
                trsv(A, p, upper, unit);
                if (upper) old_upper_trisolve(A, q, unit); else old_lower_trisolve(A, q, unit);
                if constexpr (via_blas) require_close(p, q); else require_identical(p, q);
                V r(n);
                trsv(A, r, b, upper, unit);
                if constexpr (via_blas) require_close(r, q); else require_identical(r, q);
            }
        }
}
