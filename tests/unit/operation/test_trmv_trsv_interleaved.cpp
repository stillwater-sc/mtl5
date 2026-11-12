// The generic trmv, the generic forward substitution (lower_trisolve, and
// trsv with a lower A) and the generic mat*vec (detail::mult_generic_rows,
// behind mult_generic and mult_generic_par) take W rows at a time, and the
// result is BIT-IDENTICAL to the one-row loops they replace (#749), for every
// W (#772).
//
// The one-row loops are a single serial reduction per row: each add waits on
// the last, so they ran at one add per FMA latency, 41% of the bandwidth
// roofline. W rows give W independent chains. Every row still sums the same
// terms in the same order with the same expressions, so the bits do not
// change; this test holds them to that, against verbatim copies of the old
// loops below. The public functions take W from interface::interleave_rows;
// the detail kernels are also called with every width in `widths`, for n
// below, at and past W and 2W, so the remainder rows each width hands down to
// W / 2 are covered too.
//
// On a contiguous column-major A the public functions take the column sweep
// instead (#777), which must give the same bits; the sweeps are also called
// with strips of a few rows, so the sizes below cross strip boundaries.
//
// In a BLAS build, float and double trmv and trsv go to BLAS, so those cases
// are only checked to be close; long double never reaches BLAS, and
// lower_trisolve and the detail kernels are called directly as well, so every
// build exercises the new loops bit for bit. With MTL5_NATIVE_FAST_GEMM, trmv
// and trsv block only above detail::triangular_block rows, and every n here is
// at or below it.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt): with contraction
// on, a compiler may fuse a product into its add in one form and not the other.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <mtl/detail/column_sweep.hpp>
#include <mtl/detail/scalar_chain.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/interface/interleave_rows.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

// A user's width for one type and one kernel (#772): long double gemv takes 3
// rows, which also puts the public mat*vec tests below on an odd width with a
// remainder of 1 or 2. It must precede every use of the trait in this TU.
template <>
struct mtl::interface::interleave_rows<long double, mtl::interface::kernel::gemv> {
    static constexpr std::size_t value = 3;
};

using namespace mtl;

namespace {

template <typename T> using rm = mat::dense2D<T, mat::parameters<tag::row_major>>;
template <typename T> using cm = mat::dense2D<T, mat::parameters<tag::col_major>>;

constexpr std::size_t sizes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                                 31, 32, 33, 47, 48, 49, 64};

/// The widths the detail kernels are checked at: 1 is the one-row loop, 3 and 5
/// hand odd remainders down, 6 is the float/double default, 16 the widest the
/// sweep in benchmarks/bench_interleave_rows tries.
using widths = std::index_sequence<1, 2, 3, 4, 5, 6, 8, 16>;
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

template <typename M, typename VIn, typename VOut>
void old_mult_generic(const M& A, const VIn& x, VOut& y) {
    using Result = typename VOut::value_type;
    using size_type = typename M::size_type;
    const size_type ncols = A.num_cols();
    for (std::size_t r = 0; r < static_cast<std::size_t>(A.num_rows()); ++r) {
        auto acc = math::zero<Result>();
        for (size_type c = 0; c < ncols; ++c) {
            acc = detail::generic_fma<Result>(acc, A(static_cast<size_type>(r), c), x(c));
        }
        y(static_cast<typename VOut::size_type>(r)) = acc;
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
    if constexpr (std::is_integral_v<T>) {
        return x == y;
    } else {
        if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
        return x == y && std::signbit(x) == std::signbit(y);
    }
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

TEMPLATE_TEST_CASE("generic mat*vec rows are bit-identical to the one-row loop", "[operation][749]",
                   rm<float>, rm<double>, rm<long double>, cm<float>, cm<double>, cm<long double>) {
    using M = TestType;
    using T = typename M::value_type;
    using V = vec::dense_vector<T>;
    for (std::size_t m : sizes)
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{5}, std::size_t{37}}) {
            INFO("m = " << m << ", n = " << n);
            M A(m, n);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<T>(val(5 + m, i, j));
            const V x = filled<T>(n, 6 + n);
            V y(m), z(m), w(m);
            old_mult_generic(A, x, y);
            detail::mult_generic(A, x, z);
            require_identical(z, y);
            detail::mult_generic_par(A, x, w);
            require_identical(w, y);
            // A band that starts off a multiple of 4 and ends inside a block.
            if (m >= 3) {
                V u = y;
                for (std::size_t i = 1; i + 1 < m; ++i) u(i) = T(0);
                detail::mult_generic_rows(A, x, u, 1, m - 1);
                require_identical(u, y);
            }
        }
}

TEST_CASE("generic mat*vec rows keep wrapping integers and mixed types bit for bit", "[operation][749]") {
    // int32: generic_fma wraps mod 2^32; the values overflow on purpose.
    {
        rm<std::int32_t> A(13, 9);
        vec::dense_vector<std::int32_t> x(9);
        for (std::size_t i = 0; i < 13; ++i)
            for (std::size_t j = 0; j < 9; ++j) A(i, j) = static_cast<std::int32_t>(0x7fff0000 - 977 * i * j);
        for (std::size_t j = 0; j < 9; ++j) x(j) = static_cast<std::int32_t>(65537 * (j + 3));
        vec::dense_vector<std::int32_t> y(13), z(13);
        old_mult_generic(A, x, y);
        detail::mult_generic(A, x, z);
        require_identical(z, y);
    }
    // double operands into a float y: each term rounds in the result type.
    {
        const rm<double> A = triangular<rm<double>>(11, 8);
        const vec::dense_vector<double> x = filled<double>(11, 9);
        vec::dense_vector<float> y(11), z(11);
        old_mult_generic(A, x, y);
        detail::mult_generic(A, x, z);
        require_identical(z, y);
    }
}

namespace {

/// Every detail kernel at width W against the one-row loops, all sizes.
template <std::size_t W, typename M>
void check_width() {
    using T = typename M::value_type;
    using V = vec::dense_vector<T>;
    for (std::size_t n : sizes) {
        INFO("W = " << W << ", n = " << n);
        const M A = triangular<M>(n, 11 + n);
        const V x0 = filled<T>(n, 12 + n);
        for (bool unit : {false, true}) {
            INFO((unit ? "unit" : "non-unit"));
            {   // trmv, upper
                V x = x0, y = x0;
                detail::trmv_upper_rows<W>(A, x, typename M::size_type{0}, unit);
                old_trmv(A, y, true, unit);
                require_identical(x, y);
            }
            {   // trmv, lower
                V x = x0, y = x0;
                detail::trmv_lower_rows<W>(A, x, static_cast<typename M::size_type>(n), unit);
                old_trmv(A, y, false, unit);
                require_identical(x, y);
            }
            {   // forward substitution, out of place and in place
                V x(n), y(n);
                detail::lower_trisolve_rows<W>(A, x, x0, unit);
                old_lower_trisolve(A, y, x0, unit);
                require_identical(x, y);
                V u = x0;
                detail::lower_trisolve_rows<W>(A, u, u, unit);
                require_identical(u, y);
            }
        }
        {   // mat*vec: the whole range, then a band starting and ending off a block
            V y(n), z(n);
            old_mult_generic(A, x0, y);
            detail::mult_rows_interleaved<W>(A, x0, z, 0, n);
            require_identical(z, y);
            if (n >= 3) {
                V u = y;
                for (std::size_t i = 1; i + 1 < n; ++i) u(i) = T(0);
                detail::mult_rows_interleaved<W>(A, x0, u, 1, n - 1);
                require_identical(u, y);
            }
        }
    }
}

template <typename M, std::size_t... W>
void check_widths(std::index_sequence<W...>) {
    (check_width<W, M>(), ...);
}

} // namespace

TEMPLATE_TEST_CASE("every interleave width is bit-identical to the one-row loops", "[operation][772]",
                   rm<float>, rm<double>, rm<long double>, cm<float>, cm<double>, cm<long double>) {
    check_widths<TestType>(widths{});
}

TEST_CASE("every interleave width keeps wrapping integers bit for bit", "[operation][772]") {
    rm<std::int32_t> A(19, 9);
    vec::dense_vector<std::int32_t> x(9);
    for (std::size_t i = 0; i < 19; ++i)
        for (std::size_t j = 0; j < 9; ++j) A(i, j) = static_cast<std::int32_t>(0x7fff0000 - 977 * i * j);
    for (std::size_t j = 0; j < 9; ++j) x(j) = static_cast<std::int32_t>(65537 * (j + 3));
    vec::dense_vector<std::int32_t> y(19);
    old_mult_generic(A, x, y);
    [&]<std::size_t... W>(std::index_sequence<W...>) {
        ([&] {
            INFO("W = " << W);
            vec::dense_vector<std::int32_t> z(19);
            detail::mult_rows_interleaved<W>(A, x, z, 0, 19);
            require_identical(z, y);
        }(), ...);
    }(widths{});
}

TEST_CASE("interleave_rows defaults and overrides", "[operation][772]") {
    using namespace mtl::interface;
#if !defined(MTL5_INTERLEAVE_ROWS) && !defined(MTL5_INTERLEAVE_ROWS_F32) && !defined(MTL5_INTERLEAVE_ROWS_F64)
    // Hardware float types: measured per ISA class (#780), at most half the
    // registers.
    constexpr std::size_t cap = vector_registers / 2;
    constexpr auto capped = [](std::size_t w) { return w < cap ? w : cap; };
#if defined(__x86_64__) || defined(_M_X64)
    constexpr std::size_t f64 = 8, f32_gemv = 8, f32_tri = 8;
#elif defined(__aarch64__) || defined(_M_ARM64)
    constexpr std::size_t f64 = 4, f32_gemv = 4;
#  if defined(__GNUC__) && !defined(__clang__)
    constexpr std::size_t f32_tri = 2;   // #795
#  else
    constexpr std::size_t f32_tri = 4;
#  endif
#else
    constexpr std::size_t f64 = 6, f32_gemv = 6, f32_tri = 6;
#endif
    STATIC_REQUIRE(interleave_rows_v<double, kernel::gemv> == capped(f64));
    STATIC_REQUIRE(interleave_rows_v<double, kernel::trmv> == capped(f64));
    STATIC_REQUIRE(interleave_rows_v<double, kernel::lower_trisolve> == capped(f64));
    STATIC_REQUIRE(interleave_rows_v<float, kernel::gemv> == capped(f32_gemv));
    STATIC_REQUIRE(interleave_rows_v<float, kernel::trmv> == capped(f32_tri));
    STATIC_REQUIRE(interleave_rows_v<float, kernel::lower_trisolve> == capped(f32_tri));
    STATIC_REQUIRE(interleave_rows_v<const double&, kernel::trmv> == interleave_rows_v<double, kernel::trmv>);
    // Everything else keeps the pre-#772 width of 4: no extra registers on x87,
    // and only the user knows a software type's cost.
    STATIC_REQUIRE(interleave_rows_v<long double, kernel::trmv> == 4);
    STATIC_REQUIRE(interleave_rows_v<long double, kernel::lower_trisolve> == 4);
    STATIC_REQUIRE(interleave_rows_v<std::int32_t, kernel::gemv> == 4);
#endif
    // The specialization at the top of this file, for one type and one kernel.
    STATIC_REQUIRE(interleave_rows_v<long double, kernel::gemv> == 3);
}

TEST_CASE("keep_scalar applies to row-major float at W >= 4 under GCC only", "[operation][776]") {
    // Where it applies, the row-major float cases of the width sweep above
    // check that it leaves the bits alone.
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__aarch64__))
    STATIC_REQUIRE(detail::keep_scalar_v<float, rm<float>, 4>);
    STATIC_REQUIRE(detail::keep_scalar_v<float, rm<float>, 6>);
    STATIC_REQUIRE(detail::keep_scalar_v<float, rm<float>, 16>);
#else
    STATIC_REQUIRE(!detail::keep_scalar_v<float, rm<float>, 6>);
#endif
    STATIC_REQUIRE(!detail::keep_scalar_v<float, rm<float>, 1>);
    STATIC_REQUIRE(!detail::keep_scalar_v<float, rm<float>, 3>);
    STATIC_REQUIRE(!detail::keep_scalar_v<float, cm<float>, 6>);
    STATIC_REQUIRE(!detail::keep_scalar_v<double, rm<double>, 6>);
    STATIC_REQUIRE(!detail::keep_scalar_v<long double, rm<long double>, 6>);
}

namespace {

/// The column sweeps with S rows per strip against the one-row loops.
template <std::size_t S, typename M>
void check_strip() {
    using T = typename M::value_type;
    using V = vec::dense_vector<T>;
    for (std::size_t n : sizes) {
        INFO("S = " << S << ", n = " << n);
        const M A = triangular<M>(n, 21 + n);
        const V x0 = filled<T>(n, 22 + n);
        for (bool unit : {false, true}) {
            INFO((unit ? "unit" : "non-unit"));
            for (bool upper : {false, true}) {
                INFO((upper ? "upper" : "lower"));
                V x = x0, y = x0;
                detail::trmv_column_sweep<S>(A, x, upper, unit);
                old_trmv(A, y, upper, unit);
                require_identical(x, y);
            }
            V x(n), y(n);
            detail::lower_trisolve_column_sweep<S>(A, x, x0, unit);
            old_lower_trisolve(A, y, x0, unit);
            require_identical(x, y);
            V u = x0;
            detail::lower_trisolve_column_sweep<S>(A, u, u, unit);
            require_identical(u, y);
        }
        V y(n), z(n);
        old_mult_generic(A, x0, y);
        detail::mult_rows_column_sweep<S>(A, x0, z, 0, n);
        require_identical(z, y);
        if (n >= 3) {   // a band starting and ending off a strip
            V u = y;
            for (std::size_t i = 1; i + 1 < n; ++i) u(i) = T(0);
            detail::mult_rows_column_sweep<S>(A, x0, u, 1, n - 1);
            require_identical(u, y);
        }
    }
}

template <typename M, std::size_t... S>
void check_strips(std::index_sequence<S...>) {
    (check_strip<S, M>(), ...);
}

} // namespace

TEMPLATE_TEST_CASE("the column sweeps are bit-identical to the one-row loops at every strip size", "[operation][777]",
                   cm<float>, cm<double>, cm<long double>) {
    check_strips<TestType>(std::index_sequence<1, 2, 3, 5, 8, 16>{});
}

TEST_CASE("the column-major mat*vec keeps wrapping integers and mixed types bit for bit", "[operation][777]") {
    // int32 on a column-major A: the public path is the column sweep.
    {
        cm<std::int32_t> A(19, 9);
        vec::dense_vector<std::int32_t> x(9);
        for (std::size_t i = 0; i < 19; ++i)
            for (std::size_t j = 0; j < 9; ++j) A(i, j) = static_cast<std::int32_t>(0x7fff0000 - 977 * i * j);
        for (std::size_t j = 0; j < 9; ++j) x(j) = static_cast<std::int32_t>(65537 * (j + 3));
        vec::dense_vector<std::int32_t> y(19), z(19), w(19);
        old_mult_generic(A, x, y);
        detail::mult_generic(A, x, z);
        require_identical(z, y);
        detail::mult_rows_column_sweep<4>(A, x, w, 0, 19);
        require_identical(w, y);
    }
    // double operands into a float y: each term rounds in the result type.
    {
        const cm<double> A = triangular<cm<double>>(23, 8);
        const vec::dense_vector<double> x = filled<double>(23, 9);
        vec::dense_vector<float> y(23), z(23), w(23);
        old_mult_generic(A, x, y);
        detail::mult_generic(A, x, z);
        require_identical(z, y);
        detail::mult_rows_column_sweep<5>(A, x, w, 0, 23);
        require_identical(w, y);
    }
}

TEST_CASE("the column sweep applies to contiguous column-major arithmetic types only", "[operation][777]") {
    STATIC_REQUIRE(detail::column_sweep_v<float, cm<float>>);
    STATIC_REQUIRE(detail::column_sweep_v<double, cm<double>>);
    STATIC_REQUIRE(detail::column_sweep_v<long double, cm<long double>>);
    STATIC_REQUIRE(detail::column_sweep_v<std::int32_t, cm<std::int32_t>>);
    STATIC_REQUIRE(detail::column_sweep_v<float, cm<double>>);
    STATIC_REQUIRE(!detail::column_sweep_v<float, rm<float>>);
    STATIC_REQUIRE(!detail::column_sweep_v<double, rm<double>>);
    STATIC_REQUIRE(detail::column_strip_rows<float> == 4096);
    STATIC_REQUIRE(detail::column_strip_rows<double> == 2048);
}
