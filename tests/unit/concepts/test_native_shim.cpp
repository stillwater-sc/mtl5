// MTL5 -- self-test of the test-support type native_shim (#703).
//
// float_shim and double_shim must be user-defined types that a native-only
// construct cannot accept, and must still round every operation as the native
// type underneath does: the genericity tests compare MTL5 run on the shim with
// MTL5 run on float or double, and that comparison is only meaningful if the
// shim's own arithmetic is the native arithmetic.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>

#include <mtl5_test/native_shim.hpp>

using mtl5_test::double_shim;
using mtl5_test::float_shim;

namespace {

// What native-only code does, as concepts: each must be false for the shim.
template <typename T> concept std_sqrt_ok   = requires(T t) { std::sqrt(t); };
template <typename T> concept std_abs_ok    = requires(T t) { std::abs(t); };
template <typename T> concept std_frexp_ok  = requires(T t, int* e) { std::frexp(t, e); };
template <typename T> concept std_ldexp_ok  = requires(T t) { std::ldexp(t, 3); };
template <typename T> concept std_isnan_ok  = requires(T t) { std::isnan(t); };
template <typename T> concept compares_int  = requires(T t) { t >= 0; };
template <typename T> concept mixed_double  = requires(T t) { 2.0 * t; };

// The math generic code calls through `using std::f; f(x);` resolves to the
// shim's hidden friends by ADL; the test below calls each of them that way.

// One static_assert per property, so a failure names the one that broke.
#define MTL5_SHIM_STATIC_CHECKS(S)                                              \
    static_assert(!std::is_floating_point_v<S>);                               \
    static_assert(!std::is_arithmetic_v<S>);                                   \
    static_assert(std::is_trivially_default_constructible_v<S>);               \
    static_assert(std::is_trivially_copyable_v<S>);                            \
    static_assert(!std_sqrt_ok<S>);                                            \
    static_assert(!std_abs_ok<S>);                                             \
    static_assert(!std_frexp_ok<S>);                                           \
    static_assert(!std_ldexp_ok<S>);                                           \
    MTL5_SHIM_STD_ISNAN_CHECK(S)                                               \
    static_assert(!compares_int<S>);                                           \
    static_assert(!mixed_double<S>);                                           \
    static_assert(!std::is_convertible_v<double, S>);                          \
    static_assert(!std::is_convertible_v<int, S>);                             \
    static_assert(!std::is_convertible_v<S, double>);                          \
    static_assert(!std::is_convertible_v<S, float>);                           \
    static_assert(std::is_constructible_v<S, double>);                         \
    static_assert(std::is_constructible_v<S, int>);                            \
    static_assert(std::numeric_limits<S>::is_specialized);

// Microsoft's <math.h> declares isnan, isinf, isfinite and signbit as
// unconstrained templates, so std::isnan(t) is well-formed in an unevaluated
// requires-expression for any type and only fails once instantiated. The
// check is a property of the standard library there, not of the shim; the
// others hold on every library.
#if defined(_MSVC_STL_VERSION)
#define MTL5_SHIM_STD_ISNAN_CHECK(S)
#else
#define MTL5_SHIM_STD_ISNAN_CHECK(S) static_assert(!std_isnan_ok<S>);
#endif

MTL5_SHIM_STATIC_CHECKS(float_shim)
MTL5_SHIM_STATIC_CHECKS(double_shim)
static_assert(sizeof(float_shim) == sizeof(float) && sizeof(double_shim) == sizeof(double));

} // namespace

TEMPLATE_TEST_CASE("native_shim rounds every operation as its native type does", "[native_shim]",
                   float_shim, double_shim) {
    using S = TestType;
    using N = typename S::native_type;
    using L = std::numeric_limits<N>;
    std::mt19937_64 g(703);
    std::uniform_real_distribution<double> u(-4.0, 4.0);
    std::uniform_int_distribution<int> ex(-60, 60);
    auto same = [](S s, N n) { return s.native() == n || (std::isnan(s.native()) && std::isnan(n)); };
    for (int t = 0; t < 2000; ++t) {
        const N a = static_cast<N>(std::ldexp(u(g), ex(g)));
        const N b = static_cast<N>(std::ldexp(u(g), ex(g)));
        const S sa(a), sb(b);
        CAPTURE(a, b);
        REQUIRE(same(sa + sb, a + b));
        REQUIRE(same(sa - sb, a - b));
        REQUIRE(same(sa * sb, a * b));
        REQUIRE(same(sa / sb, a / b));
        REQUIRE(same(-sa, -a));
        {
            using std::abs;
            using std::sqrt;
            REQUIRE(same(sqrt(abs(sa)), std::sqrt(std::abs(a))));
        }
        REQUIRE(same(hypot(sa, sb), std::hypot(a, b)));
        REQUIRE(same(fma(sa, sb, sa), std::fma(a, b, a)));
        int e1 = 0, e2 = 0;
        REQUIRE(same(frexp(sa, &e1), std::frexp(a, &e2)));
        REQUIRE(e1 == e2);
        REQUIRE(same(ldexp(sa, -7), std::ldexp(a, -7)));
        REQUIRE((sa < sb) == (a < b));
        REQUIRE((sa == sb) == (a == b));
        S acc = sa;
        acc += sb; acc *= sb; acc -= sa; acc /= sb;
        N nacc = a;
        nacc += b; nacc *= b; nacc -= a; nacc /= b;
        REQUIRE(same(acc, nacc));
    }
    // Limits are N's, and a NaN behaves as N's NaN.
    REQUIRE(std::numeric_limits<S>::epsilon().native() == L::epsilon());
    REQUIRE(std::numeric_limits<S>::max().native() == L::max());
    REQUIRE(std::numeric_limits<S>::min().native() == L::min());
    REQUIRE(std::numeric_limits<S>::digits == L::digits);
    REQUIRE(std::numeric_limits<S>::is_iec559 == L::is_iec559);
    const S nan = std::numeric_limits<S>::quiet_NaN();
    REQUIRE(isnan(nan));
    REQUIRE_FALSE(nan == nan);
    REQUIRE_FALSE(nan < S(0));
    REQUIRE(isinf(std::numeric_limits<S>::infinity()));
    REQUIRE(static_cast<double>(S(0.5)) == 0.5);
    std::ostringstream os;
    os << S(1.5);
    REQUIRE(os.str() == "1.5");
}
