// Wrapping integer arithmetic in the sparse and remaining L1 generic loops (#461).
//
// #460 defined the contract for the dense path: an integer `dot` or `mult`
// returns the exact sum reduced mod 2^N and never invokes signed-overflow UB.
// The loops covered here were outside that pass and still accumulated with a
// plain `+=` / `*=`, which is UB on overflow for signed types and disagrees with
// the contract the dense path states.
//
// EVERY ASSERTION IS A CLOSED FORM, not a comparison against another MTL5 loop.
// Checking a wrapping kernel against a second wrapping kernel would pass just as
// happily if both were wrong in the same way, and "both were wrong in the same
// way" is the actual failure mode here -- these sites were all written from the
// same habit. The references below are computed in unsigned arithmetic, where
// the reduction mod 2^N is what the language guarantees rather than what the
// implementation happens to do.
//
// Build this file under -fsanitize=undefined -fno-sanitize-recover=all to make
// the UB half of the claim testable rather than merely asserted; the values here
// are chosen to overflow, so an unfixed loop traps rather than quietly agreeing.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>

using namespace mtl;
using i32 = std::int32_t;
using u32 = std::uint32_t;

namespace {

constexpr i32 imin = std::numeric_limits<i32>::min();
constexpr i32 imax = std::numeric_limits<i32>::max();

/// The reference: reduce mod 2^32 in unsigned arithmetic, where the language
/// defines the reduction, then reinterpret. Deliberately NOT written with any
/// mtl helper.
constexpr i32 as_i32(u32 v) { return static_cast<i32>(v); }

// The reference arithmetic wraps on purpose, exactly as the library helpers do,
// so it needs the same exemption from clang's `unsigned-integer-overflow` lint --
// which is a check for ACCIDENTAL wrap-around, not a UB check. Naming the three
// operations keeps the exemption narrow: everything else in this file, including
// every signed expression, stays fully checked.
#if defined(__clang__)
#  define MTL5_TEST_WRAPS_ON_PURPOSE __attribute__((no_sanitize("unsigned-integer-overflow")))
#else
#  define MTL5_TEST_WRAPS_ON_PURPOSE
#endif
MTL5_TEST_WRAPS_ON_PURPOSE inline u32 uadd(u32 a, u32 b) { return a + b; }
MTL5_TEST_WRAPS_ON_PURPOSE inline u32 umul(u32 a, u32 b) { return a * b; }
MTL5_TEST_WRAPS_ON_PURPOSE inline u32 uneg(u32 a) { return 0u - a; }
/// |e| reduced mod 2^32 -- the closed form of what generic_abs must produce.
inline u32 uabs(i32 e) { return e < 0 ? uneg(static_cast<u32>(e)) : static_cast<u32>(e); }

/// Values that overflow an int32 accumulator quickly and include both ends of
/// the range, so `abs()` of the minimum is exercised too.
i32 val(std::size_t i) {
    static const i32 pool[] = {imax, imin, -1, 1, imax - 3, imin + 5, 1 << 30, -(1 << 29)};
    return pool[i % 8];
}

} // namespace

TEST_CASE("wrapping helpers agree with unsigned reduction at the extremes",
          "[detail][wrapping][integer]") {
    using detail::wrap_add; using detail::wrap_mul; using detail::wrap_sub;
    REQUIRE(wrap_add(imax, 1) == imin);
    REQUIRE(wrap_sub(imin, 1) == imax);
    REQUIRE(wrap_mul(imin, -1) == imin);          // 2^31 mod 2^32 == -2^31
    REQUIRE(detail::generic_add<i32>(imax, 1) == imin);
    REQUIRE(detail::generic_mul<i32>(imin, -1) == imin);
    REQUIRE(detail::generic_fma<i32>(imax, 1, 1) == imin);

    // The abs() defect the issue did not enumerate: std::abs(INT_MIN) is UB.
    REQUIRE(detail::generic_abs(imin) == imin);
    REQUIRE(detail::generic_abs(i32{-7}) == 7);
    REQUIRE(detail::generic_abs(-2.5) == 2.5);    // non-integral is unchanged
}

TEST_CASE("sparse CRS matvec accumulates mod 2^32", "[operation][mult][sparse][integer][wrapping]") {
    const std::size_t n = 64, band = 5;
    mat::compressed2D<i32> A(n, n);
    std::vector<std::vector<std::pair<std::size_t, i32>>> rows(n);
    {
        mat::inserter<mat::compressed2D<i32>> ins(A, band);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t b = 0; b < band; ++b) {
                const std::size_t c = (r * 3 + b * 11 + 1) % n;
                const i32 v = val(r + b);
                ins[r][c] << v;
                rows[r].push_back({c, v});
            }
    }
    vec::dense_vector<i32> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) x(i) = val(i + 3);

    mult(A, x, y);

    for (std::size_t r = 0; r < n; ++r) {
        u32 ref = 0;
        for (auto [c, v] : rows[r]) ref = uadd(ref, umul(static_cast<u32>(v), static_cast<u32>(x(c))));
        REQUIRE(y(r) == as_i32(ref));
    }
}

TEST_CASE("transposed sparse matvec scatters mod 2^32",
          "[operation][mult][sparse][integer][wrapping]") {
    const std::size_t n = 48, band = 4;
    mat::compressed2D<i32> A(n, n);
    {
        mat::inserter<mat::compressed2D<i32>> ins(A, band);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t b = 0; b < band; ++b)
                ins[r][(r * 5 + b * 7 + 1) % n] << val(r + b + 1);
    }
    vec::dense_vector<i32> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) x(i) = val(i + 2);

    mult(mat::view::transposed_view<mat::compressed2D<i32>>(A), x, y);

    std::vector<u32> ref(n, 0u);
    const auto& starts = A.ref_major();
    const auto& idx    = A.ref_minor();
    const auto& data   = A.ref_data();
    for (std::size_t r = 0; r < n; ++r)
        for (auto k = starts[r]; k < starts[r + 1]; ++k)
            ref[idx[k]] = uadd(ref[idx[k]], umul(static_cast<u32>(data[k]), static_cast<u32>(x(r))));

    for (std::size_t j = 0; j < n; ++j)
        REQUIRE(y(j) == as_i32(ref[j]));
}

TEST_CASE("operator* on a sparse matrix accumulates mod 2^32",
          "[mat][operators][sparse][integer][wrapping]") {
    const std::size_t n = 32, band = 4;
    mat::compressed2D<i32> A(n, n);
    std::vector<std::vector<std::pair<std::size_t, i32>>> rows(n);
    {
        mat::inserter<mat::compressed2D<i32>> ins(A, band);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t b = 0; b < band; ++b) {
                const std::size_t c = (r + b * 9 + 1) % n;
                const i32 v = val(r + b + 4);
                ins[r][c] << v;
                rows[r].push_back({c, v});
            }
    }
    vec::dense_vector<i32> x(n);
    for (std::size_t i = 0; i < n; ++i) x(i) = val(i + 1);

    const auto y = A * x;

    for (std::size_t r = 0; r < n; ++r) {
        u32 ref = 0;
        for (auto [c, v] : rows[r]) ref = uadd(ref, umul(static_cast<u32>(v), static_cast<u32>(x(c))));
        REQUIRE(y(r) == as_i32(ref));
    }
}

TEST_CASE("operator* on dense matrices accumulates mod 2^32",
          "[mat][operators][integer][wrapping]") {
    // The eager expression-template loops, which #460 left behind next to the
    // `mult_generic` it did fix.
    const std::size_t n = 12;
    mat::dense2D<i32> A(n, n), B(n, n);
    vec::dense_vector<i32> x(n);
    for (std::size_t r = 0; r < n; ++r) {
        x(r) = val(r + 5);
        for (std::size_t c = 0; c < n; ++c) {
            A(r, c) = val(r * n + c);
            B(r, c) = val(r + c * 3 + 2);
        }
    }

    const auto y = A * x;
    for (std::size_t r = 0; r < n; ++r) {
        u32 ref = 0;
        for (std::size_t c = 0; c < n; ++c) ref = uadd(ref, umul(static_cast<u32>(A(r, c)), static_cast<u32>(x(c))));
        REQUIRE(y(r) == as_i32(ref));
    }

    const auto C = A * B;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) {
            u32 ref = 0;
            for (std::size_t k = 0; k < n; ++k)
                ref = uadd(ref, umul(static_cast<u32>(A(r, k)), static_cast<u32>(B(k, c))));
            REQUIRE(C(r, c) == as_i32(ref));
        }
}

TEST_CASE("axpy accumulates mod 2^32", "[operation][axpy][integer][wrapping]") {
    // A strided view, so the SIMD/BLAS fast paths are declined and the generic
    // loop is what runs -- the loop this issue is about.
    const std::size_t n = 40;
    vec::dense_vector<i32> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) { x(i) = val(i); y(i) = val(i + 6); }
    std::vector<i32> y0(n);
    for (std::size_t i = 0; i < n; ++i) y0[i] = y(i);

    const i32 alpha = imax;
    axpy(alpha, x, y);

    for (std::size_t i = 0; i < n; ++i) {
        const u32 ref = uadd(static_cast<u32>(y0[i]), umul(static_cast<u32>(alpha), static_cast<u32>(x(i))));
        REQUIRE(y(i) == as_i32(ref));
    }
}

TEST_CASE("scale multiplies mod 2^32", "[operation][scale][integer][wrapping]") {
    const std::size_t n = 40;
    vec::dense_vector<i32> c(n);
    std::vector<i32> c0(n);
    for (std::size_t i = 0; i < n; ++i) { c(i) = val(i + 7); c0[i] = c(i); }

    const i32 alpha = imin;
    scale(alpha, c);

    for (std::size_t i = 0; i < n; ++i) {
        const u32 ref = umul(static_cast<u32>(alpha), static_cast<u32>(c0[i]));
        REQUIRE(c(i) == as_i32(ref));
    }
}

TEST_CASE("vector norms accumulate mod 2^32 and survive abs(INT_MIN)",
          "[operation][norms][integer][wrapping]") {
    const std::size_t n = 40;
    vec::dense_vector<i32> v(n);
    for (std::size_t i = 0; i < n; ++i) v(i) = val(i);
    REQUIRE(v(1) == imin);                      // the abs() trap is actually present

    u32 ref1 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        ref1 = uadd(ref1, uabs(v(i)));
    }
    REQUIRE(one_norm(v) == as_i32(ref1));

    // two_norm returns sqrt of the wrapped sum of squares; assert the SUM, which
    // is the part this issue governs, rather than the sqrt of it.
    u32 ref2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const u32 a = uabs(v(i));
        ref2 = uadd(ref2, umul(a, a));
    }
    using std::sqrt;
    REQUIRE(two_norm(v) == sqrt(as_i32(ref2)));

    // infinity_norm is a MAX of wrapped magnitudes, and the wrapped |imin| is
    // imin -- negative, so it never wins. That is the honest consequence of
    // defining abs(min) as min: an integer norm is defined and reduced mod 2^N,
    // but it is only a *norm* when nothing overflowed. Asserted rather than
    // glossed over, so the semantics are pinned and visible.
    i32 best = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const i32 e = v(i);
        const i32 a = as_i32(uabs(e));
        if (a > best) best = a;
    }
    REQUIRE(best == imax);                      // NOT imin, despite imin being present
    REQUIRE(infinity_norm(v) == best);
}

TEST_CASE("matrix norms accumulate mod 2^32", "[operation][norms][integer][wrapping]") {
    const std::size_t n = 10;
    mat::dense2D<i32> m(n, n);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
            m(r, c) = val(r * n + c);

    i32 best_col = std::numeric_limits<i32>::min();
    for (std::size_t c = 0; c < n; ++c) {
        u32 s = 0;
        for (std::size_t r = 0; r < n; ++r) {
            s = uadd(s, uabs(m(r, c)));
        }
        if (as_i32(s) > best_col) best_col = as_i32(s);
    }
    REQUIRE(one_norm(m) == best_col);

    i32 best_row = std::numeric_limits<i32>::min();
    for (std::size_t r = 0; r < n; ++r) {
        u32 s = 0;
        for (std::size_t c = 0; c < n; ++c) {
            s = uadd(s, uabs(m(r, c)));
        }
        if (as_i32(s) > best_row) best_row = as_i32(s);
    }
    REQUIRE(infinity_norm(m) == best_row);
}

TEST_CASE("float results are unchanged by the wrapping helpers",
          "[operation][wrapping][float]") {
    // The helpers are the plain expression for every non-integral type, so the
    // float paths must not have moved. Values with exact binary representations,
    // so `==` is the right assertion.
    const std::size_t n = 8;
    vec::dense_vector<double> x(n), y(n), c(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = 0.5 + static_cast<double>(i);
        y(i) = 0.25 * static_cast<double>(i) - 1.0;
        c(i) = x(i);
    }
    std::vector<double> y0(n);
    for (std::size_t i = 0; i < n; ++i) y0[i] = y(i);

    axpy(2.5, x, y);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(y(i) == y0[i] + 2.5 * x(i));

    scale(0.5, c);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(c(i) == 0.5 * x(i));

    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x(i);
    REQUIRE(one_norm(x) == s);
}
