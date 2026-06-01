// Tests for the mtl::simd::batch<T> abstraction (#83).
// These exercise the public API against a scalar reference and pass in both
// backends (Highway when MTL5_HAS_HIGHWAY, scalar fallback otherwise).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/simd/batch.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using Catch::Approx;

TEST_CASE("vectorizable_length tail helper", "[simd]") {
    using mtl::simd::vectorizable_length;
    CHECK(vectorizable_length(10, 4) == 8);
    CHECK(vectorizable_length(8, 4) == 8);
    CHECK(vectorizable_length(3, 4) == 0);
    CHECK(vectorizable_length(0, 4) == 0);
    CHECK(vectorizable_length(7, 1) == 7);   // scalar fallback: no tail
    CHECK(vectorizable_length(7, 0) == 0);   // guard against div-by-zero
}

TEMPLATE_TEST_CASE("batch size is a positive compile-time constant", "[simd]", float, double) {
    using B = mtl::simd::batch<TestType>;
    STATIC_REQUIRE(B::size >= 1);
    STATIC_REQUIRE(mtl::simd::width<TestType> == B::size);
}

TEMPLATE_TEST_CASE("batch load/store round-trip (aligned + unaligned)", "[simd]", float, double) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;

    alignas(64) TestType in[64];
    alignas(64) TestType out[64];
    for (std::size_t i = 0; i < N; ++i) in[i] = static_cast<TestType>(i) + TestType(1);

    SECTION("unaligned") {
        auto v = B::load_unaligned(in);
        v.store_unaligned(out);
        for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == in[i]);
    }
    SECTION("aligned") {
        auto v = B::load_aligned(in);
        v.store_aligned(out);
        for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == in[i]);
    }
    SECTION("broadcast") {
        B v(TestType(3.5));
        v.store_unaligned(out);
        for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == Approx(TestType(3.5)));
    }
}

TEMPLATE_TEST_CASE("batch elementwise arithmetic vs scalar", "[simd]", float, double) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;
    alignas(64) TestType a[64], b[64], r[64];
    for (std::size_t i = 0; i < N; ++i) { a[i] = TestType(i) + 2; b[i] = TestType(i) * TestType(0.5) + 1; }

    auto va = B::load_aligned(a);
    auto vb = B::load_aligned(b);

    (va + vb).store_aligned(r); for (std::size_t i=0;i<N;++i) CHECK(r[i] == Approx(a[i] + b[i]));
    (va - vb).store_aligned(r); for (std::size_t i=0;i<N;++i) CHECK(r[i] == Approx(a[i] - b[i]));
    (va * vb).store_aligned(r); for (std::size_t i=0;i<N;++i) CHECK(r[i] == Approx(a[i] * b[i]));
    (va / vb).store_aligned(r); for (std::size_t i=0;i<N;++i) CHECK(r[i] == Approx(a[i] / b[i]));
}

TEMPLATE_TEST_CASE("batch fma matches a*b+c", "[simd]", float, double) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;
    alignas(64) TestType a[64], b[64], c[64], r[64];
    for (std::size_t i = 0; i < N; ++i) { a[i]=TestType(i)+1; b[i]=TestType(2); c[i]=TestType(0.25)*TestType(i); }

    fma(B::load_aligned(a), B::load_aligned(b), B::load_aligned(c)).store_aligned(r);
    for (std::size_t i = 0; i < N; ++i) CHECK(r[i] == Approx(a[i]*b[i] + c[i]));
}

TEMPLATE_TEST_CASE("batch horizontal reductions", "[simd]", float, double) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;
    alignas(64) TestType a[64];
    TestType sum = 0, mn = std::numeric_limits<TestType>::max(), mx = std::numeric_limits<TestType>::lowest();
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = TestType(i) - TestType(N) / 2 + TestType(1);   // mix of signs
        sum += a[i]; mn = std::min(mn, a[i]); mx = std::max(mx, a[i]);
    }
    auto v = B::load_aligned(a);
    CHECK(reduce_add(v) == Approx(sum));
    CHECK(reduce_min(v) == Approx(mn));
    CHECK(reduce_max(v) == Approx(mx));
}

// "Written once" SIMD kernel: y = alpha*x + y, body + scalar tail. Compiles to
// the widest enabled ISA by build flags only; correct for any length.
namespace {
template <typename T>
void simd_axpy(T alpha, const T* x, T* y, std::size_t n) {
    using B = mtl::simd::batch<T>;
    const B va(alpha);
    const std::size_t vl = mtl::simd::vectorizable_length(n, B::size);
    std::size_t i = 0;
    for (; i < vl; i += B::size) {
        auto r = fma(va, B::load_unaligned(x + i), B::load_unaligned(y + i));
        r.store_unaligned(y + i);
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}
}

TEMPLATE_TEST_CASE("written-once axpy over arbitrary (incl. non-multiple) lengths", "[simd]", float, double) {
    const std::size_t lengths[] = {0, 1, 3, 7, 16, 31, 33, 100, 257};
    for (std::size_t n : lengths) {
        std::vector<TestType> x(n), y(n), ref(n);
        for (std::size_t i = 0; i < n; ++i) { x[i] = TestType(i) * TestType(0.5) + 1; y[i] = TestType(2) - TestType(i); ref[i] = y[i]; }
        const TestType alpha = TestType(1.5);
        simd_axpy(alpha, x.data(), y.data(), n);
        for (std::size_t i = 0; i < n; ++i) CHECK(y[i] == Approx(alpha * x[i] + ref[i]));
    }
}
