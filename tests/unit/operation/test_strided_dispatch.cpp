// Non-contiguous vectors must not reach the contiguous fast paths.
//
// `strided_vector_ref` stores element i at `data()[i * stride()]`, but it
// supplies both `data()` and `size()` -- which is all `BlasDenseVector` used to
// ask for. So every operation gated on that concept accepted it and then walked
// `data()` with unit stride, reading the wrong elements and returning a
// confident wrong answer. On the stride-2 case below `dot_real` gave 14 where
// the answer is 44: it summed {1,0,3}.{2,0,4} instead of {1,3,5}.{2,4,6}.
//
// This is not a corner case invented for the test. A column of a row-major
// matrix is exactly such a view -- `strided_vector_ref<double> col(A.data() + j,
// nrows, ncols)` is the documented construction in test_strided_vector_ref.cpp
// -- so "dot product of two matrix columns" was silently wrong for float and
// double alike, independent of the integer-lane work that exposed it.
//
// A stride is a runtime value, so no concept can admit only the unit-stride
// instances; the fix excludes types that advertise `stride()` outright and lets
// them take the generic element-wise loop, which indexes through `operator()`
// and is right for any stride. These cases pin that, per operation, against a
// hand-computed reference.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/strided_vector_ref.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

using Catch::Approx;

TEST_CASE("strided vectors are excluded from the contiguous dispatch concepts",
          "[operation][dispatch][strided]") {
    using SVd = mtl::vec::strided_vector_ref<double>;
    using SVi = mtl::vec::strided_vector_ref<std::int32_t>;
    using DVd = mtl::vec::dense_vector<double>;
    using DVi = mtl::vec::dense_vector<std::int32_t>;

    // The layout predicate itself: a type that advertises stride() is not
    // contiguous for dispatch purposes, whatever its stride happens to be.
    STATIC_REQUIRE_FALSE(mtl::interface::ContiguousVector<SVd>);
    STATIC_REQUIRE(mtl::interface::ContiguousVector<DVd>);

    STATIC_REQUIRE_FALSE(mtl::interface::BlasDenseVector<SVd>);
    STATIC_REQUIRE_FALSE(mtl::interface::SimdDenseVector<SVd>);
    STATIC_REQUIRE_FALSE(mtl::interface::SimdDenseVector<SVi>);

    // ... and the owning dense vectors still qualify, so nothing that was on a
    // fast path lost it.
    STATIC_REQUIRE(mtl::interface::BlasDenseVector<DVd>);
    STATIC_REQUIRE(mtl::interface::SimdDenseVector<DVd>);
    STATIC_REQUIRE(mtl::interface::SimdDenseVector<DVi>);
}

TEST_CASE("dot over strided views reads the strided elements", "[operation][dot][strided]") {
    // {1,3,5} . {2,4,6} = 2 + 12 + 30 = 44.  The contiguous misread gives 14.
    std::vector<double> a{1, 0, 3, 0, 5, 0}, b{2, 0, 4, 0, 6, 0};
    mtl::vec::strided_vector_ref<double> x(a.data(), 3, 2), y(b.data(), 3, 2);

    REQUIRE(x(0) == 1.0);
    REQUIRE(x(2) == 5.0);
    CHECK(mtl::dot_real(x, y) == Approx(44.0));
    CHECK(mtl::dot(x, y) == Approx(44.0));
}

TEST_CASE("dot over strided integer views wraps like the SIMD path",
          "[operation][dot][strided][integer]") {
    // Integer lanes reach the same generic loop, and must land on the same
    // mod-2^32 answer the contiguous kernel would give -- not on
    // signed-overflow UB.
    std::vector<std::int32_t> a(6), b(6);
    std::uint64_t acc = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        const auto ai = static_cast<std::uint32_t>(0x9E3779B9u * (k + 1));
        const auto bi = static_cast<std::uint32_t>(0x85EBCA77u * (k + 3));
        a[2 * k] = static_cast<std::int32_t>(ai);
        b[2 * k] = static_cast<std::int32_t>(bi);
        acc += static_cast<std::uint64_t>(ai) * bi;
    }
    mtl::vec::strided_vector_ref<std::int32_t> x(a.data(), 3, 2), y(b.data(), 3, 2);
    const auto expected = static_cast<std::int32_t>(static_cast<std::uint32_t>(acc));
    CHECK(mtl::dot_real(x, y) == expected);
}

TEST_CASE("two_norm over a strided view uses the strided elements",
          "[operation][norms][strided]") {
    std::vector<double> a{3, 99, 4, 99};                 // {3,4} -> 5
    mtl::vec::strided_vector_ref<double> x(a.data(), 2, 2);
    CHECK(mtl::two_norm(x) == Approx(5.0));
}

TEST_CASE("axpy and scale over strided views touch only the strided elements",
          "[operation][axpy][scale][strided]") {
    SECTION("axpy") {
        std::vector<double> xs{1, -1, 2, -1, 3, -1};
        std::vector<double> ys{10, -2, 20, -2, 30, -2};
        mtl::vec::strided_vector_ref<double> x(xs.data(), 3, 2), y(ys.data(), 3, 2);
        mtl::axpy(2.0, x, y);                            // y += 2x on the stride
        CHECK(ys[0] == Approx(12.0));
        CHECK(ys[2] == Approx(24.0));
        CHECK(ys[4] == Approx(36.0));
        CHECK(ys[1] == Approx(-2.0));                    // gaps untouched
        CHECK(ys[3] == Approx(-2.0));
        CHECK(ys[5] == Approx(-2.0));
    }
    SECTION("scale") {
        std::vector<double> cs{1, -1, 2, -1, 3, -1};
        mtl::vec::strided_vector_ref<double> c(cs.data(), 3, 2);
        mtl::scale(3.0, c);
        CHECK(cs[0] == Approx(3.0));
        CHECK(cs[2] == Approx(6.0));
        CHECK(cs[4] == Approx(9.0));
        CHECK(cs[1] == Approx(-1.0));                    // gaps untouched
        CHECK(cs[3] == Approx(-1.0));
        CHECK(cs[5] == Approx(-1.0));
    }
}
