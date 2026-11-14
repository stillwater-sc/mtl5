// detail helpers with no direct test before #838: cabs1, structure_tol,
// same_object, expr_store_t/expr_raw_t, c_index/f_index,
// transposed_orientation, machine_epsilon, and the allocator's overflow check.
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

#include <mtl/config.hpp>
#include <mtl/detail/aligned_allocator.hpp>
#include <mtl/detail/expr_storage.hpp>
#include <mtl/detail/index.hpp>
#include <mtl/detail/same_object.hpp>
#include <mtl/detail/structure_tol.hpp>
#include <mtl/detail/unit_roundoff.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/traits/transposed_orientation.hpp>

using namespace mtl;

TEST_CASE("cabs1: |Re| + |Im|", "[detail][838]") {
    CHECK(detail::cabs1(std::complex<double>(3.0, -4.0)) == 7.0);
    CHECK(detail::cabs1(-2.5) == 2.5);
    STATIC_REQUIRE(std::is_same_v<decltype(detail::cabs1(std::complex<float>{})), float>);
}

TEST_CASE("structure_tol scales with n, epsilon and the data", "[detail][838]") {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    STATIC_REQUIRE(detail::structure_tol(10, 1.0) == 10 * eps);
    STATIC_REQUIRE(detail::structure_tol(10, 4.0) == 40 * eps);
    STATIC_REQUIRE(detail::structure_tol(std::size_t{3}, 1.0f) == 3 * std::numeric_limits<float>::epsilon());
    STATIC_REQUIRE(detail::structure_tol(0, 1.0) == 0.0);
}

TEST_CASE("machine_epsilon and unit_roundoff come from numeric_limits", "[detail][838]") {
    CHECK(detail::machine_epsilon<double>() == std::numeric_limits<double>::epsilon());
    CHECK(detail::machine_epsilon<float>() == std::numeric_limits<float>::epsilon());
    CHECK(detail::unit_roundoff<double>() == std::numeric_limits<double>::epsilon() / 2);
}

TEST_CASE("same_object compares addresses, also in release builds", "[detail][838]") {
    int a = 1, b = 1;
    CHECK(detail::same_object(a, a));
    CHECK(!detail::same_object(a, b));
    const double d = 0;
    CHECK(!detail::same_object(a, d));
    STATIC_REQUIRE(detail::same_object(a, a) || true);   // usable in a constant context of its own
}

TEST_CASE("expr_store_t: lvalues by reference, rvalues by value", "[detail][838]") {
    using V = mat::dense2D<double>;
    STATIC_REQUIRE(std::is_same_v<detail::expr_store_t<V&>, const V&>);
    STATIC_REQUIRE(std::is_same_v<detail::expr_store_t<const V&>, const V&>);
    STATIC_REQUIRE(std::is_same_v<detail::expr_store_t<V>, V>);
    STATIC_REQUIRE(std::is_same_v<detail::expr_store_t<V&&>, V>);
    STATIC_REQUIRE(std::is_same_v<detail::expr_raw_t<const V&>, V>);
}

TEST_CASE("c_index and f_index", "[detail][838]") {
    STATIC_REQUIRE(detail::c_index::to_internal(3) == 3);
    STATIC_REQUIRE(detail::c_index::to_external(3) == 3);
    STATIC_REQUIRE(detail::f_index::to_internal(1) == 0);
    STATIC_REQUIRE(detail::f_index::to_external(0) == 1);
    // f_index 0 wraps; dense2D's bounds check rejects it like any bad index.
    STATIC_REQUIRE(detail::f_index::to_internal(0) == std::numeric_limits<std::size_t>::max());
    using fparams = mat::parameters<tag::row_major, detail::f_index>;
    mat::dense2D<double, fparams> A(2, 2);
    A(1, 1) = 5.0;
    A(2, 2) = 7.0;
    CHECK(A.data()[0] == 5.0);
    CHECK(A.data()[3] == 7.0);
    if constexpr (bounds_checking) {
        CHECK_THROWS_AS(A(0, 1), std::out_of_range);
        CHECK_THROWS_AS(A(1, 0), std::out_of_range);
        CHECK_THROWS_AS(A(3, 1), std::out_of_range);
    }
}

TEST_CASE("transposed_orientation", "[traits][838]") {
    STATIC_REQUIRE(std::is_same_v<traits::transposed_orientation_t<tag::row_major>, tag::col_major>);
    STATIC_REQUIRE(std::is_same_v<traits::transposed_orientation_t<tag::col_major>, tag::row_major>);
}

TEST_CASE("aligned allocation rejects a size that overflows", "[detail][838]") {
    // n * sizeof(T) used to wrap into a small allocation.
    const std::size_t huge = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
    CHECK_THROWS_AS(detail::allocate_aligned<double>(huge), std::bad_array_new_length);
    detail::aligned_allocator<double> alloc;
    CHECK_THROWS_AS(static_cast<void>(alloc.allocate(huge)), std::bad_array_new_length);
    double* p = detail::allocate_aligned<double>(4);
    REQUIRE(p != nullptr);
    detail::deallocate_aligned(p, 4);
}
