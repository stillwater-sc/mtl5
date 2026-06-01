// Tests for over-aligned storage + aligned allocator (#84, epic #82).
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/aligned_allocator.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>   // std::move
#include <vector>

namespace {
bool is_aligned(const void* p, std::size_t a) {
    return (reinterpret_cast<std::uintptr_t>(p) % a) == 0;
}
}

TEST_CASE("block_alignment is >= 64 and respects natural alignment", "[detail][simd]") {
    using mtl::detail::block_alignment;
    STATIC_REQUIRE(block_alignment<float>  >= 64);
    STATIC_REQUIRE(block_alignment<double> >= 64);
    struct alignas(128) Over { double x; };
    STATIC_REQUIRE(block_alignment<Over> == 128);   // larger natural alignment wins
}

TEST_CASE("allocate_aligned: aligned, value-initialized, round-trips", "[detail][simd]") {
    using mtl::detail::allocate_aligned;
    using mtl::detail::deallocate_aligned;

    CHECK(allocate_aligned<double>(0) == nullptr);

    for (std::size_t n : std::vector<std::size_t>{1, 3, 7, 64, 1000}) {
        double* p = allocate_aligned<double>(n);
        REQUIRE(p != nullptr);
        CHECK(is_aligned(p, 64));
        for (std::size_t i = 0; i < n; ++i) CHECK(p[i] == 0.0);   // value-initialized
        for (std::size_t i = 0; i < n; ++i) p[i] = double(i);
        for (std::size_t i = 0; i < n; ++i) CHECK(p[i] == double(i));
        deallocate_aligned(p, n);
    }
}

TEST_CASE("aligned_allocator backs an over-aligned std::vector", "[detail][simd]") {
    std::vector<double, mtl::detail::aligned_allocator<double>> v(257, 1.5);
    REQUIRE(v.size() == 257);
    CHECK(is_aligned(v.data(), 64));
    CHECK(v[0] == 1.5);
    CHECK(v[256] == 1.5);
}

TEST_CASE("dense2D<double> heap buffer is 64B-aligned", "[mat][detail][simd]") {
    for (std::size_t n : std::vector<std::size_t>{4, 5, 13, 64, 257}) {
        mtl::mat::dense2D<double> A(n, n);
        REQUIRE(A.data() != nullptr);
        CHECK(is_aligned(A.data(), 64));
    }
    mtl::mat::dense2D<float> Af(100, 100);
    CHECK(is_aligned(Af.data(), 64));
}

TEST_CASE("dense_vector<double> heap buffer is 64B-aligned", "[vec][detail][simd]") {
    for (std::size_t n : std::vector<std::size_t>{1, 7, 64, 1025}) {
        mtl::vec::dense_vector<double> v(n);
        REQUIRE(v.data() != nullptr);
        CHECK(is_aligned(v.data(), 64));
    }
}

TEST_CASE("alignment survives copy/move/realloc (still owns aligned storage)", "[mat][detail][simd]") {
    mtl::mat::dense2D<double> A(33, 17);
    A(0, 0) = 3.0;
    auto B = A;                       // copy
    CHECK(is_aligned(B.data(), 64));
    CHECK(B(0, 0) == 3.0);
    auto C = std::move(A);            // move (steals pointer; already aligned)
    CHECK(is_aligned(C.data(), 64));
    CHECK(C(0, 0) == 3.0);
}
