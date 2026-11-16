#include <catch2/catch_test_macros.hpp>
#include <mtl/recursion/base_case_test.hpp>
#include <mtl/recursion/matrix_recursator.hpp>
#include <mtl/recursion/predefined_masks.hpp>
#include <mtl/mat/dense2D.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mtl;
using namespace mtl::recursion;

// -- Utility functions --------------------------------------------------

TEST_CASE("is_power_of_2", "[recursion][utility]") {
    REQUIRE(is_power_of_2(1));
    REQUIRE(is_power_of_2(2));
    REQUIRE(is_power_of_2(4));
    REQUIRE(is_power_of_2(8));
    REQUIRE(is_power_of_2(1024));
    REQUIRE_FALSE(is_power_of_2(0));
    REQUIRE_FALSE(is_power_of_2(3));
    REQUIRE_FALSE(is_power_of_2(6));
    REQUIRE_FALSE(is_power_of_2(100));
}

TEST_CASE("first_part returns largest power of 2 <= n", "[recursion][utility]") {
    REQUIRE(first_part(0) == 0);
    REQUIRE(first_part(1) == 1);
    REQUIRE(first_part(2) == 2);
    REQUIRE(first_part(3) == 2);
    REQUIRE(first_part(4) == 4);
    REQUIRE(first_part(5) == 4);
    REQUIRE(first_part(7) == 4);
    REQUIRE(first_part(8) == 8);
    REQUIRE(first_part(15) == 8);
    REQUIRE(first_part(16) == 16);
    REQUIRE(first_part(100) == 64);
}

TEST_CASE("outer_bound returns smallest power of 2 >= n", "[recursion][utility]") {
    REQUIRE(outer_bound(0) == 0);
    REQUIRE(outer_bound(1) == 1);
    REQUIRE(outer_bound(2) == 2);
    REQUIRE(outer_bound(3) == 4);
    REQUIRE(outer_bound(4) == 4);
    REQUIRE(outer_bound(5) == 8);
    REQUIRE(outer_bound(100) == 128);
}

// -- Recursator on small dense2D ----------------------------------------

TEST_CASE("recursator wraps entire matrix", "[recursion][recursator]") {
    mat::dense2D<double> A(8, 8);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            A(i, j) = static_cast<double>(i * 8 + j);

    recursator<mat::dense2D<double>> rec(A);
    REQUIRE(rec.num_rows() == 8);
    REQUIRE(rec.num_cols() == 8);
    REQUIRE_FALSE(rec.is_empty());

    // Access through recursator matches matrix
    REQUIRE(rec(0, 0) == A(0, 0));
    REQUIRE(rec(3, 5) == A(3, 5));
    REQUIRE(rec(7, 7) == A(7, 7));
}

TEST_CASE("recursator quadrant dimensions -- power of 2",
          "[recursion][recursator]") {
    mat::dense2D<double> A(8, 8);
    recursator<mat::dense2D<double>> rec(A);

    auto nw = rec.north_west();
    auto ne = rec.north_east();
    auto sw = rec.south_west();
    auto se = rec.south_east();

    // 8x8 splits into 4x4 quadrants
    REQUIRE(nw.num_rows() == 4);
    REQUIRE(nw.num_cols() == 4);
    REQUIRE(ne.num_rows() == 4);
    REQUIRE(ne.num_cols() == 4);
    REQUIRE(sw.num_rows() == 4);
    REQUIRE(sw.num_cols() == 4);
    REQUIRE(se.num_rows() == 4);
    REQUIRE(se.num_cols() == 4);
}

TEST_CASE("recursator quadrant dimensions -- non-power of 2",
          "[recursion][recursator]") {
    mat::dense2D<double> A(6, 10);
    recursator<mat::dense2D<double>> rec(A);

    auto nw = rec.north_west();
    auto ne = rec.north_east();
    auto sw = rec.south_west();
    auto se = rec.south_east();

    // 6 rows: first_part(6)=4, split=4 since 4!=6 -> NW/NE get 4 rows, SW/SE get 2
    REQUIRE(nw.num_rows() == 4);
    REQUIRE(sw.num_rows() == 2);

    // 10 cols: first_part(10)=8, split=8 since 8!=10 -> NW/SW get 8 cols, NE/SE get 2
    REQUIRE(nw.num_cols() == 8);
    REQUIRE(ne.num_cols() == 2);

    // Total coverage
    REQUIRE(nw.num_rows() + sw.num_rows() == 6);
    REQUIRE(nw.num_cols() + ne.num_cols() == 10);
}

TEST_CASE("recursator element access through quadrants",
          "[recursion][recursator]") {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 10 + j);

    recursator<mat::dense2D<double>> rec(A);
    auto nw = rec.north_west();
    auto ne = rec.north_east();
    auto sw = rec.south_west();
    auto se = rec.south_east();

    // NW covers A(0..1, 0..1)
    REQUIRE(nw(0, 0) == A(0, 0));
    REQUIRE(nw(1, 1) == A(1, 1));

    // NE covers A(0..1, 2..3)
    REQUIRE(ne(0, 0) == A(0, 2));
    REQUIRE(ne(1, 1) == A(1, 3));

    // SW covers A(2..3, 0..1)
    REQUIRE(sw(0, 0) == A(2, 0));
    REQUIRE(sw(1, 1) == A(3, 1));

    // SE covers A(2..3, 2..3)
    REQUIRE(se(0, 0) == A(2, 2));
    REQUIRE(se(1, 1) == A(3, 3));
}

// -- Base case tests ----------------------------------------------------

TEST_CASE("min_dim_test", "[recursion][base_case]") {
    mat::dense2D<double> A(8, 8);
    recursator<mat::dense2D<double>> rec(A);

    min_dim_test test4(4);
    REQUIRE_FALSE(test4(rec));  // min(8,8) = 8 > 4

    auto nw = rec.north_west();
    REQUIRE(test4(nw));  // min(4,4) = 4 <= 4
}

TEST_CASE("max_dim_test", "[recursion][base_case]") {
    mat::dense2D<double> A(8, 4);
    recursator<mat::dense2D<double>> rec(A);

    max_dim_test test8(8);
    REQUIRE(test8(rec));  // max(8,4) = 8 <= 8

    max_dim_test test4(4);
    REQUIRE_FALSE(test4(rec));  // max(8,4) = 8 > 4
}

TEST_CASE("max_dim_test_static", "[recursion][base_case]") {
    mat::dense2D<double> A(4, 4);
    recursator<mat::dense2D<double>> rec(A);

    max_dim_test_static<4> test4;
    REQUIRE(test4(rec));

    max_dim_test_static<2> test2;
    REQUIRE_FALSE(test2(rec));
}

// -- for_each recursive traversal ---------------------------------------

TEST_CASE("for_each visits all base cases", "[recursion][for_each]") {
    mat::dense2D<double> A(8, 8);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            A(i, j) = 0.0;

    recursator<mat::dense2D<double>> rec(A);

    // Count base cases and total elements covered
    std::size_t base_count = 0;
    std::size_t total_elements = 0;

    for_each(rec,
        [&](auto& sub) {
            ++base_count;
            total_elements += sub.num_rows() * sub.num_cols();
        },
        min_dim_test(2));

    // 8x8 with min_dim<=2 base case: each base case is 2x2
    // 8/2 = 4 divisions per axis = 4*4 = 16 base cases
    REQUIRE(base_count == 16);
    REQUIRE(total_elements == 64);  // all elements covered
}

TEST_CASE("for_each applies function to base cases",
          "[recursion][for_each]") {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0;

    recursator<mat::dense2D<double>> rec(A);

    // Set each base-case element to 1.0
    for_each(rec,
        [](auto& sub) {
            for (std::size_t i = 0; i < sub.num_rows(); ++i)
                for (std::size_t j = 0; j < sub.num_cols(); ++j)
                    sub(i, j) = 1.0;
        },
        min_dim_test(1));

    // Verify all elements were set
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE(A(i, j) == 1.0);
}

// -- Predefined masks --------------------------------------------------

TEST_CASE("predefined masks are valid", "[recursion][masks]") {
    // Morton Z mask should have alternating bits
    REQUIRE(morton_z_mask == UINT64_C(0x5555555555555555));
    REQUIRE(morton_mask == ~morton_z_mask);

    // Morton + complement should cover all bits
    REQUIRE((morton_z_mask | morton_mask) == ~std::uint64_t{0});
    REQUIRE((morton_z_mask & morton_mask) == std::uint64_t{0});

    // Doppled masks: row + col should cover all bits
    REQUIRE((doppled_2_row_mask | doppled_2_col_mask) == ~std::uint64_t{0});
    REQUIRE((doppled_4_row_mask | doppled_4_col_mask) == ~std::uint64_t{0});
    REQUIRE((doppled_16_row_mask | doppled_16_col_mask) == ~std::uint64_t{0});
}

TEST_CASE("lsb_mask helper", "[recursion][masks]") {
    REQUIRE(lsb_mask(0) == std::uint64_t{0});
    REQUIRE(lsb_mask(1) == std::uint64_t{1});
    REQUIRE(lsb_mask(4) == std::uint64_t{0xF});
    REQUIRE(lsb_mask(8) == std::uint64_t{0xFF});
}

// -- Issue 877 (QA sweep #783) -------------------------------------------

TEST_CASE("row_major_mask and the doppled masks: pinned values", "[recursion][masks][877]") {
    // The old mask test checked only row | col == ~0, which every mask passes.
    CHECK(row_major_mask(1) == UINT64_C(0xAAAAAAAAAAAAAAAA));
    CHECK(row_major_mask(2) == UINT64_C(0xCCCCCCCCCCCCCCCC));
    CHECK(row_major_mask(4) == UINT64_C(0xF0F0F0F0F0F0F0F0));
    CHECK(row_major_mask(7) == UINT64_C(0x80FE03F80FE03F80));    // the last block clipped at bit 63
    CHECK(row_major_mask(33) == UINT64_C(0xFFFFFFFE00000000));   // one block, bits 33..63
    CHECK(row_major_mask(64) == UINT64_C(0));
    CHECK(col_major_mask(7) == UINT64_C(0x7F01FC07F01FC07F));
    CHECK(doppled_2_row_mask == row_major_mask(1));
    CHECK(doppled_4_row_mask == UINT64_C(0xCCCCCCCCCCCCCCCC));
    CHECK(doppled_16_row_mask == UINT64_C(0xF0F0F0F0F0F0F0F0));
    CHECK(doppled_32_row_mask == UINT64_C(0x0F83E0F83E0F83E0));
    CHECK(doppled_64_row_mask == UINT64_C(0x0FC0FC0FC0FC0FC0));
    CHECK(doppled_128_row_mask == UINT64_C(0x80FE03F80FE03F80));
    CHECK(doppled_32_col_mask == ~doppled_32_row_mask);
    CHECK(doppled_64_col_mask == ~doppled_64_row_mask);
    CHECK(doppled_128_col_mask == ~doppled_128_row_mask);
    CHECK(lsb_mask(63) == UINT64_C(0x7FFFFFFFFFFFFFFF));
    CHECK(lsb_mask(64) == ~std::uint64_t{0});
    CHECK(lsb_mask(70) == ~std::uint64_t{0});
    // Constant expressions, as the doppled constants are.
    static_assert(row_major_mask(1) == UINT64_C(0xAAAAAAAAAAAAAAAA));
}

#if MTL5_HAS_EXCEPTIONS
TEST_CASE("row_major_mask(0) and col_major_mask(0) are rejected", "[recursion][masks][877]") {
    // The loop did not advance at k = 0: a run-time call hung, a constant
    // expression hit the compiler's step limit. In a constant expression it
    // is now a call to a non-constexpr function.
    CHECK_THROWS_AS(row_major_mask(0), std::invalid_argument);
    CHECK_THROWS_AS(col_major_mask(0), std::invalid_argument);
}

TEST_CASE("outer_bound beyond the largest power of 2 is rejected", "[recursion][utility][877]") {
    constexpr std::size_t top = (std::numeric_limits<std::size_t>::max() >> 1) + 1;
    CHECK(outer_bound(top) == top);
    CHECK(outer_bound(top - 1) == top);
    CHECK_THROWS_AS(outer_bound(top + 1), std::overflow_error);
    CHECK_THROWS_AS(outer_bound(std::numeric_limits<std::size_t>::max()), std::overflow_error);
}
#endif

namespace {

/// Visits of each element of an r x c matrix by for_each with `test`, and the
/// number of base cases.
template <typename Test>
std::pair<std::vector<int>, std::size_t> visits(std::size_t r, std::size_t c, const Test& test) {
    mat::dense2D<int> A(r, c);
    recursator<mat::dense2D<int>> rec(A);
    std::vector<int> seen(r * c, 0);
    std::size_t bases = 0;
    for_each(rec,
             [&](auto& sub) {
                 ++bases;
                 for (std::size_t i = 0; i < sub.num_rows(); ++i)
                     for (std::size_t j = 0; j < sub.num_cols(); ++j)
                         ++seen[(sub.row_offset() + i) * c + sub.col_offset() + j];
             },
             test);
    return {seen, bases};
}

} // namespace

TEST_CASE("for_each visits every element exactly once, also when the test rejects 1x1", "[recursion][for_each][877]") {
    // min_dim_test(0) rejects every region, 1x1 included: north_west() of a
    // 1x1 region is itself, and for_each recursed until the stack overflowed.
    // The old coverage test wrote 1.0, which a double visit can't change.
    for (const auto& [r, c] : std::vector<std::pair<std::size_t, std::size_t>>{
             {0, 0}, {0, 3}, {1, 1}, {1, 5}, {5, 1}, {7, 5}, {8, 8}, {3, 9}}) {
        INFO(r << " x " << c);
        for (const std::size_t t : {0u, 1u, 2u}) {
            const auto [seen, bases] = visits(r, c, min_dim_test(t));
            for (const int v : seen) CHECK(v == 1);
            if (t == 0) CHECK(bases == r * c);   // every base case is 1x1
        }
        const auto [seen, bases] = visits(r, c, max_dim_test(3));
        for (const int v : seen) CHECK(v == 1);
        (void)bases;
    }
}

TEST_CASE("recursator accessors, thresholds and a const matrix", "[recursion][recursator][877]") {
    mat::dense2D<int> A(5, 7);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 7; ++j)
            A(i, j) = static_cast<int>(10 * i + j);
    recursator<mat::dense2D<int>> rec(A);
    const auto se = rec.south_east();   // rows 4..4, cols 4..6
    CHECK(se.row_offset() == 4);
    CHECK(se.col_offset() == 4);
    CHECK(se.num_rows() == 1);
    CHECK(se.num_cols() == 3);
    CHECK(se.size() == 3);
    CHECK(!se.is_empty());
    CHECK(se(0, 2) == 46);              // the const operator()
    CHECK(&se.get_matrix() == &A);
    CHECK(min_dim_test(3).threshold == 3);
    CHECK(max_dim_test(4).threshold == 4);
    CHECK(max_dim_test_static<5>::threshold == 5);

    // A const matrix: it binds an lvalue, not a temporary, which it would
    // outlive (issue 877).
    const mat::dense2D<int>& CA = A;
    recursator<const mat::dense2D<int>> crec(CA);
    CHECK(crec(4, 6) == 46);
    static_assert(std::is_constructible_v<recursator<const mat::dense2D<int>>, const mat::dense2D<int>&>);
    static_assert(!std::is_constructible_v<recursator<const mat::dense2D<int>>, mat::dense2D<int>&&>);
    static_assert(!std::is_constructible_v<recursator<mat::dense2D<int>>, mat::dense2D<int>&&>);
}
