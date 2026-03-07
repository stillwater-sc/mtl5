#include <catch2/catch_test_macros.hpp>
#include <mtl/recursion/base_case_test.hpp>
#include <mtl/recursion/matrix_recursator.hpp>
#include <mtl/recursion/predefined_masks.hpp>
#include <mtl/mat/dense2D.hpp>
#include <cstddef>
#include <cstdint>
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
