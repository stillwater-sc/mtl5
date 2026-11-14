#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/identity2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/operators.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace mtl;

TEST_CASE("coordinate2D: insert and access", "[mat][coordinate2D]") {
    mat::coordinate2D<double> coo(3, 3);
    coo.insert(0, 0, 4.0);
    coo.insert(0, 1, 1.0);
    coo.insert(1, 0, 1.0);
    coo.insert(1, 1, 4.0);
    coo.insert(2, 2, 4.0);

    REQUIRE(coo.num_rows() == 3);
    REQUIRE(coo.num_cols() == 3);
    REQUIRE(coo.nnz() == 5);

    REQUIRE_THAT(coo(0, 0), Catch::Matchers::WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(coo(0, 1), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(coo(2, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("coordinate2D: duplicate accumulation", "[mat][coordinate2D]") {
    mat::coordinate2D<double> coo(2, 2);
    coo.insert(0, 0, 1.0);
    coo.insert(0, 0, 2.0);  // duplicate
    coo.insert(1, 1, 5.0);

    auto crs = coo.compress();
    // Duplicates should be summed
    REQUIRE_THAT(crs(0, 0), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(crs(1, 1), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE(crs.nnz() == 2);
}

TEST_CASE("coordinate2D: compress to CRS", "[mat][coordinate2D]") {
    mat::coordinate2D<double> coo(3, 3);
    // Insert out of order
    coo.insert(2, 2, 3.0);
    coo.insert(0, 0, 1.0);
    coo.insert(1, 1, 2.0);
    coo.insert(0, 2, 0.5);

    auto crs = coo.compress();
    REQUIRE(crs.num_rows() == 3);
    REQUIRE(crs.num_cols() == 3);
    REQUIRE(crs.nnz() == 4);

    REQUIRE_THAT(crs(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(crs(0, 2), Catch::Matchers::WithinAbs(0.5, 1e-10));
    REQUIRE_THAT(crs(1, 1), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(crs(2, 2), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(crs(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10)); // absent
}

TEST_CASE("coordinate2D: sort", "[mat][coordinate2D]") {
    mat::coordinate2D<double> coo(3, 3);
    coo.insert(2, 0, 1.0);
    coo.insert(0, 2, 2.0);
    coo.insert(1, 1, 3.0);

    REQUIRE(!coo.is_sorted());
    coo.sort();
    REQUIRE(coo.is_sorted());

    // After sorting, entries should be in row-major order
    const auto& entries = coo.entries();
    REQUIRE(std::get<0>(entries[0]) == 0);
    REQUIRE(std::get<0>(entries[1]) == 1);
    REQUIRE(std::get<0>(entries[2]) == 2);
}

TEST_CASE("coordinate2D: sparse matvec via compress", "[mat][coordinate2D]") {
    mat::coordinate2D<double> coo(3, 3);
    coo.insert(0, 0, 2.0);
    coo.insert(1, 1, 3.0);
    coo.insert(2, 2, 4.0);

    auto crs = coo.compress();
    vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    auto y = crs * x;

    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(6.0, 1e-10));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(12.0, 1e-10));
}

TEST_CASE("identity2D: element access", "[mat][identity2D]") {
    mat::identity2D<double> I(3);

    REQUIRE(I.num_rows() == 3);
    REQUIRE(I.num_cols() == 3);

    REQUIRE_THAT(I(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(I(1, 1), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(I(2, 2), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(I(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(I(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// -- compress(): edge cases, duplicate order, index checks (#825) -----------

TEST_CASE("coordinate2D::compress on an empty matrix and on empty rows", "[mat][coordinate2D][825]") {
    const mat::coordinate2D<double> none(3, 4);
    const auto A = none.compress();
    REQUIRE(A.num_rows() == 3);
    REQUIRE(A.num_cols() == 4);
    REQUIRE(A.nnz() == 0);
    REQUIRE(A.ref_major() == std::vector<std::size_t>{0, 0, 0, 0});

    const mat::coordinate2D<double> zero(0, 0);
    REQUIRE(zero.compress().nnz() == 0);

    mat::coordinate2D<double> gaps(5, 2);   // rows 0, 2 and 4 empty
    gaps.insert(3, 0, 2.0);
    gaps.insert(1, 1, 1.0);
    const auto G = gaps.compress();
    REQUIRE(G.ref_major() == std::vector<std::size_t>{0, 0, 1, 1, 2, 2});
    REQUIRE(G(1, 1) == 1.0);
    REQUIRE(G(3, 0) == 2.0);
}

namespace {
/// Values whose float sum depends on the order they are added in.
float order_sensitive(std::size_t i) {
    const float mags[] = {1e7f, 3.0f, -1e7f, 0.5f, 7e6f, 1.25f, -3e6f, 9.0f};
    return mags[i % 8] * float(1 + i % 5);
}
} // namespace

TEST_CASE("coordinate2D duplicates are summed in insertion order", "[mat][coordinate2D][825]") {
    // Enough entries that std::sort's introsort partitions them, rather than
    // the insertion sort it uses for short ranges, which happens to be stable.
    constexpr std::size_t n = 96;
    mat::coordinate2D<float> coo(2, 2);
    float in_order00 = 0, in_order11 = 0;
    bool first00 = true, first11 = true;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = order_sensitive(i);
        if (i % 3 == 0) {
            coo.insert(1, 1, v);
            in_order11 = first11 ? v : in_order11 + v;
            first11 = false;
        } else {
            coo.insert(0, 0, v);
            in_order00 = first00 ? v : in_order00 + v;
            first00 = false;
        }
    }
    SECTION("compress() of unsorted entries") {
        const auto A = coo.compress();
        REQUIRE(A.nnz() == 2);
        REQUIRE(A(0, 0) == in_order00);
        REQUIRE(A(1, 1) == in_order11);
    }
    SECTION("sort() keeps insertion order within a position, and compress() uses it") {
        coo.sort();
        REQUIRE(coo.is_sorted());
        std::size_t k = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (i % 3 == 0) continue;
            INFO("entry " << k << " of (0, 0), inserted " << i);
            REQUIRE(std::get<2>(coo.entries()[k]) == order_sensitive(i));
            ++k;
        }
        const auto A = coo.compress();
        REQUIRE(A(0, 0) == in_order00);
        REQUIRE(A(1, 1) == in_order11);
    }
}

TEST_CASE("coordinate2D::compress rejects out-of-range entries", "[mat][coordinate2D][825]") {
    // insert() does not check; compress() does, in every build.
    mat::coordinate2D<double> bad_row(2, 2);
    bad_row.insert(0, 0, 1.0);
    bad_row.insert(2, 0, 1.0);   // was starts[3]++ on a 3-element vector
    REQUIRE_THROWS_AS(bad_row.compress(), std::out_of_range);
    mat::coordinate2D<double> bad_col(2, 2);
    bad_col.insert(0, 5, 1.0);   // was stored, and read back
    REQUIRE_THROWS_AS(bad_col.compress(), std::out_of_range);
    mat::coordinate2D<double> edge(2, 2);
    edge.insert(1, 1, 1.0);
    REQUIRE_NOTHROW(edge.compress());
}

TEST_CASE("coordinate2D::compress with a signed size_type rejects negative indices", "[mat][coordinate2D][825]") {
    using signed_params = mat::parameters<tag::row_major, detail::c_index, mat::non_fixed::dimensions, tag::on_heap, int>;
    mat::coordinate2D<double, signed_params> bad_row(2, 2);
    bad_row.insert(-2, 0, 1.0);   // was starts[-1]++
    REQUIRE_THROWS_AS(bad_row.compress(), std::out_of_range);
    mat::coordinate2D<double, signed_params> bad_col(2, 2);
    bad_col.insert(0, -1, 1.0);
    REQUIRE_THROWS_AS(bad_col.compress(), std::out_of_range);
    mat::coordinate2D<double, signed_params> ok(2, 2);
    ok.insert(1, 0, 4.0);
    const auto A = ok.compress();
    REQUIRE(A(1, 0) == 4.0);
}
