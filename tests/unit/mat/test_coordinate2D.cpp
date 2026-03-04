#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/identity2D.hpp>
#include <mtl/operation/operators.hpp>

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
