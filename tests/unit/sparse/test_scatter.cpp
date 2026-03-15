#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/sparse/util/scatter.hpp>

#include <algorithm>

using namespace mtl::sparse::util;

TEST_CASE("Sparse accumulator basic scatter", "[sparse][scatter]") {
    sparse_accumulator<double> acc(5);

    acc.scatter(0, 1.0);
    acc.scatter(3, 2.0);
    acc.scatter(0, 0.5);  // accumulate

    REQUIRE(acc.nnz() == 2);
    REQUIRE(acc(0) == 1.5);
    REQUIRE(acc(3) == 2.0);
    REQUIRE(acc(1) == 0.0);  // untouched
    REQUIRE(acc.is_set(0));
    REQUIRE(acc.is_set(3));
    REQUIRE(!acc.is_set(1));
}

TEST_CASE("Sparse accumulator store overwrites", "[sparse][scatter]") {
    sparse_accumulator<double> acc(3);

    acc.store(1, 5.0);
    acc.store(1, 7.0);  // overwrite, not accumulate

    REQUIRE(acc(1) == 7.0);
    REQUIRE(acc.nnz() == 1);
}

TEST_CASE("Sparse accumulator clear resets without touching memory", "[sparse][scatter]") {
    sparse_accumulator<double> acc(4);

    acc.scatter(0, 1.0);
    acc.scatter(2, 3.0);
    REQUIRE(acc.nnz() == 2);

    acc.clear();
    REQUIRE(acc.nnz() == 0);
    REQUIRE(!acc.is_set(0));
    REQUIRE(!acc.is_set(2));
    REQUIRE(acc(0) == 0.0);

    // Scatter again after clear
    acc.scatter(1, 9.0);
    REQUIRE(acc.nnz() == 1);
    REQUIRE(acc(1) == 9.0);
}

TEST_CASE("Sparse accumulator mutable access", "[sparse][scatter]") {
    sparse_accumulator<double> acc(5);

    acc[2] = 10.0;
    acc[2] += 5.0;

    REQUIRE(acc(2) == 15.0);
    REQUIRE(acc.is_set(2));
}

TEST_CASE("Sparse accumulator indices tracking", "[sparse][scatter]") {
    sparse_accumulator<double> acc(6);

    acc.scatter(4, 1.0);
    acc.scatter(1, 2.0);
    acc.scatter(4, 3.0);  // duplicate, should not add to indices again

    auto& idx = acc.indices();
    REQUIRE(idx.size() == 2);

    // Both 4 and 1 should appear (order depends on insertion order)
    std::vector<std::size_t> sorted_idx(idx.begin(), idx.end());
    std::sort(sorted_idx.begin(), sorted_idx.end());
    REQUIRE(sorted_idx[0] == 1);
    REQUIRE(sorted_idx[1] == 4);
}
