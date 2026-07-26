// Tests for the named test-matrix catalog migrated from Universal (universal#1210).
#include <catch2/catch_test_macros.hpp>

#include <mtl/generators/testsuite.hpp>

#include <string>

using namespace mtl;

TEST_CASE("testsuite: catalog names have condition numbers", "[generators][testsuite]") {
    const auto ns = testsuite::names();
    REQUIRE(ns.size() == 20u);
    for (const auto& n : ns) {
        REQUIRE(testsuite::kappa(n) > 0.0);   // every listed matrix has a published kappa
    }
}

TEST_CASE("testsuite: by_name loads matrices with expected shape", "[generators][testsuite]") {
    struct Case { const char* name; unsigned rows; unsigned cols; };
    const Case cases[] = {
        {"lambers_well", 2, 2}, {"faires74x3", 3, 3}, {"wilk21", 21, 21},
        {"Trefethen_20", 20, 20}, {"bcsstk01", 48, 48}, {"west0132", 132, 132},
        {"gre_343", 343, 343},
    };
    for (const auto& c : cases) {
        auto A = testsuite::by_name(c.name);
        REQUIRE(num_rows(A) == c.rows);
        REQUIRE(num_cols(A) == c.cols);
    }
}

TEST_CASE("testsuite: published condition numbers", "[generators][testsuite]") {
    REQUIRE(testsuite::kappa("lambers_well") == 10.0);
    REQUIRE(testsuite::kappa("gre_343")      > 1.0e2);
    REQUIRE(testsuite::kappa("fs_183_1")     > 1.0e12);   // notoriously ill-conditioned
}

TEST_CASE("testsuite: unknown name throws", "[generators][testsuite]") {
    REQUIRE_THROWS_AS(testsuite::kappa("does_not_exist"), std::runtime_error);
    REQUIRE_THROWS_AS(testsuite::by_name("does_not_exist"), std::runtime_error);
}
