// Tests for the named test-matrix catalog migrated from Universal (universal#1210).
#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/testsuite.hpp>
#include <mtl/operation/spectral_properties.hpp>

#include <set>

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

TEST_CASE("testsuite: every catalog matrix loads with its condition number", "[generators][testsuite]") {
    // All 20 are loaded and their 2-norm condition number computed by the SVD,
    // so a corrupt or truncated .mtx fails here; the 7 loaded above were
    // checked only for shape (#844). The reference column is that computed
    // value (4 significant digits). rump6x6ill is beyond 1/eps, where the SVD
    // is not accurate, so it only has to be that large.
    struct Case { const char* name; std::size_t n; double kappa2; };
    const Case cases[] = {
        {"lambers_well", 2, 1.000e1},   {"lambers_ill", 2, 1.869e8},  {"rump6x6ill", 6, 0.0},
        {"faires74x3", 3, 1.424e4},     {"wilk21", 21, 4.234e1},      {"Trefethen_20", 20, 6.309e1},
        {"pores_1", 30, 1.813e6},       {"gre_343", 343, 1.120e2},    {"bcsstk05", 153, 1.428e4},
        {"bcsstk22", 138, 1.107e5},     {"bcsstk01", 48, 8.823e5},    {"bcsstk03", 112, 6.792e6},
        {"bcsstk04", 132, 2.292e6},     {"steam3", 80, 4.994e10},     {"steam1", 240, 2.828e7},
        {"fs_183_1", 183, 2.193e13},    {"fs_183_3", 183, 3.266e13},  {"west0132", 132, 4.206e11},
        {"west0167", 167, 4.785e10},    {"saylr1", 238, 7.781e8},
    };
    // Published values that are not this 2-norm condition number:
    //   faires74x3 (15999) is the infinity-norm one (16000); steam3 (5.51e10)
    //   and fs_183_1 (1.5129e13) are 1-norm ones; fs_183_3 repeats fs_183_1's,
    //   and west0167 (2.827e7) repeats steam1's; rump6x6ill's is an estimate.
    const std::set<std::string> not_2norm = {"rump6x6ill", "faires74x3", "steam3",
                                             "fs_183_1", "fs_183_3", "west0167"};
    std::set<std::string> seen;
    for (const auto& c : cases) {
        CAPTURE(c.name);
        seen.insert(c.name);
        const auto A = testsuite::by_name(c.name);
        REQUIRE(num_rows(A) == c.n);
        REQUIRE(num_cols(A) == c.n);
        const double k = condition_number(A);
        if (c.kappa2 == 0.0) {
            REQUIRE(k > 1e15);
            continue;
        }
        REQUIRE_THAT(k, Catch::Matchers::WithinRel(c.kappa2, 1e-3));
        if (!not_2norm.count(c.name))
            REQUIRE_THAT(testsuite::kappa(c.name), Catch::Matchers::WithinRel(k, 1e-2));
    }
    const auto names = testsuite::names();
    REQUIRE(seen == std::set<std::string>(names.begin(), names.end()));
}

TEST_CASE("testsuite: by_name joins the directory and the file name portably", "[generators][testsuite]") {
    // with a trailing separator, and with the default directory
    const auto A = testsuite::by_name("lambers_well", testsuite::data_dir() + "/");
    REQUIRE(num_rows(A) == 2u);
}
