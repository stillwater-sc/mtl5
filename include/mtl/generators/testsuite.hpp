#pragma once
// MTL5 -- named test-matrix catalog migrated from Universal's blas/matrices suite.
//
// A Universal-free catalog of well-known reference matrices (SuiteSparse and
// textbook problems) with published condition numbers, so studies can be run on
// the SAME named problems the Universal originals used -- complementing the
// parametric generators (hilbert/lehmer/minij/...) which give controllable
// conditioning at arbitrary size. Migrated from
// include/sw/blas/matrices/{testsuite,*}.hpp (universal#1204, #1210).
//
// The matrices are stored as compact Matrix Market .mtx files (sparse stored
// sparse) under the data/testsuite directory and read via mtl::io::mm_read_dense
// into a dense2D<double>. `data_dir()` returns the build-tree path baked in by
// CMake; installed consumers pass an explicit directory.
//
// Usage:
//   #include <mtl/generators/testsuite.hpp>
//   auto A = mtl::testsuite::by_name("bcsstk01");        // dense2D<double>
//   double k = mtl::testsuite::kappa("bcsstk01");        // published cond number
//   for (const auto& n : mtl::testsuite::names()) { ... }

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>

#include <mtl/mat/dense2D.hpp>
#include <mtl/io/matrix_market.hpp>

#if __has_include(<mtl/testsuite_config.hpp>)
#  include <mtl/testsuite_config.hpp>   // MTL5_TESTSUITE_DATA_DIR (CMake-configured, in-tree builds)
#endif

namespace mtl::testsuite {

/// Published 2-norm condition numbers for the catalog (from the SuiteSparse /
/// Universal originals). rump6x6ill and wilk21 come from their header comments
/// (cond(est) ~ 1e16, and ~42 respectively); the rest from Universal's kappa().
inline const std::unordered_map<std::string, double>& kappa_table() {
    static const std::unordered_map<std::string, double> k = {
        {"lambers_well", 1.0e1},               {"lambers_ill", 1.869050824603144e8},
        {"rump6x6ill",   1.0e16},              {"faires74x3",  1.5999e4},
        {"wilk21",       4.2e1},               {"Trefethen_20", 6.308860e1},
        {"pores_1",      1.812616e6},          {"gre_343",     1.119763e2},
        {"bcsstk01",     8.8234e5},            {"bcsstk03",    6.791333e6},
        {"bcsstk04",     2.292466e6},          {"bcsstk05",    1.428114e4},
        {"bcsstk22",     1.107165e5},          {"steam1",      2.827501e7},
        {"steam3",       5.51e10},             {"fs_183_1",    1.5129e13},
        {"fs_183_3",     1.5129e13},           {"west0132",    4.2e11},
        {"west0167",     2.827e7},             {"saylr1",      7.780581e8},
    };
    return k;
}

/// The catalog entries (roughly ordered small -> large).
inline std::vector<std::string> names() {
    return { "lambers_well", "lambers_ill", "rump6x6ill", "faires74x3", "wilk21",
             "Trefethen_20", "pores_1", "gre_343", "bcsstk05", "bcsstk22",
             "bcsstk01", "bcsstk03", "bcsstk04", "steam3", "steam1",
             "fs_183_1", "fs_183_3", "west0132", "west0167", "saylr1" };
}

/// Published condition number for `name`; throws if the name is not in the catalog.
inline double kappa(const std::string& name) {
    auto it = kappa_table().find(name);
    if (it == kappa_table().end())
        throw std::runtime_error("mtl::testsuite::kappa: unknown matrix '" + name + "'");
    return it->second;
}

/// The default catalog data directory.
///
/// Header-only consumers frequently pull MTL5 in by adding `include/` to the
/// search path only (FetchContent_Populate, `-Iinclude`), so MTL5's CMake never
/// runs and the configured macro is absent. We therefore derive the data dir
/// from THIS header's own location (`__FILE__` is the absolute compile-time path
/// under CMake): include/mtl/generators/testsuite.hpp -> ../../../data/testsuite.
/// The CMake-configured macro, when present, takes precedence. Installed
/// consumers (source tree gone) should pass an explicit `dir` to by_name().
inline std::string data_dir() {
#ifdef MTL5_TESTSUITE_DATA_DIR
    return MTL5_TESTSUITE_DATA_DIR;
#else
    namespace fs = std::filesystem;
    return fs::weakly_canonical(
        fs::path(__FILE__).parent_path() / ".." / ".." / ".." / "data" / "testsuite").string();
#endif
}

/// Load a catalog matrix by name as a dense2D<double>, reading its .mtx from
/// `dir` (default: the CMake-configured catalog directory). Throws if the name is
/// unknown or the file cannot be read.
inline mat::dense2D<double> by_name(const std::string& name,
                                    const std::string& dir = data_dir()) {
    if (kappa_table().find(name) == kappa_table().end())
        throw std::runtime_error("mtl::testsuite::by_name: unknown matrix '" + name + "'");
    return mtl::io::mm_read_dense<double>(dir + "/" + name + ".mtx");
}

} // namespace mtl::testsuite
