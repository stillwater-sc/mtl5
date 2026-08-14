// Tests for mtl::util::system_info -- self-identification for benchmark tagging.
//
// These are necessarily loose: the exact CPU/OS/compiler differ per CI runner,
// so we assert on invariants that must hold on any supported platform rather
// than on specific values.

#include <catch2/catch_test_macros.hpp>
#include <mtl/util/system_info.hpp>

#include <string>

using mtl::util::identify;
using mtl::util::simd_feature_list;
using mtl::util::to_keyvals;
using mtl::util::to_string;

TEST_CASE("system_info identifies the machine", "[util][system_info]") {
    const auto si = identify();

    SECTION("CPU fields are populated") {
        // brand is never empty: CPUID or an OS fallback, else the literal "unknown".
        REQUIRE_FALSE(si.cpu.brand.empty());
        REQUIRE_FALSE(si.cpu.arch.empty());
        REQUIRE(si.cpu.logical_cores >= 1);
    }

    SECTION("architecture is one of the known values") {
        const std::string& a = si.cpu.arch;
        REQUIRE((a == "x86_64" || a == "x86" || a == "aarch64" ||
                 a == "arm" || a == "unknown"));
    }

    SECTION("x86 implies a vendor and SSE2") {
        if (si.cpu.arch == "x86_64" || si.cpu.arch == "x86") {
            // Every x86-64 part reports a 12-char CPUID vendor and has SSE2 in
            // its baseline; a build target for x86-64 mandates SSE2.
            REQUIRE(si.cpu.vendor.size() == 12);
            REQUIRE(si.cpu.sse2);
            // Feature-flag implications must be internally consistent.
            if (si.cpu.avx2)    REQUIRE(si.cpu.avx);
            if (si.cpu.avx512f) REQUIRE(si.cpu.avx2);
        }
    }
}

TEST_CASE("compiler info is resolved at compile time", "[util][system_info]") {
    const auto c = identify().compiler;

    REQUIRE_FALSE(c.name.empty());
    REQUIRE(c.name != "unknown");           // one of MSVC/GCC/Clang/Apple Clang
    REQUIRE_FALSE(c.version.empty());
    // The suite is built at C++20 or later; _MSVC_LANG/__cplusplus must reflect it.
    REQUIRE(c.cpp_standard >= 202002L);
    REQUIRE((c.build_type == "Release" || c.build_type == "Debug"));
}

TEST_CASE("os info is populated", "[util][system_info]") {
    const auto os = identify().os;
    REQUIRE_FALSE(os.name.empty());
    REQUIRE((os.name == "Windows" || os.name == "Linux" || os.name == "macOS"));
    // version is best-effort but should be non-empty on all three supported OSes.
    REQUIRE_FALSE(os.version.empty());
}

TEST_CASE("string renderings are non-empty and consistent", "[util][system_info]") {
    const auto si = identify();

    const std::string human = to_string(si);
    const std::string kv    = to_keyvals(si);

    REQUIRE(human.find(si.cpu.brand) != std::string::npos);
    REQUIRE(kv.find("cpu_brand=" + si.cpu.brand) != std::string::npos);
    REQUIRE(kv.find("compiler_version=" + si.compiler.version) != std::string::npos);

    // simd_feature_list is a subset relationship: any listed flag is set.
    const std::string feats = simd_feature_list(si.cpu);
    if (feats.find("AVX2") != std::string::npos) REQUIRE(si.cpu.avx2);
    if (feats.find("FMA") != std::string::npos)  REQUIRE(si.cpu.fma);
}

TEST_CASE("build_isa names what this binary was compiled for", "[util][system_info]") {
    // The check that is NOT a restatement of the implementation: every ISA the
    // binary was compiled for must be one the CPU actually has. It could not be
    // otherwise -- a binary using an instruction the CPU lacks would have died
    // with SIGILL long before reaching this assertion -- so a mis-mapping (say,
    // reporting AVX512F for an /arch:AVX2 build) fails here on every machine
    // that lacks the feature, which is most of them.
    //
    // This is what makes the key worth putting in a sidecar: it cannot quietly
    // claim more than the run actually used.
    //
    // The precondition is that the binary RUNS where it was built, which holds
    // for a test suite (it is compiled and executed by the same job) and for the
    // benchmark builds this key exists to describe. A cross-built binary carried
    // to a weaker machine would trip this legitimately -- and should, because its
    // build_isa would then be describing instructions that machine cannot run.
    const auto si  = mtl::util::identify();
    const auto isa = mtl::util::build_isa_list();

    REQUIRE_FALSE(isa.empty());
    REQUIRE(isa.find("  ") == std::string::npos);   // no empty slots
    REQUIRE(isa.back() != ' ');

    if (si.cpu.arch == "x86_64") {
        if (isa.find("SSE2") != std::string::npos)    REQUIRE(si.cpu.sse2);
        if (isa.find("AVX") != std::string::npos)     REQUIRE(si.cpu.avx);
        if (isa.find("AVX2") != std::string::npos)    REQUIRE(si.cpu.avx2);
        if (isa.find("FMA") != std::string::npos)     REQUIRE(si.cpu.fma);
        if (isa.find("AVX512F") != std::string::npos) REQUIRE(si.cpu.avx512f);
        // x86-64 guarantees SSE2, so a build ISA that names nothing at all on
        // this arch means the macro mapping missed the baseline.
        REQUIRE(isa != "baseline");
    }

    // It reaches the sidecar, which is the only reason it exists.
    REQUIRE(to_keyvals(si).find("build_isa=" + isa) != std::string::npos);
}
