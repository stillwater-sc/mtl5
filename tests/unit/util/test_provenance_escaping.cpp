// Provenance escaping: does a value survive the trip into build_info.hpp? (#443)
//
// The values cmake/GitInfo.cmake records are pasted between quotes in a C++
// string literal, so a perfectly ordinary Windows flag breaks the generated
// header twice at once: `/I C:\foo\bar /D S="q"` yields `\f` and `\b` as bogus
// escapes AND terminates the literal early. The first implementation kept the
// header VALID by destroying the value instead -- dropping carriage returns and
// folding newlines to a space -- which is wrong provenance rather than missing
// provenance, the exact failure this feature exists to prevent.
//
// So the assertion is byte identity, not "it compiles". The probe header is
// generated at configure time from a value carrying every hazard at once (see
// tests/unit/CMakeLists.txt), and the expected bytes reach this file through
// CMake's string(HEX), a path with no backslash handling of its own -- a broken
// escaper cannot drag the expectation along with it.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <sstream>
#include <string>

// Generated into the build tree; not the real mtl/build_info.hpp, which carries
// this build's own flags and would prove nothing.
#include "provenance_probe/build_info.hpp"
#include "provenance_probe/expected.hpp"

namespace {

/// Failure messages here are about bytes that do not print (CR, LF) or that the
/// terminal eats, so show hex.
std::string hex(const std::string& s) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        out += digits[c >> 4];
        out += digits[c & 0xF];
        out += ' ';
    }
    return out;
}

const std::string& expected() {
    static const unsigned char bytes[] = {MTL5_TEST_PROVENANCE_EXPECTED_BYTES};
    static const std::string s(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    return s;
}

} // namespace

TEST_CASE("a hostile provenance value round-trips byte for byte", "[util][provenance]") {
    const std::string got(MTL5_BUILD_CXX_FLAGS);
    INFO("expected: " << hex(expected()) << "\ngot:      " << hex(got));
    REQUIRE(got.size() == expected().size());
    REQUIRE(got == expected());
}

TEST_CASE("each escaping hazard is individually accounted for", "[util][provenance]") {
    // The byte comparison above is the contract; these name WHICH hazard broke,
    // because a single mismatched length says nothing about the cause.
    const std::string got(MTL5_BUILD_CXX_FLAGS);

    SECTION("backslashes are not consumed as escapes") {
        REQUIRE(got.find("C:\\foo\\bar") != std::string::npos);
    }
    SECTION("an embedded quote does not truncate the value") {
        REQUIRE(got.find("S=\"q\"") != std::string::npos);
    }
    SECTION("CRLF is preserved, not stripped to LF") {
        REQUIRE(got.find("\r\n") != std::string::npos);
    }
    SECTION("a bare newline is preserved, not folded to a space") {
        REQUIRE(got.find("second\nthird") != std::string::npos);
    }
}

TEST_CASE("the generated header reports the build type it was given",
          "[util][provenance]") {
    // Same escaping path, a value with no hazards in it: the escaper must leave
    // ordinary strings exactly alone.
    REQUIRE(std::string(MTL5_BUILD_CMAKE_TYPE) == "Release");
}
