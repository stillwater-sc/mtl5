# Regression test for the provenance string escaping (#442, #443).
#
# Tests the CONTRACT end to end rather than the substitution in isolation: feed a
# hostile value through GitInfo.cmake, compile the generated header, run it, and
# require the value to come back BYTE FOR BYTE. A test that only checked the
# header parses would pass on an escaping that silently mangles the value, which
# is the failure mode this feature exists to prevent -- an earlier version
# discarded \r and folded \n to a space and would have sailed through such a test.
#
# Expects: SRC_DIR, WORK_DIR, CXX

set(_template "${SRC_DIR}/include/mtl/build_info.hpp.in")
set(_script   "${SRC_DIR}/cmake/GitInfo.cmake")
file(MAKE_DIRECTORY "${WORK_DIR}")

# Every hazard at once: Windows backslashes (which produce bogus \f and \b
# escapes), an embedded quote (which terminates the literal early), and a CRLF
# plus a bare newline (which the first implementation destroyed).
set(_hostile "/I C:\\foo\\bar /D S=\"q\"\r\nsecond line\nthird -Wall")

execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DSRC_DIR=${SRC_DIR}
            -DTEMPLATE=${_template}
            -DOUTPUT=${WORK_DIR}/build_info.hpp
            -DCXX_FLAGS=${_hostile}
            -DCMAKE_TYPE=Release
            -P ${_script}
    RESULT_VARIABLE _rc
    OUTPUT_QUIET ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "GitInfo.cmake failed: ${_err}")
endif()

# Compare BYTES, not captured text. execute_process strips carriage returns from
# captured output (verified: a program emitting "A\r\nB" is captured as 3 chars),
# so comparing its stdout would have failed a correct escaping and passed one that
# dropped CR -- the test would have been measuring CMake, not the code. The probe
# writes the macro to a file in binary mode and both sides are compared as hex.
file(WRITE "${WORK_DIR}/probe.cpp"
"#include \"build_info.hpp\"\n"
"#include <cstdio>\n"
"int main() {\n"
"    std::FILE* f = std::fopen(\"${WORK_DIR}/got.bin\", \"wb\");\n"
"    if (!f) return 1;\n"
"    std::fputs(MTL5_BUILD_CXX_FLAGS, f);\n"
"    std::fclose(f);\n"
"    return 0;\n"
"}\n")

execute_process(
    COMMAND ${CXX} -std=c++20 -I ${WORK_DIR} ${WORK_DIR}/probe.cpp -o ${WORK_DIR}/probe
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    file(READ "${WORK_DIR}/build_info.hpp" _hdr)
    message(FATAL_ERROR
        "generated header does not compile -- escaping is broken.\n${_err}\n--- header ---\n${_hdr}")
endif()

execute_process(COMMAND ${WORK_DIR}/probe RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "probe failed to run")
endif()

string(HEX "${_hostile}" _expected_hex)
file(READ "${WORK_DIR}/got.bin" _got_hex HEX)
if(NOT _got_hex STREQUAL _expected_hex)
    message(FATAL_ERROR
        "provenance value was altered by escaping.\nexpected hex: ${_expected_hex}\ngot hex:      ${_got_hex}")
endif()

message(STATUS "provenance escaping: value round-trips byte for byte")
