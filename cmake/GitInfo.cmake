# Regenerate build_info.hpp at BUILD time (#442).
#
# Run in script mode from a custom target, so the recorded commit follows the
# working tree rather than whenever CMake last configured. configure_file leaves
# the file untouched when the rendered content is identical, so this does not
# force rebuilds of anything that includes it.
#
# Expects: SRC_DIR, BIN_DIR, TEMPLATE, OUTPUT, CXX_FLAGS, CMAKE_TYPE
find_package(Git QUIET)

set(MTL5_BUILD_GIT_COMMIT "unknown")
set(MTL5_BUILD_GIT_DIRTY  "unknown")

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${SRC_DIR}"
        OUTPUT_VARIABLE _sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _sha_rc
        ERROR_QUIET)
    if(_sha_rc EQUAL 0 AND _sha)
        set(MTL5_BUILD_GIT_COMMIT "${_sha}")
        # --porcelain is empty exactly when the tree is clean; it covers staged,
        # unstaged and untracked-but-tracked-path changes, which `diff --quiet`
        # alone would miss.
        execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
            WORKING_DIRECTORY "${SRC_DIR}"
            OUTPUT_VARIABLE _status
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _status_rc
            ERROR_QUIET)
        if(_status_rc EQUAL 0)
            if(_status STREQUAL "")
                set(MTL5_BUILD_GIT_DIRTY "0")
            else()
                set(MTL5_BUILD_GIT_DIRTY "1")
            endif()
        endif()
    endif()
endif()

set(MTL5_BUILD_CXX_FLAGS  "${CXX_FLAGS}")
set(MTL5_BUILD_CMAKE_TYPE "${CMAKE_TYPE}")
configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
