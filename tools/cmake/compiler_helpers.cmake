################################################################################################
# compiler_helpers.cmake
#
# Macros to auto-discover .cpp files and create test/example targets.
# Ported from Universal's compile_all() pattern, adapted for MTL5.

####
# macro to read all cpp files in a directory
# and create a target for each cpp file
#
# Parameters:
#   testing    - "true" to register as CTest, any other value to skip
#   prefix     - target name prefix (e.g. "mat", "example")
#   folder     - IDE folder for MSVC/Xcode (e.g. "Tests/mat", "Examples")
#   link_libs  - semicolon-separated list of libraries to link (e.g. "MTL5::mtl5;Catch2::Catch2WithMain")
#   ARGN       - list of source files
#
# Each source file "foo.cpp" produces target "${prefix}_foo"
macro(compile_all testing prefix folder link_libs)
    foreach(source ${ARGN})
        get_filename_component(test ${source} NAME_WE)
        string(REPLACE " " ";" new_source ${source})
        set(test_name ${prefix}_${test})
        add_executable(${test_name} ${new_source})
        target_link_libraries(${test_name} PRIVATE ${link_libs})
        set_target_properties(${test_name} PROPERTIES FOLDER ${folder})
        if(${testing} STREQUAL "true")
            if(MTL5_CMAKE_TRACE)
                message(STATUS "testing: ${test_name}")
            endif()
            add_test(NAME ${test_name} COMMAND ${test_name})
        endif()
    endforeach()
endmacro()
