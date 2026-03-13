# MTL5 Benchmark Suite -- Policy-based backend comparison
# Built when MTL5_BUILD_BENCHMARKS=ON

# Benchmark harness headers are included via <benchmarks/harness/...>
# so we add the parent of this directory to the include path.

add_executable(bench_all bench_all.cpp)
target_link_libraries(bench_all PRIVATE MTL5::mtl5)
target_include_directories(bench_all PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
set_target_properties(bench_all PROPERTIES FOLDER "Benchmarks")

# Future: individual focused benchmarks can be added here
# add_executable(bench_gemm bench_gemm.cpp)
# target_link_libraries(bench_gemm PRIVATE MTL5::mtl5)
# target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
