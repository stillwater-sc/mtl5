# Helper function to add a Catch2 test
function(mtl5_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE MTL5::mtl5 Catch2::Catch2WithMain)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Concept tests
mtl5_add_test(test_concepts concepts/test_concepts.cpp)

# Math tests
mtl5_add_test(test_zero_one math/test_zero_one.cpp)

# Detail tests
mtl5_add_test(test_contiguous_memory_block detail/test_contiguous_memory_block.cpp)

# Matrix tests
mtl5_add_test(test_dense2D mat/test_dense2D.cpp)

# Vector tests
mtl5_add_test(test_dense_vector vec/test_dense_vector.cpp)

# Operation tests
mtl5_add_test(test_dot operation/test_dot.cpp)

# ITL tests
mtl5_add_test(test_cg itl/test_cg.cpp)
