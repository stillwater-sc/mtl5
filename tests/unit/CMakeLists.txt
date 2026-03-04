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
mtl5_add_test(test_compressed2D mat/test_compressed2D.cpp)

# Vector tests
mtl5_add_test(test_dense_vector vec/test_dense_vector.cpp)

# Operation tests
mtl5_add_test(test_dot operation/test_dot.cpp)
mtl5_add_test(test_norms operation/test_norms.cpp)
mtl5_add_test(test_vector_ops operation/test_vector_ops.cpp)
mtl5_add_test(test_matrix_ops operation/test_matrix_ops.cpp)
mtl5_add_test(test_trisolve operation/test_trisolve.cpp)
mtl5_add_test(test_lu operation/test_lu.cpp)
mtl5_add_test(test_qr operation/test_qr.cpp)
mtl5_add_test(test_cholesky operation/test_cholesky.cpp)
mtl5_add_test(test_eigenvalue operation/test_eigenvalue.cpp)
mtl5_add_test(test_svd operation/test_svd.cpp)
mtl5_add_test(test_kron operation/test_kron.cpp)

# Functor tests
mtl5_add_test(test_scalar_functors functor/test_scalar_functors.cpp)

# ITL tests
mtl5_add_test(test_cg itl/test_cg.cpp)
mtl5_add_test(test_bicgstab itl/test_bicgstab.cpp)
mtl5_add_test(test_bicg itl/test_bicg.cpp)
mtl5_add_test(test_gmres itl/test_gmres.cpp)
mtl5_add_test(test_smoothers itl/test_smoothers.cpp)
mtl5_add_test(test_ilu_ic itl/test_ilu_ic.cpp)
mtl5_add_test(test_tfqmr itl/test_tfqmr.cpp)
mtl5_add_test(test_qmr itl/test_qmr.cpp)
mtl5_add_test(test_idr_s itl/test_idr_s.cpp)
