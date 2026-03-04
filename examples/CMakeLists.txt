add_executable(hello_mtl5 hello_mtl5.cpp)
target_link_libraries(hello_mtl5 PRIVATE MTL5::mtl5)

# Phase 3 examples
add_executable(phase3a_heat_equation_1d phase3a_heat_equation_1d.cpp)
target_link_libraries(phase3a_heat_equation_1d PRIVATE MTL5::mtl5)

add_executable(phase3b_convection_diffusion phase3b_convection_diffusion.cpp)
target_link_libraries(phase3b_convection_diffusion PRIVATE MTL5::mtl5)

# Phase 4 examples
add_executable(phase4a_laplacian_2d phase4a_laplacian_2d.cpp)
target_link_libraries(phase4a_laplacian_2d PRIVATE MTL5::mtl5)

add_executable(phase4b_smoother_convergence phase4b_smoother_convergence.cpp)
target_link_libraries(phase4b_smoother_convergence PRIVATE MTL5::mtl5)

# Phase 5 examples
add_executable(phase5a_least_squares_qr phase5a_least_squares_qr.cpp)
target_link_libraries(phase5a_least_squares_qr PRIVATE MTL5::mtl5)

add_executable(phase5b_solve_three_ways phase5b_solve_three_ways.cpp)
target_link_libraries(phase5b_solve_three_ways PRIVATE MTL5::mtl5)

# Phase 6 examples
add_executable(phase6a_vibrating_string phase6a_vibrating_string.cpp)
target_link_libraries(phase6a_vibrating_string PRIVATE MTL5::mtl5)

add_executable(phase6b_pca_svd phase6b_pca_svd.cpp)
target_link_libraries(phase6b_pca_svd PRIVATE MTL5::mtl5)

# Phase 7 examples
add_executable(phase7a_sparse_formats phase7a_sparse_formats.cpp)
target_link_libraries(phase7a_sparse_formats PRIVATE MTL5::mtl5)

add_executable(phase7b_structured_views phase7b_structured_views.cpp)
target_link_libraries(phase7b_structured_views PRIVATE MTL5::mtl5)
