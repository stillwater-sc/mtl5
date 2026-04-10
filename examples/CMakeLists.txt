# Auto-discover example sources organized by phase subdirectories.
# Each .cpp file becomes target "example_<filename_stem>".

file(GLOB INTRO_SRC "intro/*.cpp")
compile_all("false" "example" "Examples/intro" "MTL5::mtl5" ${INTRO_SRC})

file(GLOB PHASE01_SRC "phase01_core_types/*.cpp")
compile_all("false" "example" "Examples/phase01_core_types" "MTL5::mtl5" ${PHASE01_SRC})

file(GLOB PHASE02_SRC "phase02_basic_operations/*.cpp")
compile_all("false" "example" "Examples/phase02_basic_operations" "MTL5::mtl5" ${PHASE02_SRC})

file(GLOB PHASE03_SRC "phase03_iterative_solvers/*.cpp")
compile_all("false" "example" "Examples/phase03_iterative_solvers" "MTL5::mtl5" ${PHASE03_SRC})

file(GLOB PHASE04_SRC "phase04_sparse_assembly/*.cpp")
compile_all("false" "example" "Examples/phase04_sparse_assembly" "MTL5::mtl5" ${PHASE04_SRC})

file(GLOB PHASE05_SRC "phase05_dense_decompositions/*.cpp")
compile_all("false" "example" "Examples/phase05_dense_decompositions" "MTL5::mtl5" ${PHASE05_SRC})

file(GLOB PHASE06_SRC "phase06_eigenvalue_svd/*.cpp")
compile_all("false" "example" "Examples/phase06_eigenvalue_svd" "MTL5::mtl5" ${PHASE06_SRC})

file(GLOB PHASE07_SRC "phase07_sparse_formats/*.cpp")
compile_all("false" "example" "Examples/phase07_sparse_formats" "MTL5::mtl5" ${PHASE07_SRC})

file(GLOB PHASE08_SRC "phase08_expression_templates/*.cpp")
compile_all("false" "example" "Examples/phase08_expression_templates" "MTL5::mtl5" ${PHASE08_SRC})

file(GLOB PHASE09_SRC "phase09_advanced_types/*.cpp")
compile_all("false" "example" "Examples/phase09_advanced_types" "MTL5::mtl5" ${PHASE09_SRC})

file(GLOB PHASE10_SRC "phase10_matrix_views/*.cpp")
compile_all("false" "example" "Examples/phase10_matrix_views" "MTL5::mtl5" ${PHASE10_SRC})

file(GLOB PHASE11_SRC "phase11_assembly_reorder/*.cpp")
compile_all("false" "example" "Examples/phase11_assembly_reorder" "MTL5::mtl5" ${PHASE11_SRC})

file(GLOB PHASE12_SRC "phase12_transcendental_functions/*.cpp")
compile_all("false" "example" "Examples/phase12_transcendental_functions" "MTL5::mtl5" ${PHASE12_SRC})

file(GLOB PHASE14_SRC "phase14_io_generators/*.cpp")
compile_all("false" "example" "Examples/phase14_io_generators" "MTL5::mtl5" ${PHASE14_SRC})

file(GLOB PHASE13_SRC "phase13_krylov_multigrid/*.cpp")
compile_all("false" "example" "Examples/phase13_krylov_multigrid" "MTL5::mtl5" ${PHASE13_SRC})

file(GLOB PHASE15_SRC "phase15_sparse_direct/*.cpp")
compile_all("false" "example" "Examples/phase15_sparse_direct" "MTL5::mtl5" ${PHASE15_SRC})

file(GLOB UKF_SRC "ukf_cholesky_vs_ldlt/*.cpp")
compile_all("false" "example" "Examples/ukf_cholesky_vs_ldlt" "MTL5::mtl5" ${UKF_SRC})
