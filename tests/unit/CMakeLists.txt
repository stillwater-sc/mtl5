# Auto-discover test sources and create Catch2 test targets per category.
# Each .cpp file becomes target "${prefix}_<filename_stem>" and a CTest entry.

file(GLOB CONCEPT_SRC "concepts/*.cpp")
compile_all("true" "concept" "Tests/concepts" "MTL5::mtl5;Catch2::Catch2WithMain" ${CONCEPT_SRC})

file(GLOB MATH_SRC "math/*.cpp")
compile_all("true" "math" "Tests/math" "MTL5::mtl5;Catch2::Catch2WithMain" ${MATH_SRC})

file(GLOB DETAIL_SRC "detail/*.cpp")
compile_all("true" "detail" "Tests/detail" "MTL5::mtl5;Catch2::Catch2WithMain" ${DETAIL_SRC})

file(GLOB MAT_SRC "mat/*.cpp")
compile_all("true" "mat" "Tests/mat" "MTL5::mtl5;Catch2::Catch2WithMain" ${MAT_SRC})

file(GLOB VEC_SRC "vec/*.cpp")
compile_all("true" "vec" "Tests/vec" "MTL5::mtl5;Catch2::Catch2WithMain" ${VEC_SRC})

file(GLOB OP_SRC "operation/*.cpp")
compile_all("true" "op" "Tests/operation" "MTL5::mtl5;Catch2::Catch2WithMain" ${OP_SRC})

file(GLOB FUNCTOR_SRC "functor/*.cpp")
compile_all("true" "functor" "Tests/functor" "MTL5::mtl5;Catch2::Catch2WithMain" ${FUNCTOR_SRC})

file(GLOB ITL_SRC "itl/*.cpp")
compile_all("true" "itl" "Tests/itl" "MTL5::mtl5;Catch2::Catch2WithMain" ${ITL_SRC})

file(GLOB IO_SRC "io/*.cpp")
compile_all("true" "io" "Tests/io" "MTL5::mtl5;Catch2::Catch2WithMain" ${IO_SRC})

file(GLOB RECURSION_SRC "recursion/*.cpp")
compile_all("true" "recursion" "Tests/recursion" "MTL5::mtl5;Catch2::Catch2WithMain" ${RECURSION_SRC})

file(GLOB GENERATOR_SRC "generators/*.cpp")
compile_all("true" "generators" "Tests/generators" "MTL5::mtl5;Catch2::Catch2WithMain" ${GENERATOR_SRC})

file(GLOB INTERFACE_SRC "interface/*.cpp")
compile_all("true" "iface" "Tests/interface" "MTL5::mtl5;Catch2::Catch2WithMain" ${INTERFACE_SRC})
