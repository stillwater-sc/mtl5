# Auto-discover example sources and create targets.
# Each .cpp file becomes target "example_<filename_stem>".

file(GLOB EXAMPLE_SRC "*.cpp")
compile_all("false" "example" "Examples" "MTL5::mtl5" ${EXAMPLE_SRC})
