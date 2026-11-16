// This translation unit MUST NOT COMPILE.
//
// bind() holds a reference to its tensor, so bind(make_tensor(), i, j)
// dangled once the full-expression ended (#857). Binding a temporary is
// deleted. "deleted function" is in GCC's, Clang's and MSVC's message alike
// ("use of", "call to", "attempting to reference a deleted function").
//
// EXPECT-ERROR: deleted function
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/tensor.hpp>

int main() {
    using namespace mtl::tensor;
    Index<'i'> i; Index<'j'> j;
    auto bound = bind(tensor<double, 2, 2>{1, 2, 3, 4}, i, j);
    static_cast<void>(bound);
    return 0;
}
