// This translation unit MUST NOT COMPILE.
//
// bind() holds a reference to its tensor, so bind(make_tensor(), i, j)
// dangled once the full-expression ended (#857). Binding a temporary is
// deleted.
//
// EXPECT-ERROR: (use of deleted function|call to deleted function)
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/tensor.hpp>

int main() {
    using namespace mtl::tensor;
    Index<'i'> i; Index<'j'> j;
    auto bound = bind(tensor<double, 2, 2>{1, 2, 3, 4}, i, j);
    static_cast<void>(bound);
    return 0;
}
