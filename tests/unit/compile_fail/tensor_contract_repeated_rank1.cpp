// This translation unit MUST NOT COMPILE.
//
// contract(A(i,i), v(i)): an index repeated within one tensor (a trace) is
// not supported by the rank-1 overload (#862; issue 888 pins it).
//
// EXPECT-ERROR: an index repeated within one tensor
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/tensor.hpp>

int main() {
    using namespace mtl::tensor;
    Index<'i'> i;
    tensor<double, 2, 2> A{1, 2, 3, 4};
    tensor<double, 1, 2> v{1, 1};
    auto c = contract(bind(A, i, i), bind(v, i));
    static_cast<void>(c);
    return 0;
}
