// This translation unit MUST NOT COMPILE.
//
// contract(A(i,i), B(i,j)): an index repeated within one tensor (a trace) is
// not supported (#862; issue 888 pins it). It shares i twice, so either
// guard may fire first.
//
// EXPECT-ERROR: an index repeated within one tensor|share more than one index
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/tensor.hpp>

int main() {
    using namespace mtl::tensor;
    Index<'i'> i; Index<'j'> j;
    tensor<double, 2, 2> A{1, 2, 3, 4}, B{1, 0, 0, 1};
    auto C = contract(bind(A, i, i), bind(B, i, j));
    static_cast<void>(C);
    return 0;
}
