// This translation unit MUST NOT COMPILE.
//
// contract(A(i,j), B(i,j)) shares two indices. find_repeated took the first
// and returned a matrix: with A = {1,2,3,4} and B = I, [1 3; 2 4] instead of
// a full contraction (#857). Exactly one shared index is required.
//
// EXPECT-ERROR: the tensors share more than one index
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/tensor.hpp>

int main() {
    using namespace mtl::tensor;
    Index<'i'> i; Index<'j'> j;
    tensor<double, 2, 2> A{1, 2, 3, 4}, B{1, 0, 0, 1};
    auto C = contract(bind(A, i, j), bind(B, i, j));
    static_cast<void>(C);
    return 0;
}
