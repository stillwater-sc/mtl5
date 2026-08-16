// This translation unit MUST NOT COMPILE.
//
// The inverse of an integer matrix is not an integer matrix (it is rational),
// so inv() on one cannot be represented in its own element type -- every entry
// would be truncated toward zero. That is worse than an error because the shape
// and the type are both right; only the numbers are wrong.
//
// EXPECT-ERROR: Field
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/inv.hpp>

int main() {
    mtl::mat::dense2D<int> A(3, 3);
    auto Ainv = mtl::inv(A);
    return 0;
}
