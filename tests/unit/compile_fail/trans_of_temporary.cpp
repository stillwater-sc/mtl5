// This translation unit MUST NOT COMPILE.
//
// trans() returns a view that refers to its matrix. A * B is a temporary
// dense2D, destroyed at the end of the full expression, so `t` would refer to
// freed memory the first time it is read. This used to compile (#695). trans
// now has a deleted `const M&&` overload, which an rvalue prefers over
// `const M&`. The other view factories in mat/view/ have the same overload;
// test_view_of_temporary.cpp checks them.
//
// EXPECT-ERROR: deleted
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/trans.hpp>

int main() {
    mtl::mat::dense2D<double> A(2, 2), B(2, 2);
    auto t = mtl::trans(A * B);
    return static_cast<int>(t(0, 0));
}
