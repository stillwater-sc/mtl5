// This translation unit MUST NOT COMPILE.
//
// compressed2D is CSR-only. A col_major instantiation used to be accepted
// silently while being byte-for-byte a CSR matrix, so CSC input was read as its
// transpose with no diagnostic (#355). The container now rejects it at compile
// time, and this test pins that: if the static_assert is ever removed, this
// file starts compiling and the test turns red.
//
// The EXPECT-ERROR line below is the regex the build output must match, so the
// test cannot pass merely because compilation failed for some unrelated reason.
//
// EXPECT-ERROR: compressed2D is CSR

#include <mtl/mat/compressed2D.hpp>

int main() {
    mtl::mat::compressed2D<double,
        mtl::mat::parameters<mtl::tag::col_major>> A(2, 3);
    (void)A;
    return 0;
}
