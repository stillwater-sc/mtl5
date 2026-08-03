// This translation unit MUST NOT COMPILE.
//
// ell_matrix is row-padded ELLPACK: both arrays are nrows*width and every
// access is indices_[r * width_ + k]. A col_major instantiation was accepted
// silently while being byte-for-byte that same row-major layout (#355).
//
// Unlike the compressed2D case, this one was constructible-but-unfillable
// rather than wrong -- the only fill path is the compressed2D constructor,
// which col_major already rejects. This pins the direct rejection so that
// adding an inserter or a setter later cannot silently reopen the hole.
//
// EXPECT-ERROR: ell_matrix is row-padded

#include <mtl/mat/ell_matrix.hpp>

int main() {
    mtl::mat::ell_matrix<double,
        mtl::mat::parameters<mtl::tag::col_major>> A(2, 3, 2);
    (void)A;
    return 0;
}
