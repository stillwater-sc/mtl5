// This translation unit MUST NOT COMPILE.
//
// cholesky_factor computes A = L*L^T and orders the diagonal to test positive
// definiteness. Neither survives a complex element type, and repairing only the
// comparison would turn a compile error into a silently wrong factorization --
// the failure mode of #352. It is restricted to real element types, with a
// diagnostic naming cholesky_h_factor as the routine for Hermitian input (#353).
//
// EXPECT-ERROR: cholesky_factor computes A = L
#include <complex>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/cholesky.hpp>

int main() {
    mtl::mat::dense2D<std::complex<double>> A(4, 4);
    mtl::cholesky_factor(A);
    return 0;
}
