// This translation unit MUST NOT COMPILE.
//
// eigen_symmetric tridiagonalizes by applying H*A*H, which is a similarity
// transform only because a REAL Householder reflector is Hermitian and
// involutory. A complex reflector is unitary but not Hermitian, so H^-1 = H^H
// and the same code would compute a NON-similar matrix -- different
// eigenvalues.
//
// It already fails to compile for complex for unrelated reasons, so no silent
// wrong answer is reachable today. What this pins is that the failure stays a
// failure, and says WHY: householder() became complex-capable in #353, which
// removed several of the errors that were incidentally holding this shut.
//
// Targets eigen_symmetric specifically. The first version of this test called
// eigenvalue_symmetric_generic, which does not use householder at all -- it
// passed while covering nothing.
//
// EXPECT-ERROR: eigen_symmetric applies H
#include <complex>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>

int main() {
    mtl::mat::dense2D<std::complex<double>> A(5, 5);
    auto r = mtl::eigen_symmetric(A);
    (void)r;
    return 0;
}
