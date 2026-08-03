// This translation unit MUST NOT COMPILE.
//
// hessenberg performs a similarity transform by applying H*A*H, which is a
// similarity only because a REAL Householder reflector is Hermitian and
// involutory. A complex reflector is unitary but not Hermitian, so H^-1 = H^H
// and the same code computes a NON-similar matrix -- different eigenvalues,
// no diagnostic.
//
// householder() became complex-capable in #353, so this would now compile were
// it not rejected. This test pins the rejection so that a fix in one file
// cannot silently open a wrong-answer path in another.
//
// EXPECT-ERROR: hessenberg_factor applies H
#include <complex>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/hessenberg.hpp>

int main() {
    mtl::mat::dense2D<std::complex<double>> A(5, 5);
    mtl::vec::dense_vector<std::complex<double>> tau;
    mtl::hessenberg_factor(A, tau);
    return 0;
}
