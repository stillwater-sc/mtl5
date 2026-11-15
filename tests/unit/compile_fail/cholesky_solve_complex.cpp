// This translation unit MUST NOT COMPILE.
//
// cholesky_solve applies L and L^T. Fed the factor cholesky_h_factor computes
// for a Hermitian matrix, it applied L^T where L^H is needed and returned a
// wrong answer: for [[4, 1+2i], [1-2i, 6]] the residual was 2.0 against 2e-16
// for cholesky_h_solve (#854). It is restricted to real element types, as
// cholesky_factor is (#353), with a diagnostic naming cholesky_h_solve.
//
// EXPECT-ERROR: cholesky_solve applies L and L\^T
#include <complex>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/vec/dense_vector.hpp>

int main() {
    using C = std::complex<double>;
    mtl::mat::dense2D<C> L(2, 2);
    mtl::vec::dense_vector<C> x(2), b(2);
    mtl::cholesky_solve(L, x, b);
    return 0;
}
