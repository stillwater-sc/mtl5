// This translation unit MUST NOT COMPILE.
//
// A Krylov solver computes its step lengths as ratios in the element type --
// cg's `beta = rho / rho_1` and `alpha = rho / dot(p, q)`, and eight such ratios
// in qmr. On an integral element type that division TRUNCATES, so the iteration
// is nonsense from the first step: it runs, it converges or it does not, and it
// returns a confident wrong answer with no assertion and no flag. The integers
// are a ring, not a field -- only 1 and -1 have inverses.
//
// This used to compile. The solvers were unconstrained templates
// (`typename VecX`), so `dense_vector<int>` instantiated all ten of them. They
// now require FieldVector, the vector counterpart of the FieldMatrix that #430
// introduced for exactly this failure one level down, where
// `lu_factor(dense2D<int>)` used to compile and return truncated nonsense (#503).
//
// EXPECT-ERROR: Field
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/pc/identity.hpp>

int main() {
    mtl::mat::dense2D<int> A(4, 4);
    mtl::vec::dense_vector<int> x(4), b(4);
    mtl::itl::pc::identity<mtl::mat::dense2D<int>> M(A);
    mtl::itl::basic_iteration<int> iter(b, 100, 0);
    mtl::itl::cg(A, x, b, M, iter);
    return 0;
}
