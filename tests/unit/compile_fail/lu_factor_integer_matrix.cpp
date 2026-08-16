// This translation unit MUST NOT COMPILE.
//
// lu_factor divides by the pivot. On an integral element type that division
// TRUNCATES, so the factorization runs to completion and returns a confident,
// wrong answer -- no assertion, no singular flag, nothing to notice. The
// integers are a ring, not a field: only 1 and -1 have inverses.
//
// This used to compile. `Field` admitted `int` because the syntax fits (a / b
// yields an int), so every algorithm saying "requires Field" -- meaning "I will
// divide and expect the quotient back" -- silently accepted a type that cannot
// give it one. Field now excludes the integral types and the decompositions
// require FieldMatrix (#430 follow-up).
//
// EXPECT-ERROR: Field
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/lu.hpp>

int main() {
    mtl::mat::dense2D<int> A(4, 4);
    std::vector<mtl::mat::dense2D<int>::size_type> pivot;
    mtl::lu_factor(A, pivot);
    return 0;
}
