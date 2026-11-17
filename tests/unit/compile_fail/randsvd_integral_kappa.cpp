// This translation unit MUST NOT COMPILE.
//
// randsvd(5, 1000, 3, 42) with an integral kappa reads equally well as the
// rectangular randsvd(m = 5, n = 1000, kappa = 3, mode = 42) and the square
// seeded randsvd(n = 5, kappa = 1000, mode = 3, seed = 42), so neither binds
// (issue 888): deduced, both reject an integral kappa; with randsvd<double>
// the two are ambiguous. Write the kappa as 1000.0.
//
// EXPECT-ERROR: no matching|ambiguous|C2672|C2668
#include <mtl/generators/randsvd.hpp>

int main() {
    auto A = mtl::generators::randsvd(5, 1000, 3, 42);
    auto B = mtl::generators::randsvd<double>(5, 1000, 3, 42);
    static_cast<void>(A);
    static_cast<void>(B);
    return 0;
}
