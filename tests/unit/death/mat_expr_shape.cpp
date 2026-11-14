// The element-wise and product expressions assert that their operands' shapes
// agree (#824). An expression with mismatched shapes reads out of range, so in
// a debug build it must stop at the assertion instead. This program builds one
// mismatched expression, chosen by argv[1], and must not return normally:
// ctest passes it only when the output names the shape assertion.
//
// NDEBUG is undefined before any header, as in mult_alias.cpp, so the
// assertions are live in a Release build too; the SIGABRT handler turns the
// abort into a normal exit.
#undef NDEBUG
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <cstring>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>

extern "C" void mtl5_death_on_abort(int) { std::_Exit(0); }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::signal(SIGABRT, mtl5_death_on_abort);

    const mtl::mat::dense2D<double> A(2, 3), B(3, 2);
    using ref = const mtl::mat::dense2D<double>&;

    if (std::strcmp(argv[1], "sum") == 0) {
        const auto e = A + B;   // 2x3 + 3x2
        static_cast<void>(e);
    } else if (std::strcmp(argv[1], "product") == 0) {
        const mtl::mat::expr::mat_mat_times_expr<ref, ref> p(A, A);   // 2x3 * 2x3
        static_cast<void>(p);
    } else {
        return 2;
    }
    return 1;   // the assertion did not fire
}
