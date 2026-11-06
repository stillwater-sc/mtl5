// mult asserts that its output does not alias an input (#696).
//
// mult(A, B, C) and mult(A, x, y) overwrite C (or y) while still reading A, B
// and x, so mult(A, B, A) or mult(A, x, x) gives a wrong result with no error.
// In a debug build mult asserts against it. This program makes one aliased
// call, chosen by argv[1], and must not return normally: ctest passes it only
// when the output names the aliasing assertion (PASS_REGULAR_EXPRESSION), so an
// unrelated assertion, or none at all, fails the test.
//
// NDEBUG is undefined before any header, so the assertions are live in a
// Release build too. The SIGABRT handler turns the abort into a normal exit:
// ctest counts a signal as a failure even when the output matches.
#undef NDEBUG
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/vec/dense_vector.hpp>

extern "C" void mtl5_death_on_abort(int) { std::_Exit(0); }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::signal(SIGABRT, mtl5_death_on_abort);

    mtl::mat::dense2D<double> A(2, 2), B(2, 2);
    mtl::vec::dense_vector<double> x(2);
    for (std::size_t i = 0; i < 2; ++i) {
        x(i) = 1.0;
        for (std::size_t j = 0; j < 2; ++j) A(i, j) = B(i, j) = 1.0;
    }

    if (std::strcmp(argv[1], "matvec_x") == 0)       mtl::mult(A, x, x);
    else if (std::strcmp(argv[1], "matmat_a") == 0)  mtl::mult(A, B, A);
    else if (std::strcmp(argv[1], "matmat_b") == 0)  mtl::mult(A, B, B);
    else if (std::strcmp(argv[1], "acc_matvec_x") == 0) mtl::mult<double>(A, x, x);
    else if (std::strcmp(argv[1], "acc_matmat_a") == 0) mtl::mult<double>(A, B, A);
    else return 2;

    std::puts("the aliased call returned without asserting");
    return 1;
}
