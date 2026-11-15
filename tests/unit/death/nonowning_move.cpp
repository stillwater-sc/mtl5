// Move-assigning a block of another size into a non-owning block -- a
// dense_vector or dense2D over a caller's buffer, an ndarray view -- cannot
// write through, and a move cannot throw, so it aborts in every build (#856).
// It was an assert, and under NDEBUG it re-pointed the object at the source's
// storage: the detach #829 removed from every other assignment.
//
// This program makes one such call, chosen by argv[1], and must not return
// normally: ctest passes it only when the output names the abort's message
// (PASS_REGULAR_EXPRESSION). NDEBUG is left alone: the abort is not an assert,
// so the test means the same in a Release build. The SIGABRT handler turns the
// abort into a normal exit, because ctest counts a signal as a failure.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <mtl/array/ndarray.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

extern "C" void mtl5_death_on_abort(int) { std::_Exit(0); }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::signal(SIGABRT, mtl5_death_on_abort);

    double buf[6] = {1, 2, 3, 4, 5, 6};
    if (std::strcmp(argv[1], "vector") == 0) {
        mtl::vec::dense_vector<double> v(3, buf);
        v = mtl::vec::dense_vector<double>(5, 0.0);
    } else if (std::strcmp(argv[1], "matrix") == 0) {
        mtl::mat::dense2D<double> A(2, 3, buf);
        A = mtl::mat::dense2D<double>(4, 4);
    } else if (std::strcmp(argv[1], "ndarray") == 0) {
        mtl::array::ndarray<double, 2> view(buf, mtl::array::shape<2>{2, 3});
        view = mtl::array::ndarray<double, 2>(mtl::array::shape<2>{3, 3});
    } else {
        return 2;
    }
    std::puts("the move returned without aborting");
    return 1;
}
