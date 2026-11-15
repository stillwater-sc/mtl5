// MTL5 Phase 14 -- Tests for UMFPACK sparse direct solver
// These tests are only compiled when MTL5_HAS_UMFPACK is defined.
// When UMFPACK is not available, a placeholder test confirms the test compiles.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

#ifdef MTL5_HAS_UMFPACK

#include <stdexcept>
#include <utility>

#include <mtl/interface/umfpack.hpp>

namespace {

using csr = mat::compressed2D<double>;

// Unsymmetric: A = [4 1 0; 2 5 1; 0 3 6], x = [1 2 3] gives b = [6 15 24].
// A transposed CRS-to-CCS conversion solves A^T x = b instead, a different x.
csr unsymmetric3() {
    csr A(3, 3);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 2.0; ins[1][1] << 5.0; ins[1][2] << 1.0;
        ins[2][1] << 3.0; ins[2][2] << 6.0;
    }
    return A;
}

// Rank 2: rows 0 and 1 are proportional.
csr singular3() {
    csr A(3, 3);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][0] << 2.0; ins[1][1] << 4.0;
        ins[2][2] << 1.0;
    }
    return A;
}

void require_x123(const vec::dense_vector<double>& x) {
    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(x(1), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(x(2), WithinAbs(3.0, 1e-12));
}

} // namespace

TEST_CASE("umfpack_solve on small system", "[interface][umfpack]") {
    // A = [2 1; 1 3] (sparse), b = [5; 10] => x = [1; 3]
    csr A(2, 2);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 2.0;
        ins[0][1] << 1.0;
        ins[1][0] << 1.0;
        ins[1][1] << 3.0;
    }

    vec::dense_vector<double> b = {5.0, 10.0};
    vec::dense_vector<double> x(2, 0.0);

    interface::umfpack_solve(A, x, b);

    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(x(1), WithinAbs(3.0, 1e-10));
}

TEST_CASE("umfpack_solver RAII class", "[interface][umfpack]") {
    // 3x3 identity: x = b
    csr A(3, 3);
    {
        mat::inserter<csr> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
        ins[2][2] << 1.0;
    }

    vec::dense_vector<double> b = {7.0, 8.0, 9.0};
    vec::dense_vector<double> x(3, 0.0);

    interface::umfpack_solver solver(A);
    solver.solve(x, b);

    for (int i = 0; i < 3; ++i)
        REQUIRE_THAT(x(i), WithinAbs(b(i), 1e-10));
}

TEST_CASE("umfpack_solver on an unsymmetric matrix", "[interface][umfpack][850]") {
    vec::dense_vector<double> b = {6.0, 15.0, 24.0}, x(3, 0.0);
    interface::umfpack_solver solver(unsymmetric3());
    solver.solve(x, b);
    require_x123(x);
}

TEST_CASE("umfpack_solver throws on a singular matrix", "[interface][umfpack][850]") {
    // UMFPACK reports this as a warning with a valid Numeric object; the
    // binding throws, and frees it (run under LeakSanitizer to see the leak
    // this used to be).
    REQUIRE_THROWS_AS(interface::umfpack_solver(singular3()), std::runtime_error);

    // Structurally singular: column 1 is empty.
    csr S(3, 3);
    {
        mat::inserter<csr> ins(S);
        ins[0][0] << 1.0; ins[1][0] << 1.0; ins[2][2] << 1.0;
    }
    REQUIRE_THROWS_AS(interface::umfpack_solver(S), std::runtime_error);
}

TEST_CASE("umfpack_solver rejects a non-square matrix and wrong vector sizes",
          "[interface][umfpack][850]") {
    csr R(3, 4);
    {
        mat::inserter<csr> ins(R);
        ins[0][0] << 1.0; ins[1][1] << 1.0; ins[2][2] << 1.0;
    }
    REQUIRE_THROWS_AS(interface::umfpack_solver(R), std::invalid_argument);

    interface::umfpack_solver solver(unsymmetric3());
    vec::dense_vector<double> b3(3, 1.0), x3(3), b2(2, 1.0), x4(4);
    REQUIRE_THROWS_AS(solver.solve(x4, b3), std::invalid_argument);
    REQUIRE_THROWS_AS(solver.solve(x3, b2), std::invalid_argument);
}

TEST_CASE("umfpack_solver is movable", "[interface][umfpack][850]") {
    vec::dense_vector<double> b = {6.0, 15.0, 24.0}, x(3, 0.0);

    interface::umfpack_solver solver(unsymmetric3());
    interface::umfpack_solver moved(std::move(solver));
    REQUIRE(solver.num_rows() == 0);
    REQUIRE_THROWS_AS(solver.solve(x, b), std::logic_error);
    moved.solve(x, b);
    require_x123(x);

    // Move assignment releases the target's factorization and takes the source's.
    interface::umfpack_solver other(unsymmetric3());
    csr I(1, 1);
    { mat::inserter<csr> ins(I); ins[0][0] << 2.0; }
    interface::umfpack_solver small(I);
    small = std::move(other);
    REQUIRE(small.num_rows() == 3);
    x = vec::dense_vector<double>(3, 0.0);
    small.solve(x, b);
    require_x123(x);
}

TEST_CASE("umfpack_solver on 0x0 and 1x1 matrices", "[interface][umfpack][850]") {
    csr E(0, 0);
    interface::umfpack_solver empty(E);
    REQUIRE(empty.num_rows() == 0);
    vec::dense_vector<double> e0(0), x0(0);
    empty.solve(x0, e0);   // nothing to do; used to throw (UMFPACK rejects n = 0)

    csr A(1, 1);
    { mat::inserter<csr> ins(A); ins[0][0] << 4.0; }
    vec::dense_vector<double> b(1, 2.0), x(1, 0.0);
    interface::umfpack_solve(A, x, b);
    REQUIRE(x(0) == 0.5);
}

#else // !MTL5_HAS_UMFPACK

TEST_CASE("UMFPACK not available -- placeholder", "[interface][umfpack]") {
    // UMFPACK support not compiled in; verify test infrastructure works.
    REQUIRE(true);
}

#endif // MTL5_HAS_UMFPACK
