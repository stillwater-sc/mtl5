#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/cg.hpp>

using namespace mtl;

static mat::compressed2D<double> make_tridiagonal(std::size_t n, double diag, double off) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << diag;
            if (i > 0)     ins[i][i-1] << off;
            if (i < n - 1) ins[i][i+1] << off;
        }
    }
    return A;
}

// --- ILUT tests ---

TEST_CASE("ILUT preconditioned BiCGSTAB converges", "[itl][pc][ilut]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ilut<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("ILUT converges in fewer iterations than identity PC", "[itl][pc][ilut]") {
    const std::size_t n = 30;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // Identity PC
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> id_pc(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab(A, x1, b, id_pc, iter1);
    int iters_id = iter1.iterations();

    // ILUT PC
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::ilut<double> ilut_pc(A);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab(A, x2, b, ilut_pc, iter2);
    int iters_ilut = iter2.iterations();

    REQUIRE(iters_ilut <= iters_id);
}

TEST_CASE("ILUT with fill-in on arrowhead matrix", "[itl][pc][ilut]") {
    // Arrowhead matrix: row 0 couples to all columns, forcing fill-in at
    // low column indices when processing later rows.  This is the exact
    // pattern that triggers out-of-order column processing if fill-in
    // entries are simply appended rather than visited in ascending order.
    const std::size_t n = 30;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 10.0;
            if (i > 0) {
                ins[0][i] << 1.0;
                ins[i][0] << 1.0;
            }
        }
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    // Generous fill allowance to ensure fill-in actually happens
    itl::pc::ilut<double> pc(A, /*fill=*/20, /*threshold=*/1e-6);
    itl::basic_iteration<double> iter(b, 300, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    // Verify the actual residual is small (not just the preconditioned one)
    auto Ax = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = Ax(i) - b(i);
        res += diff * diff;
    }
    res = std::sqrt(res);
    REQUIRE(res < 1e-8);
}

// --- ILDL tests ---

TEST_CASE("ILDL preconditioned CG on SPD system converges", "[itl][pc][ildl]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ildl<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// --- Block diagonal tests ---

TEST_CASE("Block diagonal preconditioned BiCGSTAB converges", "[itl][pc][block_diagonal]") {
    const std::size_t n = 20;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -1.0;
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::block_diagonal<mat::dense2D<double>> pc(A, 5);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// --- SSOR tests ---

TEST_CASE("SSOR preconditioned BiCGSTAB converges", "[itl][pc][ssor]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ssor<mat::compressed2D<double>> pc(A, 1.0);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("SSOR converges in fewer iterations than identity PC", "[itl][pc][ssor]") {
    const std::size_t n = 30;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // Identity PC
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> id_pc(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab(A, x1, b, id_pc, iter1);
    int iters_id = iter1.iterations();

    // SSOR PC
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::ssor<mat::compressed2D<double>> ssor_pc(A, 1.2);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab(A, x2, b, ssor_pc, iter2);
    int iters_ssor = iter2.iterations();

    REQUIRE(iters_ssor <= iters_id);
}
