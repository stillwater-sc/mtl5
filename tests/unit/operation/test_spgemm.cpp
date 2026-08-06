// Sparse * sparse product (#402).
//
// Two things need asserting and only one of them is "the answer is right":
//
//  1. C = A*B holds the correct values, checked against a dense reference.
//  2. C is a WELL-FORMED compressed2D -- column indices ascending within each
//     row. compressed2D::operator()(r, c) locates entries with std::lower_bound,
//     so an unsorted row makes element access return wrong values for entries
//     that are present.
//
//     That second failure is close to invisible. Measured by removing the sort
//     from spgemm: nnz stays correct, row sums off the raw CSR arrays stay
//     correct to 4.4e-16, and a sparse matvec stays correct TO ROUNDING -- it
//     sums data[k]*x(indices[k]) over whatever order the row is in, and
//     reordering preserves the set of products, so the sum is still right.
//     (Not bit-identical -- floating-point addition is not associative -- but
//     wrong at eps, which no residual check will flag.)
//     Only reading elements back, or inspecting the index arrays, sees it. So
//     the natural tests for a matrix product -- residual of A*x, nnz, row sums
//     -- all pass on a corrupt result. require_well_formed below is what
//     catches it, and it catches it with a diagnosable message.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/spgemm.hpp>

using namespace mtl;
using SMat = mat::compressed2D<double>;

namespace {

/// Dense reference product, independent of anything under test.
mat::dense2D<double> dense_product(const SMat& A, const SMat& B) {
    const std::size_t m = A.num_rows(), n = B.num_cols(), k = A.num_cols();
    mat::dense2D<double> C(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t t = 0; t < k; ++t) s += A(i, t) * B(t, j);
            C(i, j) = s;
        }
    return C;
}

/// Every row's column indices strictly ascending, and starts_ consistent.
/// This is the invariant compressed2D::operator() depends on.
void require_well_formed(const SMat& C) {
    const auto& starts  = C.ref_major();
    const auto& indices = C.ref_minor();
    REQUIRE(starts.size() == C.num_rows() + 1);
    REQUIRE(starts[0] == 0u);
    REQUIRE(starts[C.num_rows()] == C.ref_data().size());
    for (std::size_t i = 0; i < C.num_rows(); ++i) {
        REQUIRE(starts[i] <= starts[i + 1]);
        for (std::size_t k = starts[i]; k + 1 < starts[i + 1]; ++k) {
            INFO("row " << i << ": indices[" << k << "] = " << indices[k]
                 << ", indices[" << (k + 1) << "] = " << indices[k + 1]);
            REQUIRE(indices[k] < indices[k + 1]);      // ascending AND unique
        }
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            REQUIRE(indices[k] < C.num_cols());
    }
}

void require_matches_dense(const SMat& C, const mat::dense2D<double>& R) {
    REQUIRE(C.num_rows() == R.num_rows());
    REQUIRE(C.num_cols() == R.num_cols());
    for (std::size_t i = 0; i < C.num_rows(); ++i)
        for (std::size_t j = 0; j < C.num_cols(); ++j) {
            INFO("(" << i << "," << j << ")");
            REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(R(i, j), 1e-12));
        }
}

/// Random sparse matrix with the given density.
SMat random_sparse(std::size_t m, std::size_t n, double density, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    SMat A(m, n);
    {
        mat::inserter<SMat> ins(A);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (u(gen) < density) ins[i][j] << nd(gen);
    }
    return A;
}

}  // namespace

TEST_CASE("spgemm matches a dense reference and is well-formed (#402)",
          "[operation][spgemm][sparse]") {
    // Rectangular on purpose: (4x5)*(5x3). A square-only test cannot catch a
    // result sized from the wrong operand.
    SMat A(4, 5), B(5, 3);
    {
        mat::inserter<SMat> ia(A);
        ia[0][0] << 1.0; ia[0][3] << 2.0;
        ia[1][1] << 3.0;
        ia[2][0] << -1.0; ia[2][2] << 4.0; ia[2][4] << 5.0;
        // row 3 deliberately EMPTY
    }
    {
        mat::inserter<SMat> ib(B);
        ib[0][1] << 1.0;
        ib[1][0] << 2.0; ib[1][2] << -3.0;
        ib[2][2] << 6.0;
        ib[3][0] << 7.0;
        ib[4][1] << 8.0;
    }

    const auto C = mtl::spgemm(A, B);
    require_well_formed(C);
    require_matches_dense(C, dense_product(A, B));

    // The empty row of A must give an empty row of C, not a dropped row.
    REQUIRE(C.num_rows() == 4u);
    REQUIRE(C.ref_major()[3] == C.ref_major()[4]);
}

TEST_CASE("spgemm on random sparse matrices (#402)",
          "[operation][spgemm][sparse]") {
    // Densities either side of the point where rows start overlapping heavily,
    // so both the "few collisions" and "many collisions" accumulator paths run.
    for (double density : {0.08, 0.4}) {
        for (unsigned seed : {1u, 2u, 3u}) {
            const auto A = random_sparse(17, 23, density, seed);
            const auto B = random_sparse(23, 11, density, seed + 100u);
            const auto C = mtl::spgemm(A, B);
            INFO("density = " << density << ", seed = " << seed
                 << ", nnz(C) = " << C.nnz());
            require_well_formed(C);
            require_matches_dense(C, dense_product(A, B));
        }
    }
}

TEST_CASE("operator* on two compressed2D returns compressed2D (#402)",
          "[operation][spgemm][sparse]") {
    // The type change is the point of the overload: before #402 this expression
    // yielded dense2D. static_assert states it so a silent regression to the
    // generic dense operator* is a compile error, not a slow test.
    const auto A = random_sparse(12, 12, 0.2, 7u);
    const auto B = random_sparse(12, 12, 0.2, 8u);

    auto C = A * B;
    static_assert(std::is_same_v<decltype(C), mat::compressed2D<double>>,
                  "compressed2D * compressed2D must stay sparse");

    require_well_formed(C);
    require_matches_dense(C, dense_product(A, B));
}

TEST_CASE("spgemm Galerkin triple product stays sparse (#402)",
          "[operation][spgemm][sparse]") {
    // The motivating case: A_coarse = R * A * P with all three sparse. What
    // matters is that the INTERMEDIATE is sparse -- the old path materialised
    // an n x n dense matrix at the fine-grid size.
    const std::size_t nf = 63, nc = 31;

    SMat A(nf, nf);                       // 1-D Laplacian on the fine grid
    {
        mat::inserter<SMat> ins(A);
        for (std::size_t i = 0; i < nf; ++i) {
            ins[i][i] << 2.0;
            if (i)          ins[i][i - 1] << -1.0;
            if (i + 1 < nf) ins[i][i + 1] << -1.0;
        }
    }
    SMat P(nf, nc);                       // linear interpolation
    {
        mat::inserter<SMat> ins(P);
        for (std::size_t c = 0; c < nc; ++c) {
            const std::size_t f = 2 * c + 1;
            ins[f][c] << 1.0;
            if (f)          ins[f - 1][c] << 0.5;
            if (f + 1 < nf) ins[f + 1][c] << 0.5;
        }
    }
    SMat R(nc, nf);                       // full weighting, R = P^T / 2
    {
        mat::inserter<SMat> ins(R);
        for (std::size_t c = 0; c < nc; ++c) {
            const std::size_t f = 2 * c + 1;
            ins[c][f] << 0.5;
            if (f)          ins[c][f - 1] << 0.25;
            if (f + 1 < nf) ins[c][f + 1] << 0.25;
        }
    }

    const auto AP = mtl::spgemm(A, P);
    const auto Ac = mtl::spgemm(R, AP);

    require_well_formed(AP);
    require_well_formed(Ac);
    REQUIRE(Ac.num_rows() == nc);
    REQUIRE(Ac.num_cols() == nc);

    // The intermediate is sparse, which is the whole point. A dense nf x nc
    // intermediate would hold 63*31 = 1953 entries; the sparse one holds far
    // fewer, and the coarse operator is tridiagonal-ish rather than full.
    INFO("nnz(A*P) = " << AP.nnz() << " of " << nf * nc
         << ", nnz(Ac) = " << Ac.nnz() << " of " << nc * nc);
    REQUIRE(AP.nnz() < nf * nc / 4);
    REQUIRE(Ac.nnz() <= 3 * nc);

    // And it is the right coarse operator. Working the constant out rather
    // than pattern-matching it, since getting this wrong is easy:
    //
    //   (A P) column c is -0.5 at 2c-1, 1.0 at 2c+1, -0.5 at 2c+3, and zero at
    //   the even fine rows 2c and 2c+2 -- the interior stencil cancels there.
    //   R row c is 0.25/0.5/0.25 at 2c/2c+1/2c+2, so only the 2c+1 entry meets
    //   a nonzero, giving Ac(c,c) = 0.5*1.0 = 0.5; and R row c+1 meets the
    //   2c+3 entry, giving Ac(c+1,c) = 0.5*(-0.5) = -0.25.
    //
    // So Ac = (1/4) tridiag(-1, 2, -1). Equivalently: this R is P^T/2, and
    // P^T A P = tridiag(-0.5, 1, -0.5), halved by the 1/2 in R. Asserting the
    // un-halved P^T A P values against a halved R is the mistake to avoid.
    for (std::size_t i = 0; i < nc; ++i) {
        REQUIRE_THAT(Ac(i, i), Catch::Matchers::WithinAbs(0.5, 1e-12));
        if (i + 1 < nc)
            REQUIRE_THAT(Ac(i, i + 1), Catch::Matchers::WithinAbs(-0.25, 1e-12));
    }
    // Tridiagonal and nothing else -- a coarse operator that filled in would
    // still satisfy the nnz bound above if the grid were small enough.
    for (std::size_t i = 0; i < nc; ++i)
        for (std::size_t j = 0; j < nc; ++j)
            if (j + 1 < i || j > i + 1) {
                INFO("Ac(" << i << "," << j << ") should be structurally absent");
                REQUIRE_THAT(Ac(i, j), Catch::Matchers::WithinAbs(0.0, 1e-15));
            }
}

TEST_CASE("spgemm edge cases (#402)", "[operation][spgemm][sparse][edge]") {
    SECTION("1x1") {
        SMat A(1, 1), B(1, 1);
        { mat::inserter<SMat> i(A); i[0][0] << 3.0; }
        { mat::inserter<SMat> i(B); i[0][0] << 5.0; }
        const auto C = mtl::spgemm(A, B);
        require_well_formed(C);
        REQUIRE_THAT(C(0, 0), Catch::Matchers::WithinAbs(15.0, 1e-12));
    }
    SECTION("structurally empty operands") {
        SMat A(3, 4), B(4, 2);            // no entries inserted at all
        const auto C = mtl::spgemm(A, B);
        require_well_formed(C);
        REQUIRE(C.num_rows() == 3u);
        REQUIRE(C.num_cols() == 2u);
        REQUIRE(C.nnz() == 0u);
    }
    SECTION("zero-sized result") {
        SMat A(0, 0), B(0, 0);
        const auto C = mtl::spgemm(A, B);
        REQUIRE(C.num_rows() == 0u);
        REQUIRE(C.num_cols() == 0u);
        REQUIRE(C.nnz() == 0u);
        REQUIRE(C.ref_major().size() == 1u);       // starts_ is always n+1
    }
    SECTION("inner dimension zero") {
        // (2x0) * (0x3): the contraction is over nothing, so C is 2x3 of
        // zeros -- shaped from the OUTER dimensions, which a result sized off
        // the wrong operand would get wrong.
        SMat A(2, 0), B(0, 3);
        const auto C = mtl::spgemm(A, B);
        require_well_formed(C);
        REQUIRE(C.num_rows() == 2u);
        REQUIRE(C.num_cols() == 3u);
        REQUIRE(C.nnz() == 0u);
    }
    SECTION("zero output columns") {
        // n == 0 is the case the accumulator is sized `n ? n : 1` for: a
        // zero-length workspace would be an invalid allocation to index into,
        // even though nothing is ever scattered.
        SMat A(2, 3), B(3, 0);
        const auto C = mtl::spgemm(A, B);
        REQUIRE(C.num_rows() == 2u);
        REQUIRE(C.num_cols() == 0u);
        REQUIRE(C.nnz() == 0u);
        REQUIRE(C.ref_major().size() == 3u);
    }
    SECTION("exact cancellation keeps the structural entry") {
        // (1, 1) * (1, -1)^T in the (0,0) slot cancels to exactly zero. The
        // entry STAYS in the pattern -- a Galerkin hierarchy wants a pattern
        // stable across re-assembly, not one that shifts with the numbers.
        SMat A(1, 2), B(2, 1);
        { mat::inserter<SMat> i(A); i[0][0] << 1.0; i[0][1] << 1.0; }
        { mat::inserter<SMat> i(B); i[0][0] << 1.0; i[1][0] << -1.0; }
        const auto C = mtl::spgemm(A, B);
        require_well_formed(C);
        REQUIRE(C.nnz() == 1u);
        REQUIRE_THAT(C(0, 0), Catch::Matchers::WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("spgemm mixed value types (#402)", "[operation][spgemm][sparse]") {
    // Result type is common_type_t, as for the other products.
    mat::compressed2D<float>  A(2, 2);
    mat::compressed2D<double> B(2, 2);
    { mat::inserter<mat::compressed2D<float>>  i(A); i[0][0] << 1.5f; i[1][1] << 2.0f; }
    { mat::inserter<mat::compressed2D<double>> i(B); i[0][0] << 4.0;  i[1][1] << 0.5;  }

    auto C = mtl::spgemm(A, B);
    static_assert(std::is_same_v<decltype(C), mat::compressed2D<double>>,
                  "float * double must widen to double");
    REQUIRE_THAT(C(0, 0), Catch::Matchers::WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(C(1, 1), Catch::Matchers::WithinAbs(1.0, 1e-12));
}
