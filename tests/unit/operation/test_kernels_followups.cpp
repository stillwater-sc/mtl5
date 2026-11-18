// Kernel follow-ups of issue 886 (QA sweep #872):
//
//   - matrix norms follow the storage layout, with the bits of the old loops
//     where the order allows, and walk a compressed2D's stored entries;
//   - the structural predicates on a compressed2D: the dense loop's answers,
//     in O(nnz);
//   - mult with a compressed2D A and a column-major dense or compressed B, and
//     mult<Accumulator>, through the stored entries;
//   - symm and gemmt with C aliasing A or B (both were wrong);
//   - kron with non-square operands, and shapes of the BLAS-3 family;
//   - BLAS dispatch: ger's mixed alpha, empty shapes, every gemm orientation.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/gemmt.hpp>
#include <mtl/operation/ger.hpp>
#include <mtl/operation/kron.hpp>
#include <mtl/operation/matrix_properties.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/syr2k.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/trmm.hpp>

using namespace mtl;

namespace {

using rm = mat::dense2D<double>;
using cm = mat::dense2D<double, mat::parameters<tag::col_major>>;
using crs = mat::compressed2D<double>;

double val(std::size_t i, std::size_t j) { return double(int((i * 7 + j * 3) % 11) - 5) / 4.0; }

template <typename M>
M filled(std::size_t m, std::size_t n) {
    M A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = val(i, j);
    return A;
}

/// A sparse matrix from the entries of `pattern(i, j)` (nonzero where true).
template <typename P>
crs make_sparse(std::size_t m, std::size_t n, P pattern) {
    crs S(m, n);
    {
        mat::inserter<crs> ins(S);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (pattern(i, j)) ins[i][j] << val(i, j) + (i == j ? 4.0 : 0.0);
    }
    return S;
}

rm dense_of(const crs& S) {
    rm D(S.num_rows(), S.num_cols());
    for (std::size_t i = 0; i < S.num_rows(); ++i)
        for (std::size_t j = 0; j < S.num_cols(); ++j) D(i, j) = S(i, j);
    return D;
}

/// The norms as the old loops computed them: one_norm column by column,
/// infinity_norm row by row.
double one_norm_ref(const rm& A) {
    double r = 0;
    for (std::size_t c = 0; c < A.num_cols(); ++c) {
        double s = 0;
        for (std::size_t i = 0; i < A.num_rows(); ++i) s += std::abs(A(i, c));
        r = std::max(r, s);
    }
    return r;
}
double inf_norm_ref(const rm& A) {
    double r = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double s = 0;
        for (std::size_t c = 0; c < A.num_cols(); ++c) s += std::abs(A(i, c));
        r = std::max(r, s);
    }
    return r;
}

/// The largest |A - B|; infinity if any difference is NaN (std::max kept
/// the previous value, so an all-NaN result read as 0).
template <typename MA, typename MB>
double max_diff(const MA& A, const MB& B) {
    double d = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            const double e = std::abs(double(A(i, j)) - double(B(i, j)));
            if (std::isnan(e)) return std::numeric_limits<double>::infinity();
            d = std::max(d, e);
        }
    return d;
}

} // namespace

// -- norms -------------------------------------------------------------------

TEST_CASE("matrix norms are the same on every layout and on sparse storage (issue 886)", "[operation][norms][886]") {
    const std::size_t m = 37, n = 23;
    const rm R = filled<rm>(m, n);
    const cm C = filled<cm>(m, n);
    // The buffered one_norm (row-major) and infinity_norm (column-major) sum
    // each column (row) in the old loops' order: identical bits.
    CHECK(one_norm(R) == one_norm_ref(R));
    CHECK(one_norm(C) == one_norm_ref(R));
    CHECK(infinity_norm(R) == inf_norm_ref(R));
    CHECK(infinity_norm(C) == inf_norm_ref(R));
    // frobenius_norm on column-major sums in another order: close.
    CHECK(std::abs(frobenius_norm(C) - frobenius_norm(R)) <= 1e-14 * frobenius_norm(R));

    const crs S = make_sparse(m, n, [](std::size_t i, std::size_t j) { return (i + 2 * j) % 5 == 0 || i == j; });
    const rm D = dense_of(S);
    CHECK(one_norm(S) == one_norm(D));
    CHECK(infinity_norm(S) == infinity_norm(D));
    CHECK(frobenius_norm(S) == frobenius_norm(D));   // a skipped zero adds +0
    // NaN in a stored entry still propagates.
    crs N(3, 3);
    {
        mat::inserter<crs> ins(N);
        ins[1][2] << std::numeric_limits<double>::quiet_NaN();
    }
    CHECK(std::isnan(one_norm(N)));
    CHECK(std::isnan(infinity_norm(N)));
    CHECK(std::isnan(frobenius_norm(N)));
}

TEST_CASE("sparse norms and predicates are O(nnz) (issue 886)", "[operation][norms][886]") {
    // A 200000 x 200000 diagonal: the m x n loops were 4e10 binary searches,
    // which would not finish within the test timeout; O(nnz) is instant.
    const std::size_t n = 200000;
    crs D(n, n);
    {
        mat::inserter<crs> ins(D);
        for (std::size_t i = 0; i < n; ++i) ins[i][i] << 2.0;
    }
    CHECK(frobenius_norm(D) == std::sqrt(4.0 * double(n)));
    CHECK(one_norm(D) == 2.0);
    CHECK(infinity_norm(D) == 2.0);
    CHECK(is_diagonal(D));
    CHECK(is_symmetric(D));
    CHECK(is_upper_triangular(D));
    CHECK(is_lower_triangular(D));
    CHECK(is_banded(D, 0, 0));
    CHECK(is_diagonally_dominant(D));
    // The explicit-Accumulator frobenius_norm walks the stored entries too.
    CHECK(double(frobenius_norm<long double>(D)) == std::sqrt(4.0 * double(n)));
    // A 1 x 1e8 row with one entry: a num_cols() column-sum buffer would be
    // 800 MB; the wide-matrix path sorts the stored entries instead.
    crs W(1, 100000000);
    {
        mat::inserter<crs> ins(W);
        ins[0][99999999] << -3.5;
    }
    CHECK(one_norm(W) == 3.5);
    CHECK(infinity_norm(W) == 3.5);
    // The sorted path, also on a wide matrix with several entries per column.
    // Column 7's sum (1 + 2 + 2.5 = 5.5) must win over the lone 4.0, and every
    // single entry is below both: a bug that stops summing within a column
    // gives 4 or 2.5, and one that sums across columns more (issue 909: with
    // 1, -2, 0.5 the lone 4.0 was the answer either way).
    crs V(3, 50000000);
    {
        mat::inserter<crs> ins(V);
        ins[0][7] << 1.0; ins[1][7] << -2.0; ins[2][7] << 2.5; ins[1][40000000] << 4.0;
    }
    CHECK(one_norm(V) == 5.5);
    // Empty dense matrices return 0 before any buffer: a row-major 0 x 1e8
    // one_norm allocated 800 MB, as did a column-major 1e8 x 0 infinity_norm.
    CHECK(one_norm(rm(0, 100000000)) == 0.0);
    CHECK(infinity_norm(cm(100000000, 0)) == 0.0);
    CHECK(one_norm(cm(0, 5)) == 0.0);
    CHECK(infinity_norm(rm(5, 0)) == 0.0);
}

TEST_CASE("structural predicates on a compressed2D agree with the dense loops (issue 886)", "[operation][properties][886]") {
    const std::size_t n = 12;
    auto check_same = [](const crs& S) {
        const rm D = dense_of(S);
        CHECK(is_symmetric(S) == is_symmetric(D));
        CHECK(is_hermitian(S) == is_hermitian(D));
        CHECK(is_upper_triangular(S) == is_upper_triangular(D));
        CHECK(is_lower_triangular(S) == is_lower_triangular(D));
        CHECK(is_diagonal(S) == is_diagonal(D));
        CHECK(is_banded(S, 1, 2) == is_banded(D, 1, 2));
        CHECK(is_diagonally_dominant(S) == is_diagonally_dominant(D));
        CHECK(is_diagonally_dominant(S, true) == is_diagonally_dominant(D, true));
    };
    check_same(make_sparse(n, n, [](std::size_t i, std::size_t j) { return i == j; }));
    check_same(make_sparse(n, n, [](std::size_t i, std::size_t j) { return j >= i && j - i <= 2; }));       // upper banded
    check_same(make_sparse(n, n, [](std::size_t i, std::size_t j) { return i >= j && i - j <= 1; }));       // lower banded
    check_same(make_sparse(n, n, [](std::size_t i, std::size_t j) { return (i + j) % 3 == 0; }));           // symmetric pattern
    check_same(make_sparse(n, n, [](std::size_t i, std::size_t j) { return i == j || j == i + 3; }));       // an unmatched transpose
    // Symmetric values, and one entry whose transpose is not stored.
    crs Y(3, 3);
    {
        mat::inserter<crs> ins(Y);
        ins[0][0] << 1.0; ins[0][1] << 2.0; ins[1][0] << 2.0; ins[2][2] << 5.0;
    }
    CHECK(is_symmetric(Y));
    crs Z(3, 3);
    {
        mat::inserter<crs> ins(Z);
        ins[0][2] << 1.0;
    }
    CHECK(!is_symmetric(Z));
    CHECK(!is_lower_triangular(Z));
    CHECK(is_upper_triangular(Z));
}

// -- sparse mult ----------------------------------------------------------------

TEST_CASE("mult with a compressed2D A and any B, and mult<Accumulator> (issue 886)", "[operation][mult][886]") {
    const std::size_t m = 17, k = 13, n = 5;
    const crs A = make_sparse(m, k, [](std::size_t i, std::size_t j) { return (i * 3 + j) % 4 == 0; });
    const rm Ad = dense_of(A);
    const cm Bc = filled<cm>(k, n);
    const crs Bs = make_sparse(k, n, [](std::size_t i, std::size_t j) { return (i + j) % 2 == 0; });
    rm ref(m, n);
    {
        rm Bd(k, n);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < n; ++j) Bd(i, j) = Bc(i, j);
        mult(Ad, Bd, ref);
    }
    cm Cc(m, n);
    mult(A, Bc, Cc);                           // CRS x column-major into column-major
    CHECK(max_diff(Cc, ref) == 0.0);
    rm Cr(m, n);
    mult(A, Bc, Cr);                           // CRS x column-major into row-major
    CHECK(max_diff(Cr, ref) == 0.0);
    rm Cs(m, n), refs(m, n);
    mult(A, Bs, Cs);                           // CRS x CRS into dense
    mult(Ad, dense_of(Bs), refs);
    CHECK(max_diff(Cs, refs) == 0.0);
    rm Ca(m, n);
    mult<long double>(A, Bc, Ca);              // an Accumulator
    CHECK(max_diff(Ca, ref) <= 1e-13);
}

// -- aliasing --------------------------------------------------------------------

TEST_CASE("symm and gemmt with C aliasing an operand (issue 886)", "[operation][blas3][886]") {
    rm S = filled<rm>(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j) S(j, i) = S(i, j);
    {
        // symm(1, S, C, 0, C): every entry was wrong.
        rm C = filled<rm>(4, 4), B = C, ref(4, 4);
        mult(S, B, ref);
        symm(1.0, S, C, 0.0, C);
        CHECK(max_diff(C, ref) <= 1e-13);
    }
    {
        // With beta != 0, beta * C uses C's old values.
        rm C = filled<rm>(4, 4), B = C, ref(4, 4);
        mult(S, B, ref);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) ref(i, j) += 2.0 * B(i, j);
        symm(1.0, S, C, 2.0, C);
        CHECK(max_diff(C, ref) <= 1e-13);
    }
    {
        // gemmt(1, A, C, 1, C): C read as B while written.
        const rm A = filled<rm>(4, 4);
        rm C = S, B = C, ref(4, 4);
        mult(A, B, ref);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) ref(i, j) += B(i, j);
        gemmt(1.0, A, C, 1.0, C);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C(i, j) - ref(i, j)) <= 1e-13);
    }
    {
        // syrk and syr2k with C as A, and gemmt with C as A: these were right;
        // pinned.
        rm C = S, A = C, ref(4, 4), At(4, 4);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) At(j, i) = A(i, j);
        mult(A, At, ref);
        syrk(1.0, C, 0.0, C);   // right natively, wrong under BLAS's ?syrk
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C(i, j) - ref(i, j)) <= 1e-13);
    }
    {
        // syr2k with C as A, and with beta != 0.
        rm C = S, A = C, B = filled<rm>(4, 4), ref(4, 4), t(4, 4), At(4, 4), Bt(4, 4);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) { At(j, i) = A(i, j); Bt(j, i) = B(i, j); }
        mult(A, Bt, ref);
        mult(B, At, t);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) ref(i, j) += t(i, j) + 3.0 * A(i, j);
        syr2k(1.0, C, B, 3.0, C);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C(i, j) - ref(i, j)) <= 1e-12);
    }
    {
        // Lazy expression operands that read C: storage_overlaps can't see
        // into them (mult(S, C + C, C) gave [[8,12],[22,32]], mult(C + C, S, C)
        // gave zeros).
        const rm S2 = {{1, 1}, {1, 1}};
        rm C = {{1, 2}, {3, 4}};
        mult(S2, C + C, C);
        CHECK(C(1, 0) == 8.0); CHECK(C(1, 1) == 12.0);
        C = rm{{1, 2}, {3, 4}};
        mult(C + C, S2, C);
        CHECK(C(0, 0) == 6.0); CHECK(C(1, 1) == 14.0);
        C = rm{{1, 2}, {3, 4}};
        symm(1.0, S2, C + C, 0.0, C);
        CHECK(C(1, 0) == 8.0); CHECK(C(1, 1) == 12.0);
        C = rm{{1, 2}, {3, 4}};
        gemmt(1.0, S2, C + C, 0.0, C);
        CHECK(C(1, 0) == 8.0); CHECK(C(1, 1) == 12.0);
        rm D = {{1, 2}, {3, 4}};
        syrk(1.0, D * 2.0, 0.0, D);   // (2D)(2D)^T: [[20, 44], [44, 100]]
        CHECK(D(1, 0) == 44.0); CHECK(D(1, 1) == 100.0);
        rm E = {{1, 2}, {3, 4}};
        syr2k(1.0, E + E, S2, 0.0, E);   // 2E S2^T + S2 2E^T
        CHECK(E(1, 0) == 20.0); CHECK(E(1, 1) == 28.0);
    }
    {
        // symm evaluates only the lower triangle of an expression A: here the
        // upper entries of U + U would overflow int (undefined; a UBSan build
        // flags the old full copy).
        const int big = std::numeric_limits<int>::max();
        const mat::dense2D<int> U = {{1, big}, {2, 3}};
        const mat::dense2D<int> B = {{1, 0}, {0, 1}};
        mat::dense2D<int> C(2, 2);
        symm(1, U + U, B, 0, C);   // lower of 2U, read symmetric: [[2, 4], [4, 6]]
        CHECK(C(0, 0) == 2); CHECK(C(0, 1) == 4); CHECK(C(1, 0) == 4); CHECK(C(1, 1) == 6);
    }
    {
        // An empty output returns before an expression operand is evaluated.
        const rm S3 = filled<rm>(3, 3);
        rm B(3, 0), C(3, 0);
        mult(S3 + S3, B, C);
        CHECK(C.num_cols() == 0);
    }
    {
        // trmm with an expression A keeps A's element type: a copy in B's
        // (int) type turned 1.5 into 1 and gave 2.
        const rm A = {{1.5}};
        mat::dense2D<int> B = {{2}};
        trmm(1.0, A * 1.0, B, true);
        CHECK(B(0, 0) == 3);
    }
    {
        // trmm(0, B, B): zero-filled, without copying A first.
        rm B = filled<rm>(3, 3);
        trmm(0.0, B, B, true);
        CHECK(max_diff(B, rm(3, 3)) == 0.0);
    }
    {
        // trmm(1, B, B, upper): A is B.
        rm U = filled<rm>(4, 4);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < i; ++j) U(i, j) = 0.0;
        rm B = U, ref(4, 4);
        mult(U, U, ref);
        trmm(1.0, B, B, true);
        CHECK(max_diff(B, ref) <= 1e-13);
    }
}

// -- shapes ------------------------------------------------------------------------

TEST_CASE("max_diff reports a NaN difference", "[operation][886]") {
    rm A = {{1, 2}}, B = {{1, 2}};
    CHECK(max_diff(A, B) == 0.0);
    A(0, 1) = std::numeric_limits<double>::quiet_NaN();
    CHECK(max_diff(A, B) == std::numeric_limits<double>::infinity());
}

TEST_CASE("kron with non-square operands (issue 886)", "[operation][kron][886]") {
    // 2x3 (x) 3x2: a swap of rb/cb or ra/ca passed every square case.
    const rm A = filled<rm>(2, 3);
    const rm B = filled<rm>(3, 2);
    const auto K = kron(A, B);
    REQUIRE(K.num_rows() == 6);
    REQUIRE(K.num_cols() == 6);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t p = 0; p < 3; ++p)
                for (std::size_t q = 0; q < 2; ++q) CHECK(K(i * 3 + p, j * 2 + q) == A(i, j) * B(p, q));
    const auto L = kron(rm(filled<rm>(1, 4)), rm(filled<rm>(3, 1)));
    CHECK(L.num_rows() == 3);
    CHECK(L.num_cols() == 4);
}

TEST_CASE("the BLAS-3 family on non-square and 1x1 shapes (issue 886)", "[operation][blas3][886]") {
    // syrk: C (n x n) = A A^T with A n x k, k != n; and 1x1.
    for (const auto& [nn, kk] : std::vector<std::pair<std::size_t, std::size_t>>{{3, 5}, {5, 2}, {1, 1}, {1, 4}}) {
        INFO(nn << " x " << kk);
        const rm A = filled<rm>(nn, kk);
        rm At(kk, nn), ref(nn, nn), C(nn, nn);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j < kk; ++j) At(j, i) = A(i, j);
        mult(A, At, ref);
        syrk(1.0, A, 0.0, C);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C(i, j) - ref(i, j)) <= 1e-13);
        // syr2k with B = 2A: 4 A A^T.
        rm B(nn, kk), C2(nn, nn);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j < kk; ++j) B(i, j) = 2.0 * A(i, j);
        syr2k(1.0, A, B, 0.0, C2);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C2(i, j) - 4.0 * ref(i, j)) <= 1e-12);
        // gemmt: C = A * At, lower triangle.
        rm C3(nn, nn);
        gemmt(1.0, A, At, 0.0, C3);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C3(i, j) - ref(i, j)) <= 1e-13);
    }
    // symm: C (m x n) = S B, S m x m symmetric, B m x n with n != m.
    {
        rm S = filled<rm>(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < i; ++j) S(j, i) = S(i, j);
        const rm B = filled<rm>(3, 5);
        rm C(3, 5), ref(3, 5);
        mult(S, B, ref);
        symm(1.0, S, B, 0.0, C);
        CHECK(max_diff(C, ref) <= 1e-13);
    }
    // trmm: B (m x n) = U B, U upper m x m, n != m; and 1x1.
    for (const auto& [mm, nn] : std::vector<std::pair<std::size_t, std::size_t>>{{3, 5}, {1, 1}, {4, 1}}) {
        rm U = filled<rm>(mm, mm);
        for (std::size_t i = 0; i < mm; ++i)
            for (std::size_t j = 0; j < i; ++j) U(i, j) = 0.0;
        rm B = filled<rm>(mm, nn), ref(mm, nn);
        mult(U, B, ref);
        trmm(1.0, U, B, true);
        CHECK(max_diff(B, ref) <= 1e-13);
    }
    // ger: A (m x n) += x y^T, m != n.
    {
        rm A = filled<rm>(3, 5), ref = A;
        vec::dense_vector<double> x = {1.0, -2.0, 0.5}, y = {2.0, 1.0, 0.0, -1.0, 3.0};
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 5; ++j) ref(i, j) += 1.5 * x(i) * y(j);
        ger(1.5, x, y, A);
        CHECK(max_diff(A, ref) <= 1e-13);
    }
}

// -- BLAS dispatch ----------------------------------------------------------------------

TEST_CASE("BLAS dispatch: ger's mixed alpha, empty shapes, every gemm orientation (issue 886)", "[operation][blas][886]") {
    // A double alpha on a float A does not go to ?ger (it would narrow alpha):
    // the generic path, in every build.
    STATIC_REQUIRE(!detail::ger_dispatches_blas_v<void, double, mat::dense2D<float>,
                                                  vec::dense_vector<float>, vec::dense_vector<float>>);
    // 0 x N and N x 0 through row-major gemv and gemm.
    {
        rm A0(0, 4);
        vec::dense_vector<double> x(4, 1.0), y(0);
        mult(A0, x, y);
        CHECK(y.size() == 0);
        rm An(4, 0);
        vec::dense_vector<double> x0(0), y4(4, 7.0);
        mult(An, x0, y4);
        for (std::size_t i = 0; i < 4; ++i) CHECK(y4(i) == 0.0);   // an empty sum is 0
        rm B(0, 3), C(4, 3);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 3; ++j) C(i, j) = 9.0;
        mult(An, B, C);
        CHECK(max_diff(C, rm(4, 3)) == 0.0);
    }
    // All eight A/B/C orientation combinations of gemm against one reference.
    const rm Ar = filled<rm>(4, 3), Br = filled<rm>(3, 5);
    rm ref(4, 5);
    mult(Ar, Br, ref);
    auto run = [&](auto a, auto b, auto c) {
        using MA = decltype(a); using MB = decltype(b); using MC = decltype(c);
        MA A(4, 3); MB B(3, 5); MC C(4, 5);
        for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 3; ++j) A(i, j) = Ar(i, j);
        for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 5; ++j) B(i, j) = Br(i, j);
        mult(A, B, C);
        CHECK(max_diff(C, ref) <= 1e-13);
    };
    run(rm(), rm(), rm()); run(rm(), rm(), cm()); run(rm(), cm(), rm()); run(rm(), cm(), cm());
    run(cm(), rm(), rm()); run(cm(), rm(), cm()); run(cm(), cm(), rm()); run(cm(), cm(), cm());
}

namespace order_record {
inline std::vector<double>& last_seen() { static std::vector<double> v; return v; }
/// An accumulator (the ADL hooks) that records the magnitudes in the order
/// they are added, so a test can see the walk (issue 909).
struct record_acc { std::vector<double> seen; double s; };
inline void accumulator_clear(record_acc& a) { a.seen.clear(); a.s = 0.0; }
template <typename V>
void accumulator_assign(record_acc& a, const V& v) { a.seen.assign(1, double(v)); a.s = double(v) * double(v); }
template <typename V>
void accumulator_add_product(record_acc& a, const V& x, const V& y) { a.seen.push_back(double(x)); a.s += double(x) * double(y); }
template <typename R>
R accumulator_value(const record_acc& a, std::type_identity<R>) {
    last_seen() = a.seen;
    return static_cast<R>(a.s);
}
} // namespace order_record

// frobenius_norm<Accumulator> walks a column-major matrix by columns (issue
// 909): only the default branch was made layout-aware in issue 886, and the
// Accumulator branch read a column-major matrix by rows.
TEST_CASE("frobenius_norm<Accumulator> walks the storage order (issue 909)", "[operation][norms][909]") {
    rm R(2, 3);
    cm C(2, 3);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) R(i, j) = C(i, j) = double(1 + i + 2 * j);   // 1 3 5 / 2 4 6
    const double want = std::sqrt(1.0 + 4 + 9 + 16 + 25 + 36);
    CHECK(frobenius_norm<order_record::record_acc>(C) == want);
    CHECK(order_record::last_seen() == std::vector<double>({1, 2, 3, 4, 5, 6}));   // column by column
    CHECK(frobenius_norm<order_record::record_acc>(R) == want);
    CHECK(order_record::last_seen() == std::vector<double>({1, 3, 5, 2, 4, 6}));   // row by row
}

// The dense copy of an expression operand stays (issue 909, option a), so a
// large sparse expression is formed as a compressed2D first, which mult takes
// without a dense copy: here 2.0 * S as a copy of S with its values scaled. At
// 200000 x 200000 a dense copy would be 320 GB.
TEST_CASE("a large sparse operand formed as a compressed2D is not copied dense (issue 909)",
          "[operation][mult][909]") {
    const std::size_t n = 200000;
    crs S(n, n);
    {
        mat::inserter<crs> ins(S);
        for (std::size_t i = 0; i < n; i += 1000) ins[i][i] << 1.5;
    }
    crs T = S;                                   // 2.0 * S, formed sparse
    for (auto& v : T.ref_data()) v *= 2.0;
    static_assert(!detail::holds_expression_v<crs>);
    rm B(n, 1), C(n, 1);
    for (std::size_t i = 0; i < n; ++i) B(i, 0) = 1.0;
    mult(T, B, C);
    CHECK(C(0, 0) == 3.0);
    CHECK(C(1000, 0) == 3.0);
    CHECK(C(1, 0) == 0.0);
}

// frobenius_norm<Accumulator> on empty matrices submits no work (#911 review)
// and still rounds out through the accumulator.
TEST_CASE("frobenius_norm<Accumulator> of an empty column-major matrix (issue 909)", "[operation][norms][909]") {
    CHECK(frobenius_norm<long double>(cm(0, 100000000)) == 0.0);
    CHECK(frobenius_norm<long double>(cm(100000000, 0)) == 0.0);
}
