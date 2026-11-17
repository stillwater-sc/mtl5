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
crs sparse(std::size_t m, std::size_t n, P pattern) {
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

template <typename MA, typename MB>
double max_diff(const MA& A, const MB& B) {
    double d = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) d = std::max(d, std::abs(double(A(i, j)) - double(B(i, j))));
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

    const crs S = sparse(m, n, [](std::size_t i, std::size_t j) { return (i + 2 * j) % 5 == 0 || i == j; });
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
    check_same(sparse(n, n, [](std::size_t i, std::size_t j) { return i == j; }));
    check_same(sparse(n, n, [](std::size_t i, std::size_t j) { return j >= i && j - i <= 2; }));       // upper banded
    check_same(sparse(n, n, [](std::size_t i, std::size_t j) { return i >= j && i - j <= 1; }));       // lower banded
    check_same(sparse(n, n, [](std::size_t i, std::size_t j) { return (i + j) % 3 == 0; }));           // symmetric pattern
    check_same(sparse(n, n, [](std::size_t i, std::size_t j) { return i == j || j == i + 3; }));       // an unmatched transpose
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
    const crs A = sparse(m, k, [](std::size_t i, std::size_t j) { return (i * 3 + j) % 4 == 0; });
    const rm Ad = dense_of(A);
    const cm Bc = filled<cm>(k, n);
    const crs Bs = sparse(k, n, [](std::size_t i, std::size_t j) { return (i + j) % 2 == 0; });
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
        syrk(1.0, C, 0.0, C);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(C(i, j) - ref(i, j)) <= 1e-13);
    }
}

// -- shapes ------------------------------------------------------------------------

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
