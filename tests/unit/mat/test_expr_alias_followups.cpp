// Assigning an expression that reads its own target (#824, 6022337): the
// cases the QA sweep in #832 found untested, and sparse and block operands,
// which go in place unless their storage overlaps the target's (#857).
// Also the index checks of permutation_matrix::swap_rows and map_view.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtl/config.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/mat/permutation_matrix.hpp>
#include <mtl/mat/view/map_view.hpp>
#include <mtl/operation/trans.hpp>

using namespace mtl;

namespace {

template <typename M>
M numbered(std::size_t n) {
    M A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = typename M::value_type(3 * i + j + 1);
    return A;
}

template <typename M>
bool equal(const M& A, const M& B) {
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            if (!(A(i, j) == B(i, j))) return false;
    return true;
}

} // namespace

TEST_CASE("dense2D<float> += an aliased expression (1 * trans(A))", "[mat][alias][857]") {
    auto A = numbered<mat::dense2D<float>>(3);
    mat::dense2D<float> want(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) want(i, j) = A(i, j) + A(j, i);
    A += 1.0f * trans(A);   // the scaled view is an expression; += takes expressions
    CHECK(equal(A, want));
}

TEST_CASE("a fixed-size dense2D target reads itself", "[mat][alias][fixed][857]") {
    using F = mat::dense2D<double, mat::parameters<tag::row_major, mtl::detail::c_index,
                                                   mat::fixed::dimensions<3, 3>, tag::on_stack>>;
    F A;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = double(3 * i + j + 1);
    F want;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) want(i, j) = A(i, j) + A(j, i);
    A = A + trans(A);
    CHECK(equal(A, want));
}

TEST_CASE("A -= A * A through the lazy product", "[mat][alias][857]") {
    auto A = numbered<mat::dense2D<double>>(4);
    mat::dense2D<double> want(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double s = 0;
            for (std::size_t k = 0; k < 4; ++k) s += A(i, k) * A(k, j);
            want(i, j) = A(i, j) - s;
        }
    A -= A * A;
    CHECK(equal(A, want));
}

TEST_CASE("sparse operands: in place, unless they share the target's storage", "[mat][alias][857]") {
    mat::compressed2D<double> S(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        ins[0][0] << 1.0; ins[0][1] << 2.0; ins[1][1] << 3.0;
    }
    // An ordinary dense target: the result is right either way.
    mat::dense2D<double> A(2, 2);
    A(0, 0) = 10; A(0, 1) = 20; A(1, 0) = 30; A(1, 1) = 40;
    A += S + S;
    CHECK((A(0, 0) == 12 && A(0, 1) == 24 && A(1, 0) == 30 && A(1, 1) == 46));
    // A column-major target over a full S's value array, which holds S row
    // by row: T(1, 0) is S(0, 1)'s storage. Written in place, S + S would read
    // values the assignment already overwrote, so it must take a temporary.
    mat::compressed2D<double> F(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(F);
        ins[0][0] << 1.0; ins[0][1] << 2.0; ins[1][0] << 3.0; ins[1][1] << 4.0;
    }
    REQUIRE(F.ref_data().size() == 4);
    mat::dense2D<double, mat::parameters<tag::col_major>> T(2, 2, F.ref_data().data());
    T = F + F;
    // T(r, c) = 2 F(r, c), stored column by column over F's values.
    const auto& v = F.ref_data();
    CHECK((v[0] == 2.0 && v[1] == 6.0 && v[2] == 4.0 && v[3] == 8.0));
}

TEST_CASE("swap_rows and map_view check their indices", "[mat][857]") {
    mat::permutation_matrix<double> P(3);
    if constexpr (bounds_checking) {
        CHECK_THROWS_AS(P.swap_rows(0, 3), std::out_of_range);
        CHECK_THROWS_AS(P.swap_rows(5, 1), std::out_of_range);
    }
    P.swap_rows(0, 2);
    CHECK((P.permutation()[0] == 2 && P.permutation()[2] == 0));

    const auto A = numbered<mat::dense2D<double>>(3);
    CHECK_THROWS_AS((mat::view::map_view(A, {0, 3}, {0})), std::out_of_range);
    mat::view::map_view V(A, {2, 0}, {1});
    CHECK(V(0, 0) == A(2, 1));
    if constexpr (bounds_checking) {
        CHECK_THROWS_AS(V(2, 0), std::out_of_range);
        CHECK_THROWS_AS(V(0, 1), std::out_of_range);
    }
}

namespace {

template <typename M>
mtl::detail::alias_target target_of(const M& A) {
    const auto* lo = reinterpret_cast<const unsigned char*>(A.data());
    return {&A, lo, lo + A.num_rows() * A.num_cols() * sizeof(typename M::value_type)};
}

mat::compressed2D<double> diagonal_sparse(std::size_t m, std::size_t n) {
    mat::compressed2D<double> S(m, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        for (std::size_t i = 0; i < m && i < n; ++i) ins[i][i] << double(i + 1);
    }
    return S;
}

} // namespace

TEST_CASE("the alias decision for sparse operands", "[mat][alias][857]") {
    // Disjoint sparse storage is read in place; it used to take a temporary
    // on every assignment. Checked on the decision itself, so a regression to
    // "always a temporary" fails here, not only in a benchmark.
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{0, 0}, {1, 1}, {3, 3}, {2, 5}, {5, 2}}) {
        INFO(m << " x " << n);
        const auto S1 = diagonal_sparse(m, n), S2 = diagonal_sparse(m, n);
        mat::dense2D<double> A(m, n);
        const auto t = target_of(A);
        CHECK_FALSE(mtl::detail::alias_reads_target_elsewhere(S1 + S2, t));
        CHECK_FALSE(mtl::detail::alias_reads_target_elsewhere(2.0 * S1, t));
        // The same expression assigned: the values are right.
        A = S1 + S2;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) CHECK(A(i, j) == (i == j ? 2.0 * double(i + 1) : 0.0));
    }
    // A target over a sparse operand's values: overlapping, so a temporary.
    mat::compressed2D<double> F(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(F);
        ins[0][0] << 1.0; ins[0][1] << 2.0; ins[1][0] << 3.0; ins[1][1] << 4.0;
    }
    mat::dense2D<double, mat::parameters<tag::col_major>> T(2, 2, F.ref_data().data());
    CHECK(mtl::detail::alias_reads_target_elsewhere(F + F, target_of(T)));
    // A dense operand that is not the target never needs one; one over the
    // target's buffer does, through a view.
    mat::dense2D<double> B(2, 2), C(2, 2);
    CHECK_FALSE(mtl::detail::alias_reads_target_elsewhere(B + C, target_of(B)));
    CHECK(mtl::detail::alias_reads_target_elsewhere(B + trans(B), target_of(B)));
}
