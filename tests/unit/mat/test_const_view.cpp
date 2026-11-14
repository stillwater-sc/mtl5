// Read-only matrix views over a caller's const buffer (tag::read_only).
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#include <mtl/mtl.hpp>

using namespace mtl;
using cmat = mat::const_matrix_view<double>;

namespace {
template <typename M> concept element_writable = requires(M m) { m(0, 0) = 1.0; };
template <typename M> concept resizable        = requires(M m) { m.change_dim(2, 2); };
template <typename M> concept scalable         = requires(M m) { m *= 2.0; };
template <typename M> concept expr_assignable  = requires(M m, const M& w) { m = w + w; };
template <typename M> concept data_writable    = requires(M m) { *m.data() = 1.0; };
} // namespace

TEST_CASE("const_view: writing through a read-only matrix does not compile", "[mat][const_view]") {
    STATIC_REQUIRE(cmat::read_only);
    STATIC_REQUIRE(!element_writable<cmat>);
    STATIC_REQUIRE(!resizable<cmat>);
    STATIC_REQUIRE(!scalable<cmat>);
    STATIC_REQUIRE(!expr_assignable<cmat>);
    STATIC_REQUIRE(!data_writable<cmat>);
    STATIC_REQUIRE(!std::is_copy_assignable_v<cmat>);
    STATIC_REQUIRE(!std::is_move_assignable_v<cmat>);
    STATIC_REQUIRE(!std::is_constructible_v<cmat, std::size_t, std::size_t>);
    STATIC_REQUIRE(!std::is_constructible_v<mat::dense2D<double>, std::size_t, std::size_t, const double*>);
    STATIC_REQUIRE(element_writable<mat::dense2D<double>>);
}

TEST_CASE("const_view: reads a const buffer in either orientation", "[mat][const_view]") {
    const double buf[6] = {1, 2, 3, 4, 5, 6};
    const auto A = mat::const_view(2, 3, buf);   // row-major
    REQUIRE(A.num_rows() == 2);
    REQUIRE(A.num_cols() == 3);
    REQUIRE(A.data() == buf);
    REQUIRE(!A.owns_memory());
    REQUIRE(A(1, 0) == 4.0);
    REQUIRE(A(0, 2) == 3.0);

    const mat::const_matrix_view<double, tag::col_major> B(2, 3, buf);
    REQUIRE(B(1, 0) == 2.0);
    REQUIRE(B(0, 2) == 5.0);

    const cmat C = A;   // another view of the same buffer
    REQUIRE(C.data() == buf);
    REQUIRE(C(1, 2) == 6.0);
}

TEST_CASE("const_view works with the matrix operations", "[mat][const_view]") {
    const double ab[4] = {2, 1, 1, 3};
    const double xb[2] = {1, 2};
    const auto A = mat::const_view(2, 2, ab);
    const auto x = vec::const_view(2, xb);

    const vec::dense_vector<double> y = A * x;
    REQUIRE(y(0) == 4.0);
    REQUIRE(y(1) == 7.0);

    mat::dense2D<double> C(2, 2);
    mult(A, A, C);
    REQUIRE(C(0, 0) == 5.0);
    REQUIRE(C(1, 1) == 10.0);
    const mat::dense2D<double> S = A + A;
    REQUIRE(S(1, 0) == 2.0);
    const auto T = trans(A);
    REQUIRE(T(0, 1) == 1.0);
    REQUIRE(T(1, 0) == 1.0);
    REQUIRE(frobenius_norm(A) > 3.8);

    const mat::dense2D<double> M(A);   // an owning copy of a view
    REQUIRE(M.owns_memory());
    REQUIRE(M(1, 1) == 3.0);

    // Across orientations, on a rectangular, non-symmetric source, so a
    // transposed copy or a wrong leading dimension would show.
    const double cbuf[6] = {1, 2, 3, 4, 5, 6};
    const mat::const_matrix_view<double, tag::col_major> Bc(2, 3, cbuf);
    const mat::dense2D<double> N(Bc);   // column-major view -> row-major owner
    REQUIRE(N.owns_memory());
    REQUIRE(N(0, 0) == 1.0);
    REQUIRE(N(0, 1) == 3.0);
    REQUIRE(N(0, 2) == 5.0);
    REQUIRE(N(1, 0) == 2.0);
    REQUIRE(N(1, 1) == 4.0);
    REQUIRE(N(1, 2) == 6.0);

    const auto Br = mat::const_view(2, 3, cbuf);   // row-major view -> column-major owner
    const mat::dense2D<double, mat::parameters<tag::col_major>> P(Br);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE(P(r, c) == cbuf[r * 3 + c]);
    REQUIRE(P.data()[1] == 4.0);   // column-major storage: (1, 0) is second

    mat::dense2D<double> W{{1, 2}, {3, 4}};
    const auto Wv = mat::const_view(W);
    REQUIRE(Wv.data() == W.data());
    W(0, 0) = 10;
    REQUIRE(Wv(0, 0) == 10.0);
}
