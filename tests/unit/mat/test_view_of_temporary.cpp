// The view factories reject a temporary (#695).
//
// Every view in mat/view/ holds a reference to its matrix, so a view of a
// temporary dangles once the full expression ends:
//
//     auto t = trans(A * B);   // A * B is destroyed at the semicolon
//
// Each factory has a deleted `const M&&` overload, which an rvalue prefers over
// `const M&`, so the call fails to compile instead. compile_fail/
// trans_of_temporary.cpp checks the diagnostic; this file checks every factory
// with concepts, and that lvalues still work.
#include <catch2/catch_test_macros.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/view/banded_view.hpp>
#include <mtl/mat/view/hermitian_view.hpp>
#include <mtl/mat/view/lower_view.hpp>
#include <mtl/mat/view/map_view.hpp>
#include <mtl/mat/view/strict_lower_view.hpp>
#include <mtl/mat/view/strict_upper_view.hpp>
#include <mtl/mat/view/upper_view.hpp>
#include <mtl/operation/trans.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

using D = mtl::mat::dense2D<double>;
using index_map = std::vector<D::size_type>;

template <class T> concept can_trans        = requires(T&& t) { mtl::trans(std::forward<T>(t)); };
template <class T> concept can_hermitian    = requires(T&& t) { mtl::hermitian(std::forward<T>(t)); };
template <class T> concept can_upper        = requires(T&& t) { mtl::upper(std::forward<T>(t)); };
template <class T> concept can_lower        = requires(T&& t) { mtl::lower(std::forward<T>(t)); };
template <class T> concept can_strict_upper = requires(T&& t) { mtl::strict_upper(std::forward<T>(t)); };
template <class T> concept can_strict_lower = requires(T&& t) { mtl::strict_lower(std::forward<T>(t)); };
template <class T> concept can_triu         = requires(T&& t) { mtl::triu(std::forward<T>(t), 1); };
template <class T> concept can_tril         = requires(T&& t) { mtl::tril(std::forward<T>(t), -1); };
template <class T> concept can_banded       = requires(T&& t) { mtl::banded(std::forward<T>(t), 1, 1); };
template <class T> concept can_mapped       = requires(T&& t, const index_map& r) {
    mtl::mapped(std::forward<T>(t), r, r);
};

// T is an lvalue (D&, const D&) or an rvalue (D, const D).
#define MTL5_VIEW_REJECTS_RVALUES(can)                                    \
    static_assert(can<D&>, #can ": a non-const lvalue makes a view");     \
    static_assert(can<const D&>, #can ": a const lvalue makes a view");   \
    static_assert(!can<D>, #can ": an rvalue must not make a view");      \
    static_assert(!can<const D>, #can ": a const rvalue must not make a view");

MTL5_VIEW_REJECTS_RVALUES(can_trans)
MTL5_VIEW_REJECTS_RVALUES(can_hermitian)
MTL5_VIEW_REJECTS_RVALUES(can_upper)
MTL5_VIEW_REJECTS_RVALUES(can_lower)
MTL5_VIEW_REJECTS_RVALUES(can_strict_upper)
MTL5_VIEW_REJECTS_RVALUES(can_strict_lower)
MTL5_VIEW_REJECTS_RVALUES(can_triu)
MTL5_VIEW_REJECTS_RVALUES(can_tril)
MTL5_VIEW_REJECTS_RVALUES(can_banded)
MTL5_VIEW_REJECTS_RVALUES(can_mapped)

// A view of a view: trans(upper(A)) passes upper's temporary to trans, so it
// is rejected too; a named upper view is accepted.
using Upper = decltype(mtl::upper(std::declval<D&>()));
static_assert(can_trans<Upper&>, "trans: a named view makes a view");
static_assert(can_trans<const Upper&>, "trans: a const named view makes a view");
static_assert(!can_trans<Upper>, "trans: a view rvalue must not make a view");
static_assert(!can_trans<const Upper>, "trans: a const view rvalue must not make a view");

#undef MTL5_VIEW_REJECTS_RVALUES

} // namespace

TEST_CASE("view factories still accept a named matrix", "[mat][view][695]") {
    D A(2, 3);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = static_cast<double>(10 * i + j);
    const D& cA = A;

    auto t = mtl::trans(A);
    REQUIRE(t.num_rows() == 3);
    REQUIRE(t.num_cols() == 2);
    REQUIRE(t(2, 1) == A(1, 2));
    REQUIRE(mtl::trans(cA)(0, 1) == A(1, 0));

    // A view of a view is made from a named view.
    auto tt = mtl::trans(t);
    REQUIRE(tt(1, 2) == A(1, 2));
    auto up = mtl::upper(A);
    REQUIRE(mtl::trans(up)(1, 0) == A(0, 1));

    REQUIRE(mtl::upper(A)(1, 0) == 0.0);
    REQUIRE(mtl::lower(A)(1, 0) == A(1, 0));
    REQUIRE(mtl::strict_upper(A)(0, 0) == 0.0);
    REQUIRE(mtl::strict_lower(A)(1, 0) == A(1, 0));
    REQUIRE(mtl::triu(A, 1)(0, 1) == A(0, 1));
    REQUIRE(mtl::tril(A, -1)(1, 0) == A(1, 0));
    REQUIRE(mtl::banded(A, 0, 0)(0, 1) == 0.0);
    // hermitian needs a square matrix (#826): its view of a 2x3 A read A(c, r)
    // outside A, and (2, 1) was outside the view's own two rows.
    D S(2, 2);
    S(0, 0) = 1; S(0, 1) = 2; S(1, 0) = 9; S(1, 1) = 4;
    REQUIRE(mtl::hermitian(S)(1, 0) == S(0, 1));
    const index_map rows = {1, 0}, cols = {2, 1, 0};
    REQUIRE(mtl::mapped(A, rows, cols)(0, 0) == A(1, 2));
}
