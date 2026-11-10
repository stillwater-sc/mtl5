// The -D overrides of interface::interleave_rows (#772): MTL5_INTERLEAVE_ROWS
// sets every type and kernel, and MTL5_INTERLEAVE_ROWS_F32 / _F64 set float or
// double ahead of it. The public functions must take the configured width and
// still give the one-row loop's bits; op_test_trmv_trsv_interleaved covers the
// bit identity of each width in depth, this test only that the defines reach
// the kernels.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt), as that test is.
#undef MTL5_INTERLEAVE_ROWS
#undef MTL5_INTERLEAVE_ROWS_F32
#undef MTL5_INTERLEAVE_ROWS_F64
#define MTL5_INTERLEAVE_ROWS 5
#define MTL5_INTERLEAVE_ROWS_F32 7

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

#include <mtl/interface/interleave_rows.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

template <typename T>
mat::dense2D<T> lower(std::size_t n) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(static_cast<double>((7 * i + 3 * j) % 11) / 13.0 - 0.4);
    for (std::size_t i = 0; i < n; ++i) A(i, i) = static_cast<T>(2.0 + static_cast<double>(i % 5) / 7.0);
    return A;
}

template <typename T>
vec::dense_vector<T> vec_of(std::size_t n) {
    vec::dense_vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v(i) = static_cast<T>(static_cast<double>((5 * i) % 9) / 9.0 - 0.3);
    return v;
}

template <typename V>
void require_same(const V& u, const V& w) {
    REQUIRE(u.size() == w.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
        INFO("i = " << i);
        REQUIRE(u(i) == w(i));
    }
}

/// The public mult_generic, trmv and lower_trisolve against the W = 1 kernels.
template <typename T>
void check_public_matches_one_row() {
    constexpr std::size_t n = 23;   // a remainder for 5 and for 7
    const auto A = lower<T>(n);
    const auto x0 = vec_of<T>(n);
    vec::dense_vector<T> y(n), z(n);
    detail::mult_generic(A, x0, y);
    detail::mult_rows_interleaved<1>(A, x0, z, 0, n);
    require_same(y, z);
    // A BLAS build sends float and double trmv to BLAS; long double always
    // takes the row loop.
    if constexpr (!detail::trmv_dispatches_blas_v<void, mat::dense2D<T>, vec::dense_vector<T>>)
    for (bool upper : {false, true}) {
        auto p = x0, q = x0;
        trmv(A, p, upper);
        if (upper) detail::trmv_upper_rows<1>(A, q, std::size_t{0}, false);
        else       detail::trmv_lower_rows<1>(A, q, n, false);
        require_same(p, q);
    }
    auto u = x0, w = x0;
    lower_trisolve(A, u);
    detail::lower_trisolve_rows<1>(A, w, w, false);
    require_same(u, w);
}

} // namespace

TEST_CASE("MTL5_INTERLEAVE_ROWS and its per-type defines reach the trait", "[operation][772]") {
    using namespace mtl::interface;
    STATIC_REQUIRE(interleave_rows_v<float, kernel::gemv> == 7);
    STATIC_REQUIRE(interleave_rows_v<float, kernel::trmv> == 7);
    STATIC_REQUIRE(interleave_rows_v<double, kernel::gemv> == 5);
    STATIC_REQUIRE(interleave_rows_v<double, kernel::lower_trisolve> == 5);
    STATIC_REQUIRE(interleave_rows_v<long double, kernel::trmv> == 5);
    STATIC_REQUIRE(interleave_rows_v<std::int32_t, kernel::gemv> == 5);
}

TEST_CASE("the configured widths give the one-row loop's bits", "[operation][772]") {
    check_public_matches_one_row<float>();
    check_public_matches_one_row<double>();
    check_public_matches_one_row<long double>();
}
