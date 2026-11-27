// mult()'s path in a build without the native-fast GEMM (#944): BLAS when
// MTL5_HAS_BLAS is defined, else the generic kernel. The fast-path build's
// expectations are in test_mult_dispatch.cpp; this translation unit turns that
// path off even where the build turns it on (the Highway lanes).
#define MTL5_DISPATCH_TRACE
#undef MTL5_NATIVE_FAST_GEMM

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/dispatch_trace.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cstddef>

namespace {
namespace dt = mtl::detail::dispatch_trace;
#ifdef MTL5_HAS_BLAS
constexpr bool blas = true;
#else
constexpr bool blas = false;
#endif
} // namespace

TEMPLATE_TEST_CASE("mult without the native-fast GEMM: BLAS or the generic kernel (#944)",
                   "[mult][dispatch][944]", float, double) {
    using T = TestType;
    mtl::mat::dense2D<T> A(6, 5), B(5, 4), C(6, 4);
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 5; ++j) A(i, j) = T(i + j);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 4; ++j) B(i, j) = T(i) - T(j);
    dt::reset();
    mtl::mult(A, B, C);
    CHECK(dt::count(dt::path::gemm_blas) == (blas ? 1u : 0u));
    CHECK(dt::count(dt::path::gemm_generic) == (blas ? 0u : 1u));
    CHECK(dt::count(dt::path::gemm_blocked) == 0u);
    CHECK(C(5, 3) == T(5 * -3 + 6 * -2 + 7 * -1 + 8 * 0 + 9 * 1));   // sum_k (5 + k)(k - 3) = -25

    mtl::vec::dense_vector<T> x(5, T(1)), y(6);
    dt::reset();
    mtl::mult(A, x, y);
    CHECK(dt::count(dt::path::gemv_blas) == (blas ? 1u : 0u));
    CHECK(dt::count(dt::path::gemv_generic) == (blas ? 0u : 1u));
    CHECK(dt::count(dt::path::gemv_simd) == 0u);
}
