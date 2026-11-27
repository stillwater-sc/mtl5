// mult() takes its fast path (#944): through MTL5_DISPATCH_TRACE, which kernel
// each product reached. A lost fast path gives the right answer slowly, so no
// result test sees it; this one does, on every CI platform.
//
// This translation unit turns the native-fast path on, as the GEMM tests do,
// so it is checked in every build; test_mult_dispatch_generic.cpp checks the
// default build's path. With MTL5_HAS_BLAS (the LAPACK lanes) BLAS comes first.
#define MTL5_DISPATCH_TRACE
#ifndef MTL5_NATIVE_FAST_GEMM
#define MTL5_NATIVE_FAST_GEMM 1
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/dispatch_trace.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cstddef>

namespace {
namespace dt = mtl::detail::dispatch_trace;
using colmaj = mtl::mat::parameters<mtl::tag::col_major>;

template <typename M>
M filled(std::size_t r, std::size_t c, int seed) {
    M A(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            A(i, j) = typename M::value_type((i * 3 + j * 5 + std::size_t(seed)) % 7) - typename M::value_type(3);
    return A;
}

// Exactly one hit, on `expected`, and none on any other path.
void only(dt::path expected) {
    for (std::size_t p = 0; p < static_cast<std::size_t>(dt::path::count_); ++p) {
        const auto path = static_cast<dt::path>(p);
        INFO("path " << p);
        CHECK(dt::count(path) == (path == expected ? 1u : 0u));
    }
}

#ifdef MTL5_HAS_BLAS
constexpr bool blas = true;
#else
constexpr bool blas = false;
#endif
} // namespace

TEMPLATE_TEST_CASE("mult: dense float and double reach BLAS or the blocked GEMM (#944)",
                   "[mult][dispatch][944]", float, double) {
    using T = TestType;
    SECTION("row-major") {
        const auto A = filled<mtl::mat::dense2D<T>>(9, 7, 1);
        const auto B = filled<mtl::mat::dense2D<T>>(7, 5, 2);
        mtl::mat::dense2D<T> C(9, 5);
        dt::reset();
        mtl::mult(A, B, C);
        only(blas ? dt::path::gemm_blas : dt::path::gemm_blocked);
    }
    SECTION("column-major") {
        const auto A = filled<mtl::mat::dense2D<T, colmaj>>(9, 7, 1);
        const auto B = filled<mtl::mat::dense2D<T, colmaj>>(7, 5, 2);
        mtl::mat::dense2D<T, colmaj> C(9, 5);
        dt::reset();
        mtl::mult(A, B, C);
        only(blas ? dt::path::gemm_blas : dt::path::gemm_blocked);
    }
    SECTION("C = A * B goes the same way") {
        const auto A = filled<mtl::mat::dense2D<T>>(6, 6, 3);
        const auto B = filled<mtl::mat::dense2D<T>>(6, 6, 4);
        mtl::mat::dense2D<T> C(6, 6);
        dt::reset();
        C = A * B;
        CHECK(dt::count(blas ? dt::path::gemm_blas : dt::path::gemm_blocked) >= 1u);
        CHECK(dt::count(dt::path::gemm_generic) == 0u);
    }
    SECTION("matrix-vector") {
        const auto A = filled<mtl::mat::dense2D<T>>(9, 7, 5);
        mtl::vec::dense_vector<T> x(7, T(1)), y(9);
        dt::reset();
        mtl::mult(A, x, y);
        only(blas ? dt::path::gemv_blas : dt::path::gemv_simd);
        const auto Ac = filled<mtl::mat::dense2D<T, colmaj>>(9, 7, 5);
        dt::reset();
        mtl::mult(Ac, x, y);
        only(blas ? dt::path::gemv_blas : dt::path::gemv_simd);
    }
}

TEST_CASE("mult: float operands with a double accumulator take the widening blocked GEMM (#944)",
          "[mult][dispatch][944]") {
    const auto A = filled<mtl::mat::dense2D<float>>(8, 6, 1);
    const auto B = filled<mtl::mat::dense2D<float>>(6, 4, 2);
    mtl::mat::dense2D<double> C(8, 4);
    dt::reset();
    mtl::mult<double>(A, B, C);
    only(dt::path::gemm_blocked);   // a custom accumulator never goes to BLAS
}

TEST_CASE("mult: a type no fast path supports takes the generic kernel (#944)", "[mult][dispatch][944]") {
    const auto A = filled<mtl::mat::dense2D<long double>>(5, 4, 1);
    const auto B = filled<mtl::mat::dense2D<long double>>(4, 3, 2);
    mtl::mat::dense2D<long double> C(5, 3);
    dt::reset();
    mtl::mult(A, B, C);
    only(dt::path::gemm_generic);
    mtl::vec::dense_vector<long double> x(4, 1.0L), y(5);
    dt::reset();
    mtl::mult(A, x, y);
    only(dt::path::gemv_generic);
}
