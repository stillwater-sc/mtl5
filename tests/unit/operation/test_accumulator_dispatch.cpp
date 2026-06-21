// MTL5 -- dispatch rule: a non-default accumulator forces the native kernel (#163).
// External BLAS / native-fast paths use hardware-fixed accumulation, so any
// custom accumulator must bypass them -- even for float/double operands. Proven
// with a "counting" accumulator: if the result equals the contraction count, the
// native (generic) accumulator path was taken; a BLAS path would compute A*B.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/interface/dispatch_traits.hpp>

namespace {
// Accumulator that counts the number of products instead of summing them.
struct count_acc { long n = 0; };
} // namespace

namespace mtl::math {
template <>
struct accumulator_traits<count_acc, float> {
    static void clear(count_acc& a) { a.n = 0; }
    static void assign(count_acc& a, const float&) { a.n = 1; }
    template <typename Result = float>
    static Result value(const count_acc& a) { return static_cast<Result>(a.n); }
    static void add_product(count_acc& a, const float&, const float&) { ++a.n; }
};
} // namespace mtl::math

using namespace mtl;

TEST_CASE("accumulator_allows_blas_v: only the default void permits BLAS",
          "[operation][dispatch][accumulator]") {
    static_assert(interface::accumulator_allows_blas_v<void>);
    static_assert(!interface::accumulator_allows_blas_v<float>);
    static_assert(!interface::accumulator_allows_blas_v<double>);
    static_assert(!interface::accumulator_allows_blas_v<count_acc>);
    SUCCEED();
}

TEST_CASE("dot with a custom accumulator bypasses BLAS (float operands)",
          "[operation][dispatch][accumulator]") {
    vec::dense_vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    vec::dense_vector<float> b = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    // A BLAS ?dot would return the sum of products (= 135). The counting
    // accumulator returns 5 only if the native accumulator path was taken.
    auto cnt = dot<count_acc, long>(a, b);
    static_assert(std::is_same_v<decltype(cnt), long>);
    REQUIRE(cnt == 5);
    // Default path still computes the real dot product.
    REQUIRE(dot(a, b) == 135.0f);
}

TEST_CASE("gemm with a custom accumulator bypasses BLAS/native-fast (float operands)",
          "[operation][dispatch][accumulator]") {
    const std::size_t m = 4, k = 7, n = 3;
    mat::dense2D<float> A(m, k), B(k, n), C(m, n);
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < k; ++j) A(i, j) = 2.0f;
    for (std::size_t i = 0; i < k; ++i) for (std::size_t j = 0; j < n; ++j) B(i, j) = 3.0f;

    // BLAS/native-fast would give C(i,j) = k * 2 * 3 = 42. The counting
    // accumulator gives C(i,j) = k (one count per contraction step) iff the
    // generic accumulator kernel ran instead.
    mult<count_acc>(A, B, C);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(C(i, j) == static_cast<float>(k));

    // Default path still computes the real product.
    mat::dense2D<float> Cd(m, n);
    mult(A, B, Cd);
    REQUIRE(Cd(0, 0) == 42.0f);
}
