// MTL5 -- Accumulator policy for sum and trace (#585).
//
// Each element enters as add_product(acc, x, 1), which is exact, so:
//   reach     -- a counting accumulator sees one product per element;
//   identity  -- a plain element-type accumulator gives the void bits; and
//                fma_accumulator does too, since fma(x, 1, acc) rounds the same
//                sum acc + x once, exactly as acc += x does: for a plain sum
//                configurations 1 and 2 coincide;
//   exactness -- an exact accumulator returns the correctly rounded sum
//                ([586:sum]). Measured on 20 cancelling float sums of 1000
//                terms over 2^-12 .. 2^12: exact is correctly rounded in all
//                20, plain (and so fma) in none.
// Built with -ffp-contract=off, for the bit-identity claims.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/sum.hpp>
#include <mtl/operation/trace.hpp>

#include "exact_accumulator.hpp"

using namespace mtl;

namespace counting {

struct count_acc { double s; };
inline std::size_t products = 0;

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }
template <typename V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }
template <typename V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}
template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

/// Mixed-sign values over 2^-12 .. 2^12, so the float sum cancels and rounds.
vec::dense_vector<float> cancelling(std::size_t n, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    vec::dense_vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x(i) = static_cast<float>((u(gen) < 0.5 ? -1.0 : 1.0) *
                                  std::ldexp(1.0 + u(gen), static_cast<int>(u(gen) * 24.0) - 12));
    return x;
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

/// The correctly rounded float sum: exact in a binary64 Kulisch accumulator,
/// rounded once.
float exact_sum(const vec::dense_vector<float>& x) {
    auto e = exact_test::zero<double>();
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(e, double(x(i)), 1.0);
    return accumulator_value(e, std::type_identity<float>{});
}

} // namespace

TEST_CASE("sum and trace: the accumulator sees every element (#585)",
          "[operation][sum][trace][accumulator]") {
    const auto x = cancelling(37, 1);
    counting::products = 0;
    (void)sum<counting::count_acc>(x);
    CHECK(counting::products == 37);

    mat::dense2D<float> A(6, 9);
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 9; ++j) A(i, j) = float(i + 2 * j);
    counting::products = 0;
    CHECK(trace<counting::count_acc>(A) == trace(A));
    CHECK(counting::products == 6);              // min(rows, cols)

    counting::products = 0;
    (void)sum(x);
    (void)trace(A);
    CHECK(counting::products == 0);
}

TEST_CASE("sum and trace: plain and fma accumulators are the void path bit for bit (#585)",
          "[operation][sum][trace][accumulator]") {
    using fma = math::fma_accumulator<float>;
    for (std::uint32_t seed : {2u, 3u, 4u}) {
        const auto x = cancelling(500, seed);
        INFO("seed " << seed);
        CHECK(same_bits(sum<float>(x), sum(x)));
        CHECK(same_bits(sum<fma>(x), sum(x)));
    }
    const std::size_t n = 200;
    const auto d = cancelling(n, 9);
    mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = (i == j) ? d(i) : 0.0f;
    CHECK(same_bits(trace<float>(A), trace(A)));
    CHECK(same_bits(trace<fma>(A), trace(A)));
}

TEST_CASE("[586:sum] sum and trace: an exact accumulator returns the correctly rounded sum",
          "[586:sum][operation][sum][trace][accumulator]") {
    using exact = exact_test::exact_acc<float>;
    std::size_t plain_wrong = 0;
    for (std::uint32_t seed = 10; seed < 30; ++seed) {
        const auto x = cancelling(1000, seed);
        const float want = exact_sum(x);
        INFO("seed " << seed);
        REQUIRE(sum<exact>(x) == want);
        plain_wrong += sum(x) != want;
    }
    INFO("plain misses the correctly rounded sum in " << plain_wrong << " of 20");
    CHECK(plain_wrong > 0);

    // Constructed: 2^24 + 1 - 2^24. Plain (and so fma) absorbs the 1.
    vec::dense_vector<float> c(3);
    c(0) = 16777216.0f; c(1) = 1.0f; c(2) = -16777216.0f;
    CHECK(sum(c) == 0.0f);
    CHECK(sum<exact>(c) == 1.0f);

    mat::dense2D<float> D(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) D(i, j) = (i == j) ? c(i) : 0.0f;
    CHECK(trace(D) == 0.0f);
    CHECK(trace<exact>(D) == 1.0f);
}
