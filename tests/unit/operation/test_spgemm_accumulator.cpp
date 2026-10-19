// MTL5 -- Accumulator policy for spgemm (#585).
//
//   reach     -- a counting accumulator sees exactly one product per
//                (A(i,k), B(k,j)) pair, the void default none;
//   identity  -- a plain element-type accumulator gives the void scatter's
//                bits (the same additions in the same order), which is why this
//                target is built with -ffp-contract=off. The one documented
//                exception, a -0 first product, is kept out of the inputs;
//   exactness -- with an exact accumulator every C(i,j) is the correctly
//                rounded value of its exact sum, checked against an
//                independent reference ([586:spgemm]). Measured on the 40x60 *
//                60x40 random cancelling product below: of 1589 entries,
//                plain misses the correctly rounded value at 584, fma at 446,
//                the exact accumulator at none.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/spgemm.hpp>

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

/// Random m x n sparse matrix with nonzero values of mixed sign and widely
/// spread magnitudes, so the sums cancel and round.
mat::compressed2D<float> random_sparse(std::size_t m, std::size_t n, double density,
                                       std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    mat::compressed2D<float> A(m, n);
    mat::inserter<mat::compressed2D<float>> ins(A);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (u(gen) < density) {
                // One draw per statement, exponent first: a call's arguments are evaluated in
                // an unspecified order, and GCC's (the last argument first) is the one the
                // recorded numbers were measured with.
                const int e = static_cast<int>(u(gen) * 24.0) - 12;
                const double mag  = std::ldexp(1.0 + u(gen), e);
                const double sign = u(gen) < 0.5 ? -1.0 : 1.0;
                ins[i][j] << static_cast<float>(sign * mag);
            }
    return A;
}

/// One product per (i, k, j): for each stored A(i,k), the stored row k of B.
std::size_t spgemm_products(const mat::compressed2D<float>& A, const mat::compressed2D<float>& B) {
    std::size_t count = 0;
    const auto& as = A.ref_major();
    const auto& ai = A.ref_minor();
    const auto& bs = B.ref_major();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (auto k = as[i]; k < as[i + 1]; ++k) count += bs[ai[k] + 1] - bs[ai[k]];
    return count;
}

bool same_structure_and_bits(const mat::compressed2D<float>& X, const mat::compressed2D<float>& Y) {
    if (X.ref_major() != Y.ref_major() || X.ref_minor() != Y.ref_minor()) return false;
    const auto& x = X.ref_data();
    const auto& y = Y.ref_data();
    if (x.size() != y.size()) return false;
    for (std::size_t k = 0; k < x.size(); ++k)
        if (std::memcmp(&x[k], &y[k], sizeof(float)) != 0) return false;
    return true;
}

/// The correctly rounded C(i,j): the exact sum in a binary64 Kulisch
/// accumulator, each float*float product formed exactly in double, rounded
/// once to float. Independent of the code under test: a different accumulator
/// instantiation, fed whole products.
float exact_entry(const mat::compressed2D<float>& A, const mat::compressed2D<float>& B,
                  std::size_t i, std::size_t j) {
    auto e = exact_test::zero<double>();
    const auto& as = A.ref_major();
    const auto& ai = A.ref_minor();
    const auto& ad = A.ref_data();
    const auto& bs = B.ref_major();
    const auto& bi = B.ref_minor();
    const auto& bd = B.ref_data();
    for (auto ka = as[i]; ka < as[i + 1]; ++ka) {
        const std::size_t k = ai[ka];
        for (auto kb = bs[k]; kb < bs[k + 1]; ++kb)
            if (bi[kb] == j)
                accumulator_add_product(e, double(ad[ka]) * double(bd[kb]), 1.0);
    }
    return accumulator_value(e, std::type_identity<float>{});
}

} // namespace

TEST_CASE("spgemm: the accumulator sees every product (#585)", "[operation][spgemm][accumulator]") {
    const auto A = random_sparse(30, 25, 0.2, 1);
    const auto B = random_sparse(25, 35, 0.2, 2);
    REQUIRE(spgemm_products(A, B) > 0);

    counting::products = 0;
    const auto C = spgemm<counting::count_acc>(A, B);
    REQUIRE(counting::products == spgemm_products(A, B));

    // Same pattern as the void product: the accumulator changes values only.
    const auto V = spgemm(A, B);
    REQUIRE(C.ref_major() == V.ref_major());
    REQUIRE(C.ref_minor() == V.ref_minor());

    counting::products = 0;
    (void)spgemm(A, B);
    REQUIRE(counting::products == 0);
}

TEST_CASE("spgemm: a plain element-type accumulator is the void path bit for bit (#585)",
          "[operation][spgemm][accumulator]") {
    for (std::uint32_t seed : {3u, 4u, 5u}) {
        const auto A = random_sparse(40, 30, 0.25, seed);
        const auto B = random_sparse(30, 45, 0.25, seed + 100);
        INFO("seed " << seed);
        REQUIRE(same_structure_and_bits(spgemm<float>(A, B), spgemm(A, B)));
    }
}

TEST_CASE("[586:spgemm] spgemm: an exact accumulator rounds every entry once",
          "[586:spgemm][operation][spgemm][accumulator]") {
    using exact = exact_test::exact_acc<float>;
    using fma   = math::fma_accumulator<float>;

    // Random, cancelling: configuration 3 must equal the correctly rounded
    // value at EVERY entry; 1 and 2 must not, somewhere, or the inputs do not
    // exercise rounding at all.
    const auto A = random_sparse(40, 60, 0.3, 7);
    const auto B = random_sparse(60, 40, 0.3, 8);
    const auto C1 = spgemm<float>(A, B);
    const auto C2 = spgemm<fma>(A, B);
    const auto C3 = spgemm<exact>(A, B);

    std::size_t wrong1 = 0, wrong2 = 0, entries = 0;
    const auto& s = C3.ref_major();
    const auto& c = C3.ref_minor();
    for (std::size_t i = 0; i < C3.num_rows(); ++i)
        for (auto k = s[i]; k < s[i + 1]; ++k) {
            const float want = exact_entry(A, B, i, c[k]);
            REQUIRE(C3.ref_data()[k] == want);
            wrong1 += C1.ref_data()[k] != want;
            wrong2 += C2.ref_data()[k] != want;
            ++entries;
        }
    INFO(entries << " entries; not correctly rounded: plain " << wrong1 << ", fma " << wrong2);
    CHECK(wrong1 > 0);
    CHECK(wrong2 > 0);

    // Constructed: (2^24, 1, -2^24) . (1, 1, 1). Plain and FMA both absorb the
    // 1 into 2^24 and return 0; the exact accumulator returns 1.
    mat::compressed2D<float> R(1, 3), S(3, 1);
    {
        mat::inserter<mat::compressed2D<float>> ir(R), is(S);
        ir[0][0] << 16777216.0f; ir[0][1] << 1.0f; ir[0][2] << -16777216.0f;
        is[0][0] << 1.0f;        is[1][0] << 1.0f; is[2][0] << 1.0f;
    }
    CHECK(spgemm<float>(R, S).ref_data()[0] == 0.0f);
    CHECK(spgemm<fma>(R, S).ref_data()[0] == 0.0f);
    CHECK(spgemm<exact>(R, S).ref_data()[0] == 1.0f);
}
