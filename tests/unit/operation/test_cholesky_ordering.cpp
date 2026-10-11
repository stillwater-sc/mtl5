// cholesky_factor's two summation orders (#556).
//
//   cholesky_factor(A)                               a - (sum of products)
//   cholesky_factor(A, tag::subtract_term_by_term{}) s = a; s = s - p_k ...
//
// The opt-in exists so a caller whose results are pinned to the term-by-term
// form can move to MTL5 without changing a bit, so the tests compare BITS (==)
// against in-test serial reference loops, not values within a tolerance. A
// tolerance would pass either ordering, which is exactly what must not happen.
//
// Three properties make that comparison meaningful:
//   - The pool is sized to 4 threads before first use, so the library's
//     row-partitioned column update runs threaded while the references run
//     serial. Equality therefore also pins thread-count independence.
//   - The term-by-term overload never dispatches to LAPACK, so the double and
//     float cases hold in MTL5_HAS_LAPACK builds as well.
//   - The target is built with -ffp-contract=off (tests/unit/CMakeLists.txt):
//     the library loop and the reference loop are the same source expression,
//     but a compiler may contract one into an FMA and not the other.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

#include <mtl/concepts/scalar.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/tag/summation_order.hpp>

using namespace mtl;

namespace {

// Size the process-wide pool multithreaded before its first (lazy) use.
const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

/// A class-typed scalar wrapping a double: no BLAS, no LAPACK, no SIMD lane.
struct emul {
    double v = 0.0;
    emul() = default;
    constexpr emul(double d) : v(d) {}
    friend emul operator+(emul a, emul b) { return a.v + b.v; }
    friend emul operator-(emul a, emul b) { return a.v - b.v; }
    friend emul operator*(emul a, emul b) { return a.v * b.v; }
    friend emul operator/(emul a, emul b) { return a.v / b.v; }
    friend emul operator-(emul a) { return -a.v; }
    emul& operator+=(emul o) { v += o.v; return *this; }
    emul& operator-=(emul o) { v -= o.v; return *this; }
    emul& operator*=(emul o) { v *= o.v; return *this; }
    emul& operator/=(emul o) { v /= o.v; return *this; }
    friend bool operator==(emul a, emul b) { return a.v == b.v; }
    friend auto operator<=>(emul a, emul b) { return a.v <=> b.v; }
    friend emul sqrt(emul a) { return std::sqrt(a.v); }
};
static_assert(!std::is_arithmetic_v<emul> && Field<emul>);

double bits_of(double x) { return x; }
double bits_of(emul x) { return x.v; }
float  bits_of(float x) { return x; }

template <typename T>
bool same_bits(const T& a, const T& b) {
    const auto x = bits_of(a), y = bits_of(b);
    return std::memcmp(&x, &y, sizeof(x)) == 0;
}

/// Serial reference: the term-by-term form, as a hand-written port has it.
template <typename T>
int ref_term_by_term(mat::dense2D<T>& A) {
    using std::sqrt;
    const std::size_t n = A.num_rows();
    for (std::size_t j = 0; j < n; ++j) {
        T d = A(j, j);
        for (std::size_t k = 0; k < j; ++k)
            d = d - A(j, k) * A(j, k);
        if (!(d > T(0)))
            return static_cast<int>(j + 1);
        A(j, j) = sqrt(d);
        for (std::size_t i = j + 1; i < n; ++i) {
            T s = A(i, j);
            for (std::size_t k = 0; k < j; ++k)
                s = s - A(i, k) * A(j, k);
            A(i, j) = s / A(j, j);
        }
    }
    return 0;
}

/// Serial reference: the default form, sum then subtract.
template <typename T>
int ref_subtract_sum(mat::dense2D<T>& A) {
    using std::sqrt;
    const std::size_t n = A.num_rows();
    for (std::size_t j = 0; j < n; ++j) {
        T sum = T(0);
        for (std::size_t k = 0; k < j; ++k)
            sum += A(j, k) * A(j, k);
        const T d = A(j, j) - sum;
        if (!(d > T(0)))
            return static_cast<int>(j + 1);
        A(j, j) = sqrt(d);
        for (std::size_t i = j + 1; i < n; ++i) {
            T s = T(0);
            for (std::size_t k = 0; k < j; ++k)
                s += A(i, k) * A(j, k);
            A(i, j) = (A(i, j) - s) / A(j, j);
        }
    }
    return 0;
}

/// A random SPD matrix B*B^T + n*I, stored in full.
template <typename T>
mat::dense2D<T> random_spd(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<double> B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            B(i, j) = u(g);
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = (i == j) ? static_cast<double>(n) : 0.0;
            for (std::size_t k = 0; k < n; ++k)
                s += B(i, k) * B(j, k);
            A(i, j) = T(s);
        }
    return A;
}

template <typename T>
std::size_t lower_mismatches(const mat::dense2D<T>& A, const mat::dense2D<T>& B) {
    std::size_t bad = 0;
    for (std::size_t j = 0; j < A.num_rows(); ++j)
        for (std::size_t i = j; i < A.num_rows(); ++i)
            bad += !same_bits(A(i, j), B(i, j));
    return bad;
}

template <typename T>
void check_term_by_term(std::size_t n, std::uint64_t seed) {
    auto A = random_spd<T>(n, seed), R = A;
    REQUIRE(cholesky_factor(A, tag::subtract_term_by_term{}) == 0);
    REQUIRE(ref_term_by_term(R) == 0);
    REQUIRE(lower_mismatches(A, R) == 0);
}

} // namespace

TEST_CASE("cholesky_factor term by term equals the reference loop bit for bit",
          "[operation][cholesky][ordering][556]") {
    if (detail::thread_pool::instance().size() < 2)
        WARN("single-core runner: threading not exercised");
    for (std::size_t n : {1, 2, 3, 7, 33, 81}) {
        INFO("n = " << n);
        check_term_by_term<double>(n, 556 + n);
        check_term_by_term<float>(n, 557 + n);
        check_term_by_term<emul>(n, 558 + n);
    }
    // Large enough that the column update splits across the pool.
    check_term_by_term<double>(768, 5560);
}

TEST_CASE("cholesky_factor's default order is unchanged and named by tag::subtract_sum",
          "[operation][cholesky][ordering][556]") {
    // emul never reaches LAPACK, so the default is the generic loop in every
    // build, and it must still equal the sum-then-subtract reference.
    for (std::size_t n : {1, 7, 81}) {
        auto A = random_spd<emul>(n, 600 + n), R = A;
        REQUIRE(cholesky_factor(A) == 0);
        REQUIRE(ref_subtract_sum(R) == 0);
        REQUIRE(lower_mismatches(A, R) == 0);
    }
    // The explicit tag is the same call as no tag, LAPACK or not.
    auto A = random_spd<double>(81, 610), B = A;
    REQUIRE(cholesky_factor(A) == 0);
    REQUIRE(cholesky_factor(B, tag::subtract_sum{}) == 0);
    REQUIRE(lower_mismatches(A, B) == 0);
}

TEST_CASE("the two orderings round differently", "[operation][cholesky][ordering][556]") {
    // Otherwise the tag would be selecting the same arithmetic. emul keeps both
    // calls on the generic loop in LAPACK builds too.
    auto A = random_spd<emul>(81, 620), B = A;
    REQUIRE(cholesky_factor(A) == 0);
    REQUIRE(cholesky_factor(B, tag::subtract_term_by_term{}) == 0);
    REQUIRE(lower_mismatches(A, B) > 0);
}

TEST_CASE("cholesky_factor rejects a NaN pivot in both orderings",
          "[operation][cholesky][ordering][nan][556]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // long double never dispatches to LAPACK, so this is the generic default.
    SECTION("NaN on the diagonal, default order") {
        mat::dense2D<long double> A{{4.0L, 2.0L}, {2.0L, static_cast<long double>(nan)}};
        REQUIRE(cholesky_factor(A) == 2);
    }
    SECTION("NaN on the diagonal, term by term") {
        mat::dense2D<double> A{{nan, 1.0}, {1.0, 4.0}};
        REQUIRE(cholesky_factor(A, tag::subtract_term_by_term{}) == 1);
    }
    SECTION("NaN off the diagonal reaches the next pivot") {
        mat::dense2D<long double> A{{4.0L, 0.0L, 0.0L},
                                    {static_cast<long double>(nan), 5.0L, 0.0L},
                                    {0.0L, 0.0L, 6.0L}};
        auto B = A;
        REQUIRE(cholesky_factor(A) == 2);
        REQUIRE(cholesky_factor(B, tag::subtract_term_by_term{}) == 2);
    }
}

TEST_CASE("cholesky_factor term by term reports a non-SPD pivot",
          "[operation][cholesky][ordering][556]") {
    mat::dense2D<double> A{{4.0, 2.0, 0.0}, {2.0, 1.0, 0.0}, {0.0, 0.0, 3.0}};
    auto R = A;
    REQUIRE(cholesky_factor(A, tag::subtract_term_by_term{}) == 2);
    REQUIRE(ref_term_by_term(R) == 2);
}
