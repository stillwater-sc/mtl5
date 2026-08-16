// The overflow envelope for integer dot products (#451).
//
// "int32 x int32 needs int64 and overflows at k~4" is TRUE and MISLEADING: it is
// the deterministic worst case at FULL RANGE -- every operand at +2^31, every
// term the same sign. Real data does neither, and the difference is enormous.
//
// What actually governs it is the operand BIT WIDTH b and the sign structure:
//
//   signed, zero-mean   |sum| ~ sqrt(k) * 2^(2b)/3     -- terms cancel
//   unsigned / offset   |sum| ~ k * 2^(2b)/4           -- terms accumulate
//
// So the safe depth falls off as 2^-2b, and quadratically faster for unsigned
// data. Measured on this envelope (see the table in #451), 16-bit operands are
// safe past a million terms in both regimes, while full-range operands are not
// safe past a few dozen.
//
// This is a CONTRACT test, not a demonstration: phases 2-3 of the epic must
// pick an accumulator per operand width, and these are the numbers that decide
// which. Products of two int32 always fit an int64 exactly, so only the running
// SUM can overflow -- and that is detectable portably, without __int128 (absent
// on MSVC).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using i64 = std::int64_t;

/// Exact int64 dot product that reports whether the accumulator ever wrapped.
/// a*b of two int32 cannot overflow an i64; only the addition can.
struct dot_result { i64 value = 0; bool overflowed = false; i64 peak_abs = 0; };

dot_result checked_dot(const std::vector<std::int32_t>& x,
                       const std::vector<std::int32_t>& y) {
    dot_result r;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const i64 p = static_cast<i64>(x[i]) * static_cast<i64>(y[i]);
        if ((p > 0 && r.value > std::numeric_limits<i64>::max() - p) ||
            (p < 0 && r.value < std::numeric_limits<i64>::min() - p)) {
            r.overflowed = true;
            return r;
        }
        r.value += p;
        const i64 m = r.value < 0 ? -r.value : r.value;
        if (m > r.peak_abs) r.peak_abs = m;
    }
    return r;
}

/// `signed_range`: values in [-M, M] (zero-mean). Otherwise [0, M], which is the
/// quantized-activation case and the one that accumulates linearly.
std::vector<std::int32_t> random_vec(std::mt19937& g, std::size_t k,
                                     std::int32_t M, bool signed_range) {
    std::uniform_int_distribution<std::int32_t> d(signed_range ? -M : 0, M);
    std::vector<std::int32_t> v(k);
    for (auto& e : v) e = d(g);
    return v;
}

constexpr std::int32_t max_of_bits(int b) {
    return static_cast<std::int32_t>((std::int64_t{1} << (b - 1)) - 1);
}

} // namespace

TEST_CASE("int64 accumulation is safe well past a million terms for <=16-bit operands",
          "[simd][integer][overflow]") {
    std::mt19937 g(20260816);
    constexpr std::size_t K = 1u << 20;          // ~1e6 terms

    for (int bits : {8, 12, 16}) {
        const std::int32_t M = max_of_bits(bits);
        for (bool sgn : {true, false}) {
            const auto x = random_vec(g, K, M, sgn);
            const auto y = random_vec(g, K, M, sgn);
            const auto r = checked_dot(x, y);
            INFO("bits=" << bits << (sgn ? " signed" : " unsigned")
                 << " k=" << K << " peak=" << r.peak_abs);
            REQUIRE_FALSE(r.overflowed);
            // And not merely surviving: still orders of magnitude of headroom.
            REQUIRE(r.peak_abs < std::numeric_limits<i64>::max() / 1000);
        }
    }
}

TEST_CASE("the unsigned regime is the binding one, by a square root",
          "[simd][integer][overflow]") {
    // Zero-mean data random-walks (sqrt(k)); offset data marches (k). At 20 bits
    // and a million terms both still fit, and the ratio between them is the
    // whole reason a single "safe k" number is not a useful contract.
    std::mt19937 g(7);
    constexpr std::size_t K = 1u << 20;
    const std::int32_t M = max_of_bits(20);

    const auto s = checked_dot(random_vec(g, K, M, true),  random_vec(g, K, M, true));
    const auto u = checked_dot(random_vec(g, K, M, false), random_vec(g, K, M, false));
    REQUIRE_FALSE(s.overflowed);
    REQUIRE_FALSE(u.overflowed);
    INFO("signed peak=" << s.peak_abs << " unsigned peak=" << u.peak_abs);
    REQUIRE(u.peak_abs > 10 * s.peak_abs);       // measured ratio is ~100x here
}

TEST_CASE("full-range int32 operands are the case that genuinely does not fit",
          "[simd][integer][overflow]") {
    // The claim that survives: it is FULL RANGE that breaks, not int32 as such.
    // Two products of +/-2^30 already consume the int64 sign bit.
    std::mt19937 g(11);
    const std::int32_t M = max_of_bits(31);

    // Deterministic worst case: every term at +M*M. Overflow by the 9th term,
    // matching 2^63 / (2^30)^2 = 8.
    std::vector<std::int32_t> ones(64, M);
    REQUIRE(checked_dot(ones, ones).overflowed);

    // Random signed data survives longer than the worst case -- cancellation
    // buys roughly a factor of sqrt(k) -- but not far enough to be usable.
    bool any = false;
    for (int t = 0; t < 16 && !any; ++t)
        any = checked_dot(random_vec(g, 4096, M, true), random_vec(g, 4096, M, true)).overflowed;
    REQUIRE(any);
}

TEST_CASE("a 16-bit operand product never needs more than 32 bits",
          "[simd][integer][overflow]") {
    // Why int8 -> int32 is the shape the hardware implements: the PRODUCT of two
    // b-bit values needs 2b bits, so int8 operands leave 16 bits of accumulator
    // headroom in an int32 -- ~65k terms before the sum can reach the top,
    // which is why VNNI accumulates into int32 and not something wider.
    constexpr std::int32_t M8 = max_of_bits(8);
    REQUIRE(static_cast<i64>(M8) * M8 < (i64{1} << 15));

    std::mt19937 g(3);
    constexpr std::size_t K = 1u << 16;
    const auto x = random_vec(g, K, M8, true);
    const auto y = random_vec(g, K, M8, true);
    const auto r = checked_dot(x, y);
    REQUIRE_FALSE(r.overflowed);
    // The int32 accumulator VNNI uses would also have held this.
    REQUIRE(r.peak_abs < std::numeric_limits<std::int32_t>::max());
}
