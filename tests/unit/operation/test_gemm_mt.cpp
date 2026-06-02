// Multithreaded native GEMM (#92): the ic-loop is partitioned across a
// std::thread team (C++ concurrency runtime, no OpenMP). Different ic-blocks
// write disjoint C rows and only read the shared packed B panel, so the threaded
// result is BIT-IDENTICAL to single-thread -- each C block gets the same FMAs in
// the same order regardless of which thread runs it. We assert exact equality
// (==) between nthreads=1 and several thread counts, across sizes that span many
// ic-blocks, both operand orientations, and alpha/beta.
//
// Run under TSan (-DMTL5_SANITIZE=thread) to confirm race-freedom.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Fill row-major (rs=cols,cs=1) or col-major (rs=1,cs=rows) storage with random
// values; return the (ptr-relative) strides for gemm_blocked.
template <typename T>
void fill_random(std::vector<T>& buf, std::size_t rows, std::size_t cols,
                 bool rowmaj, std::uint64_t seed) {
    buf.assign(rows * cols, T(0));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            buf[rowmaj ? i * cols + j : j * rows + i] = static_cast<T>(dist(rng));
}

// gemm_blocked with nthreads == 1 vs == nt must be bit-identical.
template <typename T>
bool mt_matches(std::size_t m, std::size_t n, std::size_t k, bool a_rowmaj, bool b_rowmaj,
                unsigned nt, T alpha, T beta, std::uint64_t seed) {
    std::vector<T> A, B;
    fill_random(A, m, k, a_rowmaj, seed);
    fill_random(B, k, n, b_rowmaj, seed + 1);
    const std::ptrdiff_t a_rs = a_rowmaj ? (std::ptrdiff_t)k : 1, a_cs = a_rowmaj ? 1 : (std::ptrdiff_t)m;
    const std::ptrdiff_t b_rs = b_rowmaj ? (std::ptrdiff_t)n : 1, b_cs = b_rowmaj ? 1 : (std::ptrdiff_t)k;

    // Identical preset C for both runs (matters when beta != 0).
    std::vector<T> C0(m * n);
    { std::mt19937_64 rng(seed + 2); std::uniform_real_distribution<double> d(-1.0, 1.0);
      for (auto& c : C0) c = static_cast<T>(d(rng)); }

    std::vector<T> C1 = C0, CN = C0;
    mtl::detail::gemm_blocked<T>(m, n, k, alpha, A.data(), a_rs, a_cs, B.data(), b_rs, b_cs,
                                 beta, C1.data(), n, 1);
    mtl::detail::gemm_blocked<T>(m, n, k, alpha, A.data(), a_rs, a_cs, B.data(), b_rs, b_cs,
                                 beta, CN.data(), n, nt);
    return C1 == CN;   // bit-exact
}

} // namespace

TEMPLATE_TEST_CASE("MT GEMM == single-thread, bit-exact, many ic-blocks", "[operation][gemm][mt]", float, double) {
    // Sizes large enough in m to span several ic (mc) blocks for any build width.
    const std::size_t ms[] = {65, 128, 200, 333};
    const std::size_t ns[] = {1, 8, 40, 96};
    const std::size_t ks[] = {1, 7, 64, 129};
    std::uint64_t seed = 1;
    for (std::size_t m : ms)
        for (std::size_t n : ns)
            for (std::size_t k : ks)
                for (unsigned nt : {2u, 3u, 4u, 8u}) {
                    INFO("m=" << m << " n=" << n << " k=" << k << " nt=" << nt);
                    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(1), TestType(0), seed++));
                }
}

TEMPLATE_TEST_CASE("MT GEMM == single-thread: orientations + alpha/beta", "[operation][gemm][mt]", float, double) {
    const std::size_t m = 257, n = 53, k = 71;   // many ic-blocks, odd/rectangular
    std::uint64_t s = 1000;
    const unsigned nt = 4;
    // orientation combos
    CHECK(mt_matches<TestType>(m, n, k, true,  true,  nt, TestType(1), TestType(0), s++));
    CHECK(mt_matches<TestType>(m, n, k, false, true,  nt, TestType(1), TestType(0), s++));
    CHECK(mt_matches<TestType>(m, n, k, true,  false, nt, TestType(1), TestType(0), s++));
    CHECK(mt_matches<TestType>(m, n, k, false, false, nt, TestType(1), TestType(0), s++));
    // alpha/beta
    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(2.5),  TestType(0),    s++));
    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(1),    TestType(-1.5), s++));
    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(-0.5), TestType(0.75), s++));
}

TEMPLATE_TEST_CASE("MT GEMM: crosses mc/kc blocks", "[operation][gemm][mt]", float, double) {
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    const std::size_t m = bp.mc * 3 + bp.mr + 1;   // several mc blocks across threads
    const std::size_t k = bp.kc + 5;               // > kc -> multiple pc iterations
    const std::size_t n = bp.nr * 2 + 1;
    INFO("m=" << m << " n=" << n << " k=" << k << " (mc=" << bp.mc << " kc=" << bp.kc << ")");
    CHECK(mt_matches<TestType>(m, n, k, true, true, 4, TestType(1), TestType(0), 7));
    CHECK(mt_matches<TestType>(m, n, k, true, true, 8, TestType(1.25), TestType(-0.5), 9));
}
