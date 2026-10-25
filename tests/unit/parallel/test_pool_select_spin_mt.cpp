// MTL5_POOL_SPIN (#636): built with the macro, pool() is the process
// spin_pool, so MTL5's kernels run on it; configure() sizes it as it sizes
// thread_pool. The kernels' own threaded tests run on it too, as
// op_test_factorization_spin_mt and op_test_gemm_spin_mt.
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/parallel/thread_pool.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace par = mtl::parallel;

#if !defined(MTL5_POOL_SPIN)
#  error "this test is built with MTL5_POOL_SPIN"
#endif

TEST_CASE("pool select: MTL5_POOL_SPIN makes pool() the spin_pool", "[parallel][spin_pool]") {
    REQUIRE(par::configure(4));   // before the first use
    STATIC_REQUIRE(std::is_same_v<par::default_pool, par::spin_pool>);
    REQUIRE(&par::pool() == &par::spin_pool::instance());
    REQUIRE(par::pool().size() == par::detail::clamp_pool_threads(4));
    REQUIRE_FALSE(par::configure(2));

    // A threaded GEMM on it equals the serial one bit for bit.
    const std::size_t n = 300;
    std::vector<double> A(n * n), B(n * n), C1(n * n, 0.0), CN(n * n, 0.0);
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = static_cast<double>((i * 7) % 13) - 6.0;
        B[i] = static_cast<double>((i * 5) % 11) - 5.0;
    }
    mtl::detail::gemm_blocked<double>(n, n, n, 1.0, A.data(), n, 1, B.data(), n, 1,
                                      0.0, C1.data(), n, 1);
    mtl::detail::gemm_blocked<double>(n, n, n, 1.0, A.data(), n, 1, B.data(), n, 1,
                                      0.0, CN.data(), n, par::pool().size());
    REQUIRE(C1 == CN);
}
