// Threaded mult on a block_matrix_view (#629).
//
// The decode-then-dot path partitions output rows across the pool, each band
// with its own decode buffer. Two claims, checked separately as in the dense
// and sparse threading tests, because bit-identity alone is satisfied by a
// kernel that quietly stayed serial:
//
//   1. BIT-IDENTITY against the serial row loop, asserted with `==`. A band
//      owns y(r) outright and decodes row r alone, so the partition cannot
//      change a value or a summation order.
//   2. THE ROWS ACTUALLY SPLIT -- the scheme's decode records the executing
//      thread.
//
// Sets MTL5_NUM_THREADS before the pool's first use.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/block_matrix_view.hpp>
#include <mtl/mat/mx_schemes.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

std::mutex g_tid_mtx;
std::unordered_set<std::size_t> g_tids;
std::atomic<bool> g_record{false};

/// MXINT8, with a decode that notes which thread ran it.
struct recording_mxint8 : mtl::mx::mxint8 {
    static void decode(const std::byte* src, float* dst) {
        if (g_record.load(std::memory_order_relaxed)) {
            const std::size_t h = std::hash<std::thread::id>{}(std::this_thread::get_id());
            std::lock_guard<std::mutex> lk(g_tid_mtx);
            g_tids.insert(h);
        }
        mtl::mx::mxint8::decode(src, dst);
    }
};

} // namespace

TEST_CASE("threaded block mult is bit-identical to the serial row loop",
          "[block][mult][threading][mt]") {
    (void)g_set_threads;
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("pool < 2 workers: the row partition is not exercised (set MTL5_NUM_THREADS)");

    const std::size_t cols = 128;
    const std::size_t grain = mtl::interface::row_grain<float>(cols);
    const std::size_t rows = 4 * grain;   // several bands' worth
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> scale(124, 130), q(-128, 127);
    std::vector<std::byte> bytes(rows * (cols / 32) * recording_mxint8::block_bytes);
    for (std::size_t b = 0; b < bytes.size(); b += 33) {
        bytes[b] = std::byte(scale(rng));
        for (std::size_t j = 1; j <= 32; ++j) bytes[b + j] = std::byte(std::uint8_t(q(rng)));
    }
    const mtl::mat::block_matrix_view<recording_mxint8> A(rows, cols, bytes);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    mtl::vec::dense_vector<float> x(cols), y(rows), yref(rows);
    for (std::size_t c = 0; c < cols; ++c) x(c) = d(rng);

    std::vector<float> buf;
    mtl::detail::block_mult_rows<void>(A, x, yref, 0, rows, buf);

    g_tids.clear();
    g_record = true;
    mtl::mult(A, x, y);
    g_record = false;

    for (std::size_t r = 0; r < rows; ++r) {
        INFO("row " << r);
        REQUIRE(y(r) == yref(r));
    }
    if (mtl::detail::thread_pool::instance().size() >= 2)
        CHECK(g_tids.size() >= 2);
}
