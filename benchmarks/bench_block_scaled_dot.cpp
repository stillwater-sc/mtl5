// Block-scaled integer gemv -- #628.
//
// y = W x over block-quantized rows, timed three ways, single-threaded:
//
//   per-block    one reduce_dot_widen per block, then a scalar scale -- what a
//                caller had to write before #628, and what branes-ai/synapse's
//                fused kernel does;
//   dequantize   each row decoded to float (w * scale_w[b]) and dotted with
//                pre-decoded float activations by reduce_dot<float> -- the
//                baseline the fused kernel is supposed to beat;
//   block-scaled reduce_dot_block_scaled: block sums kept in lanes, scaled
//                in-register, one horizontal reduce per row.
//
// The issue's claim is about block = 32: there the per-block horizontal reduce
// dominates and the fused kernel LOSES to dequantization (synapse measured
// 0.83x on AVX2). block = 256 is the control, where the reduce is amortized.
//
// Two operand pairs: int16 x int16 (synapse's, 12-bit weights against 8-bit
// activations) and uint8 x int8 (VNNI's native shape, Q8_0-like).
//
// Usage: bench_block_scaled_dot [rows] [cols] [reps]   (default 4096 4096 20)
#include <mtl/simd/algorithm.hpp>
#include <mtl/simd/batch.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

template <typename F>
double median_ms(std::size_t reps, F&& f) {
    std::vector<double> t(reps);
    f();   // warm-up: page in the operands
    for (auto& e : t) {
        const auto t0 = clock_type::now();
        f();
        e = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

template <typename T>
std::vector<T> fill(std::size_t n, long lo, long hi, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<long> d(lo, hi);
    std::vector<T> v(n);
    for (auto& e : v) e = static_cast<T>(d(rng));
    return v;
}

template <typename NW, typename NA>
void run(const char* pair, std::size_t rows, std::size_t cols, std::size_t block, std::size_t reps) {
    using Wide = typename mtl::simd::detail::block_accumulator<NA, NW>::type;
    const std::size_t nb = cols / block;
    const long whi = sizeof(NW) == 2 ? 4096 : 127;
    const auto w = fill<NW>(rows * cols, std::is_signed_v<NW> ? -whi : 0, whi, 1);
    const auto x = fill<NA>(cols, 0, std::is_signed_v<NA> ? 127 : 255, 2);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> ds(0.001f, 0.01f);
    std::vector<float> sw(rows * nb), sa(nb), s(rows * nb);
    for (auto& e : sw) e = ds(rng);
    for (auto& e : sa) e = ds(rng);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t b = 0; b < nb; ++b) s[r * nb + b] = sw[r * nb + b] * sa[b];
    std::vector<float> xf(cols);
    for (std::size_t b = 0; b < nb; ++b)
        for (std::size_t j = 0; j < block; ++j)
            xf[b * block + j] = static_cast<float>(x[b * block + j]) * sa[b];

    std::vector<float> y(rows), wf(cols);
    // The activation operand goes first: the quad op takes (unsigned, signed).
    const double t_block = median_ms(reps, [&] {
        for (std::size_t r = 0; r < rows; ++r) {
            float acc = 0.0f;
            for (std::size_t b = 0; b < nb; ++b)
                acc += s[r * nb + b] *
                       static_cast<float>(mtl::simd::reduce_dot_widen<Wide>(
                           x.data() + b * block, w.data() + r * cols + b * block, block));
            y[r] = acc;
        }
    });
    const float c_block = y[rows / 2];
    const double t_deq = median_ms(reps, [&] {
        for (std::size_t r = 0; r < rows; ++r) {
            const NW* wr = w.data() + r * cols;
            for (std::size_t b = 0; b < nb; ++b) {
                const float sc = sw[r * nb + b];
                for (std::size_t j = 0; j < block; ++j)
                    wf[b * block + j] = static_cast<float>(wr[b * block + j]) * sc;
            }
            y[r] = mtl::simd::reduce_dot(wf.data(), xf.data(), cols);
        }
    });
    const float c_deq = y[rows / 2];
    const double t_fused = median_ms(reps, [&] {
        for (std::size_t r = 0; r < rows; ++r)
            y[r] = mtl::simd::reduce_dot_block_scaled(x.data(), w.data() + r * cols,
                                                      s.data() + r * nb, nb, block);
    });
    const float c_fused = y[rows / 2];

    std::printf("| %-9s | %5zu | %10.3f | %10.3f | %12.3f | %5.2fx | %5.2fx |  %.6g / %.6g / %.6g\n",
                pair, block, t_block, t_deq, t_fused, t_deq / t_block, t_deq / t_fused,
                double(c_block), double(c_deq), double(c_fused));
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 4096;
    const std::size_t cols = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 4096;
    const std::size_t reps = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 20;
    std::printf("block-scaled gemv %zux%zu, single thread, median of %zu -- backend %s, "
                "%zu float lanes, quad dot %s\n\n",
                rows, cols, reps, mtl::simd::backend_name(), mtl::simd::batch<float>::size,
                mtl::simd::quad_dot_support_name());
    std::printf("| pair      | block | per-block ms | dequant ms | block-scaled ms | per-block vs dequant | "
                "block-scaled vs dequant |  y[mid]: per-block / dequant / block-scaled\n");
    std::printf("|---|---|---|---|---|---|---|---|\n");
    for (std::size_t block : {std::size_t(32), std::size_t(256)}) {
        if (cols % block != 0) continue;
        run<std::int16_t, std::int16_t>("i16 x i16", rows, cols, block, reps);
        run<std::int8_t, std::uint8_t>("u8 x i8", rows, cols, block, reps);
    }
    return 0;
}
