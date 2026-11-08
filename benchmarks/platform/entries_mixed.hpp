#pragma once
// mtl5_bench -- mixed precision and integer kernels (#721 milestone M6).
//
// MIXED PRECISION: narrow operands, wide accumulation. The check is what makes
// these entries mean anything -- it is set at the WIDE precision's bound, so a
// kernel that silently accumulated in the narrow type (whose error at these
// lengths is ~n * eps_f32, about 10^5 times larger) fails:
//   dot f32 -> f64, gemm f32 x f32 -> f64   gamma_n(f64) against sum |a b|; the
//                                           f32 products are exact in f64
//   block-scaled dot (Q8_0 layout)          per-block exact integer sums, scaled
//                                           and accumulated in f32: gamma(blocks)
//                                           of f32 against sum |scale * block sum|
//
// INTEGER: dot and GEMM in int32, int16 -> int32, int8 -> int32 and
// uint8 x int8 -> int32, plus the quad (VNNI / SDOT) GEMM kernel, mult_quad.
// Operands are bounded so no int32 partial sum can wrap (the library's overflow
// contract, batch.hpp), so the exact answer is known: the check is EQUALITY
// with an int64 reference. These report GOP/s (2 per multiply-add) but no
// %roof -- the probes measure floating-point peaks, and an integer ceiling
// would need an integer probe per ISA. The JSON and the sidecar carry the SIMD
// target and the native-or-emulated quad pairing, which is what an integer rate
// has to be read against (simd/batch.hpp).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/simd/algorithm.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <benchmarks/platform/common.hpp>

namespace mtl::bench::platform::mixed {

// -- mixed precision -------------------------------------------------------------

/// f32 vectors, dot accumulated in f64.
inline problem dot_f32_f64(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<float> x, y; double r = 0; };
    auto s = std::make_shared<S>(S{nonneg_vec_of<float>(n, 401), nonneg_vec_of<float>(n, 402), 0});
    problem p{vshape(n), 2.0 * n, 2.0 * n * sizeof(float), [] {},
              [s] { s->r = mtl::dot_real<double>(s->x, s->y); }, {}};
    p.check = [s, n] {
        long double ref = 0;
        for (std::size_t i = 0; i < n; ++i) ref += (long double)s->x(i) * s->y(i);
        error_tracker e(tol<double>(n), "rel err at the f64 bound (an f32 accumulation fails it)");
        e.add(s->r, ref, ref);   // nonnegative terms: sum |x y| = the sum
        return e.result();
    };
    return p;
}

/// f32 x f32 -> f64 GEMM through the documented widening interface,
/// mult<double>(A, B, C): operands converted to f64 before each product.
///
/// NOT plain mult(A, B, C) with a double C: that follows C++ promotion, so each
/// product is rounded in f32 before the f64 accumulation, and it is accurate
/// only to ~1e-8 (5e-9 to 1e-8 measured at n = 64..512, against 1e-16 for
/// mult<double>). The check, at the f64 bound, tells the two apart.
inline problem gemm_f32_f64(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<float> A, B; mat::dense2D<double> C; };
    auto s = std::make_shared<S>(S{mat_of<float>(n, n, 403), mat_of<float>(n, n, 404), mat::dense2D<double>(n, n)});
    problem p{std::to_string(n) + "x" + std::to_string(n) + "x" + std::to_string(n), 2.0 * n * n * n,
              2.0 * n * n * sizeof(float) + 1.0 * n * n * sizeof(double), [] {},
              [s] { mtl::mult<double>(s->A, s->B, s->C); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<double>(n), "rel err vs sum |a b| at the f64 bound (64 samples)");
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double ref = 0, abs = 0;
            for (std::size_t k = 0; k < n; ++k) {
                const long double t = (long double)s->A(i, k) * s->B(k, j);
                ref += t;
                abs += std::fabs(t);
            }
            e.add(s->C(i, j), ref, abs);
        });
        return e.result();
    };
    return p;
}

/// Block-scaled int8 dot, the Q8_0 layout of quantized inference: int8 values
/// in blocks of 32, one f32 scale per block on each side folded into `scales`.
inline problem block_scaled_dot(size_class c) {
    const std::size_t n = n_vec(c), block = 32, blocks = n / block;
    struct S { std::vector<std::int8_t> a, x; std::vector<float> scale; float r = 0; };
    auto s = std::make_shared<S>();
    xorshift64 g(405);
    s->a.resize(n); s->x.resize(n); s->scale.resize(blocks);
    for (std::size_t i = 0; i < n; ++i) {
        s->a[i] = static_cast<std::int8_t>(int(g.next_double() * 255.0) - 127);
        s->x[i] = static_cast<std::int8_t>(int(g.next_double() * 255.0) - 127);
    }
    for (auto& v : s->scale) v = static_cast<float>(g.next_double(0.001, 0.01));
    problem p{vshape(n) + ", block 32", 2.0 * n, 2.0 * n + 4.0 * blocks, [] {},
              [s, blocks] {
                  s->r = mtl::simd::reduce_dot_block_scaled(s->a.data(), s->x.data(), s->scale.data(), blocks, block);
              }, {}};
    p.check = [s, blocks] {
        // Per block: the kernel run on ONE block, against scale * (exact integer
        // sum). Not exact equality -- the vectorized path converts each lane's
        // integer partial sum to f32 separately and sums the lanes in f32
        // (simd/algorithm.hpp) -- but within gamma_32 of scale * sum |a_j x_j|,
        // which an overflowed or mis-paired integer sum (an O(1) error) cannot
        // meet. The total below is then held to the f32 bound for summing the
        // blocks, loose by nature (gamma of 1024 f32 terms is 2.4e-4).
        std::size_t wrong_blocks = 0;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int64_t sum = 0, mag = 0;
            for (std::size_t j = 0; j < block; ++j) {
                const std::int64_t t = std::int64_t(s->a[b * block + j]) * std::int64_t(s->x[b * block + j]);
                sum += t;
                mag += t < 0 ? -t : t;
            }
            const float one = mtl::simd::reduce_dot_block_scaled(s->a.data() + b * block, s->x.data() + b * block,
                                                                 s->scale.data() + b, 1, block);
            const long double want = (long double)s->scale[b] * (long double)sum;
            const long double bound = tol<float>(block) * (long double)s->scale[b] * (long double)mag;
            wrong_blocks += !(std::fabs((long double)one - want) <= bound);
        }
        long double ref = 0, abs = 0;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int64_t sum = 0;   // exact: |sum| <= 32 * 127^2 < 2^31
            for (std::size_t j = 0; j < block; ++j)
                sum += std::int64_t(s->a[b * block + j]) * std::int64_t(s->x[b * block + j]);
            const long double t = (long double)s->scale[b] * (long double)sum;
            ref += t;
            abs += std::fabs(t);
        }
        error_tracker e(tol<float>(blocks), "each block within gamma_32; total within the f32 scale-and-sum bound");
        e.add(s->r, ref, abs);
        auto r = e.result();
        if (wrong_blocks) {
            r.ok = false;
            r.error = std::max(r.error, 1.0);
            r.what += "; " + std::to_string(wrong_blocks) + " blocks outside the per-block bound";
        }
        return r;
    };
    return p;
}

// -- integer -----------------------------------------------------------------------

template <typename T>
std::vector<T> int_values(std::size_t n, int lim, std::uint64_t seed) {
    xorshift64 g(seed);
    std::vector<T> v(n);
    for (auto& e : v) {
        const int r = static_cast<int>(g.next_double(-1.0, 1.0) * lim);
        e = static_cast<T>(std::is_signed_v<T> ? r : (r < 0 ? -r : r));
    }
    return v;
}

/// Magnitude bound per operand type so that n products cannot wrap int32.
template <typename A> int bound_for(std::size_t n) {
    // |sum| <= n * lim^2 < 2^31  =>  lim <= sqrt(2^31 / n).
    const int lim = static_cast<int>(std::sqrt(double(1u << 31) / double(n))) - 1;
    const int type_max = int(std::numeric_limits<A>::max());
    return std::max(1, std::min(lim, type_max));
}

/// Integer dot: NA x NB accumulated in int32, compared EXACTLY with int64.
template <typename NA, typename NB>
problem int_dot(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<NA> a; vec::dense_vector<NB> b; std::int32_t r = 0; };
    const int lim = std::min(bound_for<NA>(n), bound_for<NB>(n));
    auto va = int_values<NA>(n, lim, 406);
    auto vb = int_values<NB>(n, lim, 407);
    auto s = std::make_shared<S>(S{vec::dense_vector<NA>(n), vec::dense_vector<NB>(n), 0});
    for (std::size_t i = 0; i < n; ++i) { s->a(i) = va[i]; s->b(i) = vb[i]; }
    problem p{vshape(n) + ", |v| <= " + std::to_string(lim), 2.0 * n, double(n) * (sizeof(NA) + sizeof(NB)), [] {},
              [s] {
                  if constexpr (std::is_same_v<NA, std::int32_t>) s->r = mtl::dot_real(s->a, s->b);
                  else s->r = mtl::dot_real<std::int32_t>(s->a, s->b);
              }, {}};
    p.roofline = false;
    p.check = [s, n] {
        std::int64_t ref = 0;
        for (std::size_t i = 0; i < n; ++i) ref += std::int64_t(s->a(i)) * std::int64_t(s->b(i));
        check_result r;
        r.what = "exact (int64 reference; operands bounded so int32 cannot wrap)";
        r.tolerance = 0.0;
        r.error = std::fabs(double(std::int64_t(s->r) - ref));
        r.ok = std::int64_t(s->r) == ref;
        return r;
    };
    return p;
}

/// Integer GEMM, int32 accumulation. Quad = mult_quad (VNNI / SDOT micro-kernel).
template <typename NA, typename NB, bool Quad>
problem int_gemm(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<NA> A; mat::dense2D<NB> B; mat::dense2D<std::int32_t> C; };
    const int lim = std::min(bound_for<NA>(n), bound_for<NB>(n));
    auto s = std::make_shared<S>(S{mat::dense2D<NA>(n, n), mat::dense2D<NB>(n, n), mat::dense2D<std::int32_t>(n, n)});
    auto va = int_values<NA>(n * n, lim, 408);
    auto vb = int_values<NB>(n * n, lim, 409);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) { s->A(i, j) = va[i * n + j]; s->B(i, j) = vb[i * n + j]; }
    problem p{std::to_string(n) + "^3, |v| <= " + std::to_string(lim), 2.0 * n * n * n,
              double(n) * n * (sizeof(NA) + sizeof(NB) + sizeof(std::int32_t)), [] {},
              [s] {
                  if constexpr (Quad) mtl::mult_quad<std::int32_t>(s->A, s->B, s->C);
                  else if constexpr (std::is_same_v<NA, std::int32_t>) mtl::mult(s->A, s->B, s->C);
                  else mtl::mult<std::int32_t>(s->A, s->B, s->C);
              }, {}};
    p.roofline = false;
    // Freivalds' check, exact in int64: C x = A (B x) for two random integer
    // vectors x. O(n^2) instead of the O(n^3) full reference, and -- unlike
    // sampling entries -- it sees EVERY entry: a wrong C(i, j) changes (C x)_i
    // by its error times x_j, never zero since x has no zero entries.
    p.check = [s, n] {
        std::size_t wrong = 0;
        xorshift64 g(410);
        for (int trial = 0; trial < 2; ++trial) {
            std::vector<std::int64_t> x(n), bx(n, 0);
            for (auto& v : x) v = 1 + static_cast<std::int64_t>(g.next_double() * 1000.0);
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t j = 0; j < n; ++j) bx[k] += std::int64_t(s->B(k, j)) * x[j];
            for (std::size_t i = 0; i < n; ++i) {
                std::int64_t abx = 0, cx = 0;
                for (std::size_t k = 0; k < n; ++k) abx += std::int64_t(s->A(i, k)) * bx[k];
                for (std::size_t j = 0; j < n; ++j) cx += std::int64_t(s->C(i, j)) * x[j];
                wrong += cx != abx;
            }
        }
        check_result r;
        r.what = "exact: Freivalds C x = A (B x), two random x (int64)";
        r.tolerance = 0.0;
        r.error = double(wrong);
        r.ok = wrong == 0;
        return r;
    };
    return p;
}

inline void register_all(registry& r) {
    r.add("mixed", "dot", "f32>f64", dot_f32_f64);
    r.add("mixed", "gemm", "f32>f64", gemm_f32_f64);
    r.add("mixed", "block_scaled_dot", "i8+f32", block_scaled_dot);

    r.add("integer", "dot", "i32", int_dot<std::int32_t, std::int32_t>);
    r.add("integer", "dot", "i16>i32", int_dot<std::int16_t, std::int16_t>);
    r.add("integer", "dot", "i8>i32", int_dot<std::int8_t, std::int8_t>);
    r.add("integer", "dot", "u8i8>i32", int_dot<std::uint8_t, std::int8_t>);
    r.add("integer", "gemm", "i32", int_gemm<std::int32_t, std::int32_t, false>);
    r.add("integer", "gemm", "i16>i32", int_gemm<std::int16_t, std::int16_t, false>);
    r.add("integer", "gemm", "i8>i32", int_gemm<std::int8_t, std::int8_t, false>);
    r.add("integer", "gemm_quad", "i8>i32", int_gemm<std::int8_t, std::int8_t, true>);
    r.add("integer", "gemm_quad", "u8i8>i32", int_gemm<std::uint8_t, std::int8_t, true>);
}

} // namespace mtl::bench::platform::mixed
