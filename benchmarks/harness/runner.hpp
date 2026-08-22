#pragma once
// MTL5 Benchmark Harness -- Suite runners.
//
// One binary == one backend. The benchmark calls only the public mtl:: API;
// compile-time dispatch (governed by MTL5_HAS_BLAS / MTL5_HAS_LAPACK) decides
// whether each op runs the generic C++ path or a BLAS/LAPACK path. The build
// configuration therefore *is* the backend -- exactly as in a dependent
// application that sets the BLAS/LAPACK flags once for the whole program.
//
// The `label` (e.g. "native", "openblas", "mkl") is passed in and recorded in
// the output; the harness itself does not select an implementation.

#include <benchmarks/harness/timer.hpp>
#include <benchmarks/harness/reporter.hpp>
#include <benchmarks/harness/generators.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mtl/simd/batch.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/ger.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/syr2k.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/vec/operators.hpp>   // element-wise vector expressions (a + b)
#include <mtl/mat/operators.hpp>   // element-wise matrix expressions (A + B)

namespace mtl::bench {

// ── Integer suites (#451 phase 4) ──────────────────────────────────────────
//
// WHAT THESE DO AND DO NOT MEASURE, because the obvious reading is wrong.
//
// A dot product is a streaming reduction: two operands in, two ops each, no
// reuse. At large n it is BANDWIDTH-bound, and an int8 dot moves one byte per
// element where an fp64 dot moves eight. So a large-n int8 arm is expected to
// run several times faster than fp64 whether or not the machine has VNNI at
// all -- that speedup is bytes, not arithmetic, and reading it as "VNNI is 6x
// faster" would be wrong.
//
// The instruction only shows where the kernel is compute- or latency-bound,
// which for a reduction means L1-resident sizes. Hence the sweep below spans
// L1 to DRAM: the small end isolates the instruction, the large end isolates
// the traffic, and the SHAPE of the curve across them is the actual result.
//
// The honest place for VNNI's arithmetic density is a GEMM, where operands are
// reused O(n) times and the kernel is compute-bound throughout. That is
// bench_gemm_int below, which since phase 5 runs the quad multiply-accumulate
// itself -- not merely narrow operands widened on load -- against the
// widen-on-load arm as its within-machine control.
//
// Every arm is CHECKED as well as timed, in two distinct senses that are easy
// to conflate:
//
//   the generators bound the operand magnitude so all widths stay inside the
//   int32 accumulator, which makes the arms comparable to each other and to a
//   64-bit reference. An arm that silently wrapped would otherwise still
//   produce a plausible time.
//
//   AND, for the GEMM arms, the output is actually READ afterwards and compared
//   against that reference (verify_gemm_int). The first sense alone was not
//   enough: `measure` only calls the lambda and records the time, so a kernel
//   returning nonsense still produced a well-formed CSV row, and an output
//   nobody reads is also the shape of work a compiler may delete. This PR's own
//   history is the argument -- its two int8 arms were briefly the SAME kernel,
//   and no timing could have revealed it.

/// Sizes spanning L1 -> L2 -> L3 -> DRAM for a streaming reduction. Chosen by
/// FOOTPRINT rather than by round numbers: at n = 4M the fp64 arm touches 64 MB
/// and the int8 arm 8 MB, which is the point.
inline const std::vector<std::size_t> kDefaultIntSizes = {
    1024, 4096, 16384, 65536, 262144, 1048576, 4194304};

/// dot: fp64 and fp32 baselines against the integer widths (#451 phases 0-3).
inline void bench_dot_int(reporter& rep, const std::string& label,
                          const std::vector<std::size_t>& sizes,
                          std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        const double ops = static_cast<double>(2 * n);

        // Baselines, same sizes, so the comparison is in one CSV.
        {
            auto a = make_random_vector<double>(n), b = make_random_vector<double>(n, 456);
            volatile double sink = 0.0;
            rep.add(measure([&]{ sink = mtl::dot_real(a, b); },
                            "dot_f64", label, n, ops, warmup, iterations));
            (void)sink;
        }
        {
            auto a = make_random_vector<float>(n), b = make_random_vector<float>(n, 456);
            volatile float sink = 0.0f;
            rep.add(measure([&]{ sink = mtl::dot_real(a, b); },
                            "dot_f32", label, n, ops, warmup, iterations));
            (void)sink;
        }
        // int32 lanes (phase 0). Operands bounded so the sum stays exact.
        {
            const int lim = 1 << 10;
            auto a = make_random_int_vector<std::int32_t>(n, lim);
            auto b = make_random_int_vector<std::int32_t>(n, lim, 456);
            volatile std::int32_t sink = 0;
            rep.add(measure([&]{ sink = mtl::dot_real(a, b); },
                            "dot_i32", label, n, ops, warmup, iterations));
            (void)sink;
        }
        // int16 -> int32 pairwise widen (phase 2). The magnitude bound is the
        // binding one here: at full 16-bit range this would wrap within a few
        // terms, which is the contract this arm exists to respect.
        {
            const int lim = 1 << 5;
            auto a = make_random_int_vector<std::int16_t>(n, lim);
            auto b = make_random_int_vector<std::int16_t>(n, lim, 456);
            volatile std::int32_t sink = 0;
            rep.add(measure([&]{ sink = mtl::dot_real<std::int32_t>(a, b); },
                            "dot_i16_i32", label, n, ops, warmup, iterations));
            (void)sink;
        }
        // int8 -> int32 quad widen (phase 3), symmetric form. Native only from
        // AVX10.2 on x86; elsewhere Highway decomposes it.
        {
            const int lim = 1 << 4;
            auto a = make_random_int_vector<std::int8_t>(n, lim);
            auto b = make_random_int_vector<std::int8_t>(n, lim, 456);
            volatile std::int32_t sink = 0;
            rep.add(measure([&]{ sink = mtl::dot_real<std::int32_t>(a, b); },
                            "dot_i8_i32", label, n, ops, warmup, iterations));
            (void)sink;
        }
        // uint8 x int8 -> int32: VNNI's NATIVE shape, and the arm the phase-3
        // claim actually rests on. Compare against dot_i8_i32 on a machine
        // without AVX10.2 to see what the decomposition costs.
        {
            const int lim = 1 << 4;
            auto a = make_random_int_vector<std::uint8_t>(n, lim);
            auto b = make_random_int_vector<std::int8_t>(n, lim, 456);
            volatile std::int32_t sink = 0;
            rep.add(measure([&]{ sink = mtl::dot_real<std::int32_t>(a, b); },
                            "dot_u8i8_i32", label, n, ops, warmup, iterations));
            (void)sink;
        }
    }
}

/// gemv: fp64/fp32 baselines against int32 (#451 phase 0).
///
/// Only int32 -- there is no widening gemv. Phases 2 and 3 delivered widening
/// DOT kernels; a widening gemv needs per-row accumulators in the narrow type
/// and was not part of them. Benchmarking a widening gemv that does not exist
/// is not possible, and benchmarking the generic loop under that name would be
/// a lie about what shipped.
///
/// NOTE: the native SIMD gemv is behind MTL5_NATIVE_FAST_GEMM. Without it this
/// arm measures the generic element-wise loop for EVERY type, integer or not,
/// and the comparison is between fallbacks rather than kernels.
inline void bench_gemv_int(reporter& rep, const std::string& label,
                           const std::vector<std::size_t>& sizes,
                           std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        const double ops = static_cast<double>(2 * n * n);
        {
            auto A = make_random_matrix<double>(n, n);
            auto x = make_random_vector<double>(n);
            mtl::vec::dense_vector<double> y(n);
            rep.add(measure([&]{ mtl::mult(A, x, y); },
                            "gemv_f64", label, n, ops, warmup, iterations));
        }
        {
            auto A = make_random_matrix<float>(n, n);
            auto x = make_random_vector<float>(n);
            mtl::vec::dense_vector<float> y(n);
            rep.add(measure([&]{ mtl::mult(A, x, y); },
                            "gemv_f32", label, n, ops, warmup, iterations));
        }
        {
            const int lim = 1 << 8;
            mtl::mat::dense2D<std::int32_t> A(n, n);
            auto x = make_random_int_vector<std::int32_t>(n, lim);
            mtl::vec::dense_vector<std::int32_t> y(n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    A(i, j) = static_cast<std::int32_t>((i * 31 + j * 17) % (2 * lim) - lim);
            rep.add(measure([&]{ mtl::mult(A, x, y); },
                            "gemv_i32", label, n, ops, warmup, iterations));
        }
    }
}

/// gemm: fp64/fp32 baselines against int32 and the widening 8/16-bit forms.
///
/// This is the arm the dot suite could not provide. A dot has no operand reuse,
/// so it is bandwidth-bound at any interesting size and the operand width
/// dominates everything else. A GEMM reuses each operand O(n) times and is
/// compute-bound once it is past the caches, so the balance between "moves
/// fewer bytes" and "does arithmetic faster" is genuinely different here -- and
/// measuring that difference is the point.
///
/// TWO INT8 KERNELS RUN HERE, and comparing them IS the experiment (#451 phase
/// 5). Both compute the same thing, bit for bit -- integer addition is
/// associative, so a different summation grouping is unobservable in the result:
///
///   gemm_i8_i32        widen-on-load: int8 promoted into int32 lanes, one k
///                      value per multiply-add
///   gemm_i8_i32_quad   the quad multiply-accumulate: four k values per
///                      instruction, from quad-interleaved panels
///
/// The pair is a WITHIN-MACHINE control, which is the whole reason both are
/// compiled into one binary. The programme has already paid for the absence of
/// one: the instruction's share of the dot speedup looked like 1.42x from a
/// single Zen 4 run and came out at ~1.2x once an i7 could run both arms
/// decomposed. Comparing a quad number on one machine against a widen number on
/// another would repeat that mistake with more decimal places.
///
/// `gemm_u8i8_i32_quad` is VNNI's NATIVE pairing -- unsigned activations against
/// signed weights -- and has no widen-on-load counterpart at all: the widening
/// load requires matching signedness, so before this kernel that pair fell to
/// the generic scalar loop. Its baseline is therefore the SYMMETRIC QUAD arm,
/// not the widening one: those two differ in one variable (operand signedness)
/// where the widening arm differs in two.
///
/// WHAT THE SYMMETRIC FORM COSTS DEPENDS ON THE TARGET, and the difference is
/// not cosmetic:
///
///   HWY_AVX3_DL / AVX10.2   `vpdpbusd` exists. Symmetric i8 x i8 is emulated
///                           with two of them plus a shift and subtract
///                           (native only from AVX10.2's `vpdpbssd`).
///   SSE4 / AVX2             `vpdpbusd` does NOT exist. Highway decomposes BOTH
///                           pairings through `vpmaddwd` plus sign-extension
///                           shifts, and the symmetric form carries more of that
///                           shift work -- verified by disassembly on SSE4:
///                           4 `vpsraw` + 2 `vpsllw` against 2 + 1 plus a mask.
///
/// So "two `vpdpbusd` plus a shift and subtract" describes only the first row.
/// Saying it of an SSE4 or AVX2 build would name an instruction the binary does
/// not contain.
///
/// Read `int8 quad dot:` in the header before trusting any of it. On a build
/// without the instruction these arms measure Highway's decomposition, which
/// looks like nothing in particular in a timing.
/// Check a SAMPLE of C against a 64-bit reference, and throw if it disagrees.
///
/// `measure` only calls the lambda and records elapsed time, so before this
/// existed no GEMM arm ever READ its own output: a kernel that returned nonsense
/// still produced a perfectly plausible benchmark row, and an unread output is
/// also the shape of computation a compiler is entitled to delete. That is
/// precisely the failure this PR's own history argues against -- the two int8
/// arms were briefly the SAME kernel, and no timing could have said so.
///
/// Sampled rather than exhaustive because a full O(n^3) scalar reference at
/// n = 1024 costs more than the measurement it guards. `stride` cells, each an
/// exact 64-bit dot over the full k, is O(samples * n) and catches a wrong
/// kernel, a wrong pack layout or a dropped block immediately -- the errors that
/// survive to this point are structural, not single-cell.
///
/// Called OUTSIDE the timed lambda, so it never enters the reported time.
template <typename MA, typename MB, typename MC>
void verify_gemm_int(const MA& A, const MB& B, const MC& C, const char* arm, std::size_t n) {
    using TC = typename MC::value_type;
    const std::size_t K = A.num_cols();
    const std::size_t step = (n / 8) ? (n / 8) : 1;   // ~64 cells on the diagonal grid
    for (std::size_t i = 0; i < n; i += step)
        for (std::size_t j = 0; j < n; j += step) {
            std::uint64_t acc = 0;
            for (std::size_t k = 0; k < K; ++k)
                acc += static_cast<std::uint64_t>(
                           static_cast<std::uint32_t>(static_cast<TC>(A(i, k)))) *
                       static_cast<std::uint32_t>(static_cast<TC>(B(k, j)));
            const TC want = static_cast<TC>(static_cast<std::uint32_t>(acc));
            if (C(i, j) != want) {
                // FAIL, do not warn, and do not let the run reach the CSV: a
                // sidecar written from a build that computes the wrong answer is
                // worse than no sidecar, because it looks like data. Exit rather
                // than throw so the operator sees one line instead of a
                // terminate handler -- the same convention as the guards in
                // bench_blocking_ab.cpp.
                std::fprintf(stderr,
                             "\nINTEGRITY FAILURE: %s at n=%zu -- C(%zu,%zu) = %lld, expected %lld\n"
                             "The kernel under test computes the wrong answer; no results written.\n",
                             arm, n, i, j,
                             static_cast<long long>(C(i, j)), static_cast<long long>(want));
                std::exit(2);
            }
        }
}

inline void bench_gemm_int(reporter& rep, const std::string& label,
                           const std::vector<std::size_t>& sizes,
                           std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        const double ops = 2.0 * double(n) * double(n) * double(n);
        {
            auto A = make_random_matrix<double>(n, n), B = make_random_matrix<double>(n, n, 7);
            mtl::mat::dense2D<double> C(n, n);
            rep.add(measure([&]{ mtl::mult(A, B, C); }, "gemm_f64", label, n, ops, warmup, iterations));
        }
        {
            auto A = make_random_matrix<float>(n, n), B = make_random_matrix<float>(n, n, 7);
            mtl::mat::dense2D<float> C(n, n);
            rep.add(measure([&]{ mtl::mult(A, B, C); }, "gemm_f32", label, n, ops, warmup, iterations));
        }
        // Same-type int32 through the blocked nest.
        {
            mtl::mat::dense2D<std::int32_t> A(n, n), B(n, n), C(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) = static_cast<std::int32_t>((i * 31 + j * 17) % 512 - 256);
                    B(i, j) = static_cast<std::int32_t>((i * 13 + j * 7) % 512 - 256);
                }
            rep.add(measure([&]{ mtl::mult(A, B, C); }, "gemm_i32", label, n, ops, warmup, iterations));
            verify_gemm_int(A, B, C, "gemm_i32", n);
        }
        // Widening arms: half and an eighth of fp64's bytes per operand.
        {
            mtl::mat::dense2D<std::int16_t> A(n, n), B(n, n);
            mtl::mat::dense2D<std::int32_t> C(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) = static_cast<std::int16_t>((i * 31 + j * 17) % 64 - 32);
                    B(i, j) = static_cast<std::int16_t>((i * 13 + j * 7) % 64 - 32);
                }
            rep.add(measure([&]{ mtl::mult<std::int32_t>(A, B, C); },
                            "gemm_i16_i32", label, n, ops, warmup, iterations));
            verify_gemm_int(A, B, C, "gemm_i16_i32", n);
        }
        {
            mtl::mat::dense2D<std::int8_t> A(n, n), B(n, n);
            mtl::mat::dense2D<std::int32_t> C(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) = static_cast<std::int8_t>((i * 31 + j * 17) % 32 - 16);
                    B(i, j) = static_cast<std::int8_t>((i * 13 + j * 7) % 32 - 16);
                }
            rep.add(measure([&]{ mtl::mult<std::int32_t>(A, B, C); },
                            "gemm_i8_i32", label, n, ops, warmup, iterations));
            verify_gemm_int(A, B, C, "gemm_i8_i32", n);
            // The SAME operands through the quad micro-kernel -- the arm the
            // widen-on-load number above is the control for.
            rep.add(measure([&]{ mtl::mult_quad<std::int32_t>(A, B, C); },
                            "gemm_i8_i32_quad", label, n, ops, warmup, iterations));
            verify_gemm_int(A, B, C, "gemm_i8_i32_quad", n);
        }
        // VNNI's native pairing. No widen-on-load counterpart exists (the
        // widening load requires matching signedness), so its baseline is the
        // symmetric quad arm above, which x86 below AVX10.2 emulates.
        {
            mtl::mat::dense2D<std::uint8_t> A(n, n);
            mtl::mat::dense2D<std::int8_t> B(n, n);
            mtl::mat::dense2D<std::int32_t> C(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) = static_cast<std::uint8_t>((i * 31 + j * 17) % 32);
                    B(i, j) = static_cast<std::int8_t>((i * 13 + j * 7) % 32 - 16);
                }
            rep.add(measure([&]{ mtl::mult_quad<std::int32_t>(A, B, C); },
                            "gemm_u8i8_i32_quad", label, n, ops, warmup, iterations));
            verify_gemm_int(A, B, C, "gemm_u8i8_i32_quad", n);
        }
    }
}

// ── BLAS-level suites ──────────────────────────────────────────────────────

inline void bench_dot(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto v1 = make_random_vector<double>(n);
        auto v2 = make_random_vector<double>(n, 456);
        double flops = static_cast<double>(2 * n);
        volatile double sink = 0.0;
        auto t = measure([&]{ sink = mtl::dot(v1, v2); },
                         "dot", label, n, flops, warmup, iterations);
        (void)sink;
        rep.add(t);
    }
}

inline void bench_nrm2(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto v = make_random_vector<double>(n);
        double flops = static_cast<double>(2 * n + 1);
        volatile double sink = 0.0;
        auto t = measure([&]{ sink = mtl::two_norm(v); },
                         "nrm2", label, n, flops, warmup, iterations);
        (void)sink;
        rep.add(t);
    }
}

inline void bench_axpy(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto x = make_random_vector<double>(n);
        auto y = make_random_vector<double>(n, 456);
        const double alpha = 1.0000001;   // near 1 so repeated y += alpha*x stays bounded
        double flops = static_cast<double>(2 * n);
        auto t = measure([&]{ mtl::axpy(alpha, x, y); },
                         "axpy", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_scal(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto x = make_random_vector<double>(n);
        const double alpha = 1.0000001;   // near 1 so repeated x *= alpha stays bounded
        double flops = static_cast<double>(n);
        auto t = measure([&]{ mtl::scale(alpha, x); },
                         "scal", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_gemv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto x = make_random_vector<double>(n);
        vec::dense_vector<double> y(n);
        double flops = static_cast<double>(2 * n * n);
        auto t = measure([&]{ mtl::mult(A, x, y); },
                         "gemv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_ger(reporter& rep, const std::string& label,
                      const std::vector<std::size_t>& sizes,
                      std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);   // col-major -> BLAS ger
        auto x = make_random_vector<double>(n);
        auto y = make_random_vector<double>(n, 77);
        const double alpha = 1.0000001;
        double flops = static_cast<double>(2 * n * n);
        auto t = measure([&]{ mtl::ger(alpha, x, y, A); },
                         "ger", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_symv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);   // symmetric; col-major -> BLAS symv
        auto x = make_random_vector<double>(n);
        vec::dense_vector<double> y(n, 0.0);
        double flops = static_cast<double>(2 * n * n);
        auto t = measure([&]{ mtl::symv(1.0, A, x, 0.0, y); },
                         "symv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_trmv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);   // col-major -> BLAS trmv
        auto x = make_random_vector<double>(n);
        double flops = static_cast<double>(n * n);
        auto t = measure([&]{ mtl::trmv(A, x, /*upper=*/true); },
                         "trmv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_trsv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);   // col-major -> BLAS trsv
        for (std::size_t i = 0; i < n; ++i)     // strengthen the diagonal for a stable solve
            A(i, i) += static_cast<double>(n);
        auto b = make_random_vector<double>(n);
        vec::dense_vector<double> x(n);
        double flops = static_cast<double>(n * n);
        auto t = measure([&]{ mtl::trsv(A, x, b, /*upper=*/true); },
                         "trsv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_gemm(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto B = make_random_matrix<double>(n, n, 99);
        mat::dense2D<double> C(n, n);
        double flops = static_cast<double>(2 * n * n * n);
        auto t = measure([&]{ mtl::mult(A, B, C); },
                         "gemm", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

// Rectangular GEMM shapes exercising the BLIS multi-loop (2D jc x ic) grid
// (#297 batch 9 / #311). Wide/short matrices have too few ic-blocks for the
// ic-only parallelization to fill the pool, so scaling there is the multi-loop
// payoff; tall/thin and square are the ic-parallel reference. Each shape is one
// CSV row keyed by `n` (the jc axis the 2D grid partitions) -- the shapes below
// have DISTINCT n so analyze_scaling.py (which groups series by the integer
// `size`) recovers each as its own speedup curve. The human-readable MxNxK is
// carried in the operation field. Row-major operands -> native GEMM path.
inline void bench_gemm_rect(reporter& rep, const std::string& label,
                            std::size_t warmup = 3, std::size_t iterations = 10) {
    struct shape { std::size_t m, n, k; };
    static const shape shapes[] = {
        {1024, 1024, 1024},   // square (baseline)
        {2048, 2048, 1024},   // square, larger
        {  32, 8192, 1024},   // wide/short: few ic-blocks -> jc-parallel (multi-loop)
        {  64, 4096, 1024},   // wide/short
        {8192,   64, 1024},   // tall/thin: many ic-blocks -> ic-parallel reference
    };
    for (const auto& s : shapes) {
        auto A = make_random_matrix<double>(s.m, s.k);
        auto B = make_random_matrix<double>(s.k, s.n, 99);
        mat::dense2D<double> C(s.m, s.n);
        const double flops = 2.0 * static_cast<double>(s.m)
                                 * static_cast<double>(s.n)
                                 * static_cast<double>(s.k);
        const std::string op = "gemm " + std::to_string(s.m) + "x"
                             + std::to_string(s.n) + "x" + std::to_string(s.k);
        auto t = measure([&]{ mtl::mult(A, B, C); },
                         op, label, s.n, flops, warmup, iterations);
        rep.add(t);
    }
}

// Element-wise expression sweeps (#297 batch 10 / #312): y = a + b (vector) and
// C = A + B (matrix). Pure per-index writes routed through detail::parallel_ewise
// -- memory-bandwidth bound, so scaling ceilings at the bandwidth, not the core
// count. The vector sweep is over the element index; the matrix sweep is
// row-parallel, so a WIDE/short matrix (few rows) cannot split (documents the
// #313 gap) while TALL/square shapes scale. The "gflops" column carries element
// throughput (elements/ns), whose ratio across thread counts is the speedup.
inline void bench_ewise(reporter& rep, const std::string& label,
                        std::size_t warmup = 3, std::size_t iterations = 20) {
    // Vectors: y = a + b. A volatile sink reads the last element inside the timed
    // region so an optimizer cannot elide the sweep (the output is otherwise
    // never consumed) or hoist the loop-invariant assignment out of the loop.
    volatile double sink = 0.0;
    for (std::size_t n : {std::size_t{100000}, std::size_t{1000000}, std::size_t{10000000}}) {
        auto a = make_random_vector<double>(n);
        auto b = make_random_vector<double>(n, 456);
        vec::dense_vector<double> c(n);
        const double work = static_cast<double>(n);           // 1 add / element
        auto t = measure([&]{ c = a + b; sink += c(n - 1); },
                         "ewise-vec", label, n, work, warmup, iterations);
        rep.add(t);
    }
    // Matrices: C = A + B (row-parallel sweep). Distinct nrows so analyze_scaling
    // keys each shape separately; kind + RxC in the operation field.
    struct shape { std::size_t r, c; const char* kind; };
    static const shape shapes[] = {
        {1024, 1024,   "square"},
        {4096, 4096,   "square"},
        {2000000,  8,  "tall"},   // many rows -> splits across the pool
        {1, 16000000,  "wide"},   // ONE row -> row-parallel sweep runs serial (#313)
    };
    for (const auto& s : shapes) {
        auto A = make_random_matrix<double>(s.r, s.c);
        auto B = make_random_matrix<double>(s.r, s.c, 99);
        mat::dense2D<double> C(s.r, s.c);
        const double work = static_cast<double>(s.r) * static_cast<double>(s.c);
        const std::string op = std::string("ewise-mat-") + s.kind + " "
                             + std::to_string(s.r) + "x" + std::to_string(s.c);
        auto t = measure([&]{ C = A + B; sink += C(s.r - 1, s.c - 1); },
                         op, label, s.r, work, warmup, iterations);
        rep.add(t);
    }
    (void)sink;
}

inline void bench_trmm(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);       // col-major -> BLAS trmm
        auto B = make_random_matrix_colmaj<double>(n, n, 99);
        double flops = static_cast<double>(n) * n * n;   // ~n^3 for square triangular*full
        auto t = measure([&]{ mtl::trmm(1.0, A, B, /*upper=*/true); },
                         "trmm", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_trsm(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);       // col-major -> BLAS trsm
        for (std::size_t i = 0; i < n; ++i)     // strengthen diagonal for stability
            A(i, i) += static_cast<double>(n);
        auto B_template = make_random_matrix_colmaj<double>(n, n, 99);
        double flops = static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto B = B_template;
                    mtl::trsm(1.0, A, B, /*upper=*/true);
                 }, "trsm", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_symm(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);       // symmetric; col-major -> BLAS symm
        auto B = make_random_matrix_colmaj<double>(n, n, 99);
        mat::dense2D<double, col_major_params> C(n, n);
        double flops = static_cast<double>(2 * n) * n * n;
        auto t = measure([&]{ mtl::symm(1.0, A, B, 0.0, C); },
                         "symm", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_syrk(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);       // col-major -> BLAS syrk
        mat::dense2D<double, col_major_params> C(n, n);
        double flops = static_cast<double>(n) * n * n;   // ~n^3 (half of a full gemm)
        auto t = measure([&]{ mtl::syrk(1.0, A, 0.0, C); },
                         "syrk", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_syr2k(reporter& rep, const std::string& label,
                        const std::vector<std::size_t>& sizes,
                        std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix_colmaj<double>(n, n);       // col-major -> BLAS syr2k
        auto B = make_random_matrix_colmaj<double>(n, n, 99);
        mat::dense2D<double, col_major_params> C(n, n);
        double flops = static_cast<double>(2 * n) * n * n;
        auto t = measure([&]{ mtl::syr2k(1.0, A, B, 0.0, C); },
                         "syr2k", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

// ── LAPACK-level suites ─────────────────────────────────────────────────────
// Column-major inputs so the BLAS/LAPACK dispatch is eligible (matches a
// real app that stores factorization operands column-major).

inline void bench_lu(reporter& rep, const std::string& label,
                      const std::vector<std::size_t>& sizes,
                      std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_random_matrix_colmaj<double>(n, n);
        double flops = (2.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    std::vector<std::size_t> pivot;
                    mtl::lu_factor(A, pivot);
                 }, "lu_factor", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_qr(reporter& rep, const std::string& label,
                     const std::vector<std::size_t>& sizes,
                     std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_random_matrix_colmaj<double>(n, n);
        // 2*m*n*k - 2/3*k^3 with m=n=k=n  ->  4/3 n^3
        double flops = (4.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    vec::dense_vector<double> tau;
                    mtl::qr_factor(A, tau);
                 }, "qr_factor", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_cholesky(reporter& rep, const std::string& label,
                           const std::vector<std::size_t>& sizes,
                           std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_spd_matrix_colmaj<double>(n);
        double flops = (1.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    mtl::cholesky_factor(A);
                 }, "cholesky", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_eigenvalue(reporter& rep, const std::string& label,
                             const std::vector<std::size_t>& sizes,
                             std::size_t warmup = 1, std::size_t iterations = 3) {
    for (auto n : sizes) {
        auto A_template = make_spd_matrix_colmaj<double>(n);
        double flops = (4.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    auto e = mtl::eigenvalue_symmetric(A);
                    (void)e;
                 }, "eig_sym", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

// ── Convenience: run all suites ─────────────────────────────────────────────

inline void run_all(reporter& rep, const std::string& label,
                    const std::vector<std::size_t>& blas_sizes,
                    const std::vector<std::size_t>& lapack_sizes) {
    std::cout << "=== BLAS Level 1 ===" << std::endl;
    bench_dot(rep, label, blas_sizes);
    bench_nrm2(rep, label, blas_sizes);
    bench_axpy(rep, label, blas_sizes);
    bench_scal(rep, label, blas_sizes);

    std::cout << "=== BLAS Level 2 ===" << std::endl;
    bench_gemv(rep, label, blas_sizes);
    bench_ger(rep, label, blas_sizes);
    bench_symv(rep, label, blas_sizes);
    bench_trmv(rep, label, blas_sizes);
    bench_trsv(rep, label, blas_sizes);

    std::cout << "=== BLAS Level 3 ===" << std::endl;
    bench_gemm(rep, label, blas_sizes);
    bench_trmm(rep, label, blas_sizes);
    bench_trsm(rep, label, blas_sizes);
    bench_symm(rep, label, blas_sizes);
    bench_syrk(rep, label, blas_sizes);
    bench_syr2k(rep, label, blas_sizes);

    std::cout << "=== LAPACK Factorizations ===" << std::endl;
    bench_lu(rep, label, lapack_sizes);
    bench_qr(rep, label, lapack_sizes);
    bench_cholesky(rep, label, lapack_sizes);
    bench_eigenvalue(rep, label, lapack_sizes);
}

} // namespace mtl::bench
