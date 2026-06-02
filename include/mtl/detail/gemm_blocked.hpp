#pragma once
// MTL5 -- blocked GEMM macro-kernel: the GotoBLAS/BLIS 5-loop cache-blocking
// nest around the register micro-kernel (#88) and the packing step (#89).
// (#90, epic #82, Phase 2.)
//
// gemm_blocked computes  C := beta*C + alpha * A * B  with C row-major. A and B
// are given by a base pointer plus (row stride, col stride), so every operand
// orientation -- row-major, col-major, transposed view -- packs with NO special
// casing: A(i,p) == A[i*a_rs + p*a_cs], B(p,j) == B[p*b_rs + j*b_cs]. The packing
// normalizes layout; the micro-kernel always sees its canonical packed panels.
//
// The five loops (outer..inner), per Goto/BLIS:
//   jc (step nc)  -- partition C/B columns                       -> L3 (B panel)
//     pc (step kc)  -- partition the k dimension; pack B(pc,jc)   -> L1 (B micro-panel)
//       ic (step mc)  -- partition C/A rows; pack A(ic,pc)        -> L2 (A block)
//         jr (step nr) -- macro-kernel over packed panels
//           ir (step mr) -> MICRO-KERNEL: mr x nr C tile in registers, kc deep
//
// kc/mc/nc and the mr x nr register tile come from simd::default_blocking<T>.
// Edge tiles (trailing rows < mr or cols < nr) are accumulated through a zeroed
// mr x nr temporary so the micro-kernel only ever writes full tiles.
//
// Multithreading (#92, no OpenMP -- the C++ standard concurrency runtime): the
// `ic` loop is partitioned across a std::thread team. For a fixed (jc, pc) the
// ic-blocks write DISJOINT C row ranges and only READ the shared packed B panel,
// so there is no race; each thread keeps its own packed-A buffer. Because every
// C block receives the same FMAs in the same order regardless of which thread
// runs it (and the pc loop stays sequential), the threaded result is
// BIT-IDENTICAL to the single-thread result. Thread count comes from
// gemm_default_threads() (env MTL5_NUM_THREADS; default 1 = unchanged).

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/detail/aligned_allocator.hpp>
#include <mtl/detail/gemm_microkernel.hpp>
#include <mtl/detail/gemm_pack.hpp>
#include <mtl/simd/blocking.hpp>

namespace mtl::detail {

/// SIMD-aligned scratch buffer for packed panels (reused across the nest).
template <typename T>
using packed_buffer = std::vector<T, aligned_allocator<T>>;

/// Default GEMM thread count: env MTL5_NUM_THREADS, clamped to the hardware
/// concurrency; defaults to 1 (single-thread, behaviour unchanged) when unset or
/// invalid. Read once. Set MTL5_NUM_THREADS=N to enable the parallel path.
inline unsigned gemm_default_threads() {
    static const unsigned n = [] {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        if (const char* e = std::getenv("MTL5_NUM_THREADS")) {
            char* end = nullptr;
            const unsigned long v = std::strtoul(e, &end, 10);
            if (end != e && v >= 1)
                return static_cast<unsigned>(v < hw ? v : hw);
        }
        return 1u;
    }();
    return n;
}

/// C[m x n] (row-major, leading dim ldc) := beta*C + alpha * A[m x k] * B[k x n].
/// A,B addressed by generic strides (see file header). ldc must be >= n.
template <typename T>
    requires mtl::Scalar<T>
void gemm_blocked(std::size_t m, std::size_t n, std::size_t k,
                  T alpha,
                  const T* A, std::ptrdiff_t a_rs, std::ptrdiff_t a_cs,
                  const T* B, std::ptrdiff_t b_rs, std::ptrdiff_t b_cs,
                  T beta,
                  T* C, std::size_t ldc,
                  unsigned nthreads = 1) {
    constexpr simd::blocking_params bp = simd::default_blocking<T>;
    constexpr std::size_t MR = bp.mr;
    constexpr std::size_t NR = bp.nr;
    const std::size_t KC = bp.kc, MC = bp.mc, NC = bp.nc;

    // beta: scale (or zero) C once up front; the nest then purely accumulates.
    if (beta == T(0)) {
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) C[i * ldc + j] = T(0);
    } else if (!(beta == T(1))) {
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) C[i * ldc + j] = beta * C[i * ldc + j];
    }

    if (m == 0 || n == 0 || k == 0) return;

    // Scratch for one packed A block (mc x kc) and one packed B panel (kc x nc),
    // sized to the largest block actually used (clamped to the problem) so small
    // problems don't allocate full-MC/KC/NC buffers. Reused across the nest.
    const std::size_t kc_max = std::min(KC, k);
    const std::size_t mc_max = std::min(MC, m);
    const std::size_t nc_max = std::min(NC, n);
    packed_buffer<T> Ac(packed_A_size(mc_max, kc_max, MR));
    packed_buffer<T> Bc(packed_B_size(kc_max, nc_max, NR));

    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nci = std::min(NC, n - jc);
        const std::size_t npanels = (nci + NR - 1) / NR;
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kci = std::min(KC, k - pc);
            pack_B<T, NR>(B + static_cast<std::ptrdiff_t>(pc) * b_rs
                            + static_cast<std::ptrdiff_t>(jc) * b_cs,
                          b_rs, b_cs, kci, nci, Bc.data());

            // One ic-block: pack A(ic,pc) into `Acbuf`, then run the jr/ir macro
            // over the shared Bc into this block's (disjoint) C rows. `Acbuf` is
            // caller-owned so each thread can pass its own buffer.
            auto do_ic_block = [&](std::size_t ic, T* Acbuf) {
                const std::size_t mci = std::min(MC, m - ic);
                pack_A<T, MR>(A + static_cast<std::ptrdiff_t>(ic) * a_rs
                                + static_cast<std::ptrdiff_t>(pc) * a_cs,
                              a_rs, a_cs, mci, kci, Acbuf);
                if (!(alpha == T(1))) {                       // fold alpha into A panel
                    const std::size_t na = packed_A_size(mci, kci, MR);
                    for (std::size_t t = 0; t < na; ++t) Acbuf[t] = alpha * Acbuf[t];
                }

                const std::size_t mpanels = (mci + MR - 1) / MR;
                T* Cmacro = C + static_cast<std::ptrdiff_t>(ic) * static_cast<std::ptrdiff_t>(ldc)
                              + static_cast<std::ptrdiff_t>(jc);
                for (std::size_t jr = 0; jr < npanels; ++jr) {
                    const std::size_t nr_eff = std::min(NR, nci - jr * NR);
                    const T* Bpanel = Bc.data() + jr * (NR * kci);
                    for (std::size_t ir = 0; ir < mpanels; ++ir) {
                        const std::size_t mr_eff = std::min(MR, mci - ir * MR);
                        const T* Apanel = Acbuf + ir * (MR * kci);
                        T* Cblock = Cmacro + (ir * MR) * ldc + jr * NR;
                        if (mr_eff == MR && nr_eff == NR) {
                            gemm_microkernel<T, MR, NR>(kci, Apanel, Bpanel, Cblock, ldc);
                        } else {
                            // Edge: accumulate through a zeroed full mr x nr tile so
                            // the micro-kernel's full-tile load/store stays in bounds.
                            T tile[MR * NR];
                            for (std::size_t t = 0; t < MR * NR; ++t) tile[t] = T(0);
                            for (std::size_t i = 0; i < mr_eff; ++i)
                                for (std::size_t j = 0; j < nr_eff; ++j)
                                    tile[i * NR + j] = Cblock[i * ldc + j];
                            gemm_microkernel<T, MR, NR>(kci, Apanel, Bpanel, tile, NR);
                            for (std::size_t i = 0; i < mr_eff; ++i)
                                for (std::size_t j = 0; j < nr_eff; ++j)
                                    Cblock[i * ldc + j] = tile[i * NR + j];
                        }
                    }
                }
            };

            if (nthreads <= 1) {
                for (std::size_t ic = 0; ic < m; ic += MC)   // single-thread: shared Ac (unchanged)
                    do_ic_block(ic, Ac.data());
            } else {
                // Parallel over ic-blocks: disjoint C rows, shared read-only Bc,
                // per-thread A buffer. Static round-robin assignment.
                std::vector<std::size_t> ic_starts;
                for (std::size_t ic = 0; ic < m; ic += MC) ic_starts.push_back(ic);
                const unsigned team = static_cast<unsigned>(
                    std::min<std::size_t>(nthreads, ic_starts.size()));
                auto worker = [&](unsigned tid) {
                    packed_buffer<T> Aloc(packed_A_size(mc_max, kc_max, MR));
                    for (std::size_t b = tid; b < ic_starts.size(); b += team)
                        do_ic_block(ic_starts[b], Aloc.data());
                };
                std::vector<std::thread> pool;
                pool.reserve(team > 0 ? team - 1 : 0);
                for (unsigned t = 1; t < team; ++t) pool.emplace_back(worker, t);
                worker(0);
                for (auto& th : pool) th.join();
            }
        }
    }
}

} // namespace mtl::detail
