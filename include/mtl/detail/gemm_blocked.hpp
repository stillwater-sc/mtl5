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

/// Joins a std::thread team on destruction (exception-safe: the team is joined
/// even if the main thread's share throws). Portable -- std::jthread is not
/// reliably available in Apple Clang's libc++.
struct thread_join_guard {
    std::vector<std::thread>& threads;
    ~thread_join_guard() { for (auto& t : threads) if (t.joinable()) t.join(); }
};

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
// TC = accumulator / C element type; TAB = A,B (operand) element type. With the
// default TAB == TC this is the original same-type blocked GEMM. When TAB is
// narrower (e.g. TAB=float, TC=double) the operands are packed in TAB and the
// micro-kernel widens them into TC accumulators -- the mixed-precision fast path
// (#176). Blocking is chosen for TC so the C microtile maps to TC registers.
template <typename TC, typename TAB = TC>
    requires (mtl::Scalar<TC> && mtl::Scalar<TAB>)
void gemm_blocked(std::size_t m, std::size_t n, std::size_t k,
                  TC alpha,
                  const TAB* A, std::ptrdiff_t a_rs, std::ptrdiff_t a_cs,
                  const TAB* B, std::ptrdiff_t b_rs, std::ptrdiff_t b_cs,
                  TC beta,
                  TC* C, std::size_t ldc,
                  unsigned nthreads = 1) {
    constexpr simd::blocking_params bp = simd::default_blocking<TC>;
    constexpr std::size_t MR = bp.mr;
    constexpr std::size_t NR = bp.nr;
    const std::size_t KC = bp.kc, MC = bp.mc, NC = bp.nc;

    // beta: scale (or zero) C once up front; the nest then purely accumulates.
    if (beta == TC(0)) {
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) C[i * ldc + j] = TC(0);
    } else if (!(beta == TC(1))) {
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
    packed_buffer<TAB> Ac(packed_A_size(mc_max, kc_max, MR));
    packed_buffer<TAB> Bc(packed_B_size(kc_max, nc_max, NR));

    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nci = std::min(NC, n - jc);
        const std::size_t npanels = (nci + NR - 1) / NR;
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kci = std::min(KC, k - pc);
            pack_B<TAB, NR>(B + static_cast<std::ptrdiff_t>(pc) * b_rs
                              + static_cast<std::ptrdiff_t>(jc) * b_cs,
                            b_rs, b_cs, kci, nci, Bc.data());

            // One ic-block: pack A(ic,pc) into `Acbuf`, then run the jr/ir macro
            // over the shared Bc into this block's (disjoint) C rows. `Acbuf` is
            // caller-owned so each thread can pass its own buffer.
            auto do_ic_block = [&](std::size_t ic, TAB* Acbuf) {
                const std::size_t mci = std::min(MC, m - ic);
                pack_A<TAB, MR>(A + static_cast<std::ptrdiff_t>(ic) * a_rs
                                  + static_cast<std::ptrdiff_t>(pc) * a_cs,
                                a_rs, a_cs, mci, kci, Acbuf);
                if (!(alpha == TC(1))) {                      // fold alpha into A panel (operand precision)
                    const std::size_t na = packed_A_size(mci, kci, MR);
                    for (std::size_t t = 0; t < na; ++t)
                        Acbuf[t] = static_cast<TAB>(alpha * static_cast<TC>(Acbuf[t]));
                }

                const std::size_t mpanels = (mci + MR - 1) / MR;
                TC* Cmacro = C + static_cast<std::ptrdiff_t>(ic) * static_cast<std::ptrdiff_t>(ldc)
                               + static_cast<std::ptrdiff_t>(jc);
                for (std::size_t jr = 0; jr < npanels; ++jr) {
                    const std::size_t nr_eff = std::min(NR, nci - jr * NR);
                    const TAB* Bpanel = Bc.data() + jr * (NR * kci);
                    for (std::size_t ir = 0; ir < mpanels; ++ir) {
                        const std::size_t mr_eff = std::min(MR, mci - ir * MR);
                        const TAB* Apanel = Acbuf + ir * (MR * kci);
                        TC* Cblock = Cmacro + (ir * MR) * ldc + jr * NR;
                        if (mr_eff == MR && nr_eff == NR) {
                            gemm_microkernel<TC, TAB, MR, NR>(kci, Apanel, Bpanel, Cblock, ldc);
                        } else {
                            // Edge: accumulate through a zeroed full mr x nr tile so
                            // the micro-kernel's full-tile load/store stays in bounds.
                            TC tile[MR * NR];
                            for (std::size_t t = 0; t < MR * NR; ++t) tile[t] = TC(0);
                            for (std::size_t i = 0; i < mr_eff; ++i)
                                for (std::size_t j = 0; j < nr_eff; ++j)
                                    tile[i * NR + j] = Cblock[i * ldc + j];
                            gemm_microkernel<TC, TAB, MR, NR>(kci, Apanel, Bpanel, tile, NR);
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
                    packed_buffer<TAB> Aloc(packed_A_size(mc_max, kc_max, MR));
                    for (std::size_t b = tid; b < ic_starts.size(); b += team)
                        do_ic_block(ic_starts[b], Aloc.data());
                };
                // The guard joins the team on scope exit -- including if worker(0)
                // throws (e.g. a buffer allocation failure) -- so an un-joined
                // std::thread never reaches its destructor and calls std::terminate.
                std::vector<std::thread> pool;
                pool.reserve(team > 0 ? team - 1 : 0);
                thread_join_guard guard{pool};
                for (unsigned t = 1; t < team; ++t) pool.emplace_back(worker, t);
                worker(0);
            }
        }
    }
}

} // namespace mtl::detail
