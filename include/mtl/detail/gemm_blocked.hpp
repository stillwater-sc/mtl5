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
// Multithreading (#92 + #297 batch 9, no OpenMP -- the C++ standard concurrency
// runtime): BLIS "multi-loop" 2D parallelism over a jc_nt x ic_nt thread grid.
// The n-dimension jc-blocks are partitioned across jc_nt teams; within each team
// the m-dimension ic-blocks are partitioned across ic_nt threads. A team's
// designated leader (ic_id == 0) packs the shared B panel B(pc,jc) once; the team
// members synchronize on a per-team barrier, then each packs its own A block and
// writes its DISJOINT C rows -- no race on C, one packed-B per team instead of
// per ic-block. The grid degenerates to the pure ic-parallel case (jc_nt == 1)
// for tall/square problems and to pure jc-parallel (ic_nt == 1) for wide/short
// ones, whichever the shape can fill. Because every C macro-block receives the
// same FMAs in the same order regardless of which (jc_id, ic_id) thread runs it,
// and the pc loop stays sequential per block, the threaded result is
// BIT-IDENTICAL to the single-thread result for ANY grid shape. Thread count
// comes from gemm_default_threads() (env MTL5_NUM_THREADS; default 1 = serial).

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/detail/aligned_allocator.hpp>
#include <mtl/detail/gemm_microkernel.hpp>
#include <mtl/detail/gemm_pack.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/simd/blocking.hpp>

namespace mtl::detail {

/// SIMD-aligned scratch buffer for packed panels (reused across the nest).
template <typename T>
using packed_buffer = std::vector<T, aligned_allocator<T>>;

/// Default GEMM thread count: the persistent pool's size (env MTL5_NUM_THREADS,
/// clamped to hardware concurrency; 1 when unset/invalid). Using the pool as the
/// single source of truth guarantees the GEMM's team never exceeds the pool.
inline unsigned gemm_default_threads() {
    return thread_pool::instance().size();
}

/// Reusable sense-reversing barrier for a fixed party count. Used to synchronize
/// the ic-threads of one jc-team around their shared packed-B panel: the leader
/// packs B, all wait (publish), the team computes, all wait again (so the leader
/// does not overwrite B for the next pc while members still read it). The
/// release/acquire on `sense_` establishes happens-before from the leader's pack
/// to the members' reads (and from members' reads back to the leader's repack).
/// parties <= 1 makes wait() a no-op (single-member team). The pool never nests,
/// so this is the only synchronization inside a GEMM parallel region.
class gemm_barrier {
public:
    explicit gemm_barrier(unsigned parties) noexcept : parties_(parties) {}
    void wait() noexcept {
        if (parties_ <= 1) return;
        const unsigned s = sense_.load(std::memory_order_relaxed);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == parties_) {
            arrived_.store(0, std::memory_order_relaxed);   // reset BEFORE the flip
            sense_.store(s + 1, std::memory_order_release); // publish + release spinners
        } else {
            while (sense_.load(std::memory_order_acquire) == s)
                std::this_thread::yield();
        }
    }
private:
    const unsigned parties_;
    std::atomic<unsigned> arrived_{0};
    std::atomic<unsigned> sense_{0};
};

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

    const std::size_t kc_max = std::min(KC, k);
    const std::size_t mc_max = std::min(MC, m);
    const std::size_t nc_max = std::min(NC, n);

    // One ic-block: pack A(ic,pc) into `Acbuf`, then run the jr/ir macro over the
    // shared packed-B panel `Bpack` into this block's (disjoint) C rows. `Acbuf`
    // and `Bpack` are caller-owned so each thread/team passes its own buffers.
    auto do_ic_block = [&](std::size_t ic, std::size_t jc, std::size_t nci,
                           std::size_t npanels, std::size_t kci, std::size_t pc,
                           const TAB* Bpack, TAB* Acbuf) {
        const std::size_t mci = std::min(MC, m - ic);
        pack_A<TAB, MR>(A + static_cast<std::ptrdiff_t>(ic) * a_rs
                          + static_cast<std::ptrdiff_t>(pc) * a_cs,
                        a_rs, a_cs, mci, kci, Acbuf);
        if (!(alpha == TC(1))) {                          // fold alpha into A panel (operand precision)
            const std::size_t na = packed_A_size(mci, kci, MR);
            for (std::size_t t = 0; t < na; ++t)
                Acbuf[t] = static_cast<TAB>(alpha * static_cast<TC>(Acbuf[t]));
        }

        const std::size_t mpanels = (mci + MR - 1) / MR;
        TC* Cmacro = C + static_cast<std::ptrdiff_t>(ic) * static_cast<std::ptrdiff_t>(ldc)
                       + static_cast<std::ptrdiff_t>(jc);
        for (std::size_t jr = 0; jr < npanels; ++jr) {
            const std::size_t nr_eff = std::min(NR, nci - jr * NR);
            const TAB* Bpanel = Bpack + jr * (NR * kci);
            for (std::size_t ir = 0; ir < mpanels; ++ir) {
                const std::size_t mr_eff = std::min(MR, mci - ir * MR);
                const TAB* Apanel = Acbuf + ir * (MR * kci);
                TC* Cblock = Cmacro + (ir * MR) * ldc + jr * NR;
                if (mr_eff == MR && nr_eff == NR) {
                    gemm_microkernel<TC, TAB, MR, NR>(kci, Apanel, Bpanel, Cblock, ldc);
                } else {
                    // Edge: accumulate through a zeroed full mr x nr tile so the
                    // micro-kernel's full-tile load/store stays in bounds.
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

    // Serial nest (also the fallback when the grid degenerates to one worker):
    // shared Ac/Bc buffers, jc -> pc -> ic exactly as the classic 5-loop order.
    auto serial_nest = [&]() {
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
                for (std::size_t ic = 0; ic < m; ic += MC)
                    do_ic_block(ic, jc, nci, npanels, kci, pc, Bc.data(), Ac.data());
            }
        }
    };

    if (nthreads <= 1) { serial_nest(); return; }

    // Block-start lists for the m (ic) and n (jc) dimensions.
    std::vector<std::size_t> ic_starts, jc_starts;
    for (std::size_t ic = 0; ic < m; ic += MC) ic_starts.push_back(ic);
    for (std::size_t jc = 0; jc < n; jc += NC) jc_starts.push_back(jc);
    const std::size_t nib = ic_starts.size();
    const std::size_t njb = jc_starts.size();

    // Thread budget: never exceed the pool. pool.run() clamps its worker count to
    // the pool size, so a grid sized past the pool would leave team members that
    // never run -- and their barrier would wait forever (deadlock). Cap here.
    thread_pool& pool = thread_pool::instance();
    const unsigned budget = std::min(nthreads, pool.size());
    if (budget <= 1) { serial_nest(); return; }

    // 2D grid factorization (deterministic; affects performance, not results).
    // Fill the ic loop first (keeps the shared B panel resident and maximizes its
    // reuse), then hand any leftover threads to the jc loop. Re-expand ic with
    // whatever remains so the grid uses as many threads as the shape allows.
    // grid = ic_nt * jc_nt <= budget <= pool size (surplus threads stay idle).
    unsigned ic_nt = static_cast<unsigned>(std::min<std::size_t>(budget, nib));
    if (ic_nt < 1) ic_nt = 1;
    unsigned jc_nt = static_cast<unsigned>(std::min<std::size_t>(budget / ic_nt, njb));
    if (jc_nt < 1) jc_nt = 1;
    ic_nt = static_cast<unsigned>(std::min<std::size_t>(budget / jc_nt, nib));
    if (ic_nt < 1) ic_nt = 1;
    const unsigned grid = ic_nt * jc_nt;

    if (grid <= 1) { serial_nest(); return; }

    // Pre-allocate ALL scratch outside the parallel region: one packed-B per
    // jc-team (shared within the team), one packed-A per grid thread, and one
    // barrier per team. Doing every allocation here means the region itself
    // never allocates -- so for the float/double fast path it cannot throw, and
    // a thread can never miss a barrier (which would deadlock the team).
    std::vector<packed_buffer<TAB>> team_B;   team_B.reserve(jc_nt);
    for (unsigned j = 0; j < jc_nt; ++j) team_B.emplace_back(packed_B_size(kc_max, nc_max, NR));
    std::vector<packed_buffer<TAB>> thr_A;    thr_A.reserve(grid);
    for (unsigned t = 0; t < grid; ++t) thr_A.emplace_back(packed_A_size(mc_max, kc_max, MR));
    std::vector<std::unique_ptr<gemm_barrier>> team_bar(jc_nt);
    for (unsigned j = 0; j < jc_nt; ++j) team_bar[j] = std::make_unique<gemm_barrier>(ic_nt);

    // Exception safety across the barrier: a Scalar element type may throw in
    // pack_B / do_ic_block (float/double cannot, and no allocation happens in the
    // region, but the template is Scalar-general -- posit/LNS etc.). A worker that
    // unwound out of the region would skip its remaining bar.wait() calls and its
    // teammates would spin forever. So we CATCH inside the region, record the
    // first failure atomically, skip only the *compute* on failure, and keep
    // every bar.wait() UNCONDITIONAL so barrier participation never desyncs. The
    // first exception is rethrown after the region joins. `first_exc` has a single
    // writer (the CAS winner) and is published to the caller by the pool's join.
    std::atomic<bool> failed{false};
    std::exception_ptr first_exc;
    auto record_failure = [&](std::exception_ptr e) noexcept {
        bool expected = false;
        if (failed.compare_exchange_strong(expected, true, std::memory_order_relaxed))
            first_exc = e;   // single-writer; visible to the caller after run() joins
    };

    // tid -> (jc_id, ic_id): threads sharing jc_id form one jc-team of ic_nt
    // members. Teams take jc-blocks round-robin by jc_id; members take ic-blocks
    // round-robin by ic_id. Every (ic-block, jc-block) C macro-block is owned by
    // exactly one thread -> disjoint writes, bit-identical to serial.
    pool.run(grid, [&](unsigned tid) {
        const unsigned jc_id = tid / ic_nt;
        const unsigned ic_id = tid % ic_nt;
        TAB* Aloc = thr_A[tid].data();
        TAB* Bteam = team_B[jc_id].data();
        gemm_barrier& bar = *team_bar[jc_id];
        for (std::size_t jb = jc_id; jb < njb; jb += jc_nt) {
            const std::size_t jc = jc_starts[jb];
            const std::size_t nci = std::min(NC, n - jc);
            const std::size_t npanels = (nci + NR - 1) / NR;
            for (std::size_t pc = 0; pc < k; pc += KC) {
                const std::size_t kci = std::min(KC, k - pc);
                if (ic_id == 0 && !failed.load(std::memory_order_relaxed)) {
                    try {                                         // team leader packs shared B once
                        pack_B<TAB, NR>(B + static_cast<std::ptrdiff_t>(pc) * b_rs
                                          + static_cast<std::ptrdiff_t>(jc) * b_cs,
                                        b_rs, b_cs, kci, nci, Bteam);
                    } catch (...) { record_failure(std::current_exception()); }
                }
                bar.wait();                                       // publish B to the team
                if (!failed.load(std::memory_order_relaxed)) {
                    try {
                        for (std::size_t ib = ic_id; ib < nib; ib += ic_nt)
                            do_ic_block(ic_starts[ib], jc, nci, npanels, kci, pc, Bteam, Aloc);
                    } catch (...) { record_failure(std::current_exception()); }
                }
                bar.wait();                                       // all done reading B before repack
            }
        }
    });
    if (first_exc) std::rethrow_exception(first_exc);
}

} // namespace mtl::detail
