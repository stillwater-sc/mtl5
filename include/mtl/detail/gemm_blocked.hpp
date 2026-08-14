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
// The mr x nr register tile comes from simd::default_blocking<T> (compile time --
// it is a micro-kernel template argument); kc/mc/nc come from
// simd::runtime_blocking<T>(), derived against the DETECTED cache hierarchy so
// the cache blocks follow the machine rather than the Haswell-class defaults
// (#222). On a machine whose caches match those defaults, or one where detection
// is unavailable, the two agree and nothing changes.
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
/// Choose the ic-block size that balances the threaded m-partition (#408).
///
/// `mc_max` is an UPPER bound set by the L2 budget -- any smaller value is still
/// cache-legal -- so we are free to trade a little block size for a partition
/// that divides evenly across the team. The threaded nest hands ic-blocks to the
/// `ic_nt` members round-robin, so the most-loaded thread gets
/// `ceil(nib / ic_nt)` blocks and the critical path is minimised when `nib` is a
/// MULTIPLE of `ic_nt`. This picks the largest `mc <= mc_max` achieving that.
///
/// Why this is needed at all: `mc` used to be rounded down to a multiple of `mr`
/// in derive_blocking, which coupled the L2 block size to the register tile. The
/// #382 tile change (mr 4 -> 6) therefore moved mc 64 -> 60, turning nib = 16 at
/// m = 1024 (exactly 2.00 blocks per thread on 8 threads) into nib = 18 (2.25,
/// a 1.41x critical path). That converted a +21.5% single-thread win into a
/// -7.4% eight-thread REGRESSION. The rounding is gone and the balance is chosen
/// here, where `m` and the thread count are actually known.
///
/// Returns `mc_max` unchanged for the serial case, where there is nothing to
/// balance and the largest cache-legal block is best.
inline std::size_t balanced_mc(std::size_t m, std::size_t mc_max, unsigned ic_nt,
                               std::size_t mr = 1) {
    // Serial: nothing to balance, so take the largest cache-legal block -- and
    // round it to a whole number of mr-row panels, which is worth ~3% here
    // because it removes the ragged panel from every A block rather than only
    // the last one. That rounding is only harmful when it perturbs the THREADED
    // partition, which is the case below.
    if (ic_nt <= 1 || m == 0 || mc_max == 0) {
        if (mr > 1 && mc_max >= mr) return (mc_max / mr) * mr;
        return mc_max;
    }
    std::size_t nib = (m + mc_max - 1) / mc_max;        // fewest blocks the bound allows
    nib = ((nib + ic_nt - 1) / ic_nt) * ic_nt;          // round up to a multiple of ic_nt
    if (nib == 0) return mc_max;
    const std::size_t mc = (m + nib - 1) / nib;         // rows per block; <= mc_max
    return mc == 0 ? mc_max : mc;
}

/// The threaded nest's shape decision: the ic block size and the ic x jc thread
/// grid. Pure, so it can be unit-tested and enumerated offline (#430) instead of
/// being inferred from a throughput change.
struct gemm_grid {
    std::size_t mc;      ///< ic block size actually used (<= the cache bound)
    std::size_t nib;     ///< ic blocks that mc produces
    std::size_t njb;     ///< jc blocks
    unsigned    ic_nt;   ///< threads across ic
    unsigned    jc_nt;   ///< teams across jc
};

/// Choose mc and the grid for an m x n problem on `budget` threads.
///
/// Two defects this fixes, both found by measuring three machines (#430) and both
/// costing whole cores rather than percentages:
///
/// 1. mc WAS THE CACHE BOUND ONLY, so `nib = ceil(m/mc)` could be smaller than
///    the budget and cap ic_nt permanently -- with njb == 1 (square problems,
///    where n <= nc) there is no jc parallelism to make up the difference and the
///    machine simply idles. Measured: an i7-12700K at 1024^3 ran 5 of 8 threads
///    (ratio 0.551 against the smaller-mc arm) and a Zen 4 ran 4 of 8 (0.590).
///    mc is therefore also bounded by ceil(m/budget) -- enough blocks to hand
///    every thread one -- floored at a single register tile. This only ever
///    LOWERS mc, so the L2 bound is still respected, and it binds ONLY when the
///    partition would otherwise starve: at the shipped mc = 64 with m = 1024 and
///    8 threads the cap is 128 and nothing changes.
///
/// 2. THE FACTORIZATION WAS GREEDY: fill ic first, then give jc the integer
///    quotient. That throws the remainder away -- at budget 6 with nib = 4 it
///    picked 4 x 1 = 4 and left two cores idle, while 3 x 2 and 2 x 3 were both
///    legal. Measured on a Jetson Orin Nano: 0.760 where the grid was the only
///    difference. The search below is exhaustive over ic (a handful of values)
///    and maximises ic_nt * jc_nt, preferring the larger ic_nt on ties because a
///    wider ic team keeps the shared packed-B panel resident.
constexpr gemm_grid plan_gemm_grid(std::size_t m, std::size_t n,
                                   std::size_t mc_cache, std::size_t nc,
                                   std::size_t mr, unsigned budget) {
    if (budget < 1) budget = 1;
    // mc is a divisor and a loop step, so it must never leave here as 0 even on
    // a degenerate call -- a 0 step would not terminate. Everything else about a
    // degenerate call is "nothing to do": no blocks, a 1x1 grid.
    gemm_grid g{mc_cache < 1 ? 1 : mc_cache, 0, 0, 1, 1};
    if (m == 0 || n == 0 || mc_cache == 0 || nc == 0) return g;

    // (1) Enough ic blocks to fill the budget, if m allows. Never below one
    // register tile -- a block shorter than mr would waste the micro-kernel.
    const std::size_t want = (m + budget - 1) / budget;
    if (want < g.mc) g.mc = (want < mr) ? (mr < mc_cache ? mr : mc_cache) : want;

    g.nib = (m + g.mc - 1) / g.mc;
    g.njb = (n + nc - 1) / nc;

    // (2) Exhaustive search for the largest usable grid.
    unsigned best_ic = 1, best_jc = 1, best_grid = 1;
    const unsigned ic_max = static_cast<unsigned>(g.nib < budget ? g.nib : budget);
    for (unsigned ic = 1; ic <= ic_max; ++ic) {
        unsigned jc = budget / ic;
        if (jc > g.njb) jc = static_cast<unsigned>(g.njb);
        if (jc < 1) continue;
        const unsigned cand = ic * jc;
        if (cand > best_grid || (cand == best_grid && ic > best_ic)) {
            best_grid = cand; best_ic = ic; best_jc = jc;
        }
    }
    g.ic_nt = best_ic;
    g.jc_nt = best_jc;
    return g;
}

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
    constexpr std::size_t MR = bp.mr;   // register tile: must match the compiled
    constexpr std::size_t NR = bp.nr;   // micro-kernel, so compile-time only
    // L1/L2-sized blocks from the detected hierarchy (#222). rbp.mr/rbp.nr equal
    // MR/NR by construction -- detection overrides only cache fields, never the
    // register-file or FMA terms the tile is derived from.
    const simd::blocking_params& rbp = simd::runtime_blocking<TC>();
    const std::size_t KC = rbp.kc, MC = rbp.mc;
    // NC stays compile-time: it sets the jc BLOCK COUNT, and this nest hands jc
    // blocks to teams round-robin, so changing it perturbs the thread partition
    // (measured regression; see with_detected_caches). Detection of L3 is
    // deliberately not applied until the jc loop has a balanced_nc.
    const std::size_t NC = bp.nc;
    // The ic-block size actually used. Equal to the cache bound MC everywhere
    // except the threaded nest, which lowers it to balance the m-partition once
    // ic_nt is known (#408). Captured by reference below and finalized before
    // any call, so the serial paths see MC unchanged.
    std::size_t MC_eff = balanced_mc(m, MC, 1, MR);

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
        const std::size_t mci = std::min(MC_eff, m - ic);
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
                // MUST step by MC_eff, not MC: do_ic_block sizes its block as
                // min(MC_eff, m - ic), so stepping by the larger cache bound
                // would skip (MC - MC_eff) rows per block and silently drop
                // them from C.
                for (std::size_t ic = 0; ic < m; ic += MC_eff)
                    do_ic_block(ic, jc, nci, npanels, kci, pc, Bc.data(), Ac.data());
            }
        }
    };

    if (nthreads <= 1) { serial_nest(); return; }

    // The thread budget is needed BEFORE the block lists, because it now bounds
    // the ic block size -- see plan_gemm_grid. Never exceed the pool: pool.run()
    // clamps its worker count to the pool size, so a grid sized past the pool
    // would leave team members that never run, and their barrier would wait
    // forever (deadlock).
    thread_pool& pool = thread_pool::instance();
    const unsigned budget = std::min(nthreads, pool.size());
    if (budget <= 1) { serial_nest(); return; }

    const gemm_grid plan = plan_gemm_grid(m, n, MC, NC, MR, budget);

    // Block-start lists for the m (ic) and n (jc) dimensions. The m dimension
    // steps by the PLANNED mc, which may be smaller than the cache bound MC when
    // the cache bound alone would not produce enough blocks to fill the machine.
    std::vector<std::size_t> ic_starts, jc_starts;
    for (std::size_t ic = 0; ic < m; ic += plan.mc) ic_starts.push_back(ic);
    for (std::size_t jc = 0; jc < n; jc += NC) jc_starts.push_back(jc);
    const std::size_t nib = ic_starts.size();
    const std::size_t njb = jc_starts.size();

    // Thread budget: never exceed the pool. pool.run() clamps its worker count to
    const unsigned ic_nt = plan.ic_nt;
    const unsigned jc_nt = plan.jc_nt;
    const unsigned grid  = ic_nt * jc_nt;

    if (grid <= 1) { serial_nest(); return; }

    // Now that ic_nt is fixed, re-block the m dimension so the round-robin
    // partition divides evenly (#408). balanced_mc only ever LOWERS mc, so the
    // new block count is >= nib >= ic_nt and the grid stays valid; the packed-A
    // buffers sized from mc_max = min(MC, m) stay large enough for the same
    // reason. Rebuilding ic_starts is the only state that depends on it.
    //
    // This changes neither the result nor its bit-identity to the serial nest:
    // the FMA order for a given C element is fixed by the pc (k) loop, which is
    // untouched, and ic blocking only decides which rows are grouped together.
    MC_eff = balanced_mc(m, plan.mc, ic_nt, MR);
    if (MC_eff != plan.mc) {
        ic_starts.clear();
        for (std::size_t ic = 0; ic < m; ic += MC_eff) ic_starts.push_back(ic);
    }
    const std::size_t nib_eff = ic_starts.size();

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
                // The whole team packs the shared B block cooperatively, each
                // member taking a disjoint range of NR-column panels. Packing
                // used to be done by the leader alone while every other member
                // sat on the barrier -- a serial region proportional to
                // kc*nc per pc step, which is what capped parallel efficiency
                // (#110): at N=2048 with one team of 8, thread 0 packed ~67 MB
                // while 7 cores idled.
                //
                // Panels are independent and land at computable disjoint
                // offsets, and packing is pure data movement, so the packed
                // bytes -- and therefore the GEMM result -- are identical to
                // the serial pack for any split.
                if (!failed.load(std::memory_order_relaxed)) {
                    try {
                        const std::size_t per = (npanels + ic_nt - 1) / ic_nt;
                        const std::size_t q0  = std::min<std::size_t>(npanels, ic_id * per);
                        const std::size_t q1  = std::min<std::size_t>(npanels, q0 + per);
                        if (q0 < q1)
                            pack_B_panels<TAB, NR>(B + static_cast<std::ptrdiff_t>(pc) * b_rs
                                                     + static_cast<std::ptrdiff_t>(jc) * b_cs,
                                                   b_rs, b_cs, kci, nci, Bteam, q0, q1);
                    } catch (...) { record_failure(std::current_exception()); }
                }
                bar.wait();                                       // publish B to the team
                if (!failed.load(std::memory_order_relaxed)) {
                    try {
                        for (std::size_t ib = ic_id; ib < nib_eff; ib += ic_nt)
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
