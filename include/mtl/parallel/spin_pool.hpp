#pragma once
// MTL5 -- a thread pool with mutex-free dispatch and join, for callers in a
// real-time loop (#636).
//
// spin_pool has thread_pool's interface and contract (see
// <mtl/parallel/thread_pool.hpp>): size(), run(count, task), parallel_for,
// parallel_reduce and parallel_accumulate; the caller runs tid 0; run() blocks
// until every tid completes, and every task's writes happen-before the
// caller's next action; a nested or concurrent region runs serially on its
// caller; the first exception is rethrown after the join; detail::in_region()
// behaves the same; run() does not allocate. The chunked loops are the same
// detail:: templates, so a partition is identical on either pool.
//
// What differs is the handoff. thread_pool publishes a region under a mutex
// and wakes workers through a condition variable; here:
//   * Dispatch is one store to an atomic word holding a generation and the
//     region's team size. A worker that is not part of the region only ever
//     reads that word.
//   * A worker spins on the word for `spin` iterations, each a CPU pause
//     instruction except every 64th, a yield, then blocks in
//     std::atomic::wait (on Linux a futex, for this 8-byte word one in the
//     standard library's shared table; WaitOnAddress on Windows). The yield
//     matters when the scheduler has put two of the pool's threads on one
//     core: without it, a spinner holds that core for its whole spin. The
//     caller makes the notify system call only if some worker has gone to
//     sleep.
//   * The join is an atomic countdown. The caller spins on it, then waits on
//     it; the last worker to finish notifies only a caller that is asleep.
//   * The busy flag and the first-exception slot are atomics too.
// No path through run() takes a lock, so a preempted worker holds nothing
// the caller needs: there is no priority inversion inside the pool.
//
// Worst case, per run() on a pool of n:
//   * Can block: a worker, in std::atomic::wait, once its spin runs out; the
//     caller, in its join, once its own spin runs out, until the slowest tid
//     finishes. Nothing else blocks. The common path makes no system call
//     while every worker is still spinning.
//   * Dispatch while every worker spins: one store, then one cache-line
//     transfer of the word to each worker. No system call.
//   * A worker that is asleep is woken through the kernel's scheduler, so its
//     wake-up has no bound MTL5 can state; keep regions closer together than
//     the spin, or give the workers cores of their own, to stay on the
//     spinning path.
//   * A worker preempted mid-region delays the join by as long as the
//     scheduler keeps it off a core. The pool cannot bound that; pinned
//     cores, a real-time scheduling class and a pool no larger than the cores
//     reserved for it can. The caller then spins, then sleeps, and is woken
//     by that worker when it finishes.
//   * Cost: a worker spins a full core for `spin` iterations after every
//     region before it sleeps; default_spin is about 140 us on an Alder Lake
//     P-core, where a pause takes about 35 ns, and less on cores with a
//     shorter pause.
// The measured dispatch and join latency is in
// docs/algorithms/on-node-threading.md.
//
// The spin is the constructor's second argument; the process pool reads it
// from MTL5_POOL_SPIN (default default_spin; 0 sleeps at once). Defining
// MTL5_POOL_SPIN makes pool(), and so MTL5's kernels, use the process
// spin_pool; it must then be defined in every translation unit alike. Under
// MTL5_SHARED_POOL the process spin_pool is shared across modules like the
// thread_pool, through shared_spin_pool_v1().
#include <mtl/parallel/thread_pool.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86) || defined(_M_ARM64))
#  include <intrin.h>
#endif

namespace mtl::parallel {
#if defined(MTL5_SHARED_POOL)
inline namespace pool_shared {
#else
inline namespace pool_local {
#endif

class spin_pool;

namespace detail {

#if defined(MTL5_SHARED_POOL)
/// The process's one spin_pool, defined with the shared thread_pool.
MTL5_SHARED_POOL_API spin_pool& shared_spin_pool_v1();
#endif

/// One spin-wait iteration: a pause hint to the core, nothing else.
inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

/// One iteration of a spin wait: a pause, and every 64th iteration a yield,
/// so a thread the scheduler has queued on this core behind the spinner runs
/// within a few microseconds instead of after the whole spin.
inline void spin_step(unsigned i) noexcept {
    if ((i & 63u) == 63u) std::this_thread::yield();
    else cpu_relax();
}

/// Spin iterations from MTL5_POOL_SPIN; `fallback` when unset or invalid.
inline unsigned resolve_pool_spin(unsigned fallback) {
    if (const char* e = std::getenv("MTL5_POOL_SPIN")) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(e, &end, 10);
        if (end != e && *end == '\0')
            return v > 0xFFFFFFFFul ? 0xFFFFFFFFu : static_cast<unsigned>(v);
    }
    return fallback;
}

} // namespace detail

class spin_pool {
public:
    /// Spin iterations before a waiting thread sleeps, when none is given.
    static constexpr unsigned default_spin = 4096;

    /// Construct a pool with `n` total logical workers (n-1 background threads +
    /// the caller); n < 1 is treated as 1 (serial). A waiting worker, and the
    /// caller in its join, spin `spin` iterations before they sleep.
    explicit spin_pool(unsigned n, unsigned spin = default_spin)
        : n_(n < 1 ? 1 : (n > max_pool_threads ? max_pool_threads : n)), spin_(spin) {
        workers_.reserve(n_ - 1);
        try {
            for (unsigned i = 1; i < n_; ++i)
                workers_.emplace_back([this, i] { worker_loop(i); });
        } catch (...) {
            shutdown();   // a joinable std::thread left behind would terminate
            throw;
        }
    }

    /// Process-wide spin_pool, sized on first use as thread_pool's is
    /// (configure(), else MTL5_NUM_THREADS), with its spin from MTL5_POOL_SPIN.
    static spin_pool& instance() {
#if defined(MTL5_SHARED_POOL)
        static spin_pool& pool = detail::shared_spin_pool_v1();
#else
        static spin_pool pool(detail::claim_pool_size(), detail::resolve_pool_spin(default_spin));
#endif
        return pool;
    }

    /// Total logical workers, including the calling thread (>= 1).
    unsigned size() const noexcept { return n_; }

    /// Spin iterations before a waiting thread sleeps.
    unsigned spin() const noexcept { return spin_; }

    /// Run task(tid) for tid in [0, count); thread_pool::run's contract. Takes
    /// no lock and does not allocate.
    template <typename Fn>
    void run(unsigned count, Fn&& fn) {
        if constexpr (std::is_function_v<std::remove_reference_t<Fn>>) {
            auto* fp = std::addressof(fn);
            run_impl(count, fp);
        } else {
            run_impl(count, fn);
        }
    }

    /// thread_pool::parallel_for, on this pool.
    template <typename Body>
    void parallel_for(std::size_t n, std::size_t grain, Body&& body) {
        detail::chunked_for(*this, n, grain, 0u, std::forward<Body>(body));
    }

    template <typename Body>
    void parallel_for(std::size_t n, std::size_t grain, unsigned max_threads, Body&& body) {
        detail::chunked_for(*this, n, grain, max_threads, std::forward<Body>(body));
    }

    /// thread_pool::parallel_reduce, on this pool. Allocates its partials.
    template <typename T, typename MapFn>
    T parallel_reduce(std::size_t n, std::size_t grain, MapFn&& map) {
        return detail::chunked_reduce<T>(*this, n, grain, 0u, std::forward<MapFn>(map));
    }

    template <typename T, typename MapFn>
    T parallel_reduce(std::size_t n, std::size_t grain, unsigned max_threads, MapFn&& map) {
        return detail::chunked_reduce<T>(*this, n, grain, max_threads, std::forward<MapFn>(map));
    }

    /// thread_pool::parallel_accumulate, on this pool. Allocates its partials.
    template <typename Acc, typename InitFn, typename MapFn, typename CombineFn>
    Acc parallel_accumulate(std::size_t n, std::size_t grain,
                            InitFn&& init, MapFn&& map, CombineFn&& combine) {
        return detail::chunked_accumulate<Acc>(*this, n, grain, 0u, std::forward<InitFn>(init),
                                               std::forward<MapFn>(map),
                                               std::forward<CombineFn>(combine));
    }

    template <typename Acc, typename InitFn, typename MapFn, typename CombineFn>
    Acc parallel_accumulate(std::size_t n, std::size_t grain, unsigned max_threads,
                            InitFn&& init, MapFn&& map, CombineFn&& combine) {
        return detail::chunked_accumulate<Acc>(*this, n, grain, max_threads,
                                               std::forward<InitFn>(init),
                                               std::forward<MapFn>(map),
                                               std::forward<CombineFn>(combine));
    }

    spin_pool(const spin_pool&) = delete;
    spin_pool& operator=(const spin_pool&) = delete;

    ~spin_pool() { shutdown(); }

private:
    // word_ is the generation in the high 48 bits and the region's team size
    // in the low 16, so a pool holds at most 65535 threads. A worker compares
    // the whole word with the one it last saw, so the generation must not
    // wrap while a worker is off a core: a team member preempted after its
    // countdown, while the caller runs smaller teams, would take a wrapped
    // word equal to the one it saw for no change, and the join would wait for
    // it forever. 2^48 regions is years even back to back.
    using word_type = std::uint64_t;
    static constexpr unsigned max_pool_threads = 0xFFFFu;
    static constexpr word_type team_mask = 0xFFFFu;
    static_assert(std::atomic<word_type>::is_always_lock_free,
                  "spin_pool's dispatch word must be a lock-free atomic");

    word_type next_word(unsigned team) const noexcept {
        const word_type gen = (word_.load(std::memory_order_relaxed) >> 16) + 1u;
        return (gen << 16) | (static_cast<word_type>(team) & team_mask);
    }

    /// Stop and join the workers started so far.
    void shutdown() noexcept {
        stop_.store(true, std::memory_order_relaxed);
        // The new generation publishes stop_ (release) and wakes every worker,
        // spinning or asleep; a team of 0 runs nothing.
        word_.store(next_word(0), std::memory_order_seq_cst);
        word_.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
    }

    template <typename F>
    void run_impl(unsigned count, F& fn) {
        if (count == 0) return;
        if (count > n_) count = n_;
        if (n_ <= 1 || count <= 1) {                 // serial fast path
            for (unsigned t = 0; t < count; ++t) fn(t);
            return;
        }

        bool idle = false;
        if (!busy_.compare_exchange_strong(idle, true, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            detail::region_scope in_region;           // no nesting: run serially
            for (unsigned t = 0; t < count; ++t) fn(t);
            return;
        }
        // Only the thread holding busy_ writes these, and no worker reads them
        // until the word below is published (release), or after its region's
        // join (each worker's countdown is a release, the caller's join an
        // acquire), so plain members suffice.
        ctx_ = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
        call_ = [](void* ctx, unsigned tid) { (*static_cast<F*>(ctx))(tid); };
        exc_claimed_.store(false, std::memory_order_relaxed);
        remaining_.store(count - 1, std::memory_order_relaxed);   // workers 1..count-1
        // seq_cst store, then seq_cst load of sleepers_: a worker that went to
        // sleep before this store is counted, and one that counts itself after
        // reads the new word before it sleeps (its own seq_cst pair).
        word_.store(next_word(count), std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) != 0) word_.notify_all();

        std::exception_ptr caller_exc;
        try {
            detail::region_scope in_region;
            fn(0);
        } catch (...) { caller_exc = std::current_exception(); }

        join();
        std::exception_ptr worker_exc = std::move(worker_exc_);
        worker_exc_ = nullptr;
        busy_.store(false, std::memory_order_release);

        if (caller_exc) std::rethrow_exception(caller_exc);
        if (worker_exc) std::rethrow_exception(worker_exc);
    }

    /// The caller's wait for remaining_ == 0: spin, then sleep on it.
    void join() noexcept {
        for (unsigned i = 0; i < spin_; ++i) {
            if (remaining_.load(std::memory_order_acquire) == 0) return;
            detail::spin_step(i);
        }
        // The same seq_cst pairing as dispatch, mirrored: the caller announces
        // itself, then reads remaining_; the last worker decrements, then reads
        // the announcement.
        caller_sleeping_.store(true, std::memory_order_seq_cst);
        for (;;) {
            const unsigned r = remaining_.load(std::memory_order_seq_cst);
            if (r == 0) break;
            remaining_.wait(r, std::memory_order_seq_cst);
        }
        caller_sleeping_.store(false, std::memory_order_relaxed);
    }

    /// A worker's wait for a word other than `seen`: spin, then sleep on it.
    word_type await_region(word_type seen) noexcept {
        for (unsigned i = 0; i < spin_; ++i) {
            const word_type w = word_.load(std::memory_order_acquire);
            if (w != seen) return w;
            detail::spin_step(i);
        }
        sleepers_.fetch_add(1, std::memory_order_seq_cst);
        word_type w;
        for (;;) {
            w = word_.load(std::memory_order_seq_cst);
            if (w != seen) break;
            word_.wait(seen, std::memory_order_seq_cst);
        }
        sleepers_.fetch_sub(1, std::memory_order_relaxed);
        return w;
    }

    void worker_loop(unsigned tid) {
        // A worker runs nothing but region tids, so its state is set for good.
        detail::this_region_thread().state = detail::pool_worker;
        // Workers start at word 0, before any region: a region published before
        // this worker reaches its first wait still differs from it.
        word_type seen = 0;
        for (;;) {
            seen = await_region(seen);
            if (stop_.load(std::memory_order_relaxed)) return;
            if (tid >= (seen & team_mask)) continue;   // not part of this region
            try { call_(ctx_, tid); }
            catch (...) {
                if (!exc_claimed_.exchange(true, std::memory_order_relaxed))
                    worker_exc_ = std::current_exception();
            }
            if (remaining_.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
                caller_sleeping_.load(std::memory_order_seq_cst))
                remaining_.notify_one();
        }
    }

    const unsigned n_;
    const unsigned spin_;
    std::vector<std::thread> workers_;

    // The word every worker reads, the countdown every worker writes, and the
    // rest each on a cache line of their own.
    alignas(64) std::atomic<word_type> word_{0};
    alignas(64) std::atomic<unsigned> remaining_{0};
    alignas(64) std::atomic<unsigned> sleepers_{0};   // workers asleep in word_.wait
    std::atomic<bool> caller_sleeping_{false};       // the caller is asleep in join()
    std::atomic<bool> stop_{false};
    alignas(64) std::atomic<bool> busy_{false};       // a region is in progress
    std::atomic<bool> exc_claimed_{false};            // a worker has stored worker_exc_
    void* ctx_ = nullptr;                             // the region's callable, by address
    void (*call_)(void*, unsigned) = nullptr;         // invokes *ctx_ for one tid
    std::exception_ptr worker_exc_;
};

#if defined(MTL5_POOL_SPIN)
inline namespace pool_select_spin {

/// The pool type MTL5's kernels run on: spin_pool under MTL5_POOL_SPIN.
using default_pool = spin_pool;

/// The process-wide pool that MTL5's kernels run on.
inline default_pool& pool() { return spin_pool::instance(); }

} // inline namespace pool_select_spin
#endif

} // inline namespace pool_shared / pool_local
} // namespace mtl::parallel
