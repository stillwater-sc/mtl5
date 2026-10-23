#pragma once
// MTL5 -- persistent thread pool for on-node parallel kernels (#221), public API
// since #630 (it was mtl::detail::thread_pool, which remains as an alias).
//
// A process-wide pool of long-lived worker threads, sized once: by configure()
// if that ran first, else from MTL5_NUM_THREADS (clamped to hardware
// concurrency; default 1 = serial, no workers, zero overhead). Replaces the
// per-call std::thread spawn/join that the blocked GEMM used, so a kernel that
// runs many small parallel regions (e.g. the GEMM (jc,pc) loops) pays a cheap
// condition-variable handoff instead of thread creation each time. MTL5's own
// kernels run on pool(), so code that runs its parallel work there too shares
// one set of workers with them instead of oversubscribing the cores.
//
// Contract of run(count, task):
//   * task(tid) runs once for each tid in [0, count); count is clamped to size().
//   * The CALLING thread runs tid 0; pool workers run 1..count-1.
//   * run() blocks until every tid has completed, and every task's writes
//     happen-before the caller's next action after run() returns.
//   * A region started while another is in progress -- nested inside a task, or
//     concurrently from another thread -- runs serially on its caller, all tids
//     in order. It never deadlocks and never oversubscribes.
//   * The first exception thrown by any tid is rethrown after all tids join.
//   * run() does not allocate. The task is held as a pointer to the caller's
//     callable plus a function pointer, never copied into a std::function, so
//     this holds whatever the callable captures.
// The static tid->work partition is the caller's, so a kernel that wants
// bit-identical results across thread counts (as the GEMM does) keeps that
// guarantee.
//
// No OpenMP/TBB -- just the C++ standard concurrency runtime, for portability.
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace mtl::parallel {

namespace detail {

/// Clamp a requested thread count to [1, hardware concurrency].
inline unsigned clamp_pool_threads(unsigned long v) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (v < 1) return 1u;
    return static_cast<unsigned>(v < hw ? v : hw);
}

/// Resolve the pool size from MTL5_NUM_THREADS (clamped to hardware
/// concurrency); 1 when unset/invalid.
inline unsigned resolve_pool_threads() {
    if (const char* e = std::getenv("MTL5_NUM_THREADS")) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(e, &end, 10);
        if (end != e && v >= 1)
            return clamp_pool_threads(v);
    }
    return 1u;
}

/// What configure() asked for, and whether the pool already exists. One mutex
/// orders configure() against the pool's construction, so a configure() that
/// returns true is the size the pool gets.
struct pool_config {
    std::mutex mtx;
    unsigned requested = 0;   // 0: not configured, use MTL5_NUM_THREADS
    bool created = false;
};

inline pool_config& config() {
    static pool_config c;
    return c;
}

inline unsigned claim_pool_size() {
    pool_config& c = config();
    std::lock_guard<std::mutex> lk(c.mtx);
    c.created = true;
    return c.requested != 0 ? c.requested : resolve_pool_threads();
}

} // namespace detail

class thread_pool {
public:
    /// Construct a pool with `n` total logical workers (n-1 background threads +
    /// the caller). n < 1 is treated as 1 (serial).
    explicit thread_pool(unsigned n) : n_(n < 1 ? 1 : n) {
        workers_.reserve(n_ > 0 ? n_ - 1 : 0);
        for (unsigned i = 1; i < n_; ++i)
            workers_.emplace_back([this, i] { worker_loop(i); });
    }

    /// Process-wide pool, sized on first use (see configure()).
    static thread_pool& instance() {
        static thread_pool pool(detail::claim_pool_size());
        return pool;
    }

    /// Total logical workers, including the calling thread (>= 1).
    unsigned size() const noexcept { return n_; }

    /// Run task(tid) for tid in [0, count); see the contract at the top of this
    /// header. Does not allocate.
    template <typename Fn>
    void run(unsigned count, Fn&& fn) {
        if (count == 0) return;
        if (count > n_) count = n_;
        if (n_ <= 1 || count <= 1) {                 // serial fast path
            for (unsigned t = 0; t < count; ++t) fn(t);
            return;
        }

        std::unique_lock<std::mutex> lk(mtx_);
        if (busy_) {                                  // no nesting: run serially
            lk.unlock();
            for (unsigned t = 0; t < count; ++t) fn(t);
            return;
        }
        busy_ = true;
        worker_exc_ = nullptr;
        // The task is a pointer to fn plus a trampoline, not a copy of fn: run()
        // does not return until every tid has completed, so fn outlives every
        // call, and nothing is allocated however large fn's captures are.
        using F = std::remove_reference_t<Fn>;
        ctx_ = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
        call_ = [](void* ctx, unsigned tid) { (*static_cast<F*>(ctx))(tid); };
        active_ = count;
        remaining_ = count - 1;                       // workers 1..count-1
        ++generation_;
        lk.unlock();
        wake_.notify_all();

        std::exception_ptr caller_exc;
        try { fn(0); } catch (...) { caller_exc = std::current_exception(); }

        lk.lock();
        done_.wait(lk, [&] { return remaining_ == 0; });
        ctx_ = nullptr;
        call_ = nullptr;
        busy_ = false;
        std::exception_ptr worker_exc = worker_exc_;
        lk.unlock();

        if (caller_exc) std::rethrow_exception(caller_exc);
        if (worker_exc) std::rethrow_exception(worker_exc);
    }

    /// Partition [0, n) into contiguous chunks across the pool and run
    /// body(begin, end) per chunk. Runs serially (a single body(0, n)) when the
    /// pool is serial or n is too small to amortize the handoff; otherwise caps
    /// the team so each chunk holds >= `grain` iteration units. Contiguous,
    /// deterministic chunking -> element-wise callers stay bit-identical across
    /// thread counts (each output element is produced by exactly one chunk).
    /// Does not allocate.
    template <typename Body>
    void parallel_for(std::size_t n, std::size_t grain, Body&& body) {
        parallel_for(n, grain, 0u, std::forward<Body>(body));
    }

    /// parallel_for with the team capped at `max_threads` (0: no cap, the whole
    /// pool), for a caller that runs at a thread count of its own below size().
    template <typename Body>
    void parallel_for(std::size_t n, std::size_t grain, unsigned max_threads, Body&& body) {
        if (n == 0) return;
        const unsigned team = team_for(n, grain, max_threads);
        if (team <= 1) { body(std::size_t{0}, n); return; }
        const std::size_t chunk = (n + team - 1) / team;
        run(team, [&](unsigned tid) {
            const std::size_t b = static_cast<std::size_t>(tid) * chunk;
            if (b >= n) return;
            body(b, (n < b + chunk) ? n : b + chunk);
        });
    }

    /// Reduce over [0, n): map(begin, end) returns a partial T; partials are
    /// combined with operator+ in chunk (tid) order. Runs serially (a single
    /// map(0, n)) when the pool is serial or n is below the grain.
    ///
    /// Deterministic for a fixed thread count, but NOT bit-identical to the
    /// serial reduction -- chunking changes the summation grouping/associativity.
    /// Callers that need serial-exact results must not rely on this.
    ///
    /// Allocates: a threaded call holds its partials in a std::vector<T>.
    template <typename T, typename MapFn>
    T parallel_reduce(std::size_t n, std::size_t grain, MapFn&& map) {
        return parallel_reduce<T>(n, grain, 0u, std::forward<MapFn>(map));
    }

    /// parallel_reduce with the team capped at `max_threads` (0: no cap). The
    /// result depends on the team, so it is deterministic for a fixed cap.
    template <typename T, typename MapFn>
    T parallel_reduce(std::size_t n, std::size_t grain, unsigned max_threads, MapFn&& map) {
        if (n == 0) return T{};
        const unsigned team = team_for(n, grain, max_threads);
        if (team <= 1) return map(std::size_t{0}, n);
        const std::size_t chunk = (n + team - 1) / team;
        std::vector<T> partials(team, T{});
        run(team, [&](unsigned tid) {
            const std::size_t b = static_cast<std::size_t>(tid) * chunk;
            if (b >= n) return;
            partials[tid] = map(b, (n < b + chunk) ? n : b + chunk);
        });
        T acc = T{};
        // Combine with operator+ only (not +=), honoring the documented contract
        // so plus-only result types work. Deterministic tid order.
        for (unsigned t = 0; t < team; ++t) acc = acc + partials[t];
        return acc;
    }

    /// Reduce over [0, n) into accumulator partials that are combined by the
    /// caller's `combine`, not by `+` on a result type (#582).
    ///
    /// `init()` returns an empty accumulator; `map(acc, begin, end)` adds one
    /// chunk's terms to it; `combine(into, from)` folds a partial into another.
    /// The chunking is parallel_for's: contiguous, deterministic, capped so each
    /// chunk holds >= `grain` units. Partials are folded in tid order, starting
    /// from tid 0's.
    ///
    /// Serially (one thread, or n too small to split) this is exactly
    /// `acc = init(); map(acc, 0, n);`, so a caller's single-threaded result is
    /// unchanged. Threaded, the result equals the serial one whenever `combine`
    /// and `map` are exact -- a super-accumulator such as a quire -- whatever
    /// the thread count. For a rounding accumulator it is deterministic for a
    /// fixed thread count but regrouped, as parallel_reduce is.
    ///
    /// Allocates: a threaded call holds its partials in a std::vector<Acc>.
    template <typename Acc, typename InitFn, typename MapFn, typename CombineFn>
    Acc parallel_accumulate(std::size_t n, std::size_t grain,
                            InitFn&& init, MapFn&& map, CombineFn&& combine) {
        return parallel_accumulate<Acc>(n, grain, 0u, std::forward<InitFn>(init),
                                        std::forward<MapFn>(map),
                                        std::forward<CombineFn>(combine));
    }

    /// parallel_accumulate with the team capped at `max_threads` (0: no cap).
    template <typename Acc, typename InitFn, typename MapFn, typename CombineFn>
    Acc parallel_accumulate(std::size_t n, std::size_t grain, unsigned max_threads,
                            InitFn&& init, MapFn&& map, CombineFn&& combine) {
        const unsigned team = team_for(n, grain, max_threads);
        if (team <= 1) {
            Acc acc = init();
            map(acc, std::size_t{0}, n);
            return acc;
        }
        const std::size_t chunk = (n + team - 1) / team;
        std::vector<Acc> partials;
        partials.reserve(team);
        for (unsigned t = 0; t < team; ++t) partials.push_back(init());
        run(team, [&](unsigned tid) {
            const std::size_t b = static_cast<std::size_t>(tid) * chunk;
            if (b >= n) return;
            map(partials[tid], b, (n < b + chunk) ? n : b + chunk);
        });
        for (unsigned t = 1; t < team; ++t) combine(partials[0], partials[t]);
        return std::move(partials[0]);
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    ~thread_pool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
            ++generation_;
        }
        wake_.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
    }

private:
    /// Team for a chunked loop over n units: the pool, or max_threads if that is
    /// smaller and nonzero, cut so each chunk holds >= grain units; 1 (serial)
    /// when the pool is serial or n < 2 * grain.
    unsigned team_for(std::size_t n, std::size_t grain, unsigned max_threads) const noexcept {
        unsigned team = (max_threads != 0 && max_threads < n_) ? max_threads : n_;
        if (team <= 1 || grain == 0 || n < grain * 2) return 1;
        const std::size_t max_chunks = n / grain;
        if (static_cast<std::size_t>(team) > max_chunks)
            team = static_cast<unsigned>(max_chunks);
        return team;
    }

    void worker_loop(unsigned tid) {
        std::unique_lock<std::mutex> lk(mtx_);
        // Workers are created at generation_ == 0 (which only ever increments),
        // so start "behind" 0: a region dispatched before this worker reached the
        // wait (a startup race) is still observed as generation_ != seen.
        std::uint64_t seen = 0;
        for (;;) {
            wake_.wait(lk, [&] { return generation_ != seen || stop_; });
            if (stop_) return;
            seen = generation_;
            if (tid < active_ && call_) {
                void* ctx = ctx_;             // the caller's fn, by address
                void (*call)(void*, unsigned) = call_;
                lk.unlock();
                try { call(ctx, tid); }
                catch (...) {
                    lk.lock();
                    if (!worker_exc_) worker_exc_ = std::current_exception();
                    lk.unlock();
                }
                lk.lock();
                if (--remaining_ == 0) { lk.unlock(); done_.notify_one(); lk.lock(); }
            }
            // tid >= active_: not part of this region; wait for the next one.
        }
    }

    const unsigned n_;
    std::vector<std::thread> workers_;

    std::mutex mtx_;
    std::condition_variable wake_;   // workers wait for a new region
    std::condition_variable done_;   // caller waits for the region to finish
    void* ctx_ = nullptr;            // the region's callable, by address
    void (*call_)(void*, unsigned) = nullptr;  // invokes *ctx_ for one tid
    unsigned active_ = 0;            // tids [0, active_) participate this region
    unsigned remaining_ = 0;        // worker tids still running this region
    std::uint64_t generation_ = 0;  // bumped to start a region (or to stop)
    bool busy_ = false;             // a region is in progress
    bool stop_ = false;
    std::exception_ptr worker_exc_;
};

/// The process-wide pool that MTL5's kernels run on.
inline thread_pool& pool() { return thread_pool::instance(); }

/// Size the process-wide pool from code, instead of MTL5_NUM_THREADS.
///
/// Takes effect only before the pool's first use: the first pool() call, or
/// the first MTL5 kernel that runs threaded. `threads` is clamped to the
/// hardware concurrency, as MTL5_NUM_THREADS is, and overrides it. Returns
/// true if the pool will have that size; false, changing nothing, if the pool
/// already exists or `threads` is 0. pool().size() is the size actually in use.
inline bool configure(unsigned threads) {
    if (threads == 0) return false;
    detail::pool_config& c = detail::config();
    std::lock_guard<std::mutex> lk(c.mtx);
    if (c.created) return false;
    c.requested = detail::clamp_pool_threads(threads);
    return true;
}

} // namespace mtl::parallel
