#pragma once
// MTL5 -- persistent thread pool for on-node parallel kernels (#221).
//
// A process-wide pool of long-lived worker threads, sized once from
// MTL5_NUM_THREADS (clamped to hardware concurrency; default 1 = serial, no
// workers, zero overhead). Replaces the per-call std::thread spawn/join that the
// blocked GEMM used, so a kernel that runs many small parallel regions (e.g. the
// GEMM (jc,pc) loops) pays a cheap condition-variable handoff instead of thread
// creation each time.
//
// Model: `run(count, task)` executes task(tid) for tid in [0, count) with the
// CALLING thread running tid 0 and the pool workers running 1..count-1, and
// blocks until all complete. count must be <= size(). The static tid->work
// partition is the caller's responsibility, so a kernel that wants bit-identical
// results across thread counts (as the GEMM does) keeps that guarantee.
//
// No OpenMP/TBB -- just the C++ standard concurrency runtime, for portability.
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mtl::detail {

/// Resolve the pool size from MTL5_NUM_THREADS (clamped to hardware
/// concurrency); 1 when unset/invalid. Read once.
inline unsigned resolve_pool_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (const char* e = std::getenv("MTL5_NUM_THREADS")) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(e, &end, 10);
        if (end != e && v >= 1)
            return static_cast<unsigned>(v < hw ? v : hw);
    }
    return 1u;
}

class thread_pool {
public:
    /// Construct a pool with `n` total logical workers (n-1 background threads +
    /// the caller). n < 1 is treated as 1 (serial).
    explicit thread_pool(unsigned n) : n_(n < 1 ? 1 : n) {
        workers_.reserve(n_ > 0 ? n_ - 1 : 0);
        for (unsigned i = 1; i < n_; ++i)
            workers_.emplace_back([this, i] { worker_loop(i); });
    }

    /// Process-wide pool, sized from MTL5_NUM_THREADS on first use.
    static thread_pool& instance() {
        static thread_pool pool(resolve_pool_threads());
        return pool;
    }

    /// Total logical workers, including the calling thread (>= 1).
    unsigned size() const noexcept { return n_; }

    /// Run task(tid) for tid in [0, count). The caller runs tid 0; workers run
    /// 1..count-1. Blocks until all finish. `count` is clamped to size(). If the
    /// pool is already inside a parallel region (nested or concurrent use), the
    /// region runs serially on the caller to avoid oversubscription/deadlock.
    /// The first exception thrown by any tid is rethrown after all tids join.
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
        // task_ captures fn by reference; run() does not return until every tid
        // has completed, so the reference stays valid for the workers.
        task_ = [&fn](unsigned tid) { fn(tid); };
        active_ = count;
        remaining_ = count - 1;                       // workers 1..count-1
        ++generation_;
        lk.unlock();
        wake_.notify_all();

        std::exception_ptr caller_exc;
        try { fn(0); } catch (...) { caller_exc = std::current_exception(); }

        lk.lock();
        done_.wait(lk, [&] { return remaining_ == 0; });
        task_ = nullptr;
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
    template <typename Body>
    void parallel_for(std::size_t n, std::size_t grain, Body&& body) {
        if (n == 0) return;
        unsigned team = n_;
        if (team <= 1 || grain == 0 || n < grain * 2) { body(std::size_t{0}, n); return; }
        const std::size_t max_chunks = n / grain;
        if (static_cast<std::size_t>(team) > max_chunks)
            team = static_cast<unsigned>(max_chunks);
        if (team <= 1) { body(std::size_t{0}, n); return; }
        const std::size_t chunk = (n + team - 1) / team;
        run(team, [&](unsigned tid) {
            const std::size_t b = static_cast<std::size_t>(tid) * chunk;
            if (b >= n) return;
            body(b, (n < b + chunk) ? n : b + chunk);
        });
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
            if (tid < active_ && task_) {
                auto t = task_;               // share the captured fn reference
                lk.unlock();
                try { t(tid); }
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
    std::function<void(unsigned)> task_;
    unsigned active_ = 0;            // tids [0, active_) participate this region
    unsigned remaining_ = 0;        // worker tids still running this region
    std::uint64_t generation_ = 0;  // bumped to start a region (or to stop)
    bool busy_ = false;             // a region is in progress
    bool stop_ = false;
    std::exception_ptr worker_exc_;
};

} // namespace mtl::detail
