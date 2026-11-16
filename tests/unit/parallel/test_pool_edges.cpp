// thread_pool and spin_pool edge cases of issue 876 (QA sweep #783):
//
//   - MTL5_POOL_SPIN and MTL5_NUM_THREADS parsing: signed, empty, partly
//     numeric and out-of-range values are invalid (strtoul made "-1" the
//     largest spin);
//   - a pool whose workers can't all be created throws, and does not hang
//     (Linux, in a child process with a lowered RLIMIT_AS);
//   - the chunked loops at n = 0, grain = 0, max_threads = 1 and around
//     n = 2 grain, team_for directly, and the n = 0 contract of
//     parallel_reduce and parallel_accumulate;
//   - spin_pool::instance() and spin().
#include <catch2/catch_test_macros.hpp>

#include <mtl/parallel/thread_pool.hpp>
#include <mtl/parallel/spin_pool.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#  include <csignal>
#  include <cstdio>
#  include <sys/resource.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

using namespace mtl::parallel;

namespace {

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

/// Restores an environment variable's value, or its absence, on scope exit,
/// so a test leaves the process as it found it (MTL5_NUM_THREADS=4 set for
/// the whole run must still be there for a later spin_pool::instance()).
class env_guard {
public:
    explicit env_guard(const char* name) : name_(name) {
        if (const char* v = std::getenv(name)) {
            had_ = true;
            value_ = v;
        }
    }
    ~env_guard() {
        if (had_) set_env(name_, value_.c_str());
        else unset_env(name_);
    }
    env_guard(const env_guard&) = delete;
    env_guard& operator=(const env_guard&) = delete;

private:
    const char* name_;
    bool had_ = false;
    std::string value_;
};

} // namespace

TEST_CASE("pool env parsing: MTL5_POOL_SPIN", "[parallel][876]") {
    const env_guard keep("MTL5_POOL_SPIN");
    struct cases { const char* value; unsigned expected; };
    const unsigned fallback = 77;
    for (const auto& c : {cases{"0", 0u}, cases{"4096", 4096u}, cases{"4294967295", 4294967295u},
                          cases{"-1", fallback}, cases{"", fallback}, cases{"abc", fallback},
                          cases{"12x", fallback}, cases{" 12", fallback}, cases{"+12", fallback},
                          cases{"4294967296", fallback}, cases{"99999999999", fallback}}) {
        INFO("MTL5_POOL_SPIN=\"" << c.value << "\"");
        set_env("MTL5_POOL_SPIN", c.value);
        CHECK(detail::resolve_pool_spin(fallback) == c.expected);
    }
    unset_env("MTL5_POOL_SPIN");
    CHECK(detail::resolve_pool_spin(fallback) == fallback);
}

TEST_CASE("pool env parsing: MTL5_NUM_THREADS", "[parallel][876]") {
    const env_guard keep("MTL5_NUM_THREADS");
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    set_env("MTL5_NUM_THREADS", "-4");   // was clamped to the hardware threads
    CHECK(detail::resolve_pool_threads() == 1u);
    set_env("MTL5_NUM_THREADS", "2x");
    CHECK(detail::resolve_pool_threads() == 1u);
    set_env("MTL5_NUM_THREADS", "0");
    CHECK(detail::resolve_pool_threads() == 1u);
    set_env("MTL5_NUM_THREADS", "99999999999");
    CHECK(detail::resolve_pool_threads() == 1u);
    set_env("MTL5_NUM_THREADS", "2");
    CHECK(detail::resolve_pool_threads() == (hw < 2 ? hw : 2u));
    set_env("MTL5_NUM_THREADS", "100000");   // valid, clamped to the hardware
    CHECK(detail::resolve_pool_threads() == hw);
    unset_env("MTL5_NUM_THREADS");
    CHECK(detail::resolve_pool_threads() == 1u);
}

#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
#  if defined(__has_feature)
#    if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#      define MTL5_POOL_EDGES_SANITIZED 1
#    endif
#  endif
#  if !defined(MTL5_POOL_EDGES_SANITIZED)
namespace {

/// Run `make` in a child process whose address space is capped at its
/// current size plus `headroom` bytes, so only a few worker stacks fit.
/// Returns the child's exit code: 0 the constructor threw std::system_error,
/// 1 it threw something else, 2 it did not throw, 3 setrlimit failed, -1 the
/// child had to be killed after 30 s (it hung), -2 it died on a signal, -3
/// fork() failed (no child: waitpid(-1) and kill(-1) must not be reached).
template <typename Make>
int construct_under_rlimit(Make make, unsigned long headroom) {
    const pid_t pid = fork();
    if (pid < 0) return -3;
    if (pid == 0) {
        long pages = 0;
        if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
            if (std::fscanf(f, "%ld", &pages) != 1) pages = 0;
            std::fclose(f);
        }
        const rlim_t cap = static_cast<rlim_t>(pages) * static_cast<rlim_t>(sysconf(_SC_PAGESIZE)) + headroom;
        const rlimit lim{cap, cap};
        if (setrlimit(RLIMIT_AS, &lim) != 0) _exit(3);
        try {
            make();
        } catch (const std::system_error&) {
            _exit(0);
        } catch (...) {
            _exit(1);
        }
        _exit(2);
    }
    for (int waited_ms = 0; waited_ms < 30000; waited_ms += 10) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid)
            return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);
    return -1;
}

} // namespace

TEST_CASE("a pool whose workers can't all start throws instead of hanging", "[parallel][876]") {
    // 2000 workers with 8 MB default stacks need 16 GB of address space; the
    // child has 256 MB more than it uses. Before the fix thread_pool's
    // constructor unwound with live workers waiting on a destroyed condition
    // variable, and pthread_cond_destroy blocked.
    const unsigned long headroom = 256ul << 20;
    CHECK(construct_under_rlimit([] { thread_pool p(2000); }, headroom) == 0);
    CHECK(construct_under_rlimit([] { spin_pool p(2000); }, headroom) == 0);
}
#  endif
#endif

TEST_CASE("team_for and the chunked loops at their edges", "[parallel][876]") {
    // team_for(n, grain, max_threads, pool_size)
    CHECK(detail::team_for(0, 4, 0, 4) == 1);
    CHECK(detail::team_for(100, 0, 0, 4) == 1);    // grain 0: serial
    CHECK(detail::team_for(100, 4, 1, 4) == 1);    // max_threads 1: serial
    CHECK(detail::team_for(7, 4, 0, 4) == 1);      // n = 2 grain - 1: serial
    CHECK(detail::team_for(8, 4, 0, 4) == 2);      // n = 2 grain: two chunks
    CHECK(detail::team_for(1000, 4, 3, 8) == 3);   // max_threads below the pool
    CHECK(detail::team_for(1000, 4, 0, 1) == 1);   // serial pool

    thread_pool pool(4);
    auto chunks_of = [&](std::size_t n, std::size_t grain, unsigned max_threads) {
        std::mutex m;
        std::vector<std::pair<std::size_t, std::size_t>> seen;
        pool.parallel_for(n, grain, max_threads, [&](std::size_t b, std::size_t e) {
            std::lock_guard<std::mutex> lk(m);
            seen.emplace_back(b, e);
        });
        return seen.size();
    };
    CHECK(chunks_of(0, 4, 0) == 0);     // n = 0: the body is not called
    CHECK(chunks_of(100, 0, 0) == 1);
    CHECK(chunks_of(100, 4, 1) == 1);
    CHECK(chunks_of(7, 4, 0) == 1);
    CHECK(chunks_of(8, 4, 0) == 2);

    // n = 0: parallel_reduce returns T{} without calling map, while
    // parallel_accumulate calls init() once and map(acc, 0, 0) once (Acc need
    // not be default-constructible, so init() makes the result).
    int maps = 0;
    CHECK(pool.parallel_reduce<int>(0, 4, [&](std::size_t, std::size_t) { ++maps; return 5; }) == 0);
    CHECK(maps == 0);
    int inits = 0;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    const int acc = pool.parallel_accumulate<int>(
        0, 4, [&] { ++inits; return 42; },
        [&](int&, std::size_t b, std::size_t e) { ranges.emplace_back(b, e); },
        [](int& into, const int& from) { into += from; });
    CHECK(acc == 42);
    CHECK(inits == 1);
    REQUIRE(ranges.size() == 1);
    CHECK(ranges[0] == std::pair<std::size_t, std::size_t>{0, 0});

    // The padded partials (issue 876): a threaded accumulate still sums
    // exactly, with an Acc that owns memory.
    const auto sums = pool.parallel_accumulate<std::vector<long>>(
        1000, 8, [] { return std::vector<long>(2, 0); },
        [](std::vector<long>& a, std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) { a[0] += static_cast<long>(i); a[1] += 1; }
        },
        [](std::vector<long>& into, const std::vector<long>& from) { into[0] += from[0]; into[1] += from[1]; });
    CHECK(sums[0] == 999L * 1000L / 2);
    CHECK(sums[1] == 1000);
}

TEST_CASE("parallel_accumulate with an over-aligned accumulator", "[parallel][876]") {
    // The padded wrapper keeps Acc's own alignment when it exceeds 128
    // bytes; alignas(128) alone made the wrapper ill-formed for this Acc.
    struct alignas(256) wide {
        long sum = 0;
    };
    thread_pool pool(4);
    const wide total = pool.parallel_accumulate<wide>(
        1000, 8, [] { return wide{}; },
        [](wide& a, std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) a.sum += static_cast<long>(i);
        },
        [](wide& into, const wide& from) { into.sum += from.sum; });
    CHECK(total.sum == 999L * 1000L / 2);
}

TEST_CASE("spin_pool::instance() and spin()", "[parallel][876]") {
    spin_pool& p = spin_pool::instance();
    CHECK(&p == &spin_pool::instance());
    CHECK(p.size() >= 1u);
    std::atomic<unsigned> ran{0};
    p.run(p.size(), [&](unsigned) { ran.fetch_add(1); });
    CHECK(ran.load() == p.size());
    spin_pool q(2, 123u);
    CHECK(q.spin() == 123u);
    spin_pool r(2);
    CHECK(r.spin() == spin_pool::default_spin);
}
