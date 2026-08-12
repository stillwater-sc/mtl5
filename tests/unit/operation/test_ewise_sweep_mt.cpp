// Threaded element-wise sweeps (#297 batch 10). The dense vector/matrix
// expression-template assignments (y = a + b, C += A + B, ...) evaluate one
// output element per index with no cross-element dependency, so
// detail::parallel_ewise distributes them over contiguous chunks. Each output
// element is produced by exactly one chunk with the identical per-element
// computation, so the parallel result is BIT-IDENTICAL to the serial loop: we
// assert exact equality (==) against the inline element-wise math, on sizes
// large enough to actually split across the pool.
//
// Sets MTL5_NUM_THREADS before the pool's first use (the only in-process way to
// exercise the threading). Run under TSan (-DMTL5_SANITIZE=thread) for races on
// the disjoint output writes.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <thread>
#include <unordered_set>
#include <vector>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/detail/ewise.hpp>
#include <mtl/detail/thread_pool.hpp>

using namespace mtl;

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

double vv(std::size_t i) { return 0.5 + std::sin(0.7 * static_cast<double>(i)); }
double ww(std::size_t i) { return -0.25 + std::cos(0.31 * static_cast<double>(i)); }

} // namespace

/// Minimal matrix expression that records which thread evaluated each element, so
/// a test can observe how a dense2D assignment distributed the sweep (#313).
/// operator() is const and writes one distinct slot per (r,c) -- no race.
/// At namespace scope (not unnamed) so the is_expression specialization below
/// names it unambiguously.
struct probe_expr {
    using value_type = double;
    using size_type  = std::size_t;

    std::size_t R, C;
    std::vector<std::size_t>* tid;   // R*C slots, hashed thread id per element

    size_type num_rows() const { return R; }
    size_type num_cols() const { return C; }
    size_type size()     const { return R * C; }

    value_type operator()(size_type r, size_type c) const {
        (*tid)[r * C + c] = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return static_cast<double>(r * C + c);
    }
};

namespace mtl::traits {
template <> struct is_expression<::probe_expr> : std::true_type {};
} // namespace mtl::traits

TEST_CASE("ewise vector sweep == elementwise math, bit-exact",
          "[operation][ewise][vector][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("pool < 2 workers: the parallel sweep is not exercised (set MTL5_NUM_THREADS)");

    const std::size_t n = 300000;   // > grain*2 (65536*2) -> actually splits
    vec::dense_vector<double> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) { a(i) = vv(i); b(i) = ww(i); }

    SECTION("operator= from expression") {
        vec::dense_vector<double> c(n);
        c = a + b;
        for (std::size_t i = 0; i < n; ++i) REQUIRE(c(i) == a(i) + b(i));
    }
    SECTION("construct from expression") {
        vec::dense_vector<double> c = a - b;
        REQUIRE(c.size() == n);
        for (std::size_t i = 0; i < n; ++i) REQUIRE(c(i) == a(i) - b(i));
    }
    SECTION("operator+= / operator-= from expression") {
        vec::dense_vector<double> c(n);
        for (std::size_t i = 0; i < n; ++i) c(i) = vv(i) * 2.0;
        vec::dense_vector<double> ref(n);
        for (std::size_t i = 0; i < n; ++i) ref(i) = c(i);

        c += a + b;
        for (std::size_t i = 0; i < n; ++i) REQUIRE(c(i) == ref(i) + (a(i) + b(i)));
        c -= a - b;
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(c(i) == (ref(i) + (a(i) + b(i))) - (a(i) - b(i)));
    }
}

TEST_CASE("ewise matrix sweep == elementwise math, bit-exact",
          "[operation][ewise][matrix][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("pool < 2 workers: the parallel sweep is not exercised (set MTL5_NUM_THREADS)");

    const std::size_t R = 4000, C = 64;   // rows > grain*2 (grain = 65536/C) -> splits
    mat::dense2D<double> A(R, C), B(R, C);
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < C; ++c) { A(r, c) = vv(r * C + c); B(r, c) = ww(r * C + c); }

    SECTION("operator= from expression") {
        mat::dense2D<double> D(R, C);
        D = A + B;
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) REQUIRE(D(r, c) == A(r, c) + B(r, c));
    }
    SECTION("construct from expression") {
        mat::dense2D<double> D = A - B;
        REQUIRE(D.num_rows() == R); REQUIRE(D.num_cols() == C);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) REQUIRE(D(r, c) == A(r, c) - B(r, c));
    }
    SECTION("operator+= / operator-= from expression") {
        mat::dense2D<double> D(R, C);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) D(r, c) = vv(r * C + c) * 2.0;
        mat::dense2D<double> ref = D;   // copy (default copy, serial)

        D += A + B;
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                REQUIRE(D(r, c) == ref(r, c) + (A(r, c) + B(r, c)));
    }
}

// -- flattened 2D sweep (#313) ----------------------------------------------
// The matrix sweep flattens (r,c) to a linear element index before chunking, so
// the split follows the ELEMENT count, not the row count. A row-per-unit
// decomposition leaves a 1 x N expression with a single work unit -- forever
// serial no matter how much work the row holds.

TEST_CASE("flattened 2D sweep covers the index space exactly once and splits",
          "[operation][ewise][matrix][threading][mt]") {
    const auto pool = mtl::detail::thread_pool::instance().size();
    if (pool < 2)
        WARN("pool < 2 workers: the parallel sweep is not exercised (set MTL5_NUM_THREADS)");

    // 1 x 300000: one row, > grain*2 (65536*2) elements -> splits when flattened.
    const std::size_t R = 1, C = 300000;
    std::vector<int>         visits(R * C, 0);          // each element written by one chunk
    std::vector<std::size_t> tid(R * C, 0);

    mtl::detail::parallel_ewise_2d(R, C, 1, [&](std::size_t r, std::size_t c) {
        visits[r * C + c] += 1;
        tid[r * C + c] = std::hash<std::thread::id>{}(std::this_thread::get_id());
    });

    // Every (r,c) produced exactly once: a wrong flattening would double-visit one
    // element and miss another, which this catches either way.
    for (std::size_t i = 0; i < R * C; ++i) REQUIRE(visits[i] == 1);

    // ... and the work actually reached more than one thread.
    std::unordered_set<std::size_t> threads(tid.begin(), tid.end());
    if (pool >= 2) REQUIRE(threads.size() > 1);

    // The row-major decomposition maps the linear index back correctly.
    std::vector<std::size_t> rr(R * C, 0), cc(R * C, 0);
    mtl::detail::parallel_ewise_2d(R, C, 1, [&](std::size_t r, std::size_t c) {
        rr[r * C + c] = r;
        cc[r * C + c] = c;
    });
    for (std::size_t i = 0; i < R * C; ++i) {
        REQUIRE(rr[i] == i / C);
        REQUIRE(cc[i] == i % C);
    }
}

TEST_CASE("a wide/short matrix assignment splits across the pool",
          "[operation][ewise][matrix][threading][mt]") {
    const auto pool = mtl::detail::thread_pool::instance().size();
    if (pool < 2) {
        WARN("pool < 2 workers: the parallel sweep is not exercised (set MTL5_NUM_THREADS)");
        return;
    }

    // The acceptance criterion of #313, observed through the real assignment path:
    // a 1 x 300000 expression has ONE row, so a row-per-work-unit sweep can never
    // split it. Flattened, its 300000 elements chunk across the pool.
    const std::size_t R = 1, C = 300000;
    std::vector<std::size_t> tid(R * C, 0);
    mat::dense2D<double> D(R, C);
    D = probe_expr{R, C, &tid};

    std::unordered_set<std::size_t> threads(tid.begin(), tid.end());
    REQUIRE(threads.size() > 1);

    // Every element still evaluated, exactly where it belongs.
    for (std::size_t c = 0; c < C; ++c) REQUIRE(D(0, c) == static_cast<double>(c));

    // A tall 300000 x 1 expression splits too -- the flattening did not trade one
    // shape for the other.
    std::vector<std::size_t> tid_tall(C * R, 0);
    mat::dense2D<double> T(C, R);
    T = probe_expr{C, R, &tid_tall};
    std::unordered_set<std::size_t> threads_tall(tid_tall.begin(), tid_tall.end());
    REQUIRE(threads_tall.size() > 1);
}

TEST_CASE("wide/short matrix sweep == elementwise math, bit-exact",
          "[operation][ewise][matrix][threading][mt]") {
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("pool < 2 workers: the parallel sweep is not exercised (set MTL5_NUM_THREADS)");

    // Shapes with too few rows to split row-wise, but plenty of total elements.
    auto check = [](std::size_t R, std::size_t C) {
        mat::dense2D<double> A(R, C), B(R, C);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) {
                A(r, c) = vv(r * C + c);
                B(r, c) = ww(r * C + c);
            }

        mat::dense2D<double> D(R, C);
        D = A + B;                                  // operator=
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) REQUIRE(D(r, c) == A(r, c) + B(r, c));

        mat::dense2D<double> E = A - B;             // construct
        REQUIRE(E.num_rows() == R); REQUIRE(E.num_cols() == C);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) REQUIRE(E(r, c) == A(r, c) - B(r, c));

        E += A + B;                                 // operator+=
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                REQUIRE(E(r, c) == (A(r, c) - B(r, c)) + (A(r, c) + B(r, c)));

        E -= A - B;                                 // operator-=
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                REQUIRE(E(r, c) == ((A(r, c) - B(r, c)) + (A(r, c) + B(r, c)))
                                   - (A(r, c) - B(r, c)));
    };

    SECTION("1 x 300000") { check(1, 300000); }
    SECTION("2 x 150000") { check(2, 150000); }
}

TEST_CASE("ewise sweep boundary cases", "[operation][ewise][threading][mt][edge]") {
    // Empty and tiny (below the split threshold -> serial body) must still work.
    {
        vec::dense_vector<double> a(0), b(0), c;
        c = a + b;
        REQUIRE(c.size() == 0);
    }
    {
        vec::dense_vector<double> a(3), b(3);
        for (std::size_t i = 0; i < 3; ++i) { a(i) = double(i); b(i) = double(2 * i); }
        vec::dense_vector<double> c = a + b;
        for (std::size_t i = 0; i < 3; ++i) REQUIRE(c(i) == a(i) + b(i));
    }
    {
        mat::dense2D<double> A(0, 0), B(0, 0), D;
        D = A + B;
        REQUIRE(D.num_rows() == 0);
    }
    {
        // Degenerate 2D extents: the flattened sweep must not divide by a zero
        // column count (it returns before the r = t / cols decomposition).
        int calls = 0;
        mtl::detail::parallel_ewise_2d(5, 0, 1, [&](std::size_t, std::size_t) { ++calls; });
        mtl::detail::parallel_ewise_2d(0, 5, 1, [&](std::size_t, std::size_t) { ++calls; });
        REQUIRE(calls == 0);

        // Single element: below the split threshold -> one serial body call.
        mtl::detail::parallel_ewise_2d(1, 1, 1, [&](std::size_t r, std::size_t c) {
            ++calls; REQUIRE(r == 0); REQUIRE(c == 0);
        });
        REQUIRE(calls == 1);
    }
}
