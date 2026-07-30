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

#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
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
}
