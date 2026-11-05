// MTL5 -- dense_vector and dense2D build with exceptions and RTTI disabled (#652)
//
// Firmware builds with -fno-exceptions -fno-rtti. tests/CMakeLists.txt builds
// this file with those flags and -Werror on GCC and Clang, with the default
// thread pool, with MTL5_POOL_SPIN, and with MTL5_SHARED_POOL, so every CI job on those
// compilers fails if a try, catch or throw reaches these headers again. It is
// a plain executable: Catch2 itself needs exceptions.
//
// NDEBUG is undefined first so that bounds checking is on (mtl/config.hpp) and
// the element accessors' out-of-range path is instantiated even in the Release
// builds CI uses.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

#if MTL5_HAS_EXCEPTIONS
#  error "this test must be built with exceptions disabled"
#endif

// The shared variant defines the process's pool in this translation unit, as
// the mtl5_pool library would.
#if defined(MTL5_SHARED_POOL)
#  include <mtl/parallel/shared_pool_definition.hpp>
#endif

#include <cstdio>
#include <type_traits>

// The pool's names carry the exception mode, so a module built with exceptions
// cannot link to this pool, whose workers have no catch.
#if defined(MTL5_SHARED_POOL)
static_assert(std::is_same_v<mtl::parallel::thread_pool,
                             mtl::parallel::pool_shared_noexc::thread_pool>);
#else
static_assert(std::is_same_v<mtl::parallel::thread_pool,
                             mtl::parallel::pool_local_noexc::thread_pool>);
#endif

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

} // namespace

int main() {
    static_assert(mtl::bounds_checking, "bounds checking must be on in this test");

    // Fixed-size, on-stack vector: the reflex flight-control use.
    using vec3_params = mtl::vec::parameters<mtl::tag::col_major,
                                             mtl::vec::fixed::dimension<3>,
                                             mtl::tag::on_stack>;
    mtl::vec::dense_vector<double, vec3_params> a, b;
    a(0) = 1.0; a(1) = 2.0; a(2) = 3.0;
    b(0) = 4.0; b(1) = 5.0; b(2) = 6.0;
    const auto& ca = a;
    check(ca(2) == 3.0, "const element access");
    a += b;                                       // element-wise, via the pool
    check(a(0) == 5.0 && a(1) == 7.0 && a(2) == 9.0, "fixed vector +=");

    // Fixed-size, on-stack matrix.
    using mat23_params = mtl::mat::parameters<mtl::tag::row_major,
                                              mtl::detail::c_index,
                                              mtl::mat::fixed::dimensions<2, 3>,
                                              mtl::tag::on_stack>;
    mtl::mat::dense2D<double, mat23_params> m;
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c) m(r, c) = double(r * 3 + c);
    const auto& cm = m;
    check(cm(1, 2) == 5.0, "fixed matrix element access");

    // Heap-backed containers: the aligned allocator's construct path.
    mtl::vec::dense_vector<double> d(1000, 1.0);
    mtl::vec::dense_vector<double> e(1000, 2.0);
    d += e;
    check(d(0) == 3.0 && d(999) == 3.0, "dynamic vector +=");
    mtl::mat::dense2D<double> h(4, 4);
    h(3, 3) = 7.0;
    check(h(3, 3) == 7.0, "dynamic matrix element access");

    // The pool's run() without its try/catch, on more than one thread.
    // configure() clamps to the hardware, so a one-core runner gets one tid.
    mtl::parallel::configure(2);
    const unsigned team = mtl::parallel::pool().size() < 2 ? mtl::parallel::pool().size() : 2u;
    unsigned hits[2] = {0, 0};
    mtl::parallel::pool().run(team, [&](unsigned tid) { hits[tid] += 1; });
    check(hits[0] == 1 && hits[1] == (team == 2 ? 1u : 0u), "pool run");

    if (failures == 0) std::printf("no-exceptions build: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
