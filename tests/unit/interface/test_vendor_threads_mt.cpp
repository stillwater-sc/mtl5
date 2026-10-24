// A vendor BLAS call made inside an MTL5 region runs on one thread (#640);
// one made outside keeps the vendor's thread count; results are unchanged.
#include <catch2/catch_test_macros.hpp>

#include <mtl/parallel/thread_pool.hpp>
#include <mtl/interface/vendor_threads.hpp>

#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#if defined(MTL5_HAS_CHOLMOD) || defined(MTL5_HAS_SPQR)
#include <mtl/generators/poisson.hpp>
#include <mtl/vec/dense_vector.hpp>
#endif
#ifdef MTL5_HAS_CHOLMOD
#include <mtl/interface/cholmod.hpp>
#endif
#ifdef MTL5_HAS_SPQR
#include <mtl/interface/spqr.hpp>
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace {
namespace par = mtl::parallel;
namespace iface = mtl::interface;

par::thread_pool& pool4() {
    par::configure(4);
    return par::pool();
}

/// Give the calling thread a vendor team of n where the vendor allows it on
/// one thread, so a cap to 1 is observable (a new BLIS thread starts at
/// BLIS_NUM_THREADS, usually 1). Returns false where it cannot.
bool set_thread_vendor_threads([[maybe_unused]] int n) {
#if defined(MTL5_BLAS_VENDOR_MKL)
    MKL_Set_Num_Threads_Local(n);
    return true;
#elif defined(MTL5_BLAS_VENDOR_BLIS)
    bli_thread_set_num_threads(n);
    return true;
#else
    return false;
#endif
}
} // namespace

TEST_CASE("interface: the detected BLAS vendor is reported consistently",
          "[interface][blas][threads]") {
    const std::string v = iface::blas_vendor();
    INFO("vendor: '" << v << "'");
    REQUIRE((v.empty() || v == "mkl" || v == "blis" || v == "openblas"));
    REQUIRE(iface::blas_capped_in_region() == (v == "mkl" || v == "blis"));
    REQUIRE((v.empty() == (iface::blas_vendor_threads() == 0)));
}

TEST_CASE("interface: outside a region the cap leaves the vendor's threads alone",
          "[interface][blas][threads]") {
    pool4();
    REQUIRE_FALSE(par::detail::in_region());
    const int prev = iface::blas_vendor_threads();
    if (set_thread_vendor_threads(3)) {
        {
            iface::detail::nested_blas_cap cap;
            REQUIRE(iface::blas_vendor_threads() == 3);
        }
        REQUIRE(iface::blas_vendor_threads() == 3);
        set_thread_vendor_threads(prev);
    } else {
        { iface::detail::nested_blas_cap cap; REQUIRE(iface::blas_vendor_threads() == prev); }
        REQUIRE(iface::blas_vendor_threads() == prev);
    }
}

TEST_CASE("interface: inside a region a vendor call runs on one thread; tid 0 is restored",
          "[interface][blas][threads]") {
    auto& p = pool4();
    if (p.size() < 2) {
        WARN("one hardware thread: no region is dispatched");
        return;
    }
    if (!iface::blas_capped_in_region()) {
        WARN("BLAS vendor '" << iface::blas_vendor()
             << "' has no per-thread control: vendor calls are not capped");
        return;
    }
    const unsigned n = p.size();
    std::vector<int> before(n, -1), during(n, -1), after(n, -1);
    p.run(n, [&](unsigned tid) {
        const int prev = iface::blas_vendor_threads();
        set_thread_vendor_threads(static_cast<int>(2 + tid));
        before[tid] = iface::blas_vendor_threads();
        {
            iface::detail::nested_blas_cap cap;
            during[tid] = iface::blas_vendor_threads();
        }
        after[tid] = iface::blas_vendor_threads();
        set_thread_vendor_threads(prev);
    });
    for (unsigned t = 0; t < n; ++t) {
        INFO("tid " << t);
        REQUIRE(before[t] == static_cast<int>(2 + t));
        REQUIRE(during[t] == 1);
        // The caller's setting comes back; a pool worker's stays at one.
        REQUIRE(after[t] == (t == 0 ? 2 : 1));
    }
}

#ifdef MTL5_HAS_BLAS
TEST_CASE("interface: a vendor GEMM gives the same result inside and outside a region",
          "[interface][blas][threads]") {
    auto& p = pool4();
    // Small integers: every product and partial sum is exact, so any team
    // size and any blocking gives the same bits.
    const int m = 96;
    std::vector<double> A(m * m), B(m * m), ref(m * m, 0.0);
    for (int i = 0; i < m * m; ++i) {
        A[i] = static_cast<double>((i * 7) % 11) - 5.0;
        B[i] = static_cast<double>((i * 5) % 13) - 6.0;
    }
    iface::blas::gemm('N', 'N', m, m, m, 1.0, A.data(), m, B.data(), m, 0.0, ref.data(), m);

    const unsigned n = p.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(m * m, 0.0));
    p.run(n, [&](unsigned tid) {
        iface::blas::gemm('N', 'N', m, m, m, 1.0, A.data(), m, B.data(), m, 0.0,
                          C[tid].data(), m);
    });
    for (unsigned t = 0; t < n; ++t) REQUIRE(C[t] == ref);
}
#endif

namespace {
/// Stands in for cholmod_common: the cap reads and writes only nthreads_max.
struct fake_common { int nthreads_max; };
} // namespace

TEST_CASE("interface: the SuiteSparse cap sets nthreads_max to 1 only inside a region",
          "[interface][suitesparse][threads]") {
    auto& p = pool4();
    fake_common c{6};
    {
        iface::detail::nested_suitesparse_cap<fake_common> cap(c);
        REQUIRE(c.nthreads_max == 6);       // outside a region: unchanged
    }
    REQUIRE(c.nthreads_max == 6);
    if (p.size() < 2) return;

    std::vector<fake_common> per(p.size(), fake_common{6});
    std::vector<int> during(p.size(), -1);
    p.run(p.size(), [&](unsigned tid) {
        iface::detail::nested_suitesparse_cap<fake_common> cap(per[tid]);
        during[tid] = per[tid].nthreads_max;
    });
    for (unsigned t = 0; t < p.size(); ++t) {
        REQUIRE(during[t] == 1);
        REQUIRE(per[t].nthreads_max == 6);  // restored
    }
}

#if defined(MTL5_HAS_CHOLMOD) || defined(MTL5_HAS_SPQR)
namespace {
/// Solve the same SPD system on every tid of a region and once outside one.
template <typename Solve>
void same_inside_and_outside(Solve&& solve) {
    auto& p = pool4();
    const auto A = mtl::generators::poisson2d_dirichlet(24, 24);
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<double> b(n, 1.0), ref(n, 0.0);
    solve(A, ref, b);
    std::vector<mtl::vec::dense_vector<double>> x(p.size(), mtl::vec::dense_vector<double>(n, 0.0));
    p.run(p.size(), [&](unsigned tid) { solve(A, x[tid], b); });
    for (unsigned t = 0; t < p.size(); ++t)
        for (std::size_t i = 0; i < n; ++i) REQUIRE(x[t](i) == ref(i));
}
} // namespace
#endif

#ifdef MTL5_HAS_CHOLMOD
TEST_CASE("interface: CHOLMOD solves the same inside and outside a region",
          "[interface][cholmod][threads]") {
    same_inside_and_outside([](const auto& A, auto& x, const auto& b) {
        mtl::interface::cholmod_solve(A, x, b);
    });
}
#endif

#ifdef MTL5_HAS_SPQR
TEST_CASE("interface: SPQR solves the same inside and outside a region",
          "[interface][spqr][threads]") {
    same_inside_and_outside([](const auto& A, auto& x, const auto& b) {
        mtl::interface::spqr_solver s(A);
        s.solve(x, b);
    });
}
#endif
