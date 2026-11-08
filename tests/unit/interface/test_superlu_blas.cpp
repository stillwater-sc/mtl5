// MTL5 -- SuperLU and BLAS in one translation unit (#725)
//
// SuperLU's slu_ddefs.h declares its own prototypes for the BLAS routines it
// calls, with signatures that differ from MTL5's in interface/blas.hpp, so a
// translation unit that included both -- <mtl/mtl.hpp> with MTL5_HAS_BLAS and
// MTL5_HAS_SUPERLU among them -- did not compile. interface/superlu.hpp now
// hides SuperLU's prototypes while its header is parsed.
//
// This file is the regression test: it includes the umbrella header, which
// pulls in both interfaces, and calls a SuperLU solve and BLAS-dispatched
// dense operations in the same translation unit. It only does real work when
// both MTL5_HAS_SUPERLU and MTL5_HAS_BLAS are defined; CI builds it that way
// in the LAPACK lanes. Otherwise a placeholder confirms it compiles and links.
#include <mtl/mtl.hpp>
#include <mtl/interface/superlu.hpp>
#include <mtl/interface/blas.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>

using namespace mtl;

#if defined(MTL5_HAS_SUPERLU) && defined(MTL5_HAS_BLAS)

// The renaming must not leak: after the headers, the BLAS names are the real
// symbols again (for the usual trailing-underscore Fortran convention, the
// one slu_Cnames.h leaves unmapped).
#  if defined(dgemm_) || defined(dgemv_) || defined(dtrsv_) || defined(dtrsm_) || \
      defined(daxpy_) || defined(dcopy_) || defined(ddot_)
#    error "superlu.hpp leaked a BLAS-name macro"
#  endif

TEST_CASE("SuperLU solve and BLAS-dispatched operations in one translation unit", "[interface][superlu][blas][725]") {
    // Sparse: a 1D Laplacian, solved by SuperLU.
    const std::size_t n = 50;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0;
            if (i > 0)     ins[i][i - 1] << -1.0;
            if (i + 1 < n) ins[i][i + 1] << -1.0;
        }
    }
    vec::dense_vector<double> b(n, 1.0), x(n, 0.0);
    interface::superlu_solver slu(A);
    slu.solve(x, b);

    vec::dense_vector<double> r(n);
    r = A * x;
    for (std::size_t i = 0; i < n; ++i) REQUIRE(std::fabs(r(i) - b(i)) < 1e-10);

    // Dense: products and a dot that go to dgemm_, dgemv_ and ddot_.
    mat::dense2D<double> D(n, n), E(n, n), F(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            D(i, j) = (i == j) ? 2.0 : 0.0;
            E(i, j) = double(i + j) / double(n);
        }
    F = D * E;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) REQUIRE(F(i, j) == 2.0 * E(i, j));

    vec::dense_vector<double> y(n);
    y = D * x;
    for (std::size_t i = 0; i < n; ++i) REQUIRE(y(i) == 2.0 * x(i));
    REQUIRE(std::fabs(dot(y, b) - 2.0 * dot(x, b)) < 1e-9 * std::fabs(dot(x, b)));
}

#else

TEST_CASE("SuperLU + BLAS translation unit (placeholder)", "[interface][superlu][blas][725]") {
    SUCCEED("MTL5_HAS_SUPERLU and MTL5_HAS_BLAS not both defined -- the combined test is skipped");
}

#endif
