#pragma once
// MTL5 Benchmark Harness -- Suite runners.
//
// One binary == one backend. The benchmark calls only the public mtl:: API;
// compile-time dispatch (governed by MTL5_HAS_BLAS / MTL5_HAS_LAPACK) decides
// whether each op runs the generic C++ path or a BLAS/LAPACK path. The build
// configuration therefore *is* the backend -- exactly as in a dependent
// application that sets the BLAS/LAPACK flags once for the whole program.
//
// The `label` (e.g. "native", "openblas", "mkl") is passed in and recorded in
// the output; the harness itself does not select an implementation.

#include <benchmarks/harness/timer.hpp>
#include <benchmarks/harness/reporter.hpp>
#include <benchmarks/harness/generators.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/operation/mult.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/ger.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>

namespace mtl::bench {

// ── BLAS-level suites ──────────────────────────────────────────────────────

inline void bench_dot(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto v1 = make_random_vector<double>(n);
        auto v2 = make_random_vector<double>(n, 456);
        double flops = static_cast<double>(2 * n);
        volatile double sink = 0.0;
        auto t = measure([&]{ sink = mtl::dot(v1, v2); },
                         "dot", label, n, flops, warmup, iterations);
        (void)sink;
        rep.add(t);
    }
}

inline void bench_nrm2(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto v = make_random_vector<double>(n);
        double flops = static_cast<double>(2 * n + 1);
        volatile double sink = 0.0;
        auto t = measure([&]{ sink = mtl::two_norm(v); },
                         "nrm2", label, n, flops, warmup, iterations);
        (void)sink;
        rep.add(t);
    }
}

inline void bench_axpy(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto x = make_random_vector<double>(n);
        auto y = make_random_vector<double>(n, 456);
        const double alpha = 1.0000001;   // near 1 so repeated y += alpha*x stays bounded
        double flops = static_cast<double>(2 * n);
        auto t = measure([&]{ mtl::axpy(alpha, x, y); },
                         "axpy", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_scal(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto x = make_random_vector<double>(n);
        const double alpha = 1.0000001;   // near 1 so repeated x *= alpha stays bounded
        double flops = static_cast<double>(n);
        auto t = measure([&]{ mtl::scale(alpha, x); },
                         "scal", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_gemv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto x = make_random_vector<double>(n);
        vec::dense_vector<double> y(n);
        double flops = static_cast<double>(2 * n * n);
        auto t = measure([&]{ mtl::mult(A, x, y); },
                         "gemv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_ger(reporter& rep, const std::string& label,
                      const std::vector<std::size_t>& sizes,
                      std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto x = make_random_vector<double>(n);
        auto y = make_random_vector<double>(n, 77);
        const double alpha = 1.0000001;
        double flops = static_cast<double>(2 * n * n);
        auto t = measure([&]{ mtl::ger(alpha, x, y, A); },
                         "ger", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_symv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);   // treated as symmetric
        auto x = make_random_vector<double>(n);
        vec::dense_vector<double> y(n, 0.0);
        double flops = static_cast<double>(2 * n * n);
        auto t = measure([&]{ mtl::symv(1.0, A, x, 0.0, y); },
                         "symv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_trmv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);   // upper triangle used
        auto x = make_random_vector<double>(n);
        double flops = static_cast<double>(n * n);
        auto t = measure([&]{ mtl::trmv(A, x, /*upper=*/true); },
                         "trmv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_trsv(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        for (std::size_t i = 0; i < n; ++i)     // strengthen the diagonal for a stable solve
            A(i, i) += static_cast<double>(n);
        auto b = make_random_vector<double>(n);
        vec::dense_vector<double> x(n);
        double flops = static_cast<double>(n * n);
        auto t = measure([&]{ mtl::trsv(A, x, b, /*upper=*/true); },
                         "trsv", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_gemm(reporter& rep, const std::string& label,
                       const std::vector<std::size_t>& sizes,
                       std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto B = make_random_matrix<double>(n, n, 99);
        mat::dense2D<double> C(n, n);
        double flops = static_cast<double>(2 * n * n * n);
        auto t = measure([&]{ mtl::mult(A, B, C); },
                         "gemm", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

// ── LAPACK-level suites ─────────────────────────────────────────────────────
// Column-major inputs so the BLAS/LAPACK dispatch is eligible (matches a
// real app that stores factorization operands column-major).

inline void bench_lu(reporter& rep, const std::string& label,
                      const std::vector<std::size_t>& sizes,
                      std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_random_matrix_colmaj<double>(n, n);
        double flops = (2.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    std::vector<std::size_t> pivot;
                    mtl::lu_factor(A, pivot);
                 }, "lu_factor", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_qr(reporter& rep, const std::string& label,
                     const std::vector<std::size_t>& sizes,
                     std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_random_matrix_colmaj<double>(n, n);
        // 2*m*n*k - 2/3*k^3 with m=n=k=n  ->  4/3 n^3
        double flops = (4.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    vec::dense_vector<double> tau;
                    mtl::qr_factor(A, tau);
                 }, "qr_factor", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_cholesky(reporter& rep, const std::string& label,
                           const std::vector<std::size_t>& sizes,
                           std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_spd_matrix_colmaj<double>(n);
        double flops = (1.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    mtl::cholesky_factor(A);
                 }, "cholesky", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

inline void bench_eigenvalue(reporter& rep, const std::string& label,
                             const std::vector<std::size_t>& sizes,
                             std::size_t warmup = 1, std::size_t iterations = 3) {
    for (auto n : sizes) {
        auto A_template = make_spd_matrix_colmaj<double>(n);
        double flops = (4.0 / 3.0) * static_cast<double>(n) * n * n;
        auto t = measure([&]{
                    auto A = A_template;
                    auto e = mtl::eigenvalue_symmetric(A);
                    (void)e;
                 }, "eig_sym", label, n, flops, warmup, iterations);
        rep.add(t);
    }
}

// ── Convenience: run all suites ─────────────────────────────────────────────

inline void run_all(reporter& rep, const std::string& label,
                    const std::vector<std::size_t>& blas_sizes,
                    const std::vector<std::size_t>& lapack_sizes) {
    std::cout << "=== BLAS Level 1 ===" << std::endl;
    bench_dot(rep, label, blas_sizes);
    bench_nrm2(rep, label, blas_sizes);
    bench_axpy(rep, label, blas_sizes);
    bench_scal(rep, label, blas_sizes);

    std::cout << "=== BLAS Level 2 ===" << std::endl;
    bench_gemv(rep, label, blas_sizes);
    bench_ger(rep, label, blas_sizes);
    bench_symv(rep, label, blas_sizes);
    bench_trmv(rep, label, blas_sizes);
    bench_trsv(rep, label, blas_sizes);

    std::cout << "=== BLAS Level 3 ===" << std::endl;
    bench_gemm(rep, label, blas_sizes);

    std::cout << "=== LAPACK Factorizations ===" << std::endl;
    bench_lu(rep, label, lapack_sizes);
    bench_qr(rep, label, lapack_sizes);
    bench_cholesky(rep, label, lapack_sizes);
    bench_eigenvalue(rep, label, lapack_sizes);
}

} // namespace mtl::bench
