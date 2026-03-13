#pragma once
// MTL5 Benchmark Harness -- Suite runner with compile-time backend expansion
// Uses fold expressions over backend_list to benchmark all available backends.

#include <benchmarks/harness/backend.hpp>
#include <benchmarks/harness/timer.hpp>
#include <benchmarks/harness/reporter.hpp>
#include <benchmarks/harness/generators.hpp>
#include <benchmarks/harness/op_blas.hpp>
#include <benchmarks/harness/op_lapack.hpp>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace mtl::bench {

// ── Helper: expand a backend_list into a fold over a callable ──────────────

template <typename Fn, typename... Backends>
void for_each_backend(backend_list<Backends...>, Fn&& fn) {
    (fn.template operator()<Backends>(), ...);
}

// ── BLAS-level benchmark suites ───────────────────────────────────────────

/// Run GEMM across all sizes for all backends in BackendList
template <typename BackendList = dense_backends>
void bench_gemm(reporter& rep, const std::vector<std::size_t>& sizes,
                std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto B = make_random_matrix<double>(n, n, 99);
        mat::dense2D<double> C(n, n);
        double flops = op::gemm<Native>::flops(n, n, n);

        for_each_backend(BackendList{}, [&]<typename Backend>() {
            auto t = measure(
                [&]{ op::gemm<Backend>::run(A, B, C); },
                "gemm", Backend::name, n, flops,
                warmup, iterations);
            rep.add(t);
        });
    }
}

/// Run GEMV across all sizes for all backends in BackendList
template <typename BackendList = dense_backends>
void bench_gemv(reporter& rep, const std::vector<std::size_t>& sizes,
                std::size_t warmup = 3, std::size_t iterations = 10) {
    for (auto n : sizes) {
        auto A = make_random_matrix<double>(n, n);
        auto x = make_random_vector<double>(n);
        vec::dense_vector<double> y(n);
        double flops = op::gemv<Native>::flops(n, n);

        for_each_backend(BackendList{}, [&]<typename Backend>() {
            auto t = measure(
                [&]{ op::gemv<Backend>::run(A, x, y); },
                "gemv", Backend::name, n, flops,
                warmup, iterations);
            rep.add(t);
        });
    }
}

/// Run dot product across all sizes
template <typename BackendList = dense_backends>
void bench_dot(reporter& rep, const std::vector<std::size_t>& sizes,
               std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto v1 = make_random_vector<double>(n);
        auto v2 = make_random_vector<double>(n, 456);
        double flops = op::dot_op<Native>::flops(n);

        for_each_backend(BackendList{}, [&]<typename Backend>() {
            volatile double sink = 0.0;
            auto t = measure(
                [&]{ sink = op::dot_op<Backend>::run(v1, v2); },
                "dot", Backend::name, n, flops,
                warmup, iterations);
            rep.add(t);
            (void)sink;
        });
    }
}

/// Run two_norm across all sizes
template <typename BackendList = dense_backends>
void bench_nrm2(reporter& rep, const std::vector<std::size_t>& sizes,
                std::size_t warmup = 3, std::size_t iterations = 20) {
    for (auto n : sizes) {
        auto v = make_random_vector<double>(n);
        double flops = op::two_norm_op<Native>::flops(n);

        for_each_backend(BackendList{}, [&]<typename Backend>() {
            volatile double sink = 0.0;
            auto t = measure(
                [&]{ sink = op::two_norm_op<Backend>::run(v); },
                "nrm2", Backend::name, n, flops,
                warmup, iterations);
            rep.add(t);
            (void)sink;
        });
    }
}

// ── LAPACK-level benchmark suites ─────────────────────────────────────────

/// Run LU factorization across all sizes (column-major for LAPACK eligibility)
template <typename BackendList = factor_backends>
void bench_lu(reporter& rep, const std::vector<std::size_t>& sizes,
              std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        // Pre-generate template matrix outside the timing loop
        auto A_template = make_random_matrix_colmaj<double>(n, n);
        for_each_backend(BackendList{}, [&]<typename Backend>() {
            auto t = measure(
                [&]{
                    auto A = A_template; // copy
                    std::vector<std::size_t> pivot;
                    op::lu_factor_op<Backend>::run(A, pivot);
                },
                "lu_factor", Backend::name, n,
                op::lu_factor_op<Native>::flops(n),
                warmup, iterations);
            rep.add(t);
        });
    }
}

/// Run QR factorization across all sizes (column-major for LAPACK eligibility)
template <typename BackendList = factor_backends>
void bench_qr(reporter& rep, const std::vector<std::size_t>& sizes,
              std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_random_matrix_colmaj<double>(n, n);
        for_each_backend(BackendList{}, [&]<typename Backend>() {
            auto t = measure(
                [&]{
                    auto A = A_template; // copy
                    vec::dense_vector<double> tau;
                    op::qr_factor_op<Backend>::run(A, tau);
                },
                "qr_factor", Backend::name, n,
                op::qr_factor_op<Native>::flops(n, n),
                warmup, iterations);
            rep.add(t);
        });
    }
}

/// Run Cholesky factorization across all sizes (column-major SPD matrices)
template <typename BackendList = factor_backends>
void bench_cholesky(reporter& rep, const std::vector<std::size_t>& sizes,
                    std::size_t warmup = 2, std::size_t iterations = 5) {
    for (auto n : sizes) {
        auto A_template = make_spd_matrix_colmaj<double>(n);
        for_each_backend(BackendList{}, [&]<typename Backend>() {
            auto t = measure(
                [&]{
                    auto A = A_template; // copy
                    op::cholesky_factor_op<Backend>::run(A);
                },
                "cholesky", Backend::name, n,
                op::cholesky_factor_op<Native>::flops(n),
                warmup, iterations);
            rep.add(t);
        });
    }
}

/// Run eigenvalue solver across all sizes (column-major symmetric matrices)
template <typename BackendList = factor_backends>
void bench_eigenvalue(reporter& rep, const std::vector<std::size_t>& sizes,
                      std::size_t warmup = 1, std::size_t iterations = 3) {
    for (auto n : sizes) {
        auto A_template = make_spd_matrix_colmaj<double>(n);
        for_each_backend(BackendList{}, [&]<typename Backend>() {
            auto t = measure(
                [&]{
                    auto A = A_template; // copy
                    op::eigenvalue_sym_op<Backend>::run(A);
                },
                "eig_sym", Backend::name, n,
                op::eigenvalue_sym_op<Native>::flops(n),
                warmup, iterations);
            rep.add(t);
        });
    }
}

// ── Convenience: run all suites ───────────────────────────────────────────

inline void run_all(reporter& rep,
                    const std::vector<std::size_t>& blas_sizes = {64, 128, 256, 512, 1024},
                    const std::vector<std::size_t>& lapack_sizes = {64, 128, 256, 512}) {
    std::cout << "=== BLAS Level 1 ===" << std::endl;
    bench_dot(rep, blas_sizes);
    bench_nrm2(rep, blas_sizes);

    std::cout << "=== BLAS Level 2 ===" << std::endl;
    bench_gemv(rep, blas_sizes);

    std::cout << "=== BLAS Level 3 ===" << std::endl;
    bench_gemm(rep, blas_sizes);

    std::cout << "=== LAPACK Factorizations ===" << std::endl;
    bench_lu(rep, lapack_sizes);
    bench_qr(rep, lapack_sizes);
    bench_cholesky(rep, lapack_sizes);
    bench_eigenvalue(rep, lapack_sizes);
}

} // namespace mtl::bench
