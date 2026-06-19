// MTL5 -- Native KLU on circuit-simulation matrices (double precision)
//
// Loads a sparse matrix from the SuiteSparse Matrix Collection
// (https://sparse.tamu.edu/) in Matrix Market format and solves A*x = b with
// MTL5's native KLU sparse direct solver. Circuit-simulation matrices (Modified
// Nodal Analysis) are typically reducible -- exactly the block-triangular
// structure native KLU's BTF exploits.
//
// When MTL5 is built with SuiteSparse KLU (-DMTL5_WITH_SUITESPARSE_KLU=ON), the
// same system is also solved with the external KLU binding and the two results
// are compared.
//
// Usage:
//   circuit_matrix_klu [matrix.mtx]
//
// With no argument a small synthetic block-triangular circuit-like matrix is
// used, so the example runs without any download. Fetch real matrices with
// examples/phase15_sparse_direct/fetch_circuit_matrices.sh, e.g.:
//
//   ./circuit_matrix_klu data/rajat30/rajat30.mtx
//
// Targets: Rajat/rajat30 (small/medium) and Freescale/circuit5M (very large).

#include <mtl/mtl.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Sparse = mtl::mat::compressed2D<double>;

// Small reducible (block-triangular) matrix mimicking circuit structure: two
// decoupled 3x3 blocks linked by an upper coupling. BTF should find 2 blocks.
Sparse make_synthetic_circuit() {
    Sparse A(6, 6);
    mtl::mat::inserter<Sparse> ins(A);
    ins[0][0] << 3.0;  ins[0][1] << -1.0;
    ins[1][0] << -1.0; ins[1][1] << 3.0;  ins[1][2] << -1.0;
    ins[2][1] << -1.0; ins[2][2] << 3.0;
    ins[3][3] << 4.0;  ins[3][4] << -1.0;
    ins[4][3] << -1.0; ins[4][4] << 4.0;  ins[4][5] << -1.0;
    ins[5][4] << -1.0; ins[5][5] << 4.0;
    ins[0][5] << -2.0;  // upper coupling: block {0,1,2} -> {3,4,5}
    return A;
}

// Residual ||A*x - b||_inf.
double residual_inf(const Sparse& A,
                    const mtl::vec::dense_vector<double>& x,
                    const mtl::vec::dense_vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double m = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ax += dat[k] * x(static_cast<int>(ci[k]));
        m = std::max(m, std::abs(ax - b(static_cast<int>(r))));
    }
    return m;
}

template <typename F>
double time_ms(F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    Sparse A;
    try {
        if (argc > 1) {
            std::cout << "Loading Matrix Market file: " << argv[1] << '\n';
            A = mtl::io::mm_read<double>(argv[1]);
        } else {
            std::cout << "No matrix given; using synthetic block-triangular "
                         "circuit matrix.\n(Pass a .mtx file -- see "
                         "fetch_circuit_matrices.sh -- to run on a real circuit.)\n";
            A = make_synthetic_circuit();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load matrix: " << e.what() << '\n';
        return 1;
    }

    const std::size_t n = A.num_rows();
    if (A.num_rows() != A.num_cols()) {
        std::cerr << "Matrix is not square (" << A.num_rows() << "x"
                  << A.num_cols() << "); KLU requires a square system.\n";
        return 1;
    }
    std::cout << "Matrix: " << n << " x " << A.num_cols()
              << ", nnz = " << A.nnz() << "\n\n";

    // Reproducible RHS: b = A * ones, so the exact solution is all-ones.
    mtl::vec::dense_vector<double> ones(n, 1.0), b(n, 0.0);
    {
        const auto& rp = A.ref_major();
        const auto& ci = A.ref_minor();
        const auto& dat = A.ref_data();
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
                s += dat[k] * ones(static_cast<int>(ci[k]));
            b(static_cast<int>(r)) = s;
        }
    }

    // --- Native KLU (always available, any value type) ---
    std::cout << "Native KLU (BTF + block-wise Gilbert-Peierls LU):\n";
    try {
        mtl::vec::dense_vector<double> x(n, 0.0);
        double factor_ms = 0.0, solve_ms = 0.0;
        mtl::sparse::factorization::klu_numeric<double> fac;
        factor_ms = time_ms([&] {
            fac = mtl::sparse::factorization::native_klu_factor(A);
        });
        solve_ms = time_ms([&] { fac.solve(x, b); });

        std::cout << "  BTF blocks   : " << fac.nblocks() << '\n';
        std::cout << "  factor time  : " << std::fixed << std::setprecision(2)
                  << factor_ms << " ms\n";
        std::cout << "  solve time   : " << solve_ms << " ms\n";
        std::cout << "  ||Ax-b||_inf : " << std::scientific << std::setprecision(4)
                  << residual_inf(A, x, b) << '\n';
    } catch (const std::exception& e) {
        std::cout << "  native KLU failed: " << e.what() << '\n';
    }

    // --- External SuiteSparse KLU (optional) + cross-check ---
#ifdef MTL5_HAS_KLU
    std::cout << "\nExternal SuiteSparse KLU:\n";
    try {
        mtl::vec::dense_vector<double> x_ext(n, 0.0), x_nat(n, 0.0);
        double ext_ms = time_ms([&] { mtl::interface::klu_solve(A, x_ext, b); });
        std::cout << "  total time   : " << std::fixed << std::setprecision(2)
                  << ext_ms << " ms\n";
        std::cout << "  ||Ax-b||_inf : " << std::scientific << std::setprecision(4)
                  << residual_inf(A, x_ext, b) << '\n';

        // Cross-check native vs external (only if native succeeded).
        try {
            mtl::sparse::factorization::native_klu_solve(A, x_nat, b);
            double diff = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                diff = std::max(diff, std::abs(x_nat(static_cast<int>(i))
                                               - x_ext(static_cast<int>(i))));
            std::cout << "  max|native - external| = " << diff << '\n';
        } catch (const std::exception&) {
            std::cout << "  (native KLU unavailable for this matrix; skipping cross-check)\n";
        }
    } catch (const std::exception& e) {
        std::cout << "  external KLU failed: " << e.what() << '\n';
    }
#else
    std::cout << "\n[External SuiteSparse KLU not available: build with "
                 "-DMTL5_WITH_SUITESPARSE_KLU=ON to enable the cross-check.]\n";
#endif

    return 0;
}
