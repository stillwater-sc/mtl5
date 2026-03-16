// MTL5 -- Sparse Direct Solvers: Unified Dispatch Demonstration
//
// This example demonstrates MTL5's unified solve dispatch, which
// automatically selects the best solver backend at compile time:
//
//   Sparse double + SuiteSparse → external solver (UMFPACK, CHOLMOD, SPQR)
//   Sparse (any type)           → native solver + fill-reducing ordering
//   Dense                       → dense factorization (with optional LAPACK)
//
// The user writes one function call. The compiler picks the backend.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstddef>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a symmetric positive definite sparse matrix (2D Laplacian on n x n grid)
// This produces a banded matrix of size N = n*n with bandwidth n.
mtl::mat::compressed2D<double> make_laplacian_2d(std::size_t grid_n) {
    std::size_t N = grid_n * grid_n;
    mtl::mat::compressed2D<double> A(N, N);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < grid_n; ++i) {
            for (std::size_t j = 0; j < grid_n; ++j) {
                std::size_t row = i * grid_n + j;
                ins[row][row] << 4.0;  // diagonal
                if (j + 1 < grid_n) {  // right neighbor
                    ins[row][row + 1] << -1.0;
                    ins[row + 1][row] << -1.0;
                }
                if (i + 1 < grid_n) {  // bottom neighbor
                    ins[row][row + grid_n] << -1.0;
                    ins[row + grid_n][row] << -1.0;
                }
            }
        }
    }
    return A;
}

// Build an unsymmetric sparse matrix (convection-diffusion)
mtl::mat::compressed2D<double> make_convection_diffusion(std::size_t n) {
    mtl::mat::compressed2D<double> A(n, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;   // diffusion
                ins[i + 1][i] << -1.5;   // diffusion + convection (unsymmetric)
            }
        }
        // Wrap-around coupling (makes it more interesting)
        if (n > 2) {
            ins[0][n - 1] << -0.3;
            ins[n - 1][0] << -0.5;
        }
    }
    return A;
}

// Build a dense SPD matrix
mtl::mat::dense2D<double> make_dense_spd(std::size_t n) {
    mtl::mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(n + 1);
        for (std::size_t j = i + 1; j < n; ++j) {
            double v = 1.0 / static_cast<double>(1 + i + j);
            A(i, j) = v;
            A(j, i) = v;
        }
    }
    return A;
}

// Build an overdetermined sparse system (4n x n)
mtl::mat::compressed2D<double> make_overdetermined(std::size_t n) {
    std::size_t m = 4 * n;
    mtl::mat::compressed2D<double> A(m, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
        // First n rows: identity
        for (std::size_t i = 0; i < n; ++i)
            ins[i][i] << 1.0;
        // Next n rows: shifted identity
        for (std::size_t i = 0; i < n; ++i)
            ins[n + i][i] << 2.0;
        // Next n rows: tridiagonal
        for (std::size_t i = 0; i < n; ++i) {
            ins[2 * n + i][i] << 3.0;
            if (i + 1 < n)
                ins[2 * n + i][i + 1] << -1.0;
        }
        // Last n rows: another pattern
        for (std::size_t i = 0; i < n; ++i) {
            ins[3 * n + i][i] << 1.5;
            if (i > 0)
                ins[3 * n + i][i - 1] << 0.5;
        }
    }
    return A;
}

template <typename MatT, typename VecT>
double compute_residual(const MatT& A, const VecT& x, const VecT& b) {
    std::size_t m = A.num_rows();
    std::size_t n = A.num_cols();
    double res = 0.0, bnorm = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        double Ax_i = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            Ax_i += A(i, j) * x(j);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bnorm += b(i) * b(i);
    }
    if (bnorm == 0.0) return std::sqrt(res);
    return std::sqrt(res / bnorm);
}

void print_separator() {
    std::cout << std::string(70, '-') << '\n';
}

// ---------------------------------------------------------------------------
// Demonstrations
// ---------------------------------------------------------------------------

void demo_unified_solve() {
    std::cout << "1. UNIFIED SOLVE: mtl::solve(A, x, b)\n";
    std::cout << "   One function call, automatic backend selection.\n\n";

    // Sparse system
    std::size_t n = 25;
    auto A_sparse = make_convection_diffusion(n);
    mtl::vec::dense_vector<double> b_sparse(n, 1.0);
    mtl::vec::dense_vector<double> x_sparse(n, 0.0);

    mtl::solve(A_sparse, x_sparse, b_sparse);
    std::cout << "   Sparse " << n << "x" << n << " (nnz=" << A_sparse.nnz() << "): ";
#ifdef MTL5_HAS_UMFPACK
    std::cout << "dispatched to UMFPACK\n";
#else
    std::cout << "dispatched to native sparse LU + COLAMD\n";
#endif
    // Compute residual manually for sparse
    double res_s = 0.0, bn_s = 0.0;
    const auto& starts = A_sparse.ref_major();
    const auto& indices = A_sparse.ref_minor();
    const auto& data = A_sparse.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x_sparse(indices[k]);
        double ri = Ax_i - b_sparse(i);
        res_s += ri * ri;
        bn_s += b_sparse(i) * b_sparse(i);
    }
    std::cout << "   Relative residual: " << std::scientific
              << std::sqrt(res_s / bn_s) << '\n';

    // Dense system
    auto A_dense = make_dense_spd(8);
    mtl::vec::dense_vector<double> b_dense(8, 1.0);
    mtl::vec::dense_vector<double> x_dense(8, 0.0);

    mtl::solve(A_dense, x_dense, b_dense);
    std::cout << "\n   Dense 8x8: dispatched to dense LU"
#ifdef MTL5_HAS_LAPACK
              << " (LAPACK)"
#endif
              << '\n';
    std::cout << "   Relative residual: "
              << compute_residual(A_dense, x_dense, b_dense) << '\n';
}

void demo_cholesky_dispatch() {
    std::cout << "\n2. CHOLESKY DISPATCH: mtl::cholesky_solve_dispatch(A, x, b)\n";
    std::cout << "   For symmetric positive definite systems.\n\n";

    // 2D Laplacian (SPD)
    std::size_t grid = 6;
    std::size_t N = grid * grid;
    auto A = make_laplacian_2d(grid);

    mtl::vec::dense_vector<double> b(N, 1.0);
    mtl::vec::dense_vector<double> x(N, 0.0);

    mtl::cholesky_solve_dispatch(A, x, b);

    std::cout << "   " << grid << "x" << grid << " Laplacian (" << N << "x" << N
              << ", nnz=" << A.nnz() << "): ";
#ifdef MTL5_HAS_CHOLMOD
    std::cout << "dispatched to CHOLMOD\n";
#else
    std::cout << "dispatched to native sparse Cholesky + AMD\n";
#endif

    double res = 0.0, bn = 0.0;
    const auto& st = A.ref_major();
    const auto& id = A.ref_minor();
    const auto& da = A.ref_data();
    for (std::size_t i = 0; i < N; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = st[i]; k < st[i + 1]; ++k)
            Ax_i += da[k] * x(id[k]);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bn += b(i) * b(i);
    }
    std::cout << "   Relative residual: " << std::sqrt(res / bn) << '\n';

    // Compare orderings
    auto sym_nat = mtl::sparse::factorization::sparse_cholesky_symbolic(A);
    auto sym_rcm = mtl::sparse::factorization::sparse_cholesky_symbolic(A,
        mtl::sparse::ordering::rcm{});
    auto sym_amd = mtl::sparse::factorization::sparse_cholesky_symbolic(A,
        mtl::sparse::ordering::amd{});

    std::cout << "\n   Fill-in comparison (nnz in L):\n";
    std::cout << "     Natural ordering: " << sym_nat.nnz_L << '\n';
    std::cout << "     RCM ordering:     " << sym_rcm.nnz_L << '\n';
    std::cout << "     AMD ordering:     " << sym_amd.nnz_L << '\n';
}

void demo_lu_dispatch() {
    std::cout << "\n3. LU DISPATCH: mtl::lu_solve_dispatch(A, x, b)\n";
    std::cout << "   For general unsymmetric systems.\n\n";

    std::size_t n = 30;
    auto A = make_convection_diffusion(n);

    mtl::vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b(i) = std::sin(static_cast<double>(i));

    mtl::vec::dense_vector<double> x(n, 0.0);
    mtl::lu_solve_dispatch(A, x, b);

    std::cout << "   Unsymmetric " << n << "x" << n << " (nnz=" << A.nnz() << "): ";
#ifdef MTL5_HAS_UMFPACK
    std::cout << "dispatched to UMFPACK\n";
#else
    std::cout << "dispatched to native sparse LU + COLAMD\n";
#endif

    double res = 0.0, bn = 0.0;
    const auto& st = A.ref_major();
    const auto& id = A.ref_minor();
    const auto& da = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = st[i]; k < st[i + 1]; ++k)
            Ax_i += da[k] * x(id[k]);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bn += b(i) * b(i);
    }
    std::cout << "   Relative residual: " << std::sqrt(res / bn) << '\n';
}

void demo_qr_dispatch() {
    std::cout << "\n4. QR DISPATCH: mtl::qr_solve_dispatch(A, x, b)\n";
    std::cout << "   For least-squares and square systems.\n\n";

    // Overdetermined system: min ||Ax - b||
    std::size_t n = 5;
    auto A = make_overdetermined(n);
    std::size_t m = A.num_rows();

    // RHS: exact solution is x = [1, 2, 3, 4, 5]
    mtl::vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = static_cast<double>(i + 1);

    // b = A * x_true
    mtl::vec::dense_vector<double> b(m, 0.0);
    const auto& st = A.ref_major();
    const auto& id = A.ref_minor();
    const auto& da = A.ref_data();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = st[i]; k < st[i + 1]; ++k)
            b(i) += da[k] * x_true(id[k]);
    }

    mtl::vec::dense_vector<double> x(n, 0.0);
    mtl::qr_solve_dispatch(A, x, b);

    std::cout << "   Overdetermined " << m << "x" << n << " (consistent system):\n";
#ifdef MTL5_HAS_SPQR
    std::cout << "   dispatched to SPQR\n";
#else
    std::cout << "   dispatched to native sparse QR\n";
#endif
    std::cout << "   Solution: [";
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << x(i);
    }
    std::cout << "]\n";
    std::cout << "   Expected: [1, 2, 3, 4, 5]\n";

    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = x(i) - x_true(i);
        err += d * d;
    }
    std::cout << "   Solution error: " << std::scientific << std::sqrt(err) << '\n';
}

void demo_backend_summary() {
    std::cout << "\n5. BACKEND AVAILABILITY\n\n";
    std::cout << "   Compile-time dispatch selects the best available backend:\n\n";
    std::cout << "   BLAS:    "
#ifdef MTL5_HAS_BLAS
              << "ENABLED"
#else
              << "not available (generic fallback)"
#endif
              << '\n';
    std::cout << "   LAPACK:  "
#ifdef MTL5_HAS_LAPACK
              << "ENABLED"
#else
              << "not available (generic fallback)"
#endif
              << '\n';
    std::cout << "   UMFPACK: "
#ifdef MTL5_HAS_UMFPACK
              << "ENABLED  -> sparse LU dispatch target"
#else
              << "not available (native sparse LU)"
#endif
              << '\n';
    std::cout << "   CHOLMOD: "
#ifdef MTL5_HAS_CHOLMOD
              << "ENABLED  -> sparse Cholesky dispatch target"
#else
              << "not available (native sparse Cholesky)"
#endif
              << '\n';
    std::cout << "   SPQR:    "
#ifdef MTL5_HAS_SPQR
              << "ENABLED  -> sparse QR dispatch target"
#else
              << "not available (native sparse QR)"
#endif
              << '\n';
    std::cout << "   SuperLU: "
#ifdef MTL5_HAS_SUPERLU
              << "ENABLED"
#else
              << "not available"
#endif
              << '\n';
    std::cout << "   KLU:     "
#ifdef MTL5_HAS_KLU
              << "ENABLED"
#else
              << "not available"
#endif
              << '\n';

    std::cout << "\n   Native solvers (always available, any value type):\n";
    std::cout << "     sparse::factorization::sparse_cholesky  (SPD, LL^T)\n";
    std::cout << "     sparse::factorization::sparse_lu        (general, PA=LU)\n";
    std::cout << "     sparse::factorization::sparse_qr        (least-squares, QR)\n";
    std::cout << "\n   Fill-reducing orderings:\n";
    std::cout << "     sparse::ordering::rcm     (bandwidth reduction)\n";
    std::cout << "     sparse::ordering::amd     (symmetric fill, for Cholesky)\n";
    std::cout << "     sparse::ordering::colamd  (column fill, for LU/QR)\n";
}

// ---------------------------------------------------------------------------

int main() {
    std::cout << "MTL5 Sparse Direct Solvers — Unified Dispatch Demo\n";
    std::cout << "==================================================\n\n";

    demo_unified_solve();
    print_separator();
    demo_cholesky_dispatch();
    print_separator();
    demo_lu_dispatch();
    print_separator();
    demo_qr_dispatch();
    print_separator();
    demo_backend_summary();

    std::cout << '\n';
    return 0;
}
