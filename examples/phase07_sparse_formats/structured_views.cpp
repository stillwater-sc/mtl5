// structured_views.cpp - Structured Matrix Views
//
// This example demonstrates:
//   1. Hermitian view: store upper triangle, mirror with conjugation
//   2. Banded view: extract tridiagonal/diagonal bands from dense
//   3. Map view: zero-copy submatrices and row/column permutations
//   4. Identity matrix: implicit, zero storage
//   5. Using views with iterative solvers (hermitian view + CG)
//   6. Matrix Market I/O round-trip
//
// Views are lightweight wrappers that reinterpret existing storage
// without copying data - zero-cost abstractions.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace mtl;

void print_matrix(const std::string& name, auto const& M,
                  std::size_t rows, std::size_t cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (std::size_t i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < cols; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << M(i, j);
        }
        std::cout << "]\n";
    }
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 7B: Structured Matrix Views - Zero-Copy Abstractions\n";
    std::cout << "=============================================================\n\n";

    const std::size_t n = 6;

    // ======================================================================
    // Part 1: Hermitian View
    // ======================================================================
    std::cout << "=== Part 1: Hermitian View ===\n";
    std::cout << "Store only the upper triangle; the view mirrors it below.\n";
    std::cout << "For real matrices: H(i,j) = H(j,i). No extra storage.\n\n";

    // Build SPD matrix - store only upper triangle
    mat::dense2D<double> A_upper(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A_upper(i, i) = 4.0 + i;  // positive diagonal
        for (std::size_t j = i + 1; j < n; ++j) {
            double val = 1.0 / (1.0 + (j - i));
            A_upper(i, j) = val;
            A_upper(j, i) = 0.0;  // lower triangle is zero in storage
        }
    }

    std::cout << "Upper triangle only:\n";
    print_matrix("A_upper", A_upper, n, n);

    auto H = hermitian(A_upper);
    std::cout << "\nHermitian view (mirrors upper to lower):\n";
    print_matrix("H = hermitian(A)", H, n, n);

    // Verify symmetry
    bool is_sym = true;
    for (std::size_t i = 0; i < n && is_sym; ++i)
        for (std::size_t j = 0; j < n && is_sym; ++j)
            if (std::abs(H(i,j) - H(j,i)) > 1e-15)
                is_sym = false;
    std::cout << "\nSymmetry check H(i,j) == H(j,i): " << (is_sym ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 2: Banded View
    // ======================================================================
    std::cout << "=== Part 2: Banded View ===\n";
    std::cout << "Extract a band around the diagonal. Elements outside = 0.\n\n";

    // Build a full matrix
    mat::dense2D<double> A_full(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A_full(i, i) = 10.0 + i;
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j)
                A_full(i, j) = 1.0 / (1.0 + std::abs(static_cast<double>(i) - static_cast<double>(j)));
        }
    }

    // Tridiagonal view (lower=1, upper=1)
    auto tri = banded(A_full, 1, 1);
    std::cout << "Tridiagonal view banded(A, 1, 1):\n";
    print_matrix("tri", tri, n, n);

    // Diagonal-only view (lower=0, upper=0)
    auto diag_view = banded(A_full, 0, 0);
    std::cout << "\nDiagonal-only view banded(A, 0, 0):\n";
    print_matrix("diag", diag_view, n, n);

    // Verify out-of-band elements are zero
    bool band_ok = true;
    for (std::size_t i = 0; i < n && band_ok; ++i)
        for (std::size_t j = 0; j < n && band_ok; ++j) {
            auto diff = static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>(i);
            if (diff < -1 || diff > 1) {
                if (tri(i, j) != 0.0) band_ok = false;
            }
        }
    std::cout << "\nOut-of-band elements zero: " << (band_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 3: Map View
    // ======================================================================
    std::cout << "=== Part 3: Map View (Index Remapping) ===\n";
    std::cout << "Reindex rows and columns without copying data.\n\n";

    // Extract 3x3 submatrix (rows 1,3,5 and cols 0,2,4)
    std::vector<std::size_t> row_sel = {1, 3, 5};
    std::vector<std::size_t> col_sel = {0, 2, 4};

    auto sub = mapped(A_full, row_sel, col_sel);
    std::cout << "Submatrix A[{1,3,5}, {0,2,4}]:\n";
    print_matrix("sub", sub, 3, 3);

    // Verify: sub(i,j) == A_full(row_sel[i], col_sel[j])
    bool sub_ok = true;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (sub(i, j) != A_full(row_sel[i], col_sel[j]))
                sub_ok = false;
    std::cout << "\nSubmatrix indexing: " << (sub_ok ? "PASS" : "FAIL") << "\n\n";

    // Row permutation: reverse order
    std::vector<std::size_t> rev_rows(n), all_cols(n);
    for (std::size_t i = 0; i < n; ++i) {
        rev_rows[i] = n - 1 - i;
        all_cols[i] = i;
    }
    auto perm = mapped(A_full, rev_rows, all_cols);
    std::cout << "Row-reversed view (rows " << n-1 << "..0):\n";
    print_matrix("perm", perm, n, n);

    // ======================================================================
    // Part 4: Identity Matrix
    // ======================================================================
    std::cout << "\n=== Part 4: Identity Matrix (Zero Storage) ===\n\n";

    mat::identity2D<double> I(n);
    std::cout << "identity2D(" << n << "):\n";
    print_matrix("I", I, n, n);

    // Verify: I(i,i) = 1, I(i,j) = 0 for i != j
    bool id_ok = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (I(i, j) != expected) id_ok = false;
        }
    std::cout << "\nIdentity check: " << (id_ok ? "PASS" : "FAIL")
              << " (0 bytes of storage!)\n\n";

    // ======================================================================
    // Part 5: Solve with Hermitian View
    // ======================================================================
    std::cout << "=== Part 5: CG on Hermitian View ===\n";
    std::cout << "The hermitian view satisfies the Matrix concept,\n";
    std::cout << "so it works directly with iterative solvers.\n\n";

    vec::dense_vector<double> b_h(n);
    for (std::size_t i = 0; i < n; ++i) b_h(i) = 1.0 + i;

    vec::dense_vector<double> x_h(n, 0.0);
    itl::pc::identity<decltype(H)> pc_h(H);
    itl::noisy_iteration<double> iter_h(b_h, 200, 1.0e-10);
    int info = itl::cg(H, x_h, b_h, pc_h, iter_h);
    std::cout << "\nCG on hermitian view: " << iter_h.iterations()
              << " iterations, code = " << info << "\n";

    // Verify solution
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Hx_i = 0.0;
        for (std::size_t j = 0; j < n; ++j) Hx_i += H(i, j) * x_h(j);
        res += (b_h(i) - Hx_i) * (b_h(i) - Hx_i);
    }
    std::cout << "Residual ||b - Hx||_2 = " << std::scientific << std::sqrt(res) << "\n\n";

    // ======================================================================
    // Part 6: Matrix Market I/O
    // ======================================================================
    std::cout << "=== Part 6: Matrix Market I/O ===\n\n";

    // Write a small dense matrix
    mat::dense2D<double> W(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            W(i, j) = H(i, j);

    std::string fname = (std::filesystem::temp_directory_path() / "mtl5_hermitian_block.mtx").string();
    io::mm_write(fname, W, "4x4 block from hermitian view");
    std::cout << "Written " << fname << "\n";

    auto W_read = io::mm_read_dense<double>(fname);
    bool mm_ok = true;
    for (std::size_t i = 0; i < 4 && mm_ok; ++i)
        for (std::size_t j = 0; j < 4 && mm_ok; ++j)
            if (std::abs(W(i,j) - W_read(i,j)) > 1e-12)
                mm_ok = false;
    std::cout << "Round-trip check: " << (mm_ok ? "PASS" : "FAIL") << "\n\n";
    std::remove(fname.c_str());

    // -- Commentary -------------------------------------------------------
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. Views are zero-copy wrappers that reinterpret existing data.\n";
    std::cout << "   No memory allocation, no data movement.\n";
    std::cout << "2. Hermitian view: store only upper (or lower) triangle,\n";
    std::cout << "   saving ~50% storage for symmetric matrices.\n";
    std::cout << "3. Banded view: enforce band structure, useful for extracting\n";
    std::cout << "   preconditioners (tridiagonal from pentadiagonal, etc.).\n";
    std::cout << "4. Map view: submatrices and permutations without copies.\n";
    std::cout << "   Useful for domain decomposition and reordering.\n";
    std::cout << "5. Identity matrix: O(1) storage, O(1) element access.\n";
    std::cout << "   Used in Kronecker products and as default preconditioner.\n";
    std::cout << "6. All views satisfy the Matrix concept, so they work with\n";
    std::cout << "   solvers, norms, and other operations transparently.\n";

    return EXIT_SUCCESS;
}
