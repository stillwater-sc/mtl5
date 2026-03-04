// phase7a_sparse_formats.cpp — Sparse Format Shootout: COO vs CRS vs ELL
//
// This example demonstrates:
//   1. Three sparse matrix formats and their storage tradeoffs
//   2. COO assembly → sort → compress to CRS pipeline
//   3. CRS assembly via the inserter pattern
//   4. ELL conversion from CRS for GPU-friendly layout
//   5. Matrix Market round-trip I/O
//
// We build a 2D Laplacian (5-point stencil) on a 6x6 grid (36 unknowns)
// in all three formats and compare their storage requirements.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 7A: Sparse Format Shootout — COO vs CRS vs ELL\n";
    std::cout << "=============================================================\n\n";

    const std::size_t grid = 6;   // 6x6 interior grid
    const std::size_t N = grid * grid;  // 36 unknowns

    std::cout << "Problem: 2D Laplacian on " << grid << "x" << grid
              << " grid = " << N << " unknowns\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Format 1: COO (Coordinate) — best for unstructured assembly
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Format 1: COO (Coordinate / Triplet) ===\n";
    std::cout << "Storage: 3 arrays (row, col, val) x nnz entries.\n";
    std::cout << "Best for: assembly phase, when structure is unknown.\n\n";

    mat::coordinate2D<double> coo(N, N);
    coo.reserve(5 * N);  // estimate: ~5 nnz per row

    for (std::size_t iy = 0; iy < grid; ++iy) {
        for (std::size_t ix = 0; ix < grid; ++ix) {
            std::size_t row = ix + iy * grid;
            coo.insert(row, row, 4.0);  // center
            if (ix > 0)         coo.insert(row, row - 1, -1.0);     // left
            if (ix + 1 < grid)  coo.insert(row, row + 1, -1.0);     // right
            if (iy > 0)         coo.insert(row, row - grid, -1.0);   // down
            if (iy + 1 < grid)  coo.insert(row, row + grid, -1.0);   // up
        }
    }

    std::cout << "COO entries: " << coo.nnz() << "\n";
    std::cout << "COO storage: 3 x " << coo.nnz() << " = "
              << 3 * coo.nnz() << " values (row + col + data)\n";

    // Sort and compress to CRS
    coo.sort();
    auto crs_from_coo = coo.compress();
    std::cout << "After compress(): CRS with " << crs_from_coo.nnz() << " nnz\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Format 2: CRS (Compressed Row Storage) — best for computation
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Format 2: CRS (Compressed Row Storage) ===\n";
    std::cout << "Storage: data[nnz] + col_idx[nnz] + row_ptr[nrows+1].\n";
    std::cout << "Best for: matrix-vector products, iterative solvers.\n\n";

    mat::compressed2D<double> crs(N, N);
    {
        mat::inserter<mat::compressed2D<double>> ins(crs, 5);
        for (std::size_t iy = 0; iy < grid; ++iy) {
            for (std::size_t ix = 0; ix < grid; ++ix) {
                std::size_t row = ix + iy * grid;
                ins[row][row] << 4.0;
                if (ix > 0)         ins[row][row - 1] << -1.0;
                if (ix + 1 < grid)  ins[row][row + 1] << -1.0;
                if (iy > 0)         ins[row][row - grid] << -1.0;
                if (iy + 1 < grid)  ins[row][row + grid] << -1.0;
            }
        }
    }

    std::cout << "CRS nnz: " << crs.nnz() << "\n";
    std::cout << "CRS storage: " << crs.nnz() << " (data) + " << crs.nnz()
              << " (col_idx) + " << N + 1 << " (row_ptr) = "
              << 2 * crs.nnz() + N + 1 << " values\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Format 3: ELL (ELLPACK) — best for GPU / uniform sparsity
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Format 3: ELL (ELLPACK) ===\n";
    std::cout << "Storage: indices[nrows*width] + data[nrows*width].\n";
    std::cout << "Best for: GPU kernels, matrices with uniform row widths.\n\n";

    mat::ell_matrix<double> ell(crs);  // convert from CRS

    std::cout << "ELL max_width: " << ell.max_width() << " (max nnz per row)\n";
    std::cout << "ELL storage: 2 x " << N << " x " << ell.max_width()
              << " = " << 2 * N * ell.max_width() << " values\n";
    std::cout << "  (includes " << N * ell.max_width() - crs.nnz()
              << " padding entries for short rows)\n\n";

    // ── Verify all three formats agree ───────────────────────────────────
    std::cout << "=== Verification: All formats give identical results ===\n";

    bool all_match = true;
    for (std::size_t i = 0; i < N && all_match; ++i) {
        for (std::size_t j = 0; j < N && all_match; ++j) {
            double v_coo = crs_from_coo(i, j);
            double v_crs = crs(i, j);
            double v_ell = ell(i, j);
            if (std::abs(v_coo - v_crs) > 1e-14 || std::abs(v_crs - v_ell) > 1e-14) {
                std::cout << "MISMATCH at (" << i << "," << j << "): "
                          << v_coo << " vs " << v_crs << " vs " << v_ell << "\n";
                all_match = false;
            }
        }
    }
    std::cout << (all_match ? "All entries match across COO, CRS, and ELL.\n\n"
                            : "ERROR: Format mismatch detected!\n\n");

    // ── Storage comparison table ─────────────────────────────────────────
    std::cout << "=== Storage Comparison ===\n";
    std::size_t nnz = crs.nnz();
    std::cout << std::setw(10) << "Format"
              << std::setw(12) << "Values"
              << std::setw(12) << "Overhead"
              << std::setw(10) << "Total"
              << std::setw(12) << "vs Dense" << "\n";
    std::cout << std::string(56, '-') << "\n";
    std::cout << std::setw(10) << "Dense"
              << std::setw(12) << N*N << std::setw(12) << 0
              << std::setw(10) << N*N << std::setw(12) << "100%\n";
    std::cout << std::setw(10) << "COO"
              << std::setw(12) << nnz << std::setw(12) << 2*nnz
              << std::setw(10) << 3*nnz
              << std::setw(11) << std::fixed << std::setprecision(1)
              << 100.0*3*nnz/(N*N) << "%\n";
    std::cout << std::setw(10) << "CRS"
              << std::setw(12) << nnz << std::setw(12) << nnz+N+1
              << std::setw(10) << 2*nnz+N+1
              << std::setw(11) << 100.0*(2*nnz+N+1)/(N*N) << "%\n";
    std::size_t ell_total = 2 * N * ell.max_width();
    std::cout << std::setw(10) << "ELL"
              << std::setw(12) << N*ell.max_width()
              << std::setw(12) << N*ell.max_width()
              << std::setw(10) << ell_total
              << std::setw(11) << 100.0*ell_total/(N*N) << "%\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Matrix Market Round-Trip I/O
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Matrix Market I/O Round-Trip ===\n\n";

    // Write sparse (CRS → coordinate .mtx)
    std::string sparse_file = (std::filesystem::temp_directory_path() / "mtl5_laplacian_sparse.mtx").string();
    io::mm_write_sparse(sparse_file, crs, "2D Laplacian 6x6 grid");
    std::cout << "Written: " << sparse_file << " (coordinate format)\n";

    // Read back
    auto crs_read = io::mm_read<double>(sparse_file);
    std::cout << "Read back: " << crs_read.num_rows() << "x" << crs_read.num_cols()
              << ", nnz=" << crs_read.nnz() << "\n";

    // Verify round-trip
    bool rt_ok = true;
    for (std::size_t i = 0; i < N && rt_ok; ++i)
        for (std::size_t j = 0; j < N && rt_ok; ++j)
            if (std::abs(crs(i,j) - crs_read(i,j)) > 1e-12)
                rt_ok = false;
    std::cout << "Sparse round-trip: " << (rt_ok ? "PASS" : "FAIL") << "\n\n";

    // Write dense (small submatrix for demonstration)
    mat::dense2D<double> small(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            small(i, j) = crs(i, j);

    std::string dense_file = (std::filesystem::temp_directory_path() / "mtl5_laplacian_dense.mtx").string();
    io::mm_write(dense_file, small, "Top-left 4x4 block");
    std::cout << "Written: " << dense_file << " (array format)\n";

    auto dense_read = io::mm_read_dense<double>(dense_file);
    bool rt_dense_ok = true;
    for (std::size_t i = 0; i < 4 && rt_dense_ok; ++i)
        for (std::size_t j = 0; j < 4 && rt_dense_ok; ++j)
            if (std::abs(small(i,j) - dense_read(i,j)) > 1e-12)
                rt_dense_ok = false;
    std::cout << "Dense round-trip:  " << (rt_dense_ok ? "PASS" : "FAIL") << "\n\n";

    // Clean up temp files
    std::remove(sparse_file.c_str());
    std::remove(dense_file.c_str());

    // ── Commentary ───────────────────────────────────────────────────────
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. COO: simplest format, good for assembly. O(nnz) for insert,\n";
    std::cout << "   but O(nnz) for element lookup. Sort + compress → CRS.\n";
    std::cout << "2. CRS: the workhorse format. O(log(nnz/row)) element access,\n";
    std::cout << "   O(nnz) SpMV, minimal storage overhead.\n";
    std::cout << "3. ELL: GPU-friendly due to coalesced memory access patterns.\n";
    std::cout << "   Wastes space when row lengths vary widely.\n";
    std::cout << "4. Pipeline: COO (assembly) → CRS (compute) → ELL (GPU).\n";
    std::cout << "5. Matrix Market is the standard exchange format for sparse\n";
    std::cout << "   matrices (used by SuiteSparse, NIST, etc.).\n";

    return EXIT_SUCCESS;
}
