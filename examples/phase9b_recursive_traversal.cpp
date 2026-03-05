// phase9b_recursive_traversal.cpp — Block-Recursive Matrix Traversal
//
// This example demonstrates:
//   1. The recursator: a lightweight handle for quad-tree matrix subdivision
//   2. How recursive subdivision works on power-of-2 and non-power-of-2 matrices
//   3. for_each traversal with configurable base-case thresholds
//   4. Why block-recursive algorithms improve cache locality
//
// Key insight: Cache-oblivious algorithms recursively subdivide a matrix into
// quadrants until reaching a base-case size that fits in cache. The recursator
// provides this subdivision without copying data — it simply tracks offsets
// into the original matrix.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace mtl;
using namespace mtl::recursion;

/// Print a recursator's region information
template <typename Matrix>
void print_region(const std::string& label, const recursator<Matrix>& rec) {
    std::cout << "  " << std::setw(12) << label
              << ": offset=(" << rec.row_offset() << "," << rec.col_offset()
              << "), size=" << rec.num_rows() << "x" << rec.num_cols()
              << (rec.is_empty() ? " [EMPTY]" : "") << "\n";
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 9B: Block-Recursive Matrix Traversal\n";
    std::cout << "=============================================================\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 1. Basic Recursator — Power-of-2 Matrix
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 1. Quad-Tree Subdivision (8x8 matrix) ===\n\n";

    mat::dense2D<double> A(8, 8);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            A(i, j) = static_cast<double>(i * 10 + j);

    recursator<mat::dense2D<double>> rec(A);
    std::cout << "Original matrix: " << rec.num_rows() << "x"
              << rec.num_cols() << "\n\n";

    std::cout << "Level 1 — split into quadrants:\n";
    print_region("north_west", rec.north_west());
    print_region("north_east", rec.north_east());
    print_region("south_west", rec.south_west());
    print_region("south_east", rec.south_east());

    std::cout << "\nLevel 2 — split NW further:\n";
    auto nw = rec.north_west();
    print_region("NW.nw", nw.north_west());
    print_region("NW.ne", nw.north_east());
    print_region("NW.sw", nw.south_west());
    print_region("NW.se", nw.south_east());
    std::cout << "\n";

    // Verify element access through recursator
    std::cout << "Element verification:\n";
    auto se = rec.south_east();
    std::cout << "  A(6,6) = " << A(6, 6) << "\n";
    std::cout << "  SE(2,2) = " << se(2, 2) << "  (same element via recursator)\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 2. Non-Power-of-2 Matrix
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 2. Non-Power-of-2 Subdivision (6x10 matrix) ===\n\n";

    mat::dense2D<double> B(6, 10);
    recursator<mat::dense2D<double>> recB(B);

    std::cout << "Original: " << recB.num_rows() << "x" << recB.num_cols() << "\n";
    std::cout << "  Row split point: first_part(6) = " << first_part(6)
              << " (largest power of 2 <= 6)\n";
    std::cout << "  Col split point: first_part(10) = " << first_part(10)
              << " (largest power of 2 <= 10)\n\n";

    print_region("north_west", recB.north_west());
    print_region("north_east", recB.north_east());
    print_region("south_west", recB.south_west());
    print_region("south_east", recB.south_east());

    // Verify coverage
    auto bnw = recB.north_west();
    auto bne = recB.north_east();
    auto bsw = recB.south_west();
    std::cout << "\n  Coverage: " << bnw.num_rows() << "+" << bsw.num_rows()
              << " = " << (bnw.num_rows() + bsw.num_rows()) << " rows, "
              << bnw.num_cols() << "+" << bne.num_cols()
              << " = " << (bnw.num_cols() + bne.num_cols()) << " cols\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 3. for_each Traversal with Base Case Tests
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 3. Recursive Traversal with for_each ===\n\n";

    // Initialize matrix to zeros
    mat::dense2D<double> C(8, 8);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            C(i, j) = 0.0;

    // Different base-case thresholds
    for (std::size_t threshold : {4, 2, 1}) {
        std::size_t base_count = 0;
        for_each(recursator<mat::dense2D<double>>(C),
            [&](auto& sub) {
                ++base_count;
            },
            min_dim_test(threshold));

        std::cout << "  min_dim_test(" << threshold << "): "
                  << base_count << " base cases"
                  << " (each ~" << threshold << "x" << threshold << ")\n";
    }
    std::cout << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // 4. Block-Recursive Algorithm: Set Diagonal via Recursion
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 4. Block-Recursive Diagonal Setting ===\n\n";

    // Zero out C and set diagonal via recursive traversal
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            C(i, j) = 0.0;

    // Use recursion to set diagonal elements within each base case
    for_each(recursator<mat::dense2D<double>>(C),
        [](auto& sub) {
            auto rows = sub.num_rows();
            auto cols = sub.num_cols();
            auto diag = std::min(rows, cols);
            for (std::size_t i = 0; i < diag; ++i) {
                // Only set diagonal if this block is on the diagonal
                if (sub.row_offset() + i == sub.col_offset() + i) {
                    sub(i, i) = 1.0;
                }
            }
        },
        min_dim_test(2));

    std::cout << "Result (should be identity-like on diagonal):\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < 8; ++j) {
            std::cout << std::setw(4) << static_cast<int>(C(i, j));
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // 5. Cache Locality Explanation
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 5. Why Block-Recursive Algorithms Win ===\n\n";

    std::cout << "Traditional row-by-row traversal of an NxN matrix:\n";
    std::cout << "  - Touches N elements per row before moving to next row\n";
    std::cout << "  - For large N, each row access causes cache misses\n";
    std::cout << "  - Working set: O(N) cache lines\n\n";

    std::cout << "Block-recursive traversal:\n";
    std::cout << "  - Recursively divides matrix into quadrants\n";
    std::cout << "  - Base case processes a small BxB block that fits in cache\n";
    std::cout << "  - Working set: O(B^2) << O(N) for B << N\n";
    std::cout << "  - 'Cache-oblivious': adapts to any cache size without tuning\n\n";

    std::cout << "The recursator enables this by tracking (offset, size) pairs\n";
    std::cout << "without copying matrix data. Subdivision is O(1) — just pointer\n";
    std::cout << "arithmetic on the original matrix storage.\n";

    return EXIT_SUCCESS;
}
