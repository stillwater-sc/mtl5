// permutation_block_diagonal.cpp -- Permutation Matrices & Block Diagonal
//
// This example demonstrates:
//   1. Permutation matrix construction and efficient O(n) matvec
//   2. Row swapping for LU pivoting
//   3. Permutation inverse and the identity P^{-1} * P = I
//   4. Block diagonal matrices for decoupled systems
//   5. Block diagonal as a preconditioner concept
//
// Key insight: permutation_matrix and block_diagonal2D store NO matrix
// entries -- they use implicit structure for O(n) operations instead of
// the O(n^2) generic matvec.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

void print_vector(const std::string& name, const vec::dense_vector<double>& v) {
    std::cout << "  " << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << v(i);
    }
    std::cout << "]\n";
}

void print_perm_matrix(const std::string& name, const permutation_matrix<double>& P) {
    std::cout << name << " (" << P.num_rows() << "x" << P.num_cols() << "):\n";
    for (std::size_t i = 0; i < P.num_rows(); ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < P.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::setw(2) << static_cast<int>(P(i, j));
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void print_dense(const std::string& name, const mat::dense2D<double>& M) {
    std::cout << name << ":\n";
    for (std::size_t i = 0; i < M.num_rows(); ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < M.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << M(i, j);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 10B: Permutation Matrices & Block Diagonal\n";
    std::cout << "=============================================================\n\n";

    // ======================================================================
    // Part 1: Permutation Matrix Basics
    // ======================================================================
    std::cout << "=== Part 1: Permutation Matrix Basics ===\n\n";

    // A permutation matrix stores only a vector perm[] where
    // P(i,j) = 1 if perm[i] == j.  Storage: O(n), not O(n^2).
    std::cout << "  A permutation matrix P stores only an index vector.\n";
    std::cout << "  perm = [2, 0, 3, 1] means:\n";
    std::cout << "    row 0 -> col 2,  row 1 -> col 0,\n";
    std::cout << "    row 2 -> col 3,  row 3 -> col 1\n\n";

    permutation_matrix<double> P(std::vector<std::size_t>{2, 0, 3, 1});
    print_perm_matrix("P", P);

    // P * x is O(n): just reorder elements
    vec::dense_vector<double> x = {10.0, 20.0, 30.0, 40.0};
    auto y = P * x;

    print_vector("x      ", x);
    print_vector("P * x  ", y);
    std::cout << "  y[i] = x[perm[i]], so y = [x[2], x[0], x[3], x[1]]\n\n";

    // ======================================================================
    // Part 2: Inverse Permutation
    // ======================================================================
    std::cout << "=== Part 2: Inverse Permutation (P^{-1} = P^T) ===\n\n";

    auto Pinv = P.inverse();
    print_perm_matrix("P^{-1}", Pinv);

    // P^{-1} * P * x should give back x
    auto roundtrip = Pinv * (P * x);
    print_vector("x              ", x);
    print_vector("P^{-1} * P * x ", roundtrip);

    double err = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        err += std::abs(x(i) - roundtrip(i));
    std::cout << "  Round-trip error: " << std::scientific << err << "\n\n";

    // ======================================================================
    // Part 3: Row Swapping for LU Pivoting
    // ======================================================================
    std::cout << "=== Part 3: Building Permutation via Row Swaps ===\n\n";

    std::cout << "  LU with partial pivoting builds P from successive row swaps.\n";
    std::cout << "  Starting from identity, apply swap_rows() at each pivot step.\n\n";

    // Simulate 3-step pivoting
    permutation_matrix<double> P_lu(4);
    std::cout << "  Step 0: swap rows 0 and 2 (pivot row for column 0)\n";
    P_lu.swap_rows(0, 2);
    std::cout << "  Step 1: swap rows 1 and 3 (pivot row for column 1)\n";
    P_lu.swap_rows(1, 3);
    std::cout << "  Step 2: no swap needed (pivot already in place)\n\n";

    print_perm_matrix("P_lu (accumulated pivots)", P_lu);

    // Apply to a vector
    vec::dense_vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto Pv = P_lu * v;
    print_vector("v       ", v);
    print_vector("P_lu * v", Pv);
    std::cout << "\n";

    // ======================================================================
    // Part 4: Block Diagonal Matrix
    // ======================================================================
    std::cout << "=== Part 4: Block Diagonal Matrix ===\n\n";

    std::cout << "  A block diagonal matrix stores independent diagonal blocks.\n";
    std::cout << "  Off-diagonal blocks are implicitly zero.\n";
    std::cout << "  Matvec is O(sum of block sizes), not O(n^2).\n\n";

    // Build two blocks representing decoupled subsystems:
    // Block 1: 2x2 rotation-like matrix
    mat::dense2D<double> B1(2, 2);
    B1(0, 0) =  0.8; B1(0, 1) = -0.6;
    B1(1, 0) =  0.6; B1(1, 1) =  0.8;

    // Block 2: 3x3 tridiagonal SPD block
    mat::dense2D<double> B2(3, 3);
    B2(0, 0) =  2.0; B2(0, 1) = -1.0; B2(0, 2) =  0.0;
    B2(1, 0) = -1.0; B2(1, 1) =  2.0; B2(1, 2) = -1.0;
    B2(2, 0) =  0.0; B2(2, 1) = -1.0; B2(2, 2) =  2.0;

    block_diagonal2D<double> BD({B1, B2});

    std::cout << "  Block 1 (2x2 rotation):\n";
    print_dense("    B1", B1);
    std::cout << "  Block 2 (3x3 tridiagonal SPD):\n";
    print_dense("    B2", B2);

    std::cout << "  Full block diagonal matrix BD (" << BD.num_rows()
              << "x" << BD.num_cols() << "):\n";
    // Print the full implicit matrix
    for (std::size_t i = 0; i < BD.num_rows(); ++i) {
        std::cout << "    [";
        for (std::size_t j = 0; j < BD.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(1) << std::setw(5) << BD(i, j);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    // Block diagonal matvec
    vec::dense_vector<double> w = {1.0, 0.0, 1.0, 1.0, 1.0};
    auto Bw = BD * w;

    print_vector("w     ", w);
    print_vector("BD * w", Bw);
    std::cout << "\n";

    std::cout << "  Block 1 applied to [1, 0]: [0.8*1 + (-0.6)*0, 0.6*1 + 0.8*0] = [0.8, 0.6]\n";
    std::cout << "  Block 2 applied to [1, 1, 1]: [2-1, -1+2-1, -1+2] = [1, 0, 1]\n\n";

    // ======================================================================
    // Part 5: Block Diagonal as Preconditioner Concept
    // ======================================================================
    std::cout << "=== Part 5: Block Diagonal Preconditioning Concept ===\n\n";

    std::cout << "  For a matrix with block structure:\n";
    std::cout << "\n";
    std::cout << "      [A11  A12]             [A11   0 ]^{-1}\n";
    std::cout << "  A = [        ],   M^{-1} = [        ]\n";
    std::cout << "      [A21  A22]             [ 0   A22]\n";
    std::cout << "\n";
    std::cout << "  The block diagonal preconditioner inverts each block\n";
    std::cout << "  independently. Cost: sum of O(block_size^3) instead of O(n^3).\n\n";

    // Demonstrate: solve BD * x = b by inverting each block
    // For the rotation block, inv = transpose (it's orthogonal-ish)
    // For the tridiagonal, use LU

    vec::dense_vector<double> rhs = {0.8, 0.6, 1.0, 0.0, 1.0};
    std::cout << "  Solving BD * x = b with block-independent solves:\n";
    print_vector("b", rhs);

    // Solve block 1: B1 * x1 = [0.8, 0.6]
    mat::dense2D<double> B1_copy(2, 2);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            B1_copy(i, j) = B1(i, j);

    vec::dense_vector<double> b1(2), x1(2);
    b1(0) = rhs(0); b1(1) = rhs(1);
    std::vector<std::size_t> piv;
    lu_factor(B1_copy, piv);
    lu_solve(B1_copy, piv, x1, b1);

    // Solve block 2: B2 * x2 = [1, 0, 1]
    mat::dense2D<double> B2_copy(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            B2_copy(i, j) = B2(i, j);

    vec::dense_vector<double> b2(3), x2(3);
    b2(0) = rhs(2); b2(1) = rhs(3); b2(2) = rhs(4);
    std::vector<std::size_t> piv2;
    lu_factor(B2_copy, piv2);
    lu_solve(B2_copy, piv2, x2, b2);

    // Combine
    vec::dense_vector<double> x_sol(5);
    x_sol(0) = x1(0); x_sol(1) = x1(1);
    x_sol(2) = x2(0); x_sol(3) = x2(1); x_sol(4) = x2(2);
    print_vector("x", x_sol);

    // Verify: BD * x should equal rhs
    auto check = BD * x_sol;
    print_vector("BD * x (verify)", check);

    double resid = 0.0;
    for (std::size_t i = 0; i < 5; ++i)
        resid += (check(i) - rhs(i)) * (check(i) - rhs(i));
    std::cout << "  ||b - BD*x|| = " << std::scientific << std::sqrt(resid) << "\n\n";

    // -- Takeaways --------------------------------------------------------
    std::cout << "=== Takeaways ===\n\n";
    std::cout << "  1. permutation_matrix: O(n) storage, O(n) matvec (index remap)\n";
    std::cout << "  2. P.inverse() = P^T (orthogonal), P^{-1}*P*x = x always\n";
    std::cout << "  3. swap_rows() builds pivots incrementally (LU partial pivoting)\n";
    std::cout << "  4. block_diagonal2D: stores blocks, off-diag is implicitly zero\n";
    std::cout << "  5. Block diagonal solves decouple into independent small solves\n";
    std::cout << "  6. Both types satisfy the Matrix concept -- usable everywhere\n";

    return EXIT_SUCCESS;
}
