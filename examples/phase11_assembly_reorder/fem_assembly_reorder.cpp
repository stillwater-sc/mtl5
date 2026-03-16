// fem_assembly_reorder.cpp -- FEM Assembly & Matrix Reordering
//
// This example demonstrates:
//   1. shifted_inserter for FEM-style assembly of local element matrices
//   2. Overlapping element contributions with update_plus accumulation
//   3. Permutation-based matrix reordering (Cuthill-McKee bandwidth reduction)
//   4. Symmetric reorder PAP^T preserving eigenvalues (trace check)
//   5. Row/column reorder for non-symmetric systems
//
// Key insight: shifted_inserter lets you write element assembly loops that
// don't know their global position -- offsets are set externally, enabling
// clean separation between local element physics and global mesh topology.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace mtl;

void print_matrix(const std::string& name, const mat::dense2D<double>& M) {
    std::cout << name << " (" << M.num_rows() << "x" << M.num_cols() << "):\n";
    for (std::size_t i = 0; i < M.num_rows(); ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < M.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << std::setw(7) << M(i, j);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void print_sparse(const std::string& name, const mat::compressed2D<double>& M) {
    std::cout << name << " (" << M.num_rows() << "x" << M.num_cols() << "):\n";
    for (std::size_t i = 0; i < M.num_rows(); ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < M.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << std::setw(7) << M(i, j);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void print_vector(const std::string& name, const vec::dense_vector<double>& v) {
    std::cout << "  " << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << v(i);
    }
    std::cout << "]\n";
}

/// Compute the bandwidth of a dense matrix (max |i-j| where A(i,j) != 0).
std::size_t bandwidth(const mat::dense2D<double>& A) {
    std::size_t bw = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            if (A(i, j) != 0.0)
                bw = std::max(bw, static_cast<std::size_t>(
                    std::abs(static_cast<long>(i) - static_cast<long>(j))));
    return bw;
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 11B: FEM Assembly & Matrix Reordering\n";
    std::cout << "=============================================================\n\n";

    // ======================================================================
    // Part 1: FEM Assembly with shifted_inserter
    // ======================================================================
    std::cout << "=== Part 1: 1D FEM Assembly with shifted_inserter ===\n\n";

    std::cout << "  Consider a 1D mesh with 4 elements and 5 nodes:\n\n";
    std::cout << "    Node:    0-----1-----2-----3-----4\n";
    std::cout << "    Element:   [0]   [1]   [2]   [3]\n\n";

    std::cout << "  Each element contributes a 2x2 local stiffness matrix:\n";
    std::cout << "    k_e = [ 1, -1]\n";
    std::cout << "          [-1,  1]\n\n";
    std::cout << "  shifted_inserter handles global placement automatically.\n\n";

    const std::size_t n_nodes = 5;
    const std::size_t n_elem = 4;

    mat::compressed2D<double> K(n_nodes, n_nodes);
    {
        // Use update_plus to accumulate overlapping contributions
        mat::shifted_inserter<
            mat::inserter<mat::compressed2D<double>, mat::update_plus<double>>
        > ins(K, 4, 0, 0);

        for (std::size_t e = 0; e < n_elem; ++e) {
            // Element e connects nodes e and e+1
            // Set offset so local indices (0,1) map to global (e, e+1)
            ins.set_row_offset(e);
            ins.set_col_offset(e);

            // Insert 2x2 local stiffness
            ins[0][0] << 1.0;
            ins[0][1] << -1.0;
            ins[1][0] << -1.0;
            ins[1][1] << 1.0;
        }
        // Destructor finalizes the sparse matrix
    }

    print_sparse("K (assembled global stiffness)", K);

    std::cout << "  The tridiagonal pattern emerges naturally from element\n";
    std::cout << "  overlap: nodes shared between elements get accumulated.\n";
    std::cout << "  Interior nodes have K(i,i)=2 (two element contributions).\n\n";

    // Verify structure: K should be tridiagonal with 1,2,...,2,1 diagonal
    std::cout << "  Diagonal: [";
    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << K(i, i);
    }
    std::cout << "]\n";
    std::cout << "  Expected: [1, 2, 2, 2, 1]\n\n";

    // ======================================================================
    // Part 2: 2D FEM with Non-Sequential Numbering
    // ======================================================================
    std::cout << "=== Part 2: 2D Mesh Element Assembly ===\n\n";

    std::cout << "  Consider a small 2D mesh with 6 nodes and 2 triangular\n";
    std::cout << "  elements. Node numbering may not be sequential:\n\n";
    std::cout << "    2---5       Element 0: nodes {0, 1, 2}\n";
    std::cout << "    |\\  |       Element 1: nodes {1, 5, 2}\n";
    std::cout << "    | \\ |  \n";
    std::cout << "    0---1  \n\n";

    // Element connectivity (global node indices for each element)
    std::size_t connectivity[][3] = {{0, 1, 2}, {1, 5, 2}};
    const std::size_t n_dof = 6;

    // 3x3 local stiffness for a reference triangle
    // (simplified: k_e = ones(3,3) + 2*eye(3))
    double ke[3][3] = {
        { 3.0, 1.0, 1.0},
        { 1.0, 3.0, 1.0},
        { 1.0, 1.0, 3.0}
    };

    mat::dense2D<double> K2(n_dof, n_dof);
    for (std::size_t i = 0; i < n_dof; ++i)
        for (std::size_t j = 0; j < n_dof; ++j)
            K2(i, j) = 0.0;

    // Manual assembly using direct index mapping
    for (std::size_t e = 0; e < 2; ++e) {
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                K2(connectivity[e][i], connectivity[e][j]) += ke[i][j];
    }

    print_matrix("K2 (assembled from 2 triangular elements)", K2);

    std::cout << "  Nodes 1, 2 are shared between elements => their entries\n";
    std::cout << "  accumulate contributions from both elements.\n";
    std::cout << "  Nodes 3, 4 are not part of any element => zero rows/cols.\n\n";

    // ======================================================================
    // Part 3: Bandwidth and Reordering Motivation
    // ======================================================================
    std::cout << "=== Part 3: Matrix Reordering for Bandwidth Reduction ===\n\n";

    std::cout << "  Poor node numbering leads to large bandwidth, which\n";
    std::cout << "  means more fill-in during factorization.\n\n";

    // Create a matrix with intentionally bad numbering
    // This simulates a mesh where adjacent nodes have distant indices
    mat::dense2D<double> A_bad = {
        { 4, 0, 0,-1, 0,-1},
        { 0, 4,-1, 0,-1, 0},
        { 0,-1, 4,-1, 0, 0},
        {-1, 0,-1, 4, 0, 0},
        { 0,-1, 0, 0, 4,-1},
        {-1, 0, 0, 0,-1, 4}
    };

    std::cout << "  Original bandwidth: " << bandwidth(A_bad) << "\n";
    print_matrix("A (bad numbering)", A_bad);

    // Apply a permutation that groups connected nodes together
    // (simulating a Cuthill-McKee-like reordering)
    std::vector<std::size_t> perm = {0, 3, 2, 5, 4, 1};
    std::cout << "  Applying permutation: [";
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << perm[i];
    }
    std::cout << "]\n";
    std::cout << "  (new node i was old node perm[i])\n\n";

    auto A_good = mat::reorder(A_bad, perm);
    std::cout << "  Reordered bandwidth: " << bandwidth(A_good) << "\n";
    print_matrix("PAP^T (reordered)", A_good);

    std::cout << "  The non-zeros are now closer to the diagonal.\n";
    std::cout << "  Less fill-in during Cholesky or LU factorization!\n\n";

    // ======================================================================
    // Part 4: Trace Preservation (Eigenvalue Invariant)
    // ======================================================================
    std::cout << "=== Part 4: Symmetric Reorder Preserves Trace ===\n\n";

    std::cout << "  The symmetric reorder B = PAP^T is a similarity transform.\n";
    std::cout << "  It preserves eigenvalues, and therefore the trace.\n\n";

    double trace_orig = 0.0, trace_reord = 0.0;
    for (std::size_t i = 0; i < A_bad.num_rows(); ++i) {
        trace_orig += A_bad(i, i);
        trace_reord += A_good(i, i);
    }

    std::cout << "  trace(A)     = " << std::fixed << std::setprecision(1) << trace_orig << "\n";
    std::cout << "  trace(PAP^T) = " << std::fixed << std::setprecision(1) << trace_reord << "\n";
    std::cout << "  Difference:    " << std::scientific << std::abs(trace_orig - trace_reord) << "\n\n";

    // Verify via Frobenius norm preservation too
    double frob_orig = 0.0, frob_reord = 0.0;
    for (std::size_t i = 0; i < A_bad.num_rows(); ++i)
        for (std::size_t j = 0; j < A_bad.num_cols(); ++j) {
            frob_orig += A_bad(i, j) * A_bad(i, j);
            frob_reord += A_good(i, j) * A_good(i, j);
        }

    std::cout << "  ||A||_F     = " << std::fixed << std::setprecision(4) << std::sqrt(frob_orig) << "\n";
    std::cout << "  ||PAP^T||_F = " << std::fixed << std::setprecision(4) << std::sqrt(frob_reord) << "\n";
    std::cout << "  Frobenius norm is also preserved (unitary similarity).\n\n";

    // ======================================================================
    // Part 5: Row/Column Reorder and P * A
    // ======================================================================
    std::cout << "=== Part 5: Row vs Column vs Symmetric Reorder ===\n\n";

    mat::dense2D<double> M = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    print_matrix("M", M);

    std::vector<std::size_t> p = {2, 0, 1};  // cyclic shift
    std::cout << "  Permutation: [2, 0, 1] (cyclic shift)\n\n";

    auto M_rows = mat::reorder_rows(M, p);
    print_matrix("reorder_rows(M, p): row i <- row p[i]", M_rows);

    auto M_cols = mat::reorder_cols(M, p);
    print_matrix("reorder_cols(M, p): col j <- col p[j]", M_cols);

    auto M_sym = mat::reorder(M, p);
    print_matrix("reorder(M, p): B(i,j) = M(p[i],p[j])", M_sym);

    // Show that P * M gives same result as reorder_rows
    permutation_matrix<double> P(p);
    auto PM = P * M;

    std::cout << "  Verifying P * M == reorder_rows(M, p):\n";
    double diff = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            diff += std::abs(PM(i, j) - M_rows(i, j));
    std::cout << "  Total absolute difference: " << std::scientific << diff << "\n\n";

    // -- Takeaways --------------------------------------------------------
    std::cout << "=== Takeaways ===\n\n";
    std::cout << "  1. shifted_inserter decouples local element indexing from global\n";
    std::cout << "  2. update_plus accumulates overlapping element contributions\n";
    std::cout << "  3. set_row_offset/set_col_offset can change between elements\n";
    std::cout << "  4. reorder(A, perm) computes PAP^T -- a similarity transform\n";
    std::cout << "  5. Similarity preserves eigenvalues, trace, and Frobenius norm\n";
    std::cout << "  6. Good numbering reduces bandwidth => less fill-in => faster solves\n";
    std::cout << "  7. reorder_rows/reorder_cols handle non-symmetric cases\n";

    return EXIT_SUCCESS;
}
