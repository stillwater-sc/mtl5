// strided_views_unit_vectors.cpp -- Strided Views & Unit Vectors
//
// This example demonstrates:
//   1. Unit vector construction and their role as canonical basis vectors
//   2. Extracting columns from row-major matrices via strided_vector_ref
//   3. Custom stride iteration and sub-vector extraction
//   4. Building projection matrices from unit vectors
//   5. Combining strided views with dot products for column norms
//
// Key insight: strided_vector_ref is a zero-copy view -- it provides
// vector semantics over non-contiguous memory by skipping elements
// with a fixed stride. This makes column extraction from row-major
// matrices O(1) to construct and O(n) to traverse.

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

void print_strided(const std::string& name, const vec::strided_vector_ref<double>& v) {
    std::cout << "  " << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << v(i);
    }
    std::cout << "]  (stride=" << v.stride() << ")\n";
}

void print_matrix(const std::string& name, const mat::dense2D<double>& M) {
    std::cout << name << " (" << M.num_rows() << "x" << M.num_cols() << "):\n";
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
    std::cout << " Phase 11A: Strided Views & Unit Vectors\n";
    std::cout << "=============================================================\n\n";

    // ======================================================================
    // Part 1: Unit Vectors as Canonical Basis
    // ======================================================================
    std::cout << "=== Part 1: Unit Vectors as Canonical Basis ===\n\n";

    std::cout << "  unit_vector(n, k) creates e_k: a vector of length n\n";
    std::cout << "  with 1 at position k and 0 elsewhere.\n";
    std::cout << "  These are the standard basis of R^n.\n\n";

    const std::size_t n = 4;
    for (std::size_t k = 0; k < n; ++k) {
        auto ek = unit_vector(n, k);
        std::cout << "  e_" << k << " = [";
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << static_cast<int>(ek(i));
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    // Show that A * e_k extracts column k
    mat::dense2D<double> A = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    print_matrix("A", A);

    std::cout << "  A * e_k extracts column k of A:\n";
    for (std::size_t k = 0; k < A.num_cols(); ++k) {
        auto ek = unit_vector(A.num_cols(), k);
        auto col = A * ek;
        std::cout << "  A * e_" << k << " = [";
        for (std::size_t i = 0; i < col.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(0) << col(i);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    // ======================================================================
    // Part 2: Strided Column Extraction
    // ======================================================================
    std::cout << "=== Part 2: Strided Column Extraction (Zero-Copy) ===\n\n";

    std::cout << "  In row-major storage, column elements are separated\n";
    std::cout << "  by num_cols positions in memory:\n\n";
    std::cout << "    Memory layout: [1,2,3,4, 5,6,7,8, 9,10,11,12]\n";
    std::cout << "                    ^       ^         ^           \n";
    std::cout << "    Column 0:      1       5         9     (stride=4)\n\n";

    std::cout << "  strided_vector_ref provides vector access without copying:\n\n";

    auto nrows = A.num_rows();
    auto ncols = A.num_cols();

    for (std::size_t j = 0; j < ncols; ++j) {
        // Column j starts at data + j, with stride = ncols
        vec::strided_vector_ref<double> col_j(A.data() + j, nrows, ncols);
        std::string label = "col(" + std::to_string(j) + ")";
        print_strided(label, col_j);
    }
    std::cout << "\n";

    // Demonstrate that modifying through the view changes the matrix
    std::cout << "  Modifying through the view updates the matrix directly:\n";
    vec::strided_vector_ref<double> col2(A.data() + 2, nrows, ncols);
    double old_val = col2(0);
    col2(0) = 99.0;
    std::cout << "  col(2)[0] = 99  =>  A(0,2) is now " << A(0, 2) << "\n";
    col2(0) = old_val;  // restore
    std::cout << "  (restored)\n\n";

    // ======================================================================
    // Part 3: Iterator-Based Column Processing
    // ======================================================================
    std::cout << "=== Part 3: Column Norms via Strided Iteration ===\n\n";

    std::cout << "  strided_vector_ref supports range-based for loops.\n";
    std::cout << "  Computing the 2-norm of each column:\n\n";

    for (std::size_t j = 0; j < ncols; ++j) {
        vec::strided_vector_ref<double> col_j(A.data() + j, nrows, ncols);

        // Compute norm using range-based for
        double sum_sq = 0.0;
        for (double val : col_j) {
            sum_sq += val * val;
        }
        double norm = std::sqrt(sum_sq);

        std::cout << "  ||col(" << j << ")||_2 = sqrt(";
        bool first = true;
        for (double val : col_j) {
            if (!first) std::cout << " + ";
            std::cout << val << "^2";
            first = false;
        }
        std::cout << ") = " << std::fixed << std::setprecision(4) << norm << "\n";
    }
    std::cout << "\n";

    // ======================================================================
    // Part 4: Sub-Vector Extraction
    // ======================================================================
    std::cout << "=== Part 4: Sub-Vector Extraction ===\n\n";

    std::cout << "  sub_vector(v, start, finish) creates a narrower view\n";
    std::cout << "  into the same strided data.\n\n";

    // Build a larger matrix to show this clearly
    mat::dense2D<double> B(5, 4);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            B(i, j) = static_cast<double>(10 * (i + 1) + (j + 1));

    print_matrix("B", B);

    vec::strided_vector_ref<double> full_col(B.data() + 1, 5, 4);
    print_strided("col(1) full   ", full_col);

    auto sub = vec::sub_vector(full_col, 1, 4);
    print_strided("col(1)[1:4]   ", sub);
    std::cout << "  (rows 1-3 of column 1, same underlying memory)\n\n";

    // ======================================================================
    // Part 5: Building a Projection Matrix
    // ======================================================================
    std::cout << "=== Part 5: Projection from Unit Vectors ===\n\n";

    std::cout << "  The projection onto coordinate k is P_k = e_k * e_k^T.\n";
    std::cout << "  For a vector v, P_k * v = v_k * e_k (keeps only component k).\n\n";

    vec::dense_vector<double> v = {3.0, 7.0, 2.0};
    print_vector("v         ", v);

    // Build P_1 = e_1 * e_1^T (projects onto the y-axis)
    auto e1 = unit_vector(3, 1);
    mat::dense2D<double> P1(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            P1(i, j) = e1(i) * e1(j);

    print_matrix("P_1 = e_1 * e_1^T", P1);

    auto projected = P1 * v;
    print_vector("P_1 * v   ", projected);
    std::cout << "  Only the y-component survives: v_1 * e_1 = "
              << v(1) << " * [0, 1, 0]\n\n";

    // Verify: sum of all projections reconstructs v
    std::cout << "  Verifying: P_0*v + P_1*v + P_2*v = v\n";
    vec::dense_vector<double> reconstructed(3, 0.0);
    for (std::size_t k = 0; k < 3; ++k) {
        auto ek = unit_vector(3, k);
        mat::dense2D<double> Pk(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                Pk(i, j) = ek(i) * ek(j);
        auto component = Pk * v;
        for (std::size_t i = 0; i < 3; ++i)
            reconstructed(i) += component(i);
    }
    print_vector("sum P_k*v ", reconstructed);
    double err = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        err += std::abs(v(i) - reconstructed(i));
    std::cout << "  Error: " << std::scientific << err << "\n\n";

    // -- Takeaways --------------------------------------------------------
    std::cout << "=== Takeaways ===\n\n";
    std::cout << "  1. unit_vector(n, k) returns e_k: the k-th standard basis vector\n";
    std::cout << "  2. A * e_k extracts column k; e_k^T * A extracts row k\n";
    std::cout << "  3. strided_vector_ref: zero-copy column view from row-major data\n";
    std::cout << "  4. Stride = num_cols for row-major column extraction\n";
    std::cout << "  5. sub_vector() narrows the view without additional copies\n";
    std::cout << "  6. Strided iterators enable range-based for and STL algorithms\n";
    std::cout << "  7. P_k = e_k * e_k^T is the coordinate projection; sum(P_k) = I\n";

    return EXIT_SUCCESS;
}
