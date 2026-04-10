// vectors_and_matrices.cpp -- Dense Vectors and Matrices: The Building Blocks
//
// This example demonstrates the core data types of MTL5:
//   - Dense vectors (column and row orientation)
//   - Dense matrices (row-major and column-major)
//   - Sparse matrices via the inserter pattern
//   - Construction, element access, and basic properties

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>

int main() {
    using namespace mtl;

    std::cout << "MTL5 Phase 1: Core Types -- Vectors and Matrices\n";
    std::cout << "================================================\n\n";

    // ---- Dense Vectors ----
    std::cout << "1. Dense Vectors\n\n";

    // Construction methods
    vec::dense_vector<double> v1(5, 0.0);           // size 5, filled with 0
    vec::dense_vector<double> v2 = {1.0, 2.0, 3.0, 4.0, 5.0};  // initializer list
    vec::dense_vector<int>    v3(4, 42);             // integer vector

    std::cout << "   v2 (initializer list): [";
    for (std::size_t i = 0; i < v2.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << v2(i);
    }
    std::cout << "]\n";
    std::cout << "   v2.size() = " << v2.size() << '\n';

    // Element access: v(i) with bounds checking, v[i] without
    v1(0) = 10.0;
    v1(4) = 50.0;
    std::cout << "   v1(0) = " << v1(0) << ", v1(4) = " << v1(4) << '\n';

    // Arithmetic
    auto v4 = v2;
    v4 += v2;  // v4 = 2 * v2
    std::cout << "   v2 + v2 = [";
    for (std::size_t i = 0; i < v4.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << v4(i);
    }
    std::cout << "]\n";

    // Scalar multiplication
    v4 *= 0.5;
    std::cout << "   (v2+v2) * 0.5 = [";
    for (std::size_t i = 0; i < v4.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << v4(i);
    }
    std::cout << "]\n";

    // ---- Dense Matrices ----
    std::cout << "\n2. Dense Matrices\n\n";

    // Construction
    mat::dense2D<double> A(3, 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    std::cout << "   A (3x4):\n";
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        std::cout << "     [";
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::setw(4) << A(i, j);
        }
        std::cout << "]\n";
    }
    std::cout << "   num_rows = " << A.num_rows()
              << ", num_cols = " << A.num_cols()
              << ", size = " << A.size() << '\n';

    // ---- Sparse Matrices ----
    std::cout << "\n3. Sparse Matrices (Compressed Row Storage)\n\n";

    // The inserter pattern: RAII-based sparse matrix construction
    mat::compressed2D<double> S(5, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        // Build a tridiagonal matrix
        for (std::size_t i = 0; i < 5; ++i) {
            ins[i][i] << 4.0;                      // diagonal
            if (i + 1 < 5) ins[i][i + 1] << -1.0;  // super-diagonal
            if (i > 0)     ins[i][i - 1] << -1.0;  // sub-diagonal
        }
    }  // inserter destructor finalizes the CRS structure

    std::cout << "   S (5x5 tridiagonal, nnz=" << S.nnz() << "):\n";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << "     [";
        for (std::size_t j = 0; j < 5; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::setw(4) << S(i, j);
        }
        std::cout << "]\n";
    }

    // ---- Concepts ----
    std::cout << "\n4. C++20 Concepts\n\n";

    std::cout << "   Scalar<double>:   " << std::boolalpha << Scalar<double> << '\n';
    std::cout << "   Field<double>:    " << Field<double> << '\n';
    std::cout << "   Matrix<dense2D>:  " << Matrix<mat::dense2D<double>> << '\n';
    std::cout << "   SparseMatrix<compressed2D>: "
              << SparseMatrix<mat::compressed2D<double>> << '\n';
    std::cout << "   DenseMatrix<dense2D>:       "
              << DenseMatrix<mat::dense2D<double>> << '\n';

    // ---- Math Identities ----
    std::cout << "\n5. Algebraic Identities\n\n";

    std::cout << "   zero<double>() = " << math::zero<double>() << '\n';
    std::cout << "   one<double>()  = " << math::one<double>() << '\n';
    std::cout << "   zero<int>()    = " << math::zero<int>() << '\n';
    std::cout << "   one<int>()     = " << math::one<int>() << '\n';

    std::cout << '\n';
    return 0;
}
