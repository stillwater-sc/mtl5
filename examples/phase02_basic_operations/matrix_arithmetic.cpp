// matrix_arithmetic.cpp — Matrix Addition, Scaling, and Matrix-Matrix Multiply
//
// This example demonstrates matrix-level arithmetic operations:
//   - Matrix addition and subtraction (expression templates)
//   - Scalar-matrix multiplication
//   - Matrix-matrix multiplication
//   - Trace and diagonal extraction
//   - Combining dense and sparse matrices

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>

void print_dense(const char* name, const mtl::mat::dense2D<double>& M) {
    std::cout << "   " << name << " (" << M.num_rows() << "x" << M.num_cols() << "):\n";
    for (std::size_t i = 0; i < M.num_rows(); ++i) {
        std::cout << "     [";
        for (std::size_t j = 0; j < M.num_cols(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << M(i, j);
        }
        std::cout << "]\n";
    }
}

int main() {
    using namespace mtl;

    std::cout << "MTL5 Phase 2: Matrix Arithmetic\n";
    std::cout << "================================\n\n";

    // ---- Matrix Addition ----
    std::cout << "1. Matrix Addition and Subtraction\n\n";

    mat::dense2D<double> A(3, 3);
    mat::dense2D<double> B(3, 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(i * 3 + j + 1);
            B(i, j) = static_cast<double>((i == j) ? 10.0 : 0.0);
        }
    }

    print_dense("A", A);
    print_dense("B (10*I)", B);

    auto C = A + B;  // expression template — evaluated when assigned
    mat::dense2D<double> C_mat = C;
    print_dense("A + B", C_mat);

    // ---- Scalar Multiplication ----
    std::cout << "\n2. Scalar-Matrix Multiplication\n\n";

    mat::dense2D<double> S_mat = A;
    scale(0.5, S_mat);
    print_dense("0.5 * A", S_mat);

    // ---- Matrix-Matrix Multiply ----
    std::cout << "\n3. Matrix-Matrix Multiply\n\n";

    mat::dense2D<double> P(2, 3);
    mat::dense2D<double> Q(3, 2);
    P(0, 0) = 1; P(0, 1) = 2; P(0, 2) = 3;
    P(1, 0) = 4; P(1, 1) = 5; P(1, 2) = 6;

    Q(0, 0) = 7;  Q(0, 1) = 8;
    Q(1, 0) = 9;  Q(1, 1) = 10;
    Q(2, 0) = 11; Q(2, 1) = 12;

    print_dense("P (2x3)", P);
    print_dense("Q (3x2)", Q);

    mat::dense2D<double> PQ(2, 2);
    mult(P, Q, PQ);
    print_dense("P * Q (2x2)", PQ);

    // Verify: PQ(0,0) = 1*7 + 2*9 + 3*11 = 58
    std::cout << "   Verify: PQ(0,0) = 1*7 + 2*9 + 3*11 = " << PQ(0, 0) << '\n';

    // ---- Trace ----
    std::cout << "\n4. Trace and Diagonal\n\n";

    mat::dense2D<double> D(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            D(i, j) = static_cast<double>(i == j ? i + 1 : 0);

    std::cout << "   D = diag(1, 2, 3, 4)\n";
    std::cout << "   trace(D) = " << trace(D) << "  (sum of diagonal = 10)\n";

    auto diag_vec = diagonal(D);
    std::cout << "   diagonal(D) = [";
    for (std::size_t i = 0; i < diag_vec.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << diag_vec(i);
    }
    std::cout << "]\n";

    // ---- Mixed Dense-Sparse Matvec ----
    std::cout << "\n5. Sparse and Dense Produce the Same Results\n\n";

    // Same matrix in dense and sparse form
    mat::dense2D<double> M_dense(3, 3);
    mat::compressed2D<double> M_sparse(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(M_sparse);
        for (std::size_t i = 0; i < 3; ++i) {
            double diag = 4.0;
            M_dense(i, i) = diag;
            ins[i][i] << diag;
            if (i + 1 < 3) {
                M_dense(i, i + 1) = -1.0;
                M_dense(i + 1, i) = -1.0;
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
            // zero out remaining dense entries
            for (std::size_t j = 0; j < 3; ++j) {
                if (j != i && j != i + 1 && j + 1 != i)
                    M_dense(i, j) = 0.0;
            }
        }
    }

    vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    auto y_d = M_dense * x;
    auto y_s = M_sparse * x;

    std::cout << "   Dense  A*x = [" << y_d(0) << ", " << y_d(1) << ", " << y_d(2) << "]\n";
    std::cout << "   Sparse A*x = [" << y_s(0) << ", " << y_s(1) << ", " << y_s(2) << "]\n";
    std::cout << "   (Identical results — same algorithm, different storage)\n";

    std::cout << '\n';
    return 0;
}
