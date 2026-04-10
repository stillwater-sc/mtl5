// norms_and_products.cpp -- Vector Norms, Dot Products, and Matrix-Vector Multiply
//
// This example demonstrates the fundamental numerical operations:
//   - Vector norms: one_norm, two_norm, infinity_norm
//   - Dot products and inner products
//   - Matrix-vector multiplication (dense and sparse)
//   - Expression templates: lazy evaluation for efficiency

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    using namespace mtl;

    std::cout << "MTL5 Phase 2: Norms, Products, and Matrix-Vector Multiply\n";
    std::cout << "==========================================================\n\n";

    // ---- Vector Norms ----
    std::cout << "1. Vector Norms\n\n";

    vec::dense_vector<double> v = {3.0, -4.0, 0.0, 5.0, -2.0};

    std::cout << "   v = [3, -4, 0, 5, -2]\n";
    std::cout << "   one_norm(v)      = " << one_norm(v)
              << "  (sum of |v_i|)\n";
    std::cout << "   two_norm(v)      = " << std::fixed << std::setprecision(6)
              << two_norm(v) << "  (Euclidean length)\n";
    std::cout << "   infinity_norm(v) = " << infinity_norm(v)
              << "  (max |v_i|)\n";

    // ---- Dot Products ----
    std::cout << "\n2. Dot Products\n\n";

    vec::dense_vector<double> a = {1.0, 2.0, 3.0};
    vec::dense_vector<double> b = {4.0, 5.0, 6.0};

    std::cout << "   a = [1, 2, 3]\n";
    std::cout << "   b = [4, 5, 6]\n";
    std::cout << "   dot(a, b) = " << dot(a, b)
              << "  (1*4 + 2*5 + 3*6 = 32)\n";

    // ---- Dense Matrix-Vector Multiply ----
    std::cout << "\n3. Dense Matrix-Vector Multiply\n\n";

    mat::dense2D<double> A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;

    vec::dense_vector<double> x = {1.0, 1.0, 1.0};

    auto y_dense = A * x;  // expression template: evaluated lazily

    std::cout << "   A = [[1,2,3],[4,5,6],[7,8,9]]\n";
    std::cout << "   x = [1, 1, 1]\n";
    std::cout << "   A*x = [" << y_dense(0) << ", " << y_dense(1)
              << ", " << y_dense(2) << "]  (row sums)\n";

    // ---- Sparse Matrix-Vector Multiply ----
    std::cout << "\n4. Sparse Matrix-Vector Multiply\n\n";

    // Build a 5x5 tridiagonal sparse matrix
    std::size_t n = 5;
    mat::compressed2D<double> S(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }

    vec::dense_vector<double> xs = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto ys = S * xs;

    std::cout << "   S = tridiag(-1, 2, -1), size 5x5, nnz=" << S.nnz() << '\n';
    std::cout << "   x = [1, 2, 3, 4, 5]\n";
    std::cout << "   S*x = [";
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ys(i);
    }
    std::cout << "]\n";

    // ---- Element-wise Operations ----
    std::cout << "\n5. Element-wise Vector Operations\n\n";

    vec::dense_vector<double> p = {1.0, 4.0, 9.0, 16.0};
    auto p_abs = abs(p);
    auto p_sqrt = sqrt(p);
    auto p_neg = negate(p);

    std::cout << "   p       = [1, 4, 9, 16]\n";
    std::cout << "   abs(p)  = [" << p_abs(0) << ", " << p_abs(1)
              << ", " << p_abs(2) << ", " << p_abs(3) << "]\n";
    std::cout << "   sqrt(p) = [" << p_sqrt(0) << ", " << p_sqrt(1)
              << ", " << p_sqrt(2) << ", " << p_sqrt(3) << "]\n";
    std::cout << "   -p      = [" << p_neg(0) << ", " << p_neg(1)
              << ", " << p_neg(2) << ", " << p_neg(3) << "]\n";

    // ---- Transposed Matrix-Vector ----
    std::cout << "\n6. Transposed Matrix-Vector Multiply\n\n";

    mat::compressed2D<double> R(3, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(R);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][0] << 3.0; ins[1][1] << 4.0;
        ins[2][0] << 5.0; ins[2][1] << 6.0;
    }

    vec::dense_vector<double> w = {1.0, 1.0, 1.0};

    auto Rt_w = trans(R) * w;  // 2x3 * 3x1 = 2x1
    std::cout << "   R = [[1,2],[3,4],[5,6]]  (3x2)\n";
    std::cout << "   w = [1, 1, 1]\n";
    std::cout << "   trans(R)*w = [" << Rt_w(0) << ", " << Rt_w(1)
              << "]  (column sums: 9, 12)\n";

    std::cout << '\n';
    return 0;
}
