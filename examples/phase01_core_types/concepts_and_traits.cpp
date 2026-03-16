// concepts_and_traits.cpp — C++20 Concepts and Type Traits in MTL5
//
// This example demonstrates the concept hierarchy and type trait system
// that enables MTL5's generic programming model:
//   - Scalar, Field, OrderedField concepts
//   - Matrix and Vector concepts with dense/sparse specializations
//   - Category traits for compile-time dispatch
//   - Mixed-type expressions via std::common_type

#include <mtl/mtl.hpp>
#include <iostream>
#include <complex>

// A minimal custom number type that satisfies Scalar and Field
struct MyScalar {
    double value;

    MyScalar() : value(0) {}
    MyScalar(double v) : value(v) {}
    explicit MyScalar(int v) : value(static_cast<double>(v)) {}

    MyScalar operator+(const MyScalar& b) const { return {value + b.value}; }
    MyScalar operator-(const MyScalar& b) const { return {value - b.value}; }
    MyScalar operator*(const MyScalar& b) const { return {value * b.value}; }
    MyScalar operator/(const MyScalar& b) const { return {value / b.value}; }
    MyScalar operator-() const { return {-value}; }

    bool operator==(const MyScalar& b) const { return value == b.value; }
    bool operator<(const MyScalar& b) const { return value < b.value; }
    bool operator>(const MyScalar& b) const { return value > b.value; }
    bool operator<=(const MyScalar& b) const { return value <= b.value; }
    bool operator>=(const MyScalar& b) const { return value >= b.value; }

    friend std::ostream& operator<<(std::ostream& os, const MyScalar& s) {
        return os << s.value;
    }
};

int main() {
    using namespace mtl;

    std::cout << "MTL5 Phase 1: Concepts and Traits\n";
    std::cout << "==================================\n\n";

    // ---- Scalar concept hierarchy ----
    std::cout << "1. Scalar Concept Hierarchy\n\n";

    std::cout << "   Scalar<T> — arithmetic ops (+, -, *, unary-, zero)\n";
    std::cout << "   Field<T>  — Scalar + division\n";
    std::cout << "   OrderedField<T> — Field + total ordering (<, >, <=, >=)\n\n";

    std::cout << "   Built-in types:\n";
    std::cout << "     Scalar<int>:    " << std::boolalpha << Scalar<int> << '\n';
    std::cout << "     Field<int>:     " << Field<int> << '\n';
    std::cout << "     Scalar<double>: " << Scalar<double> << '\n';
    std::cout << "     Field<double>:  " << Field<double> << '\n';
    std::cout << "     OrderedField<double>: " << OrderedField<double> << '\n';

    std::cout << "\n   std::complex<double>:\n";
    std::cout << "     Scalar:       " << Scalar<std::complex<double>> << '\n';
    std::cout << "     Field:        " << Field<std::complex<double>> << '\n';
    std::cout << "     OrderedField: " << OrderedField<std::complex<double>>
              << "  (complex is not ordered!)\n";

    std::cout << "\n   Custom type (MyScalar):\n";
    std::cout << "     Scalar:       " << Scalar<MyScalar> << '\n';
    std::cout << "     Field:        " << Field<MyScalar> << '\n';
    std::cout << "     OrderedField: " << OrderedField<MyScalar> << '\n';

    // ---- Matrix/Vector concepts ----
    std::cout << "\n2. Matrix and Vector Concepts\n\n";

    std::cout << "   Collection<T> — has value_type, size_type, size()\n";
    std::cout << "   Vector<T>     — Collection + v(i) access\n";
    std::cout << "   Matrix<T>     — Collection + m(r,c), num_rows, num_cols\n";
    std::cout << "   DenseMatrix<T>  — Matrix + category == tag::dense\n";
    std::cout << "   SparseMatrix<T> — Matrix + category == tag::sparse\n\n";

    using DenseVec  = vec::dense_vector<double>;
    using DenseMat  = mat::dense2D<double>;
    using SparseMat = mat::compressed2D<double>;

    std::cout << "   dense_vector<double>:\n";
    std::cout << "     Collection: " << Collection<DenseVec> << '\n';
    std::cout << "     Vector:     " << Vector<DenseVec> << '\n';

    std::cout << "   dense2D<double>:\n";
    std::cout << "     Matrix:      " << Matrix<DenseMat> << '\n';
    std::cout << "     DenseMatrix: " << DenseMatrix<DenseMat> << '\n';

    std::cout << "   compressed2D<double>:\n";
    std::cout << "     Matrix:       " << Matrix<SparseMat> << '\n';
    std::cout << "     SparseMatrix: " << SparseMatrix<SparseMat> << '\n';
    std::cout << "     DenseMatrix:  " << DenseMatrix<SparseMat> << '\n';

    // ---- Type traits for dispatch ----
    std::cout << "\n3. Category Traits (Compile-Time Dispatch)\n\n";

    std::cout << "   traits::category_t<dense2D<double>> is tag::dense:  "
              << std::is_same_v<traits::category_t<DenseMat>, tag::dense> << '\n';
    std::cout << "   traits::category_t<compressed2D<double>> is tag::sparse: "
              << std::is_same_v<traits::category_t<SparseMat>, tag::sparse> << '\n';

    std::cout << "\n   These traits drive if-constexpr dispatch in operations:\n";
    std::cout << "   - Dense float/double → BLAS/LAPACK when available\n";
    std::cout << "   - Sparse float/double → SuiteSparse when available\n";
    std::cout << "   - Any type → generic template implementation\n";

    // ---- Mixed-type expressions ----
    std::cout << "\n4. Mixed-Type Expressions\n\n";

    mat::dense2D<int> A_int(2, 2);
    A_int(0, 0) = 1; A_int(0, 1) = 2;
    A_int(1, 0) = 3; A_int(1, 1) = 4;

    vec::dense_vector<double> x_dbl = {1.5, 2.5};

    // int matrix * double vector → double result (via std::common_type)
    auto y = A_int * x_dbl;
    std::cout << "   int matrix * double vector → double result:\n";
    std::cout << "   [1 2] * [1.5] = [" << y(0) << "]\n";
    std::cout << "   [3 4]   [2.5]   [" << y(1) << "]\n";

    // ---- Custom type in a vector ----
    std::cout << "\n5. Custom Number Types in MTL5 Containers\n\n";

    vec::dense_vector<MyScalar> cv(3);
    cv(0) = MyScalar(1.0);
    cv(1) = MyScalar(2.0);
    cv(2) = MyScalar(3.0);

    std::cout << "   dense_vector<MyScalar>: [";
    for (std::size_t i = 0; i < cv.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << cv(i);
    }
    std::cout << "]\n";
    std::cout << "   Any type satisfying Scalar works in MTL5 containers.\n";
    std::cout << "   This is how posit<32,2>, lns<16>, cfloat<16,5> from\n";
    std::cout << "   the Universal library plug in without modification.\n";

    std::cout << '\n';
    return 0;
}
