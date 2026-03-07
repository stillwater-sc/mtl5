#pragma once
// MTL5 -- Pretty-print vectors and matrices to ostream
#include <ostream>
#include <iomanip>
#include <string>
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl {

/// Print a vector: [v0, v1, ..., vn]
template <Vector V>
std::ostream& operator<<(std::ostream& os, const V& v) {
    os << '[';
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        if (i > 0) os << ", ";
        os << v(i);
    }
    os << ']';
    return os;
}

/// Print a matrix in row-by-row format
template <Matrix M>
std::ostream& operator<<(std::ostream& os, const M& m) {
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        os << '[';
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            if (c > 0) os << ", ";
            os << m(r, c);
        }
        os << "]\n";
    }
    return os;
}

/// Print a vector with configurable precision
template <Vector V>
void print(std::ostream& os, const V& v, int precision = 6) {
    auto old = os.precision(precision);
    os << v;
    os.precision(old);
}

/// Print a matrix with configurable precision
template <Matrix M>
void print(std::ostream& os, const M& m, int precision = 6) {
    auto old = os.precision(precision);
    os << m;
    os.precision(old);
}

/// Print sparse matrix in triplet format: (row, col) = value
template <typename Value, typename Parameters>
void print_sparse(std::ostream& os,
                  const mat::compressed2D<Value, Parameters>& A,
                  int precision = 6) {
    auto old = os.precision(precision);
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();

    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            os << '(' << i << ", " << indices[k] << ") = " << data[k] << '\n';
    os.precision(old);
}

/// Print matrix in MATLAB format: name = [v1 v2 ...; v3 v4 ...; ...]
template <Matrix M>
void print_matlab(std::ostream& os, const M& m,
                  const std::string& name = "A", int precision = 6) {
    auto old = os.precision(precision);
    os << name << " = [";
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        if (r > 0) os << "; ";
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            if (c > 0) os << ' ';
            os << m(r, c);
        }
    }
    os << "];\n";
    os.precision(old);
}

} // namespace mtl
