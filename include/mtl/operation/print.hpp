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

/// @brief Print a vector to a stream with configurable precision.
///
/// @details MTL4 intentionally omitted operator<<() for vectors and matrices,
/// providing only explicit print() functions. The reasons were:
///
/// 1. **Inserter conflict** -- MTL uses operator<< for sparse element insertion
///    (e.g., `ins[r][c] << 3.14`). A stream operator<< on the same types risks
///    overload ambiguity or surprising ADL interactions.
/// 2. **Expression template safety** -- `std::cout << A * B` would silently force
///    evaluation of a lazy expression. An explicit print() call makes the cost visible.
/// 3. **Format control** -- There is no single right way to format a matrix (dense vs
///    sparse, dimensions, precision). A free function is easier to parameterize.
/// 4. **Compile-time cost** -- Including \<iostream\> in a header-only math library
///    adds overhead to every translation unit, even when no output is needed.
///
/// MTL5 provides both operator<< (for convenience) and print() (for precision control),
/// but print() remains the recommended explicit API for production code.
///
/// @param os        Output stream to write to.
/// @param v         The vector to print.
/// @param precision Number of significant digits (default 6).
template <Vector V>
void print(std::ostream& os, const V& v, int precision = 6) {
    auto old = os.precision(precision);
    os << v;
    os.precision(old);
}

/// @brief Print a matrix to a stream with configurable precision.
/// @param os        Output stream to write to.
/// @param m         The matrix to print in row-by-row format.
/// @param precision Number of significant digits (default 6).
/// @see print(std::ostream&, const V&, int) for design rationale.
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
