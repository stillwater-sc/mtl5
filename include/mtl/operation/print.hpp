#pragma once
// MTL5 — Pretty-print vectors and matrices to ostream
#include <ostream>
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>

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

} // namespace mtl
