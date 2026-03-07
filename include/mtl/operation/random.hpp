#pragma once
// MTL5 -- Fill collections with random values using <random>
#include <random>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>

namespace mtl {

/// Fill a vector with uniform random values in [lo, hi)
template <Vector V>
void fill_random(V& v, typename V::value_type lo = typename V::value_type(0),
                       typename V::value_type hi = typename V::value_type(1)) {
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<typename V::value_type> dist(lo, hi);
    for (typename V::size_type i = 0; i < v.size(); ++i)
        v(i) = dist(gen);
}

/// Fill a matrix with uniform random values in [lo, hi)
template <Matrix M>
void fill_random(M& A, typename M::value_type lo = typename M::value_type(0),
                       typename M::value_type hi = typename M::value_type(1)) {
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<typename M::value_type> dist(lo, hi);
    for (typename M::size_type r = 0; r < A.num_rows(); ++r)
        for (typename M::size_type c = 0; c < A.num_cols(); ++c)
            A(r, c) = dist(gen);
}

/// Return a random dense_vector of size n with values in [lo, hi)
template <typename T = double>
auto random_vector(std::size_t n, T lo = T(0), T hi = T(1)) {
    vec::dense_vector<T> v(n);
    fill_random(v, lo, hi);
    return v;
}

/// Return a random dense2D of size m x n with values in [lo, hi)
template <typename T = double>
auto random_matrix(std::size_t m, std::size_t n, T lo = T(0), T hi = T(1)) {
    mat::dense2D<T> A(m, n);
    fill_random(A, lo, hi);
    return A;
}

} // namespace mtl
