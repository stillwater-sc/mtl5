#pragma once
// MTL5 -- Kronecker product of two matrices
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

namespace detail {

/// kron's element type: the common type of the operands' value types, as in
/// dot, axpy and the other mixed-type operations (#694). Taking A's value type
/// alone narrowed a mixed product: kron(dense2D<int>{2}, dense2D<double>{0.25})
/// gave 0. A pair of custom types with no std::common_type keeps A's value
/// type, as before, rather than failing to compile.
template <typename TA, typename TB, typename = void>
struct kron_value { using type = TA; };

template <typename TA, typename TB>
struct kron_value<TA, TB, std::void_t<std::common_type_t<TA, TB>>> {
    using type = std::common_type_t<TA, TB>;
};

} // namespace detail

/// Kronecker product: C = A (x) B, where C(i*rb+p, j*cb+q) = A(i,j)*B(p,q).
/// C's value type is std::common_type_t of A's and B's value types, so a
/// mixed-type product is not narrowed to A's type; for two operands of the
/// same type it is that type.
template <Matrix MA, Matrix MB>
auto kron(const MA& A, const MB& B) {
    using value_type = typename detail::kron_value<typename MA::value_type,
                                                   typename MB::value_type>::type;
    using size_type  = typename MA::size_type;
    const size_type ra = A.num_rows(), ca = A.num_cols();
    const size_type rb = B.num_rows(), cb = B.num_cols();

    mat::dense2D<value_type> C(ra * rb, ca * cb);
    for (size_type i = 0; i < ra; ++i)
        for (size_type j = 0; j < ca; ++j)
            for (size_type p = 0; p < rb; ++p)
                for (size_type q = 0; q < cb; ++q)
                    C(i * rb + p, j * cb + q) = static_cast<value_type>(A(i, j) * B(p, q));
    return C;
}

} // namespace mtl
