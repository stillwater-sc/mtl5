#pragma once
// MTL5 -- argument checks shared by the dense eigensolvers and the SVD (#853)
//
// A non-square matrix used to be only asserted, so under NDEBUG the iterations
// read past its last column or used its leading block, and a NaN or infinite
// entry reached iterations that then returned NaN or never converged. These
// checks run in every build, once per call, at O(m n) -- small beside the
// O(n^3) solvers they guard.
#include <stdexcept>
#include <string>
#include <string_view>

namespace mtl::detail {

/// Throws std::invalid_argument unless every entry of A is finite.
template <typename M>
void require_finite(const M& A, std::string_view who) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    for (size_type i = 0; i < A.num_rows(); ++i)
        for (size_type j = 0; j < A.num_cols(); ++j) {
            const value_type a = A(i, j);
            if (!(a - a == value_type(0)))
                throw std::invalid_argument(std::string(who) + ": matrix has a NaN or infinite entry");
        }
}

/// Throws std::invalid_argument unless A is square.
template <typename M>
void require_square(const M& A, std::string_view who) {
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument(std::string(who) + ": matrix must be square");
}

/// Throws std::invalid_argument unless A is square and every entry finite.
template <typename M>
void require_square_finite(const M& A, std::string_view who) {
    require_square(A, who);
    require_finite(A, who);
}

} // namespace mtl::detail
