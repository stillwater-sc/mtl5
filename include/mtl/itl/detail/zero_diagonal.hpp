#pragma once
// MTL5 -- the error for a zero or missing diagonal in a preconditioner or
// smoother (issue 867).
//
// ILU(0), ILUT, the diagonal (Jacobi) preconditioner and the Jacobi,
// Gauss-Seidel and SOR smoothers divide by A's diagonal (or U's, for the ILUs).
// A zero or unstored one used to leave a 0 in the inverse, or divide by it:
// solve() then returned inf, NaN or a silently wrong x. They check it once, at
// construction, in every build, and throw this, naming the row.
#include <cstddef>
#include <stdexcept>
#include <string>

#include <mtl/detail/exceptions.hpp>

namespace mtl::itl::detail {

/// Through throw_or_abort, so a build with exceptions disabled still compiles
/// and aborts instead (#870 review).
[[noreturn]] inline void throw_zero_diagonal(const char* who, std::size_t row) {
    ::mtl::detail::throw_or_abort(std::runtime_error(
        std::string(who) + ": zero or missing diagonal at row " + std::to_string(row)));
}

/// A pivot that is not positive where the factorization needs it to be: IC(0)
/// takes the square root of each pivot, so a zero or negative one (A not
/// positive definite, or IC(0) breaking down on it) gave NaN in solve() (issue
/// 925). Checked at construction, as the zero diagonal is.
[[noreturn]] inline void throw_nonpositive_pivot(const char* who, std::size_t row) {
    ::mtl::detail::throw_or_abort(std::runtime_error(
        std::string(who) + ": non-positive pivot at row " + std::to_string(row)
        + " (the matrix is not positive definite, or the incomplete factorization broke down)"));
}

/// A matrix that must be square, checked once at construction in every build
/// (issue 925; the index-check policy of #825). The preconditioners checked it
/// with assert only, and a non-square A then indexed past the work arrays,
/// which are sized by rows and indexed by column.
template <typename Matrix>
void require_square(const Matrix& A, const char* who) {
    if (A.num_rows() != A.num_cols())
        ::mtl::detail::throw_or_abort(std::invalid_argument(
            std::string(who) + ": the matrix must be square, it is "
            + std::to_string(A.num_rows()) + " x " + std::to_string(A.num_cols())));
}

} // namespace mtl::itl::detail
