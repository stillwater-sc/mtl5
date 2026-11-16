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

} // namespace mtl::itl::detail
