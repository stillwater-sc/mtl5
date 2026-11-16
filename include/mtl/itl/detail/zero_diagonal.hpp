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

namespace mtl::itl::detail {

[[noreturn]] inline void throw_zero_diagonal(const char* who, std::size_t row) {
    throw std::runtime_error(std::string(who) + ": zero or missing diagonal at row "
                             + std::to_string(row));
}

} // namespace mtl::itl::detail
