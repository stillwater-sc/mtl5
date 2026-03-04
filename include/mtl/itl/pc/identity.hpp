#pragma once
// MTL5 — Identity preconditioner (no-op): solve(x, b) copies b to x
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>

namespace mtl::itl::pc {

/// Identity preconditioner — solve(x, b) simply copies b into x.
/// Satisfies the Preconditioner concept. Stores nothing.
template <typename Matrix>
class identity {
public:
    explicit identity(const Matrix&) {} // ignores input

    /// solve: x = b  (identity preconditioning)
    template <typename VectorOut, typename VectorIn>
    void solve(VectorOut& x, const VectorIn& b) const {
        assert(x.size() == b.size());
        for (typename VectorIn::size_type i = 0; i < b.size(); ++i)
            x(i) = b(i);
    }

    /// adjoint_solve: same as solve (self-adjoint)
    template <typename VectorOut, typename VectorIn>
    void adjoint_solve(VectorOut& x, const VectorIn& b) const {
        solve(x, b);
    }
};

} // namespace mtl::itl::pc
