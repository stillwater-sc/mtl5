#pragma once
// MTL5 -- trsv: solve A*x = b for triangular A  (BLAS level-2)
// A BLAS-style facade over the forward/back-substitution solvers, with an
// in-place (solve A*x = x) form. A is upper (upper=true) or lower triangular;
// unit_diag treats the diagonal as all ones. No transpose in this variant.
// BLAS dispatch for column-major dense float/double when available.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Solve A*x = x in place, A upper/lower triangular.
template <Matrix M, Vector VX>
void trsv(const M& A, VX& x, bool upper, bool unit_diag = false) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n);

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<M> && interface::BlasDenseVector<VX> &&
                  !interface::is_row_major_v<M> &&
                  std::is_same_v<value_type, typename M::value_type>) {
        using T = value_type;
        if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
            interface::blas::trsv(upper ? 'U' : 'L', 'N', unit_diag ? 'U' : 'N',
                                  static_cast<int>(n), A.data(), static_cast<int>(n),
                                  x.data(), 1);
            return;
        }
    }
#endif
    if (upper)
        upper_trisolve(A, x, unit_diag);
    else
        lower_trisolve(A, x, unit_diag);
}

/// Solve A*x = b, writing the solution into x (b unchanged).
template <Matrix M, Vector VX, Vector VB>
void trsv(const M& A, VX& x, const VB& b, bool upper, bool unit_diag = false) {
    using size_type = typename M::size_type;
    for (size_type i = 0; i < b.size(); ++i) x(i) = b(i);
    trsv(A, x, upper, unit_diag);
}

} // namespace mtl
