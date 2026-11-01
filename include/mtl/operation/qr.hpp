#pragma once
// MTL5 -- QR factorization via Householder reflections
// A = Q*R. A is overwritten with R (upper triangular) on+above diagonal,
// Householder vectors stored below diagonal. tau stores the beta scalars.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/tag/reflector_form.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/math/accumulator_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// QR factorization: A (m x n, m >= n) is overwritten with R on+above diagonal
/// and Householder vectors below diagonal. tau(k) stores beta for column k.
/// Returns 0 on success.
///
/// `Accumulator` (leading, default void) selects how each reflector's sums are
/// rounded: the sum of squares in householder() and the per-column v^H*A(:,j)
/// in apply_householder_left() (#536). void is today's arithmetic, including
/// the LAPACK geqrf dispatch; any other accumulator runs the generic loop,
/// since geqrf's sums are its own.
template <typename Accumulator = void, FieldMatrix M>
int qr_factor(M& A, vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);
    tau.change_dim(k);

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::accumulator_allows_blas_v<Accumulator>) {
        // A zero dimension makes lda == 0, which LAPACK rejects (XERBLA); an
        // empty factorization is trivial, so skip the call and fall through.
        if (m > 0 && n > 0) {
            int m_int = static_cast<int>(m);
            int n_int = static_cast<int>(n);
            // geqrf works in-place on a column-major buffer. A column-major
            // dense2D IS that buffer -- point straight at A.data(). A row-major
            // dense2D is not, so copy it into a column-major scratch through the
            // (i,j) accessor, factor there, and copy the result (R + packed
            // reflectors) back through (i,j); the generic qr_extract_Q / qr_solve
            // then read it exactly as on the column-major path, and tau is filled
            // the same way (#417).
            std::vector<value_type> buf;
            value_type* aptr;
            if constexpr (interface::is_row_major_v<M>) {
                buf.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                for (size_type i = 0; i < m; ++i)
                    for (size_type j = 0; j < n; ++j)
                        buf[static_cast<std::size_t>(j) * m + i] = A(i, j);
                aptr = buf.data();
            } else {
                aptr = const_cast<value_type*>(A.data());
            }
            // Workspace query
            value_type work_opt;
            interface::lapack::geqrf(m_int, n_int, aptr, m_int, tau.data(), &work_opt, -1);
            int lwork = static_cast<int>(work_opt);
            std::vector<value_type> work(lwork);
            int info = interface::lapack::geqrf(m_int, n_int, aptr, m_int,
                tau.data(), work.data(), lwork);
            if constexpr (interface::is_row_major_v<M>) {
                for (size_type i = 0; i < m; ++i)
                    for (size_type j = 0; j < n; ++j)
                        A(i, j) = buf[static_cast<std::size_t>(j) * m + i];
            }
            return info;
        }
    }
#endif

    for (size_type j = 0; j < k; ++j) {
        // Extract column A(j:m-1, j)
        size_type len = m - j;
        vec::dense_vector<value_type> col(len);
        for (size_type i = 0; i < len; ++i)
            col(i) = A(j + i, j);

        // Compute Householder reflection
        auto [v, beta] = householder<Accumulator>(col);
        tau(j) = beta;

        // Apply reflection to A(j:m-1, j:n-1)
        apply_householder_left<Accumulator>(A, v, beta, j, j);

        // Store Householder vector below diagonal (v(0) is implicit 1)
        for (size_type i = 1; i < len; ++i)
            A(j + i, j) = v(i);
    }
    return 0;
}

/// QR factorization with the textbook Householder reflector (#559). Real
/// element types only; never dispatches to LAPACK, whose reflector is the
/// default form. For each column k (k < min(m, n)):
///
///   auto [v, vtv, alpha] = householder(A(k:m-1, k), tag::textbook_reflector{});
///   if !(vtv > 0): skip the column (tau(k) = 0, column k untouched)
///   apply_householder_left(A, v, vtv, k, k+1, tag::textbook_reflector{});
///   R(k,k) = alpha
///
/// so R on and above the diagonal is bit-identical to a hand-written textbook
/// QR. R's diagonal sign, -sign(x_k)*||x||, is the default's too from 5.22.0
/// (#672); the values differ in rounding. Below the diagonal and in
/// tau the same reflector is stored in MTL5's packed form, v/v(0) with
/// tau(k) = 2*v(0)^2/vtv, so qr_extract_Q and qr_solve work on the result
/// unchanged. Their Q agrees with the textbook Q to rounding, not to the bit:
/// the packed form is a rescaling of v. Returns 0.
template <FieldMatrix M>
int qr_factor(M& A, vec::dense_vector<typename M::value_type>& tau, tag::textbook_reflector) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    static_assert(!is_complex_v<value_type>,
        "qr_factor(A, tau, tag::textbook_reflector): the textbook form is defined "
        "for real element types only. Use qr_factor(A, tau) for complex matrices.");
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);
    tau.change_dim(k);

    for (size_type j = 0; j < k; ++j) {
        const size_type len = m - j;
        vec::dense_vector<value_type> col(len);
        for (size_type i = 0; i < len; ++i)
            col(i) = A(j + i, j);

        auto [v, vtv, alpha] = householder(col, tag::textbook_reflector{});
        if (!(vtv > math::zero<value_type>())) {
            // Skipped (zero or NaN column): the identity reflection. Column j
            // keeps its entries, as with the default's tau == 0.
            tau(j) = math::zero<value_type>();
            continue;
        }

        // Columns j+1.. get the reflector. Column j's result is known exactly
        // (alpha*e_1), and is what a hand-written QR stores after applying it.
        apply_householder_left(A, v, vtv, j, j + 1, tag::textbook_reflector{});
        A(j, j) = alpha;

        // Packed storage: v(0) = x_0 - alpha has |v(0)| = |x_0| + ||x|| > 0.
        const value_type v0 = v(0);
        for (size_type i = 1; i < len; ++i)
            A(j + i, j) = v(i) / v0;
        tau(j) = value_type(2) * v0 * v0 / vtv;
    }
    return 0;
}

/// Extract explicit Q (m x m) from QR factorization stored in A and tau.
/// Q = H_0^H * H_1^H * ... * H_{k-1}^H where H_j = I - tau(j) * v_j * v_j^H.
/// (For real element types conj is the identity and this is the familiar
/// Q = H_0 * H_1 * ... * H_{k-1}.) `Accumulator` is forwarded to each
/// apply_householder_left (#536).
template <typename Accumulator = void, Matrix M>
auto qr_extract_Q(const M& A, const vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);

    // Start with Q = I
    mat::dense2D<value_type> Q(m, m);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            Q(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    // Apply Householder reflections in reverse: Q = H_0 * H_1 * ... * H_{k-1}
    // which means Q *= H_{k-1}, ..., Q *= H_0
    for (size_type jj = 0; jj < k; ++jj) {
        size_type j = k - 1 - jj;
        size_type len = m - j;
        vec::dense_vector<value_type> v(len);
        v(0) = math::one<value_type>();
        for (size_type i = 1; i < len; ++i)
            v(i) = A(j + i, j);

        // Apply H_j^H to Q from the left.
        //
        // qr_factor formed R = H_{k-1}...H_0 A, so A = H_0^-1 ... H_{k-1}^-1 R
        // and therefore Q = H_0^H ... H_{k-1}^H. The reflectors are UNITARY but
        // not Hermitian for complex tau, so the inverse is H^H = I -
        // conj(tau)*v*v^H, NOT H. Passing tau here instead of conj(tau) would
        // silently return the wrong Q for complex input while remaining exactly
        // right for real input, where conj is the identity (#353).
        apply_householder_left<Accumulator>(Q, v, functor::scalar::conj<value_type>::apply(tau(j)), j, 0);
    }
    return Q;
}

/// Extract explicit R (m x n) from QR factorization stored in A.
template <Matrix M>
auto qr_extract_R(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();

    mat::dense2D<value_type> R(m, n);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            R(i, j) = (j >= i) ? A(i, j) : math::zero<value_type>();
    return R;
}

/// Make R's diagonal non-negative: for each k < min(m, n) with R(k,k) < 0,
/// negate row k of R and column k of Q. Q*R is unchanged (Q D D R with
/// D = diag(+-1)), and for a full-rank A the result is the unique QR with a
/// positive diagonal.
///
/// qr_factor reflects onto -sign(x_k)*||x||, LAPACK's sign, so R(k,k) < 0
/// wherever the pivot entry was positive (#672). This is for callers that
/// relied on the non-negative diagonal qr_factor gave before 5.22.0. It needs
/// the explicit Q and R (qr_extract_Q, qr_extract_R): the packed factor has no
/// way to store a negated column of Q. For a complex A, R's diagonal is real
/// by construction and its real part is tested.
template <Matrix MQ, Matrix MR>
void qr_normalize_signs(MQ& Q, MR& R) {
    using value_type = typename MR::value_type;
    using size_type  = typename MR::size_type;
    const size_type m = R.num_rows();
    const size_type n = R.num_cols();
    assert(Q.num_rows() == m && Q.num_cols() == m);
    for (size_type k = 0; k < std::min(m, n); ++k) {
        if (!(functor::scalar::real<value_type>::apply(R(k, k)) < math::zero<magnitude_t<value_type>>()))
            continue;
        for (size_type j = k; j < n; ++j) R(k, j) = -R(k, j);
        for (size_type i = 0; i < m; ++i) Q(i, k) = -Q(i, k);
    }
}

/// Solve A*x = b via QR factorization (least-squares for m > n).
/// A must already be factored via qr_factor. x has size n, b has size m.
/// `Accumulator` (leading, default void) rounds the two sums this solve forms
/// itself: each reflector's v^H*y when forming Q^H*b, and each row's
/// sum of R(i,j)*x(j) in the back substitution (#536). The subtraction from
/// y(i) and the division by R(i,i) happen after the sum is rounded out.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
void qr_solve(const M& QR, const vec::dense_vector<typename M::value_type>& tau,
              VecX& x, const VecB& b) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = QR.num_rows();
    const size_type n = QR.num_cols();
    const size_type k = std::min(m, n);
    assert(x.size() == n && b.size() == m);

    // y = Q^T * b: apply Householder reflections to b
    vec::dense_vector<value_type> y(m);
    for (size_type i = 0; i < m; ++i)
        y(i) = b(i);

    for (size_type j = 0; j < k; ++j) {
        // tau == 0 is the identity reflection, and the column below the
        // diagonal is then not a reflector: a skipped column keeps its own
        // entries there. Applying it anyway would compute y -= 0*v*dot, which
        // is NaN as soon as one of those entries is. apply_householder_left
        // returns early on the same test, so qr_extract_Q already skips it.
        if (tau(j) == math::zero<value_type>())
            continue;
        size_type len = m - j;
        vec::dense_vector<value_type> v(len);
        v(0) = math::one<value_type>();
        for (size_type i = 1; i < len; ++i)
            v(i) = QR(j + i, j);

        // Apply H_j to y(j:m-1): y -= tau * v * (v^H * y).
        // Here we want Q^H b = H_{k-1}...H_0 b, so the reflectors are applied
        // as H (not H^H) in forward order -- the mirror of qr_extract_Q above.
        value_type dot = math::zero<value_type>();
        if constexpr (std::is_void_v<Accumulator>) {
            for (size_type i = 0; i < len; ++i)
                dot += functor::scalar::conj<value_type>::apply(v(i)) * y(j + i);
        } else {
            using AT = math::accumulator_traits<Accumulator, value_type>;
            Accumulator acc;
            AT::clear(acc);
            for (size_type i = 0; i < len; ++i)
                AT::add_product(acc, functor::scalar::conj<value_type>::apply(v(i)), y(j + i));
            dot = AT::template value<value_type>(acc);
        }
        for (size_type i = 0; i < len; ++i)
            y(j + i) -= tau(j) * v(i) * dot;
    }

    // Back substitution: R*x = y(0:n-1) where R is upper triangular in QR
    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        if constexpr (std::is_void_v<Accumulator>) {
            for (size_type j = i + 1; j < n; ++j)
                sum += QR(i, j) * x(j);
        } else {
            using AT = math::accumulator_traits<Accumulator, value_type>;
            Accumulator acc;
            AT::clear(acc);
            for (size_type j = i + 1; j < n; ++j)
                AT::add_product(acc, value_type(QR(i, j)), value_type(x(j)));
            sum = AT::template value<value_type>(acc);
        }
        x(i) = (y(i) - sum) / QR(i, i);
    }
}

} // namespace mtl
