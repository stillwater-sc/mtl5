#pragma once
// MTL5 -- Sparse LU factorization with partial pivoting
//
// Implements the left-looking sparse LU algorithm from Davis's
// "Direct Methods for Sparse Linear Systems", Chapter 6.
//
// Key features:
//   - Symbolic/numeric phase separation
//   - Column pre-ordering via pluggable fill-reducing ordering
//   - Threshold partial pivoting for numerical stability
//   - Produces L (unit lower triangular) and U (upper triangular) in CSC
//   - Row permutation tracked from pivoting
//
// Algorithm (left-looking LU for column k):
//   1. Solve L(1:k-1, 1:k-1) * x = A(:, q[k]) via sparse triangular solve
//   2. x now contains U(1:k-1, k) and candidate L(:, k) entries
//   3. Select pivot: row i >= k maximizing |x[i]| (threshold pivoting)
//   4. Swap pivot row with row k
//   5. U(1:k, k) = x[1:k], L(k+1:n, k) = x[k+1:n] / U(k,k)
//
// The factorization computes P*A*Q = L*U where:
//   P = row permutation (from pivoting)
//   Q = column permutation (from fill-reducing ordering)
//   L = unit lower triangular
//   U = upper triangular
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.
//            CSparse: cs_lu, cs_sqr.
//            Gilbert & Peierls, "Sparse Partial Pivoting in Time Proportional
//            to Arithmetic Operations", SIAM J. Sci. Stat. Comput., 1988.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/level_schedule.hpp>

namespace mtl::sparse::factorization {

/// Result of symbolic LU analysis.
/// Captures the column ordering. For LU, the symbolic phase is lighter
/// than Cholesky because row pivoting during numeric factorization can
/// change the fill pattern.
struct lu_symbolic {
    std::vector<std::size_t> col_perm;   // column permutation q[new]=old
    std::vector<std::size_t> col_pinv;   // inverse column permutation
    std::size_t n{0};                    // matrix dimension
};

namespace detail {
/// Folds |u_kk| over U's diagonal (the last entry of each column) into
/// [lo, hi]; `seen` says whether lo and hi hold a value yet. At a NaN pivot it
/// stops and returns false, with lo set to that NaN.
template <typename Value, typename SizeType>
bool lu_fold_pivots(const util::csc_matrix<Value, SizeType>& U, Value& lo, Value& hi, bool& seen) {
    using std::abs;
    for (SizeType k = 0; k < U.ncols; ++k) {
        if (U.col_ptr[k + 1] == U.col_ptr[k]) continue;
        const Value a = abs(U.values[U.col_ptr[k + 1] - 1]);
        if (!(a == a)) { lo = a; return false; }
        if (!seen) { lo = a; hi = a; seen = true; }
        else { if (a < lo) lo = a; if (a > hi) hi = a; }
    }
    return true;
}

/// min(g, min over U's columns k of amax_of(k) / max_i |U(i, k)|), where
/// amax_of(k) is the largest |entry| of the factored matrix's column that
/// became U's column k. A NaN, in U or in a ratio, is returned as it is.
template <typename Value, typename SizeType, typename ColMax>
Value lu_rgrowth(const util::csc_matrix<Value, SizeType>& U, ColMax&& amax_of, Value g = Value{1}) {
    using std::abs;
    for (SizeType k = 0; k < U.ncols; ++k) {
        Value umax{0};
        for (SizeType p = U.col_ptr[k]; p < U.col_ptr[k + 1]; ++p) {
            const Value a = abs(U.values[p]);
            if (!(a == a)) return a;
            if (a > umax) umax = a;
        }
        if (umax == Value{0}) continue;   // not in an installed factor: its diagonal is nonzero
        const Value r = amax_of(static_cast<std::size_t>(k)) / umax;
        if (!(r == r)) return r;
        if (r < g) g = r;
    }
    return g;
}

/// ||A||_1, the largest column sum of |A(i, j)|, for a square CSR A. NaN if A
/// holds one.
template <typename Value, typename Parameters>
Value csr_norm1(const mat::compressed2D<Value, Parameters>& A) {
    using std::abs;
    std::vector<Value> col(A.num_cols(), Value{0});
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t k = 0; k < A.nnz(); ++k) {
        const Value a = abs(dat[k]);
        if (!(a == a)) return a;
        col[ci[k]] += a;
    }
    Value m{0};
    for (const Value& c : col) {
        if (!(c == c)) return c;
        if (c > m) m = c;
    }
    return m;
}

/// An estimate of ||A^{-1}||_1 from solves with A and A^T: Hager's method with
/// Higham's refinements (Higham, "FORTRAN codes for estimating the one-norm of
/// a real or complex matrix", ACM TOMS 14(4), 1988; Algorithm 4.1), as
/// SuiteSparse's klu_condest. At most five pairs of solves and one more, so
/// O(5 nnz(L + U)). The estimate is a lower bound of ||A^{-1}||_1, and is
/// exact or within a factor of 3 in practice. `solve(y, x)` computes A^{-1} x
/// and `solve_t(z, x)` A^{-T} x, on dense_vector<Value>. NaN if a solve gives a
/// NaN.
template <typename Value, typename Solve, typename SolveT>
Value inverse_norm1_estimate(std::size_t n, Solve&& solve, SolveT&& solve_t) {
    using std::abs;
    if (n == 0) return Value{0};
    using vec_t = vec::dense_vector<Value>;
    vec_t x(n, Value{1} / Value(n)), y(n), xi(n), xi_old(n), z(n);
    auto norm1 = [&](const vec_t& v) {
        Value t{0};
        for (std::size_t i = 0; i < n; ++i) t += abs(v(i));
        return t;
    };
    Value est{0};
    for (int k = 0; k < 5; ++k) {
        solve(y, x);
        const Value ny = norm1(y);
        if (!(ny == ny)) return ny;
        if (k > 0 && !(ny > est)) break;      // no increase: the last estimate stands
        est = ny;
        bool same_sign = (k > 0);
        for (std::size_t i = 0; i < n; ++i) {
            xi(i) = (y(i) < Value{0}) ? -Value{1} : Value{1};
            if (k > 0 && !(xi(i) == xi_old(i))) same_sign = false;
        }
        if (same_sign) break;                  // the sign pattern repeated: converged
        solve_t(z, xi);
        std::size_t j = 0;
        Value zmax = abs(z(0)), ztx{0};
        for (std::size_t i = 0; i < n; ++i) {
            const Value a = abs(z(i));
            if (!(a == a)) return a;
            if (a > zmax) { zmax = a; j = i; }
            ztx += z(i) * x(i);
        }
        if (k > 0 && !(zmax > ztx)) break;     // Hager's test: x is a local maximum
        for (std::size_t i = 0; i < n; ++i) { x(i) = Value{0}; xi_old(i) = xi(i); }
        x(j) = Value{1};
    }
    // Higham's extra vector, x_i = (-1)^i (1 + i / (n - 1)): it catches the
    // matrices on which the iteration above stops at a poor local maximum.
    for (std::size_t i = 0; i < n; ++i) {
        const Value t = n > 1 ? Value{1} + Value(i) / Value(n - 1) : Value{1};
        x(i) = (i % 2 == 0) ? t : -t;
    }
    solve(y, x);
    const Value alt = Value(2) * norm1(y) / (Value(3) * Value(n));
    if (!(alt == alt)) return alt;
    return alt > est ? alt : est;
}
} // namespace detail

/// Result of numeric LU factorization.
/// Contains L (unit lower triangular) and U (upper triangular) in CSC,
/// plus the row and column permutations.
template <typename Value>
struct lu_numeric {
    std::vector<std::size_t> row_perm;   // row permutation from pivoting (p[new]=old)
    std::vector<std::size_t> row_pinv;   // inverse row permutation
    std::size_t num_perturbed = 0;       // pivots replaced by perturbation (0 = clean factor)
    lu_symbolic symbolic;                // symbolic analysis used

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }

    /// Read-only access to the factors (L unit lower, U upper; both CSC).
    const util::csc_matrix<Value>& factorL() const { return L_; }
    const util::csc_matrix<Value>& factorU() const { return U_; }

    /// Install the factors and (re)build the coupled forward (L) and upper (U)
    /// solve schedules from them. L, U and their schedules are always set
    /// together here, so the schedules' cached structure always matches the
    /// factors' patterns. Both must be square and match the symbolic dimension.
    /// Strongly exception safe: both schedules are built into locals before any
    /// member is replaced, so a throw leaves the object unchanged.
    void set_factor(util::csc_matrix<Value> L, util::csc_matrix<Value> U) {
        const std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(L.ncols) != n || static_cast<std::size_t>(L.nrows) != n ||
            static_cast<std::size_t>(U.ncols) != n || static_cast<std::size_t>(U.nrows) != n)
            throw std::invalid_argument(
                "lu_numeric::set_factor: factor dimensions do not match the symbolic analysis");
        lower_solve_schedule fsched = build_lower_solve_schedule(L);   // may throw
        upper_solve_schedule usched = build_upper_solve_schedule(U);   // may throw
        // The diagonals are checked here, once, so the solves skip the pass
        // (diagonal_check::done_by_caller, issue 926).
        detail::require_nonzero_diagonal(L, false, L.values, "lu_numeric::set_factor");   // may throw
        detail::require_nonzero_diagonal(U, true, U.values, "lu_numeric::set_factor");    // may throw
        L_ = std::move(L);            // noexcept
        U_ = std::move(U);            // noexcept
        fwd_sched_ = std::move(fsched);  // noexcept
        up_sched_  = std::move(usched);  // noexcept
        has_factor_ = true;              // commit last: solve() is now valid
    }

    /// True once factors (and their coupled schedules) have been installed.
    bool has_factor() const { return has_factor_; }

    /// A cheap estimate of the reciprocal condition number: min |u_kk| /
    /// max |u_kk| over U's diagonal, as SuiteSparse's klu_rcond. It costs O(n).
    /// It is a crude first check, not a bound: a small value means U's pivots
    /// span many orders of magnitude. 1 for n = 0; NaN if a pivot is NaN.
    /// \throws std::logic_error if no factor is installed (n > 0).
    Value rcond() const {
        if (symbolic.n == 0) return Value{1};
        require_factor("lu_numeric::rcond");
        Value lo{0}, hi{0};
        bool seen = false;
        if (!detail::lu_fold_pivots(U_, lo, hi, seen)) return lo;   // NaN
        return lo / hi;
    }

    /// The reciprocal pivot growth: the minimum over U's columns k of
    /// max_i |A(i, q[k])| / max_i |U(i, k)|, capped at 1, as SuiteSparse's
    /// klu_rgrowth. Near 1 means the pivots fit A; a tiny value means elements
    /// of U grew far past A's, and the factorization (a refactorization, which
    /// keeps an earlier pivot sequence, above all) cannot be trusted. It costs
    /// O(nnz(A) + nnz(U)). `A` must be the matrix this factorization is of.
    /// 1 for n = 0; NaN if A or U holds a NaN.
    /// \throws std::invalid_argument if A's dimensions are not n x n.
    /// \throws std::logic_error      if no factor is installed (n > 0).
    template <typename Parameters>
    Value rgrowth(const mat::compressed2D<Value, Parameters>& A) const {
        using std::abs;
        const std::size_t n = symbolic.n;
        if (A.num_rows() != n || A.num_cols() != n)
            throw std::invalid_argument(
                "lu_numeric::rgrowth: matrix dimensions do not match the factorization");
        if (n == 0) return Value{1};
        require_factor("lu_numeric::rgrowth");
        std::vector<Value> amax(n, Value{0});   // by A's column
        const auto& ci  = A.ref_minor();
        const auto& dat = A.ref_data();
        for (std::size_t k = 0; k < A.nnz(); ++k) {
            const Value a = abs(dat[k]);
            if (!(a == a)) return a;
            if (a > amax[ci[k]]) amax[ci[k]] = a;
        }
        return detail::lu_rgrowth(U_, [&](std::size_t k) { return amax[symbolic.col_perm[k]]; });
    }

    /// Solve A^T x = b (the transpose, not the conjugate transpose). With
    /// P A Q = L U, A^T = Q U^T L^T P, so U^T L^T (P x) = Q^T b: a forward
    /// solve with U^T, a back solve with L^T, then the row permutation. x and
    /// b may be the same vector. Serial: the transposed solves have no schedule.
    /// \throws std::logic_error if no factor is installed (n > 0).
    /// \throws std::invalid_argument on a size mismatch.
    template <typename VecX, typename VecB>
    void solve_transpose(VecX& x, const VecB& b) const {
        const std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n || static_cast<std::size_t>(b.size()) != n)
            throw std::invalid_argument(
                "lu_numeric::solve_transpose: vector size mismatch (expected " + std::to_string(n) + ")");
        if (n == 0) return;
        require_factor("lu_numeric::solve_transpose");
        std::vector<Value> w(n);
        for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<Value>(b(symbolic.col_perm[i]));
        // U^T v = w, forward: column k of U holds U(i, k) for i <= k, diagonal last.
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t last = U_.col_ptr[k + 1] - 1;
            Value t = w[k];
            for (std::size_t p = U_.col_ptr[k]; p < last; ++p) t -= U_.values[p] * w[U_.row_ind[p]];
            w[k] = t / U_.values[last];
        }
        // L^T u = v, backward: column k of L holds L(i, k) for i >= k, diagonal
        // first (1, divided by as the forward solve does).
        for (std::size_t k = n; k-- > 0; ) {
            const std::size_t first = L_.col_ptr[k];
            Value t = w[k];
            for (std::size_t p = first + 1; p < L_.col_ptr[k + 1]; ++p) t -= L_.values[p] * w[L_.row_ind[p]];
            w[k] = t / L_.values[first];
        }
        for (std::size_t i = 0; i < n; ++i)
            x(row_perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }

    /// An estimate of the 1-norm condition number ||A||_1 ||A^{-1}||_1, as
    /// SuiteSparse's klu_condest: Hager's method with Higham's refinements
    /// (detail::inverse_norm1_estimate), from at most 11 solves. A lower bound,
    /// exact or within a factor of 3 in practice. Far more telling than rcond()
    /// and far cheaper than forming A^{-1}. `A` must be the matrix this
    /// factorization is of. 1 for n = 0; NaN if A or a solve holds a NaN.
    template <typename Parameters>
    Value condest(const mat::compressed2D<Value, Parameters>& A) const {
        const std::size_t n = symbolic.n;
        if (A.num_rows() != n || A.num_cols() != n)
            throw std::invalid_argument(
                "lu_numeric::condest: matrix dimensions do not match the factorization");
        if (n == 0) return Value{1};
        require_factor("lu_numeric::condest");
        const Value anorm = detail::csr_norm1(A);
        const Value inv = detail::inverse_norm1_estimate<Value>(
            n, [&](auto& y, const auto& x) { solve(y, x); },
               [&](auto& z, const auto& x) { solve_transpose(z, x); });
        return anorm * inv;
    }

    /// Install recomputed VALUES on the already-installed factor pattern, keeping
    /// the existing solve schedules. The schedules are value-agnostic -- they cache
    /// positions into L_.values / U_.values, not the values -- so a same-pattern
    /// refactorization can swap in new numbers and skip the O(nnz) schedule rebuild
    /// set_factor pays. Only the value arrays change; col_ptr / row_ind are kept.
    ///
    /// Requires a factor already installed, and value arrays of exactly the
    /// installed nnz -- a PATTERN change must go through set_factor. The schedules'
    /// built_nnz staleness guards (the ones the solves assert on) are re-checked
    /// here, so a schedule can never be left silently stale.
    /// Strongly exception safe: all validation precedes any member replacement.
    void set_factor_values(std::vector<Value> Lvals, std::vector<Value> Uvals) {
        if (!has_factor_)
            throw std::logic_error(
                "lu_numeric::set_factor_values: no factor installed (call set_factor first)");
        if (Lvals.size() != L_.values.size() || Uvals.size() != U_.values.size() ||
            fwd_sched_.built_nnz != Lvals.size() || up_sched_.built_nnz != Uvals.size())
            throw std::invalid_argument(
                "lu_numeric::set_factor_values: value count does not match the installed "
                "factor pattern (use set_factor for a pattern change)");
        detail::require_nonzero_diagonal(L_, false, Lvals, "lu_numeric::set_factor_values");   // may throw
        detail::require_nonzero_diagonal(U_, true, Uvals, "lu_numeric::set_factor_values");    // may throw
        L_.values = std::move(Lvals);   // noexcept
        U_.values = std::move(Uvals);   // noexcept
    }

    /// Solve A*x = b using the LU factorization.
    /// P*A*Q = L*U, so A = P^T * L * U * Q^T
    /// A*x = b => P^T * L * U * Q^T * x = b
    /// => L * U * Q^T * x = P * b
    /// Let y = Q^T * x, then L * U * y = P * b
    /// 1. w = P * b
    /// 2. L * z = w (forward solve)
    /// 3. U * y = z (back solve)
    /// 4. x = Q * y
    /// `Accumulator` (default `void`, leading as in the numeric phase) selects
    /// the accumulator policy of the triangular-solve row reductions (#535):
    /// `void` is the raw Value loop; see level_schedule.hpp for the contract.
    template <typename Accumulator = void, typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        std::vector<Value> w;
        solve<Accumulator>(x, b, w);
    }

    /// solve(x, b) with the caller's workspace `w`, resized to n: a caller
    /// that solves many small systems (native_klu's blocks) reuses one
    /// buffer instead of allocating per solve (issue 908). Same arithmetic.
    /// `w` must not share storage with x or b: b is read through the row
    /// permutation while w is written. x and b may be the same vector (all of
    /// b is read into w before x is written).
    template <typename Accumulator = void, typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b, std::vector<Value>& w) const {
        // A caller can set `symbolic` (n > 0) yet never install a factor; the
        // triangular solves below would then read empty factor arrays. Reject
        // that explicitly instead of walking off the end.
        if (!has_factor_)
            throw std::logic_error(
                "lu_numeric::solve: no factor installed (call set_factor "
                "or sparse_lu_numeric first)");
        std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "lu_numeric::solve: vector size mismatch (expected "
                + std::to_string(n) + ")");
        }

        // Step 1: Apply row permutation: w = P * b
        w.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = static_cast<Value>(b(row_perm[i]));

        // Step 2: Forward solve: L * z = w. Level-scheduled (parallel,
        // bit-identical to dense_lower_solve); serial at MTL5_NUM_THREADS=1.
        level_scheduled_lower_solve<Accumulator>(L_, fwd_sched_, w, diagonal_check::done_by_caller);

        // Step 3: Back solve: U * y = z. Level-scheduled (parallel, bit-identical
        // to dense_upper_solve); serial at MTL5_NUM_THREADS=1.
        level_scheduled_upper_solve<Accumulator>(U_, up_sched_, w, diagonal_check::done_by_caller);

        // Step 4: Apply column permutation: x = Q * y
        // Q maps new col -> old col, so x[q[i]] = y[i]
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.col_perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }

private:
    void require_factor(const char* who) const {
        if (!has_factor_)
            throw std::logic_error(std::string(who) + ": no factor installed");
    }

    util::csc_matrix<Value> L_;          // unit lower triangular factor (CSC), diagonal first
    util::csc_matrix<Value> U_;          // upper triangular factor (CSC), diagonal last
    lower_solve_schedule    fwd_sched_;  // forward (L z = w) schedule, bound to L_
    upper_solve_schedule    up_sched_;   // upper  (U y = z) schedule, bound to U_
    bool                    has_factor_ = false;  // set by set_factor; solve() requires it
};

/// Perform symbolic LU analysis on a square sparse matrix.
/// Computes the column permutation for fill reduction.
///
/// \param A        Square sparse matrix in CRS format
/// \param ordering Fill-reducing ordering functor
/// \return         Symbolic analysis result
template <typename Value, typename Parameters, typename Ordering>
lu_symbolic sparse_lu_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering)
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument(
            "sparse_lu_symbolic: matrix must be square");
    }
    std::size_t n = A.num_rows();

    lu_symbolic sym;
    sym.n = n;
    sym.col_perm = ordering(A);
    if (!util::is_valid_permutation(sym.col_perm) || sym.col_perm.size() != n) {
        throw std::invalid_argument(
            "sparse_lu_symbolic: ordering returned invalid permutation");
    }
    sym.col_pinv = util::invert_permutation(sym.col_perm);
    return sym;
}

/// Overload without ordering: uses identity permutation (natural ordering).
template <typename Value, typename Parameters>
lu_symbolic sparse_lu_symbolic(
    const mat::compressed2D<Value, Parameters>& A)
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument(
            "sparse_lu_symbolic: matrix must be square");
    }
    std::size_t n = A.num_rows();

    lu_symbolic sym;
    sym.n = n;
    sym.col_perm = util::identity_permutation(n);
    sym.col_pinv = sym.col_perm;
    return sym;
}

/// Accumulator policy for the numeric workspace of the left-looking solve.
///
/// The canonical, cross-cutting trait lives in `mtl::math::accumulator_traits`
/// (#158); this name inherits it for the sparse kernels and source compatibility.
/// Specialize `mtl::math::accumulator_traits<Acc, Value>` to affect both the
/// sparse factorizations and the dense BLAS ops at once (it propagates here via
/// the base); specialize this one for a sparse-only customization.
template <typename Acc, typename Value>
struct accumulator_traits : mtl::math::accumulator_traits<Acc, Value> {};

/// Perform numeric LU factorization with threshold partial pivoting.
///
/// Uses the left-looking algorithm: for each column k, performs a sparse
/// triangular solve with the already-factored columns of L, then selects
/// a pivot from the remaining entries.
///
/// \param A         Square sparse matrix in CRS format
/// \param sym       Symbolic analysis from sparse_lu_symbolic()
/// \param threshold Pivoting threshold in (0, 1]. A pivot candidate at row i
///                  is acceptable if |a(i,k)| >= threshold * max|a(j,k)| for j>=k.
///                  Default 1.0 = full partial pivoting. Smaller values trade
///                  stability for sparsity preservation.
/// \param pivot_perturb Opt-in zero-pivot perturbation (default 0 = off). When
///                  > 0, a chosen pivot whose magnitude is below
///                  `pivot_perturb * ||column||_inf` is replaced (sign-preserving)
///                  by that floor and `num_perturbed` is incremented, instead of
///                  throwing. This keeps a low-precision factorization going when
///                  cancellation drives a structurally-nonzero pivot numerically
///                  to zero; the result is meant to be cleaned up by iterative
///                  refinement (see mtl::sparse::iterative_refine). With the
///                  default 0, behavior is byte-identical to before (hard throw).
/// \return          Numeric factorization result containing L, U, and permutations
///
/// \throws std::runtime_error if a zero pivot is encountered and perturbation is
///                  off, or the whole pivot column is numerically zero (truly
///                  singular: nothing to perturb).
template <typename Accumulator = void, typename Value, typename Parameters>
    requires OrderedField<Value>
lu_numeric<Value> sparse_lu_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const lu_symbolic& sym,
    Value threshold = Value{1},
    Value pivot_perturb = Value{0})
{
    // void (the default) accumulates in Value, as the former `Accumulator = Value` did.
    using Acc = std::conditional_t<std::is_void_v<Accumulator>, Value, Accumulator>;
    using size_type = std::size_t;
    using std::abs;  // ADL: also find abs() for custom number types (e.g. posit/cfloat)
    using AT = accumulator_traits<Acc, Value>;  // numeric workspace policy
    size_type n = sym.n;
    // Row and node indices are stored as int32 below (pinv, xi): a larger n
    // would wrap them (issue 907). Rejected first, as supernodal_lu_numeric
    // does, before the checks that would need an A and a permutation of that
    // size, so a test can reach it (issue 926).
    if (n > static_cast<size_type>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument(
            "sparse_lu_numeric: n = " + std::to_string(n)
            + " exceeds the int32 index range of the factorization");
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "sparse_lu_numeric: matrix dimensions ("
            + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols())
            + ") do not match symbolic analysis (n=" + std::to_string(n) + ")");
    }
    if (sym.col_perm.size() != n || !util::is_valid_permutation(sym.col_perm)) {
        throw std::invalid_argument(
            "sparse_lu_numeric: symbolic permutation is invalid or wrong size");
    }
    // threshold <= 0 accepts any candidate, a zero one included (issue 864).
    if (!(threshold > Value{0} && threshold <= Value{1})) {
        throw std::invalid_argument("sparse_lu_numeric: threshold must be in (0, 1]");
    }

    // Column-permute A (apply the fill-reducing ordering) and convert to CSC.
    auto AQ = util::column_permute(A, sym.col_perm);
    auto C = util::crs_to_csc(AQ);

    // Left-looking Gilbert-Peierls LU with threshold partial pivoting
    // (Davis, "Direct Methods for Sparse Linear Systems", cs_lu). Each column
    // is computed by a sparse triangular solve over the *reach* of C(:,k) in
    // the partially-built L, so the total work is O(flops), not O(n^2).
    //
    //   pinv[old_row] = pivot position of that row, or -1 if not yet pivotal.
    //
    // L and U are accumulated in flat CSC arrays. L row indices are kept in
    // original-row space during factorization and remapped to pivot positions
    // at the end. L is unit lower (diagonal stored first per column, value 1);
    // U has its diagonal stored last per column -- the layouts dense_lower_solve
    // / dense_upper_solve expect.
    // Row-index / node arrays use 32-bit indices to halve memory bandwidth in
    // the (memory-bound) numeric gather/scatter -- all values are node/row ids
    // in [0, n) with a -1 sentinel, and any in-memory block has n < 2^31 (KLU
    // uses int32 here too). Column pointers (Lp/Up/xprune) and the DFS resume
    // stack (pstack, which holds positions into Li up to the fill count) stay
    // 64-bit.
    using idx32 = std::int32_t;
    std::vector<idx32>          pinv(n, -1);
    std::vector<Acc>    x(n);             // dense numeric workspace (accumulator policy)
    std::vector<idx32>          xi(n);            // DFS stack + reach output
    std::vector<std::ptrdiff_t> pstack(n);        // DFS resume-pointer stack (Li positions)
    std::vector<char>           marked(n, 0);     // reach marks
    std::vector<idx32>          reach_nodes;      // marked nodes, for O(reach) reset
    reach_nodes.reserve(n);

    std::vector<size_type>      Lp(n + 1, 0), Up(n + 1, 0);
    // Eisenstat-Liu symmetric pruning: xprune[j] is the end of the *pruned*
    // prefix of column j of L used by the symbolic reach DFS (the numeric update
    // still uses the full column). It restricts the DFS to the pivotal-row
    // prefix once a symmetric entry (j,k)+(k,j) appears, which is the dominant
    // constant-factor win in scalar left-looking LU (SuperLU pruneL).
    std::vector<size_type>      xprune(n, 0);
    std::vector<idx32> Li;  std::vector<Value> Lx;
    std::vector<idx32> Ui;  std::vector<Value> Ux;
    Li.reserve(C.values.size() * 2); Lx.reserve(C.values.size() * 2);
    Ui.reserve(C.values.size() * 2); Ux.reserve(C.values.size() * 2);
    std::size_t num_perturbed = 0;     // count of perturbed pivots (pivot_perturb > 0)

    for (size_type k = 0; k < n; ++k) {
        Lp[k] = Li.size();
        Up[k] = Ui.size();

        // --- reach(C(:,k)) in L: iterative DFS, topological order in xi[top..n) ---
        reach_nodes.clear();
        size_type top = n;
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            std::ptrdiff_t start = static_cast<std::ptrdiff_t>(C.row_ind[p]);
            if (marked[start]) continue;
            std::ptrdiff_t head = 0;
            xi[0] = start;
            while (head >= 0) {
                std::ptrdiff_t node = xi[head];
                std::ptrdiff_t jcol = pinv[node];          // column of L, if pivotal
                if (!marked[node]) {
                    marked[node] = 1;
                    reach_nodes.push_back(node);
                    pstack[head] = (jcol < 0) ? 0
                                 : static_cast<std::ptrdiff_t>(Lp[jcol]);
                }
                bool done = true;
                // Symbolic DFS traverses only the pruned prefix of column jcol.
                std::ptrdiff_t pend = (jcol < 0) ? 0
                                    : static_cast<std::ptrdiff_t>(xprune[jcol]);
                for (std::ptrdiff_t p2 = pstack[head]; p2 < pend; ++p2) {
                    std::ptrdiff_t child = Li[p2];
                    if (marked[child]) continue;
                    pstack[head] = p2;                     // resume here on return
                    xi[++head] = child;
                    done = false;
                    break;
                }
                if (done) {
                    --head;
                    xi[--top] = node;
                }
            }
        }

        // --- scatter C(:,k) over the reach, then x = L \ C(:,k) ---
        for (size_type p = top; p < n; ++p) AT::clear(x[xi[p]]);
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            AT::assign(x[C.row_ind[p]], C.values[p]);

        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t j = xi[px];
            std::ptrdiff_t jcol = pinv[j];
            if (jcol < 0) continue;                        // column not yet formed
            Value xj = AT::value(x[j]);                    // round U(j,k) once (single rounding)
            // L(j,j) == 1 (unit lower, stored first) -> no divide needed. The
            // elimination subtraction x[i] -= L(i,j)*xj is an accumulate of the
            // negated product (negating a Value is exact).
            for (size_type p = Lp[jcol] + 1; p < Lp[jcol + 1]; ++p)
                AT::add_product(x[Li[p]], -Lx[p], xj);
        }

        // --- threshold partial pivoting + emit U(:,k) for already-pivotal rows ---
        Value a = Value{0};                                // max |candidate|
        Value colnorm = Value{0};                          // ||column k||_inf over the reach
        std::ptrdiff_t ipiv = -1;
        bool have = false;
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            Value mag = abs(AT::value(x[i]));
            if (mag > colnorm) colnorm = mag;
            if (pinv[i] < 0) {                             // pivot candidate / L entry
                if (!have || mag > a) { a = mag; ipiv = i; have = true; }
            } else {                                       // U(pinv[i], k)
                Ui.push_back(pinv[i]);
                Ux.push_back(AT::value(x[i]));
            }
        }
        const bool perturb = pivot_perturb > Value{0};
        if (ipiv < 0) {                                    // no candidate row at all
            throw std::runtime_error(
                "sparse_lu_numeric: zero pivot at column " + std::to_string(k)
                + " (matrix is singular)");
        }
        if (!perturb && a == Value{0}) {                   // default: hard throw, unchanged
            throw std::runtime_error(
                "sparse_lu_numeric: zero pivot at column " + std::to_string(k)
                + " (matrix is singular)");
        }
        // Prefer the diagonal if it meets the threshold. Column k of C is
        // A(:, q[k]) with rows still in the original numbering, so its diagonal
        // is row q[k], not row k (issue 916; CSparse cs_lu does the same).
        // A zero diagonal never displaces a nonzero maximum, even when
        // threshold * a underflows to zero.
        const auto kd = static_cast<std::ptrdiff_t>(sym.col_perm[k]);
        if (pinv[kd] < 0) {
            const Value dmag = abs(AT::value(x[kd]));
            if (dmag >= threshold * a && (dmag != Value{0} || a == Value{0}))
                ipiv = kd;
        }

        Value pivot = AT::value(x[ipiv]);
        // Opt-in zero-pivot perturbation: if the chosen pivot has collapsed below
        // pivot_perturb * ||column||, replace it (sign-preserving) by that floor
        // rather than failing. A truly zero column (colnorm == 0) cannot be
        // perturbed -- it is genuinely singular, so still throw.
        if (perturb) {
            const Value floor = pivot_perturb * colnorm;
            if (abs(pivot) < floor) {
                pivot = (pivot < Value{0}) ? -floor : floor;
                ++num_perturbed;
            } else if (abs(pivot) == Value{0}) {           // colnorm == 0 too
                throw std::runtime_error(
                    "sparse_lu_numeric: zero pivot at column " + std::to_string(k)
                    + " (matrix is singular; pivot column is numerically zero)");
            }
        }
        Ui.push_back(static_cast<std::ptrdiff_t>(k));      // U(k,k) stored last
        Ux.push_back(pivot);
        pinv[ipiv] = static_cast<std::ptrdiff_t>(k);

        Li.push_back(ipiv);                                // L(k,k) = 1 stored first
        Lx.push_back(Value{1});
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {                             // remaining -> L(i,k)
                Li.push_back(i);
                Lx.push_back(AT::value(x[i]) / pivot);
            }
            AT::clear(x[i]);                               // clear workspace (reach only)
        }
        xprune[k] = Li.size();                             // column k initially unpruned

        // --- symmetric pruning (Eisenstat-Liu / SuperLU pruneL) ---
        // For each column j contributing to U(:,k): if the symmetric entry is
        // present (pivot row of k appears in L(:,j)), partition L(:,j) so the
        // already-pivotal rows form a prefix, and prune the DFS to that prefix.
        {
            size_type u_diag = Ui.size() - 1;              // last U entry is diag k
            for (size_type up = Up[k]; up < u_diag; ++up) {
                size_type j = static_cast<size_type>(Ui[up]);
                if (xprune[j] != Lp[j + 1]) continue;      // already pruned
                bool sym = false;                          // pivot row ipiv in L(:,j)?
                for (size_type p = Lp[j]; p < Lp[j + 1]; ++p)
                    if (Li[p] == ipiv) { sym = true; break; }
                if (!sym) continue;
                size_type kmin = Lp[j] + 1, kmax = Lp[j + 1] - 1;  // skip diagonal
                while (kmin <= kmax) {
                    if (pinv[Li[kmax]] < 0)            { --kmax; }
                    else if (pinv[Li[kmin]] >= 0)      { ++kmin; }
                    else {
                        std::swap(Li[kmin], Li[kmax]);
                        std::swap(Lx[kmin], Lx[kmax]);
                        ++kmin; --kmax;
                    }
                }
                xprune[j] = kmin;                          // [Lp[j], kmin) = pivotal prefix
            }
        }

        for (std::ptrdiff_t node : reach_nodes) marked[node] = 0;  // O(reach) reset
    }
    Lp[n] = Li.size();
    Up[n] = Ui.size();

    // Remap L row indices from original-row space to pivot positions.
    for (size_type p = 0; p < Li.size(); ++p)
        Li[p] = pinv[Li[p]];

    lu_numeric<Value> result;
    result.symbolic = sym;
    result.num_perturbed = num_perturbed;

    // Row permutation: pinv[old] = position; row_perm[position] = old.
    result.row_perm.resize(n);
    result.row_pinv.resize(n);
    for (size_type i = 0; i < n; ++i) {
        size_type pos = static_cast<size_type>(pinv[i]);
        result.row_pinv[i] = pos;
        result.row_perm[pos] = i;
    }

    // Build CSC, sorting each column by row index (ascending). Sorting keeps the
    // diagonal first in L (rows >= k, min = k) and last in U (rows <= k, max = k)
    // -- the layout the triangular solves expect -- and gives each column a
    // topological order so sparse_lu_refactor() can replay the numeric values
    // without a reach DFS. The cost is O(fill * log(col_len)), negligible vs the
    // numeric flops.
    std::vector<size_type> ord;
    auto to_csc = [&](const std::vector<size_type>& Mp,
                      const std::vector<idx32>& Mi,
                      const std::vector<Value>& Mx) -> util::csc_matrix<Value> {
        util::csc_matrix<Value> M;
        M.nrows = n;
        M.ncols = n;
        M.col_ptr = Mp;
        M.row_ind.resize(Mi.size());
        M.values.resize(Mx.size());
        for (size_type c = 0; c < n; ++c) {
            size_type b = Mp[c], e = Mp[c + 1];
            ord.resize(e - b);
            for (size_type t = 0; t < e - b; ++t) ord[t] = b + t;
            std::sort(ord.begin(), ord.end(),
                      [&](size_type a, size_type c2) { return Mi[a] < Mi[c2]; });
            for (size_type t = 0; t < e - b; ++t) {
                M.row_ind[b + t] = static_cast<size_type>(Mi[ord[t]]);
                M.values[b + t]  = Mx[ord[t]];
            }
        }
        return M;
    };
    // Install the factors and build their coupled solve schedules atomically.
    result.set_factor(to_csc(Lp, Li, Lx), to_csc(Up, Ui, Ux));
    return result;
}

/// Refactorize IN PLACE: recompute `fac`'s numeric values from a matrix with the
/// SAME sparsity pattern, reusing its symbolic structure and pivot sequence --
/// no BTF/ordering, no reach DFS, no pivot search -- and install them while
/// KEEPING the factor pattern and the coupled solve schedules (#310). This is the
/// allocation-light form of the fast path for repeated solves of a fixed pattern
/// with changing values (e.g. SPICE transient analysis: one factor, many
/// refactor); nothing is copied but the recomputed values.
///
/// Preconditions: A has the same column ordering (fac.symbolic) and its nonzero
/// pattern lies inside the factor's (A's pattern at factor time, plus fill).
/// Checked: an entry outside it throws std::invalid_argument. Relies on fac.L / fac.U columns being
/// stored in ascending row order (as sparse_lu_numeric produces).
///
/// Strong exception guarantee: the values are computed into scratch arrays and
/// installed only after the last column succeeds, so a throw leaves `fac` exactly
/// as it was -- still solvable with its previous values.
///
/// A reused pivot that the new values make tiny but not zero is accepted, and
/// the solve can then be wrong (issue 946). Check `fac.rgrowth(A)` afterwards:
/// a tiny value means the kept pivots no longer fit; factor afresh.
///
/// \throws std::runtime_error if a reused pivot is exactly zero for the new
///         values (the prior pivot sequence is no longer valid).
/// \throws std::invalid_argument if A has an entry outside the factor's pattern.
template <typename Value, typename Parameters>
    requires OrderedField<Value>
void sparse_lu_refactor_in_place(
    const mat::compressed2D<Value, Parameters>& A,
    lu_numeric<Value>& fac)
{
    using size_type = std::size_t;
    size_type n = fac.symbolic.n;
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "sparse_lu_refactor: matrix dimensions do not match prior factorization");
    }
    // The replay below indexes fac's factor pattern; with no factor installed
    // those arrays are empty and the walk would run off the end.
    if (!fac.has_factor()) {
        throw std::logic_error(
            "sparse_lu_refactor: prior factorization has no factor installed");
    }

    auto AQ = util::column_permute(A, fac.symbolic.col_perm);
    auto C  = util::crs_to_csc(AQ);

    const auto& pinv = fac.row_pinv;     // original row -> pivot position (fixed)
    // The installed factors' PATTERN is reused verbatim (read-only); only the
    // values are recomputed, into scratch arrays laid out on that pattern. Every
    // position is written by the column that owns it before any later column reads
    // it, so the zero-initialization is just placeholder storage. Installing these
    // via set_factor_values keeps the existing solve schedules (value-agnostic,
    // and valid for an unchanged pattern) instead of rebuilding them. Throws on a
    // zero pivot; no perturbation.
    const util::csc_matrix<Value>& L = fac.factorL();
    const util::csc_matrix<Value>& U = fac.factorU();
    std::vector<Value> Lx(L.values.size(), Value{0});
    std::vector<Value> Ux(U.values.size(), Value{0});

    std::vector<Value> x(n, Value{0});
    // mark[i] == k: pivot-space row i is in L(:,k) or U(:,k), the positions the
    // replay below computes for column k.
    std::vector<size_type> mark(n, n);

    for (size_type k = 0; k < n; ++k) {
        // An entry of A outside the factor's pattern would be scattered and never
        // consumed: the refactor would return a wrong factor with no error (issue
        // 864). Entries in the fill are fine: U(:,k)'s pattern is closed under the
        // reach in L, so the replay computes them exactly.
        for (size_type p = U.col_ptr[k]; p < U.col_ptr[k + 1]; ++p) mark[U.row_ind[p]] = k;
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) mark[L.row_ind[p]] = k;
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            if (mark[pinv[C.row_ind[p]]] != k) {
                throw std::invalid_argument(
                    "sparse_lu_refactor: A has an entry in row "
                    + std::to_string(C.row_ind[p]) + ", column "
                    + std::to_string(fac.symbolic.col_perm[k])
                    + " outside the prior factor's pattern; factor it afresh");
            }
        }

        // Scatter column k of the (column-permuted) A into the pivot-space
        // workspace.
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            x[pinv[C.row_ind[p]]] = C.values[p];

        // Left-looking updates from contributing columns j < k. U(:,k) is sorted
        // ascending with the diagonal (row k) last, so the strict-upper entries
        // are exactly the contributing columns in topological order.
        size_type ubeg = U.col_ptr[k], uend = U.col_ptr[k + 1];
        for (size_type p = ubeg; p + 1 < uend; ++p) {   // exclude diagonal (last)
            size_type j = U.row_ind[p];
            Value xj = x[j];
            if (xj == Value{0}) continue;
            // L(:,j): diagonal (row j, value 1) is first -> skip it. j < k, so
            // Lx(:,j) already holds this refactorization's values.
            for (size_type q = L.col_ptr[j] + 1; q < L.col_ptr[j + 1]; ++q)
                x[L.row_ind[q]] -= Lx[q] * xj;
        }

        Value pivot = x[k];
        if (pivot == Value{0}) {
            throw std::runtime_error(
                "sparse_lu_refactor: zero pivot at column " + std::to_string(k)
                + " (prior pivot sequence invalid for the new values)");
        }

        // Write U(:,k) values (all rows <= k, including the diagonal = pivot).
        for (size_type p = ubeg; p < uend; ++p)
            Ux[p] = x[U.row_ind[p]];

        // Write L(:,k): unit diagonal first, then x[row]/pivot below.
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) {
            size_type r = L.row_ind[p];
            Lx[p] = (r == k) ? Value{1} : x[r] / pivot;
        }

        // Clear the workspace. Clear the U/L pattern AND the scattered A column:
        // for an exact same-pattern refactor these coincide, but if A carries an
        // entry outside the prior pattern (a pattern mismatch) it is scattered
        // here yet never consumed by the replay -- clearing the scatter positions
        // too keeps such an entry from polluting later columns.
        for (size_type p = ubeg; p < uend; ++p)               x[U.row_ind[p]] = Value{0};
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) x[L.row_ind[p]] = Value{0};
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) x[pinv[C.row_ind[p]]] = Value{0};
    }

    // Install only the values: the symbolic analysis, permutations, factor pattern
    // and both solve schedules stay exactly as they were. The schedules are
    // value-agnostic and the pattern is unchanged, so rebuilding them here would
    // reproduce them entry for entry at O(nnz) cost per refactor.
    // The validating install goes first: everything after it is noexcept, so a
    // rejected install leaves `fac` wholly unchanged.
    fac.set_factor_values(std::move(Lx), std::move(Ux));
    fac.num_perturbed = 0;      // refactor recomputes values without perturbing
}

/// Refactorize into a NEW factorization object, leaving `prev` untouched: a copy
/// of prev (symbolic analysis, permutations, factor pattern and solve schedules)
/// refactorized in place. Prefer sparse_lu_refactor_in_place in a transient loop
/// that overwrites its factorization anyway -- it skips this copy.
/// Same preconditions and exceptions as sparse_lu_refactor_in_place; on a throw
/// `prev` is untouched (the copy is discarded).
template <typename Value, typename Parameters>
    requires OrderedField<Value>
lu_numeric<Value> sparse_lu_refactor(
    const mat::compressed2D<Value, Parameters>& A,
    const lu_numeric<Value>& prev)
{
    lu_numeric<Value> result = prev;
    sparse_lu_refactor_in_place(A, result);
    return result;
}

/// One-shot sparse LU solve: factor and solve A*x = b.
/// `Accumulator` (default `void`) is forwarded to both the numeric factorization
/// and the solve, so one policy covers the whole solve path (#535).
template <typename Accumulator = void, typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
    requires OrderedField<Value>
void sparse_lu_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering,
    Value threshold = Value{1})
{
    auto sym = sparse_lu_symbolic(A, ordering);
    auto num = sparse_lu_numeric<Accumulator>(A, sym, threshold);
    num.template solve<Accumulator>(x, b);
}

/// One-shot sparse LU solve without ordering (natural ordering).
/// `Accumulator` (default `void`) is forwarded to both the numeric factorization
/// and the solve, so one policy covers the whole solve path (#535).
template <typename Accumulator = void, typename Value, typename Parameters, typename VecX, typename VecB>
void sparse_lu_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = sparse_lu_symbolic(A);
    auto num = sparse_lu_numeric<Accumulator>(A, sym);
    num.template solve<Accumulator>(x, b);
}

} // namespace mtl::sparse::factorization
