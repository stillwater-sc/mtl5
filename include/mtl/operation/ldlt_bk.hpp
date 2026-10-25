#pragma once
// MTL5 -- Bunch-Kaufman pivoted LDL^T factorization for symmetric matrices
// P * A * P^T = L * D * L^T where:
//   L is unit lower triangular
//   D is block-diagonal with 1x1 and 2x2 blocks
//   P is a permutation matrix
//
// The Bunch-Kaufman partial pivoting strategy selects the largest available
// pivot at each step, using 2x2 blocks when no single diagonal entry is
// a good pivot. This handles symmetric indefinite matrices without breakdown
// and provides bounded element growth (factor <= (1+sqrt(17))/8 ~= 2.57).
//
// The pivot sequence is stored in LAPACK convention:
//   ipiv[k] > 0  -> 1x1 pivot, row/col swap with ipiv[k]-1
//   ipiv[k] < 0 && ipiv[k+1] < 0  -> 2x2 pivot using rows |ipiv[k]|-1, |ipiv[k+1]|-1
//
// Reference:
//   Bunch & Kaufman, "Some Stable Methods for Calculating Inertia and
//   Solving Symmetric Linear Systems", Math. Comp. 31(137), 1977.
//   Golub & Van Loan, "Matrix Computations", Section 4.4.
//
// Mixed precision (#585): ldlt_bk_factor and ldlt_bk_solve take a leading
// `Accumulator` (default void, see math/accumulator_traits.hpp); void is
// byte-identical to the routines before #585. The void factorization is
// right-looking -- every trailing entry is updated once per step -- so there is
// no sum for an accumulator to hold. A non-void accumulator therefore runs a
// LEFT-looking Bunch-Kaufman: column k of the Schur complement is formed on
// demand, each entry one reduction
//
//     S(i,k) = A(i,k) - sum_{p<k} L(i,p) W(k,p),     W = L * D,
//
// rounded once (W is the unscaled pivot columns), and so is column r when the
// pivot tests need it. Same tests, same alpha, same 1x1 / 2x2 choices on the
// computed values, same ipiv encoding. While factoring it keeps L and W in one
// row order (interchanges permute whole rows); at the end it undoes the later
// interchanges on the earlier columns, which is the per-step layout
// ldlt_bk_solve reads (see symmetric_swap). The solve likewise rebuilds the
// one-row-order L, permutes b once, and substitutes by rows, forming each
// b(i) - sum in the accumulator, rounded once, as the sparse solves do (#535).
// The accumulator path stores W (n x n) in the factorization and a copy of L
// in the solve.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/thread_pool.hpp>

namespace mtl {

/// Pivot information from Bunch-Kaufman factorization.
/// Uses LAPACK ipiv convention for 1x1 and 2x2 block pivots.
struct bk_pivot_info {
    std::vector<int> ipiv;
};

/// Swap rows and columns r1 and r2 in the lower triangle of a symmetric matrix,
/// touching only columns >= first_col. Only the lower triangle (and diagonal)
/// is accessed and modified.
///
/// `first_col` matters for Bunch-Kaufman. The factorization is a PRODUCT of
/// per-step permutations and elementary transforms,
///
///     A = (P_0 L_0)(P_1 L_1)... D ...(P_1 L_1)^T (P_0 L_0)^T
///
/// not a single global permutation, and `ldlt_bk_solve` replays those
/// permutations one step at a time (as LAPACK's dsytrs does). So the L columns
/// already written by earlier steps must be left ALONE: they are recorded in the
/// row ordering that was current when they were computed, and the solve reaches
/// them before the later interchanges are replayed.
///
/// Permuting them here instead produces a factor in the "single global P"
/// convention while the solve still reads the per-step one. The two disagree
/// the moment any interchange happens, which silently returned wrong solutions
/// with info == 0 (#335). Passing first_col = k keeps the swap inside the
/// trailing submatrix, which is also what makes the stored factor match
/// LAPACK's dsytrf byte for byte.
template <Matrix M>
void symmetric_swap(M& A, std::size_t r1, std::size_t r2, std::size_t first_col = 0) {
    if (r1 == r2) return;
    if (r1 > r2) std::swap(r1, r2);
    std::size_t n = A.num_rows();

    // Swap diagonal entries
    auto tmp = A(r1, r1);
    A(r1, r1) = A(r2, r2);
    A(r2, r2) = tmp;

    // Swap entries in rows r1 and r2 for columns in [first_col, r1)
    for (std::size_t j = first_col; j < r1; ++j) {
        tmp = A(r1, j);
        A(r1, j) = A(r2, j);
        A(r2, j) = tmp;
    }

    // Swap entries in column r1 (rows between r1 and r2) with row r2
    for (std::size_t i = r1 + 1; i < r2; ++i) {
        tmp = A(i, r1);
        A(i, r1) = A(r2, i);  // A(r2, i) is in upper triangle -> use A(i, r2) conceptually
        A(r2, i) = tmp;
    }

    // Swap entries in rows r1 and r2 for columns between r1 and r2 handled above
    // Swap entries below r2 in columns r1 and r2
    for (std::size_t i = r2 + 1; i < n; ++i) {
        tmp = A(i, r1);
        A(i, r1) = A(i, r2);
        A(i, r2) = tmp;
    }
}

namespace detail {

/// One interchange of the factorization: rows r1 < r2, made at step k (so it
/// applies to the L columns p < k in the one-row-order convention).
struct bk_interchange {
    std::size_t k, r1, r2;
};

/// The interchanges ipiv records, in the order the factorization made them.
inline std::vector<bk_interchange> bk_interchanges(const std::vector<int>& ipiv) {
    std::vector<bk_interchange> out;
    const std::size_t n = ipiv.size();
    for (std::size_t k = 0; k < n;) {
        if (ipiv[k] > 0) {
            const auto p = static_cast<std::size_t>(ipiv[k] - 1);
            if (p != k) out.push_back({k, std::min(k, p), std::max(k, p)});
            ++k;
        } else {
            const auto p = static_cast<std::size_t>(-ipiv[k] - 1);
            if (p != k + 1) out.push_back({k, std::min(k + 1, p), std::max(k + 1, p)});
            k += 2;
        }
    }
    return out;
}

/// ldlt_bk_factor's accumulator path: left-looking Bunch-Kaufman (top of file).
template <typename Accumulator, typename M>
int ldlt_bk_factor_acc(M& A, bk_pivot_info& pivots) {
    using value_type = typename M::value_type;
    using AT = math::accumulator_traits<Accumulator, value_type>;
    using std::abs;
    const std::size_t n = A.num_rows();
    const value_type alpha = value_type((1.0 + std::sqrt(17.0)) / 8.0);

    std::vector<value_type> W(n * n);              // W(j,p) = W[j * n + p], p < k
    std::vector<value_type> ck(n), cr(n);          // columns k and r of S, rows >= k
    std::vector<bk_interchange> swaps;

    // S(i,j) at step k, from the lower triangle's not-yet-factored entry.
    auto schur = [&](std::size_t i, std::size_t j, std::size_t k) {
        Accumulator acc{};
        AT::assign(acc, value_type(i >= j ? A(i, j) : A(j, i)));
        for (std::size_t p = 0; p < k; ++p)
            AT::add_product(acc, value_type(-A(i, p)), W[j * n + p]);
        return AT::template value<value_type>(acc);
    };
    // c[i] = S(i,j) for i in [k, n); entries are independent.
    auto column = [&](std::vector<value_type>& c, std::size_t j, std::size_t k) {
        const std::size_t grain =
            std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / (k == 0 ? 1 : k));
        ::mtl::parallel::pool().parallel_for(n - k, grain,
            [&](std::size_t b, std::size_t e) {
                for (std::size_t t = b; t < e; ++t) c[k + t] = schur(k + t, j, k);
            });
    };
    // Interchange rows/columns r1 < r2 everywhere: L, W and the unfactored part.
    auto interchange = [&](std::size_t k, std::size_t r1, std::size_t r2) {
        symmetric_swap(A, r1, r2, 0);
        for (std::size_t p = 0; p < k; ++p) std::swap(W[r1 * n + p], W[r2 * n + p]);
        swaps.push_back({k, r1, r2});
    };

    std::size_t k = 0;
    while (k < n) {
        column(ck, k, k);
        if (k == n - 1) {
            if (ck[k] == math::zero<value_type>())
                return static_cast<int>(k + 1);
            A(k, k) = ck[k];
            pivots.ipiv[k] = static_cast<int>(k + 1);
            ++k;
            continue;
        }

        std::size_t r = k + 1;
        value_type lambda_val = abs(ck[k + 1]);
        for (std::size_t i = k + 2; i < n; ++i) {
            value_type aik = abs(ck[i]);
            if (aik > lambda_val) {
                lambda_val = aik;
                r = i;
            }
        }

        bool use_2x2 = false;
        std::size_t swap_r = k;
        if (lambda_val == math::zero<value_type>()) {
            if (ck[k] == math::zero<value_type>())
                return static_cast<int>(k + 1);
        } else if (abs(ck[k]) >= alpha * lambda_val) {
            // Test 1: 1x1, no interchange.
        } else {
            column(cr, r, k);
            cr[k] = ck[r];                         // S(k,r) == S(r,k): one value for both
            value_type sigma = math::zero<value_type>();
            for (std::size_t i = k; i < n; ++i)
                if (i != r && abs(cr[i]) > sigma) sigma = abs(cr[i]);

            if (abs(ck[k]) * sigma >= alpha * lambda_val * lambda_val) {
                // Test 2: 1x1, no interchange.
            } else if (abs(cr[r]) >= alpha * sigma) {
                // Test 3: 1x1 on S(r,r); interchange k <-> r. The new column k is
                // column r with entries k and r exchanged.
                swap_r = r;
                interchange(k, k, r);
                std::swap(cr[k], cr[r]);
                std::swap(ck, cr);
            } else {
                // 2x2 on {k, r}; interchange k+1 <-> r, which permutes both columns.
                use_2x2 = true;
                if (r != k + 1) {
                    interchange(k, k + 1, r);
                    std::swap(ck[k + 1], ck[r]);
                    std::swap(cr[k + 1], cr[r]);
                }
            }
        }

        if (!use_2x2) {
            const value_type dkk = ck[k];
            if (dkk == math::zero<value_type>())
                return static_cast<int>(k + 1);
            pivots.ipiv[k] = static_cast<int>(swap_r + 1);
            A(k, k) = dkk;
            for (std::size_t i = k + 1; i < n; ++i) {
                W[i * n + k] = ck[i];
                A(i, k) = ck[i] / dkk;
            }
            ++k;
        } else {
            const value_type d00 = ck[k], d10 = ck[k + 1], d11 = cr[k + 1];
            const value_type det = d00 * d11 - d10 * d10;
            if (det == math::zero<value_type>())
                return static_cast<int>(k + 1);
            pivots.ipiv[k]     = -static_cast<int>(r + 1);
            pivots.ipiv[k + 1] = -static_cast<int>(r + 1);
            const value_type inv_det = math::one<value_type>() / det;
            const value_type di00 =  d11 * inv_det;
            const value_type di01 = -d10 * inv_det;
            const value_type di11 =  d00 * inv_det;
            A(k, k) = d00;
            A(k + 1, k) = d10;
            A(k + 1, k + 1) = d11;
            for (std::size_t i = k + 2; i < n; ++i) {
                W[i * n + k]     = ck[i];
                W[i * n + k + 1] = cr[i];
                A(i, k)     = ck[i] * di00 + cr[i] * di01;
                A(i, k + 1) = ck[i] * di01 + cr[i] * di11;
            }
            k += 2;
        }
    }

    // To the per-step layout: undo each interchange on the columns before it,
    // latest first.
    for (auto it = swaps.rbegin(); it != swaps.rend(); ++it)
        for (std::size_t p = 0; p < it->k; ++p)
            std::swap(A(it->r1, p), A(it->r2, p));
    return 0;
}

/// ldlt_bk_solve's accumulator path (top of file).
template <typename Accumulator, typename M, typename VecX, typename VecB>
void ldlt_bk_solve_acc(const M& A, const bk_pivot_info& pivots, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using AT = math::accumulator_traits<Accumulator, value_type>;
    const std::size_t n = A.num_rows();
    const auto swaps = bk_interchanges(pivots.ipiv);

    // L in one row order: replay each interchange on the columns before it.
    std::vector<value_type> L(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) L[i * n + j] = value_type(A(i, j));
    for (const auto& s : swaps)
        for (std::size_t p = 0; p < s.k; ++p) std::swap(L[s.r1 * n + p], L[s.r2 * n + p]);
    // (k+1, k) of a 2x2 block holds D, not L.
    std::vector<bool> block2(n, false);
    for (std::size_t k = 0; k < n;) {
        if (pivots.ipiv[k] > 0) { ++k; } else { block2[k] = true; k += 2; }
    }

    for (std::size_t i = 0; i < n; ++i) x(i) = b(i);
    for (const auto& s : swaps) std::swap(x(s.r1), x(s.r2));

    for (std::size_t i = 0; i < n; ++i) {          // L y = P b
        Accumulator acc{};
        AT::assign(acc, value_type(x(i)));
        for (std::size_t j = 0; j < i; ++j)
            if (!(j + 1 == i && block2[j]))
                AT::add_product(acc, value_type(-L[i * n + j]), value_type(x(j)));
        x(i) = AT::template value<value_type>(acc);
    }

    for (std::size_t k = 0; k < n;) {              // D z = y, as the void solve
        if (!block2[k]) {
            if (A(k, k) == math::zero<value_type>())
                throw std::domain_error("ldlt_bk_solve: zero 1x1 pivot in D");
            x(k) = x(k) / A(k, k);
            ++k;
        } else {
            value_type d00 = A(k, k);
            value_type d10 = A(k + 1, k);
            value_type d11 = A(k + 1, k + 1);
            value_type det = d00 * d11 - d10 * d10;
            if (det == math::zero<value_type>())
                throw std::domain_error("ldlt_bk_solve: singular 2x2 pivot in D");
            value_type xk  = x(k);
            value_type xk1 = x(k + 1);
            x(k)     = (d11 * xk - d10 * xk1) / det;
            x(k + 1) = (d00 * xk1 - d10 * xk) / det;
            k += 2;
        }
    }

    for (std::size_t ii = n; ii-- > 0;) {          // L^T u = z
        Accumulator acc{};
        AT::assign(acc, value_type(x(ii)));
        for (std::size_t j = ii + 1; j < n; ++j)
            if (!(j == ii + 1 && block2[ii]))
                AT::add_product(acc, value_type(-L[j * n + ii]), value_type(x(j)));
        x(ii) = AT::template value<value_type>(acc);
    }

    for (auto it = swaps.rbegin(); it != swaps.rend(); ++it) std::swap(x(it->r1), x(it->r2));
}

} // namespace detail

/// Bunch-Kaufman pivoted LDL^T factorization: P*A*P^T = L*D*L^T.
/// In-place: lower triangle of A is overwritten with L (unit diagonal implicit),
/// D occupies the diagonal and first subdiagonal for 2x2 blocks.
/// Returns 0 on success, k+1 if a singular block is encountered at step k.
/// A non-void `Accumulator` runs the left-looking form (top of file).
template <typename Accumulator = void, FieldMatrix M>
int ldlt_bk_factor(M& A, bk_pivot_info& pivots) {
    using value_type = typename M::value_type;
    using std::abs;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n);

    pivots.ipiv.resize(n, 0);

    if constexpr (!std::is_void_v<Accumulator>)
        return detail::ldlt_bk_factor_acc<Accumulator>(A, pivots);

    // Bunch-Kaufman threshold: alpha = (1 + sqrt(17)) / 8
    const value_type alpha = value_type((1.0 + std::sqrt(17.0)) / 8.0);

    std::size_t k = 0;
    while (k < n) {
        if (k == n - 1) {
            // Last 1x1 pivot - no choice
            if (A(k, k) == math::zero<value_type>())
                return static_cast<int>(k + 1);
            pivots.ipiv[k] = static_cast<int>(k + 1);  // 1-based, positive = 1x1
            ++k;
            continue;
        }

        // Step 1: Find largest off-diagonal magnitude in column k (rows > k)
        std::size_t r = k + 1;
        value_type lambda_val = abs(A(k + 1, k));
        for (std::size_t i = k + 2; i < n; ++i) {
            value_type aik = abs(A(i, k));
            if (aik > lambda_val) {
                lambda_val = aik;
                r = i;
            }
        }

        // Step 2: Determine pivot type
        bool use_1x1 = false;
        bool use_2x2 = false;
        std::size_t swap_r = r;

        if (lambda_val == math::zero<value_type>()) {
            // Column k is zero below diagonal - 1x1 pivot (might be zero)
            if (A(k, k) == math::zero<value_type>())
                return static_cast<int>(k + 1);
            use_1x1 = true;
            swap_r = k;  // no swap needed
        } else if (abs(A(k, k)) >= alpha * lambda_val) {
            // Test 1: diagonal is large enough for 1x1 pivot, no swap needed
            use_1x1 = true;
            swap_r = k;
        } else {
            // Step 3: Find largest off-diagonal magnitude in row r (cols != r)
            value_type sigma = math::zero<value_type>();
            for (std::size_t j = k; j < r; ++j) {
                value_type arj = abs(A(r, j));
                if (arj > sigma) sigma = arj;
            }
            for (std::size_t i = r + 1; i < n; ++i) {
                value_type air = abs(A(i, r));
                if (air > sigma) sigma = air;
            }

            if (abs(A(k, k)) * sigma >= alpha * lambda_val * lambda_val) {
                // Test 2: diagonal good enough relative to both column maxima, no swap
                use_1x1 = true;
                swap_r = k;
            } else if (abs(A(r, r)) >= alpha * sigma) {
                // Test 3: A(r,r) is a good pivot - swap r<->k, use 1x1
                use_1x1 = true;
                swap_r = r;
                // first_col = k: leave the L columns written by steps < k
                // untouched; the solve replays this interchange at step k.
                symmetric_swap(A, k, r, k);
            } else {
                // Use 2x2 pivot from rows/cols {k, r}
                // Swap r <-> k+1 to put the 2x2 block at positions {k, k+1}
                use_2x2 = true;
                if (r != k + 1)
                    // first_col = k: column k belongs to the 2x2 block being
                    // formed and must be swapped; columns < k must not.
                    symmetric_swap(A, k + 1, r, k);
            }
        }

        if (use_1x1) {
            // 1x1 pivot at position k
            value_type dkk = A(k, k);
            if (dkk == math::zero<value_type>())
                return static_cast<int>(k + 1);

            // Record pivot (1-based, positive = 1x1 with swap target)
            pivots.ipiv[k] = static_cast<int>(swap_r + 1);

            // Perform rank-1 update on the trailing submatrix:
            //   A(i,j) -= A(i,k) * A(j,k) / D(k,k)  for i,j > k
            // Then store L(i,k) = A(i,k) / D(k,k).
            // Must update A(i,j) using original A(j,k) BEFORE overwriting
            // column k with L values. Process column-by-column:
            for (std::size_t j = k + 1; j < n; ++j) {
                value_type ajk = A(j, k);
                value_type ajk_over_d = ajk / dkk;
                for (std::size_t i = j; i < n; ++i)
                    A(i, j) = A(i, j) - A(i, k) * ajk_over_d;
            }
            // Now store L column
            for (std::size_t i = k + 1; i < n; ++i)
                A(i, k) = A(i, k) / dkk;
            ++k;
        } else {
            // 2x2 pivot at positions {k, k+1}
            value_type d00 = A(k, k);
            value_type d10 = A(k + 1, k);
            value_type d11 = A(k + 1, k + 1);
            value_type det = d00 * d11 - d10 * d10;
            if (det == math::zero<value_type>())
                return static_cast<int>(k + 1);

            // Record pivot (negative = 2x2 block)
            // After swap, the 2x2 block uses positions k and k+1
            // ipiv[k] = -(original row that was swapped to k+1) 1-based
            // ipiv[k+1] = same (both negative signals 2x2)
            std::size_t actual_r = (r != k + 1) ? r : k + 1;
            pivots.ipiv[k]     = -static_cast<int>(actual_r + 1);
            pivots.ipiv[k + 1] = -static_cast<int>(actual_r + 1);

            // Inverse of 2x2 D block: D^{-1} = (1/det) * [[d11, -d10], [-d10, d00]]
            value_type inv_det = math::one<value_type>() / det;
            value_type di00 =  d11 * inv_det;
            value_type di01 = -d10 * inv_det;
            value_type di11 =  d00 * inv_det;

            // Perform rank-2 update on the trailing submatrix:
            //   A(i,j) -= [A(i,k) A(i,k+1)] * D^{-1} * [A(j,k) A(j,k+1)]^T
            // Must use original A(:,k) and A(:,k+1) before overwriting with L.
            // Process column-by-column in the trailing submatrix:
            for (std::size_t j = k + 2; j < n; ++j) {
                value_type ajk0 = A(j, k);
                value_type ajk1 = A(j, k + 1);
                // Compute D^{-1} * [ajk0, ajk1]^T
                value_type t0 = di00 * ajk0 + di01 * ajk1;
                value_type t1 = di01 * ajk0 + di11 * ajk1;
                for (std::size_t i = j; i < n; ++i)
                    A(i, j) = A(i, j) - A(i, k) * t0 - A(i, k + 1) * t1;
            }
            // Now store L columns: L(i,:) = [A(i,k) A(i,k+1)] * D^{-1}
            for (std::size_t i = k + 2; i < n; ++i) {
                value_type aik0 = A(i, k);
                value_type aik1 = A(i, k + 1);
                A(i, k)     = aik0 * di00 + aik1 * di01;
                A(i, k + 1) = aik0 * di01 + aik1 * di11;
            }
            k += 2;
        }
    }
    return 0;
}

/// Solve A*x = b using precomputed Bunch-Kaufman LDL^T factors.
/// The factored A contains L (unit lower, implicit diagonal) and D
/// (diagonal and subdiagonal for 2x2 blocks). pivots records the
/// permutation and block structure. A non-void `Accumulator` substitutes by
/// rows, forming each b(i) - sum in one reduction (top of file).
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
void ldlt_bk_solve(const M& A, const bk_pivot_info& pivots, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && b.size() == n);

    if constexpr (!std::is_void_v<Accumulator>) {
        detail::ldlt_bk_solve_acc<Accumulator>(A, pivots, x, b);
        return;
    }

    // Copy b into x (we work in-place on x)
    for (std::size_t i = 0; i < n; ++i)
        x(i) = b(i);

    // Step 1: Apply forward permutation and forward substitution (L * y = P * b)
    // Process blocks from top to bottom
    std::size_t k = 0;
    while (k < n) {
        if (pivots.ipiv[k] > 0) {
            // 1x1 pivot - swap if needed
            std::size_t p = static_cast<std::size_t>(pivots.ipiv[k] - 1);
            if (p != k) {
                auto tmp = x(k);
                x(k) = x(p);
                x(p) = tmp;
            }

            // Forward substitution for column k: x(i) -= L(i,k) * x(k)
            for (std::size_t i = k + 1; i < n; ++i)
                x(i) = x(i) - A(i, k) * x(k);
            ++k;
        } else {
            // 2x2 pivot - swap k+1 with |ipiv[k]|-1
            std::size_t p = static_cast<std::size_t>(-pivots.ipiv[k] - 1);
            if (p != k + 1) {
                auto tmp = x(k + 1);
                x(k + 1) = x(p);
                x(p) = tmp;
            }

            // Forward substitution for columns k and k+1
            for (std::size_t i = k + 2; i < n; ++i) {
                x(i) = x(i) - A(i, k) * x(k) - A(i, k + 1) * x(k + 1);
            }
            k += 2;
        }
    }

    // Step 2: Solve D * z = y (block-diagonal solve)
    k = 0;
    while (k < n) {
        if (pivots.ipiv[k] > 0) {
            // 1x1 block
            if (A(k, k) == math::zero<value_type>())
                throw std::domain_error("ldlt_bk_solve: zero 1x1 pivot in D");
            x(k) = x(k) / A(k, k);
            ++k;
        } else {
            // 2x2 block at {k, k+1}
            value_type d00 = A(k, k);
            value_type d10 = A(k + 1, k);
            value_type d11 = A(k + 1, k + 1);
            value_type det = d00 * d11 - d10 * d10;
            if (det == math::zero<value_type>())
                throw std::domain_error("ldlt_bk_solve: singular 2x2 pivot in D");

            value_type xk  = x(k);
            value_type xk1 = x(k + 1);
            x(k)     = (d11 * xk - d10 * xk1) / det;
            x(k + 1) = (d00 * xk1 - d10 * xk) / det;
            k += 2;
        }
    }

    // Step 3: Backward substitution (L^T * u = z) and reverse permutation
    // Process blocks from bottom to top
    if (n == 0) return;
    k = n;
    while (k > 0) {
        if (k >= 1 && pivots.ipiv[k - 1] > 0) {
            // 1x1 pivot at k-1
            --k;
            // Backward substitution: x(k) -= sum L(i,k) * x(i) for i > k
            for (std::size_t i = k + 1; i < n; ++i)
                x(k) = x(k) - A(i, k) * x(i);

            // Reverse permutation
            std::size_t p = static_cast<std::size_t>(pivots.ipiv[k] - 1);
            if (p != k) {
                auto tmp = x(k);
                x(k) = x(p);
                x(p) = tmp;
            }
        } else if (k >= 2) {
            // 2x2 pivot at {k-2, k-1}
            k -= 2;
            // Backward substitution for columns k and k+1
            for (std::size_t i = k + 2; i < n; ++i) {
                x(k)     = x(k)     - A(i, k) * x(i);
                x(k + 1) = x(k + 1) - A(i, k + 1) * x(i);
            }

            // Reverse permutation: swap k+1 with |ipiv[k]|-1
            std::size_t p = static_cast<std::size_t>(-pivots.ipiv[k] - 1);
            if (p != k + 1) {
                auto tmp = x(k + 1);
                x(k + 1) = x(p);
                x(p) = tmp;
            }
        } else {
            break;
        }
    }
}

} // namespace mtl
