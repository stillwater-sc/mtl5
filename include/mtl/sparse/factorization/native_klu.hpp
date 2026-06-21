#pragma once
// MTL5 -- Native KLU: BTF + block-wise Gilbert-Peierls LU
//
// A header-only, value-type-generic KLU sparse direct solver (cf. the external
// SuiteSparse binding in mtl/interface/klu.hpp, which is double-only). The
// algorithm follows Davis & Palamadai Natarajan, "Algorithm 907: KLU", ACM
// TOMS 2010:
//
//   1. Permute A to Block Triangular Form: B = P*A*Q is block upper triangular
//      with a zero-free diagonal (mtl::sparse::ordering::block_triangular_form).
//   2. Factor each diagonal block B_bb with the existing left-looking
//      Gilbert-Peierls sparse LU (sparse_lu_numeric).
//   3. Solve A*x = b by block back-substitution over the block structure.
//
// Solve derivation. With B = P*A*Q block upper triangular, A*x = b becomes
//   B*y = c,   c[i] = b[p[i]],   x[q[j]] = y[j].
// Because B is block UPPER triangular (nonzeros B[i,j] only when the block of i
// is <= the block of j), we solve the last block first and work upward: for
// block b,  B_bb * y_b = c_b - sum_{b' > b} B_{b,b'} * y_{b'}.
//
// Each diagonal block is ordered with AMD on A+A^T (KLU's default), the matrix is
// row-equilibrated before factorization, and native_klu_refactor reuses a prior
// factorization's structure + pivots for fast repeated solves of a fixed pattern.
// Iterative refinement remains out of scope here.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006;
//            Davis & Palamadai Natarajan, "Algorithm 907: KLU", ACM TOMS 2010.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>
#include <mtl/sparse/util/permutation.hpp>

namespace mtl::sparse::factorization {

/// Numeric KLU factorization: BTF permutation, the permuted matrix B = P*A*Q
/// (block upper triangular), the per-new-index block id, and a left-looking LU
/// factorization of each diagonal block.
template <typename Value>
struct klu_numeric {
    ordering::btf_result          btf;            ///< row/col permutations + blocks
    mat::compressed2D<Value>      B;              ///< P*(R*A)*Q, block upper triangular
    std::vector<std::size_t>      block_of;       ///< new index -> block id
    std::vector<lu_numeric<Value>> block_numeric; ///< LU of each diagonal block
    std::vector<Value>            row_scale;      ///< r[orig row]; empty = unscaled
                                                  ///< (factored R*A, so solve scales b)

    std::size_t num_rows() const { return btf.row_perm.size(); }
    std::size_t num_cols() const { return num_rows(); }
    std::size_t nblocks()  const { return btf.nblocks(); }

    /// Solve A*x = b via block back-substitution. x and b have the dimension of
    /// the original matrix A.
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        const std::size_t n = num_rows();
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "klu_numeric::solve: vector size mismatch");
        }

        // c[i] = (R*b)[p[i]]; y is updated in place to the final solution per
        // block. We factored R*A, so the RHS is row-scaled by the same R; the
        // resulting x solves the original A*x = b unchanged (no x unscaling).
        std::vector<Value> y(n);
        const bool scaled = !row_scale.empty();
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t orow = btf.row_perm[i];
            Value bi = static_cast<Value>(b(orow));
            y[i] = scaled ? row_scale[orow] * bi : bi;
        }

        const auto& rp  = B.ref_major();
        const auto& ci  = B.ref_minor();
        const auto& dat = B.ref_data();

        // Back-substitution: last block first.
        for (std::size_t blk = nblocks(); blk-- > 0; ) {
            const std::size_t r0 = btf.blocks[blk];
            const std::size_t r1 = btf.blocks[blk + 1];
            const std::size_t m  = r1 - r0;

            vec::dense_vector<Value> rhs(m), sol(m, Value{0});
            for (std::size_t i = r0; i < r1; ++i) {
                Value val = y[i];
                for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
                    const std::size_t j = ci[k];
                    if (block_of[j] > blk)          // coupling to a solved block
                        val -= dat[k] * y[j];
                    // block_of[j] == blk: part of the diagonal block (handled by
                    //   the block LU below). block_of[j] < blk: impossible for
                    //   block-upper-triangular B.
                }
                rhs(static_cast<int>(i - r0)) = val;
            }

            block_numeric[blk].solve(sol, rhs);

            for (std::size_t i = r0; i < r1; ++i)
                y[i] = sol(static_cast<int>(i - r0));
        }

        // x[q[j]] = y[j]
        for (std::size_t j = 0; j < n; ++j)
            x(btf.col_perm[j]) = static_cast<typename VecX::value_type>(y[j]);
    }
};

/// Factor A using native KLU: BTF, then a left-looking LU of each diagonal
/// block with threshold partial pivoting.
///
/// \throws std::invalid_argument if A is not square.
/// \throws std::runtime_error    if A is structurally singular (no perfect
///                               matching) or a diagonal block is numerically
///                               singular.
template <typename Value, typename Parameters>
    requires OrderedField<Value>
klu_numeric<Value> native_klu_factor(
    const mat::compressed2D<Value, Parameters>& A,
    Value threshold = Value{1},
    bool scale = true)
{
    using std::abs;  // ADL for custom number types
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument("native_klu_factor: matrix must be square");
    }
    const std::size_t n = A.num_rows();

    klu_numeric<Value> result;
    if (n == 0) {
        result.btf.blocks = {0};
        return result;
    }

    // BTF is structural; scaling does not change the pattern, so compute it on A.
    result.btf = ordering::block_triangular_form(A);
    if (result.btf.structurally_singular) {
        throw std::runtime_error(
            "native_klu_factor: matrix is structurally singular");
    }

    // Row scaling (KLU default): r[i] = 1 / max_j |A(i,j)|. Equilibrating the
    // rows keeps threshold partial pivoting close to the fill-reducing order on
    // badly-scaled indefinite blocks, which otherwise inflates fill. We factor
    // R*A and row-scale the RHS in solve(); x is unchanged.
    if (scale) {
        const auto& rp  = A.ref_major();
        const auto& ci  = A.ref_minor();
        const auto& dat = A.ref_data();
        result.row_scale.assign(n, Value{1});
        for (std::size_t r = 0; r < n; ++r) {
            Value m = Value{0};
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k) {
                Value a = abs(dat[k]);
                if (a > m) m = a;
            }
            if (m > Value{0}) result.row_scale[r] = Value{1} / m;
        }
    }

    // Build B = P*(R*A)*Q. New column of old column c is q_inv[c].
    auto q_inv = util::invert_permutation(result.btf.col_perm);
    mat::compressed2D<Value> B(n, n);
    {
        const auto& rp  = A.ref_major();
        const auto& ci  = A.ref_minor();
        const auto& dat = A.ref_data();
        mat::inserter<mat::compressed2D<Value>> ins(B);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t orow = result.btf.row_perm[i];
            Value s = scale ? result.row_scale[orow] : Value{1};
            for (std::size_t k = rp[orow]; k < rp[orow + 1]; ++k)
                ins[i][q_inv[ci[k]]] << s * dat[k];
        }
    }
    result.B = std::move(B);

    // new index -> block id
    result.block_of.assign(n, 0);
    for (std::size_t blk = 0; blk < result.btf.nblocks(); ++blk)
        for (std::size_t k = result.btf.blocks[blk];
             k < result.btf.blocks[blk + 1]; ++k)
            result.block_of[k] = blk;

    // Factor each diagonal block.
    const auto& Brp  = result.B.ref_major();
    const auto& Bci  = result.B.ref_minor();
    const auto& Bdat = result.B.ref_data();
    result.block_numeric.reserve(result.btf.nblocks());
    // Reusable raw-CSR workspace for block extraction. Building each block
    // directly with a two-pass counting sort avoids the inserter's per-entry
    // hashing/sorting/finalization and the per-block allocation churn (notable
    // over the many blocks of a reducible matrix).
    std::vector<std::size_t> bstart, bind;
    std::vector<Value>       bval;
    for (std::size_t blk = 0; blk < result.btf.nblocks(); ++blk) {
        const std::size_t r0 = result.btf.blocks[blk];
        const std::size_t r1 = result.btf.blocks[blk + 1];
        const std::size_t m  = r1 - r0;

        // pass 1: count diagonal-block entries per local row
        bstart.assign(m + 1, 0);
        for (std::size_t i = r0; i < r1; ++i)
            for (std::size_t k = Brp[i]; k < Brp[i + 1]; ++k) {
                std::size_t j = Bci[k];
                if (j >= r0 && j < r1) ++bstart[(i - r0) + 1];
            }
        for (std::size_t t = 0; t < m; ++t) bstart[t + 1] += bstart[t];
        std::size_t bnnz = bstart[m];
        bind.resize(bnnz);
        bval.resize(bnnz);
        // pass 2: scatter. B's rows are column-sorted (built sorted), so each
        // block row's slot fills in ascending block-column order.
        for (std::size_t i = r0; i < r1; ++i) {
            std::size_t d = bstart[i - r0];
            for (std::size_t k = Brp[i]; k < Brp[i + 1]; ++k) {
                std::size_t j = Bci[k];
                if (j >= r0 && j < r1) { bind[d] = j - r0; bval[d] = Bdat[k]; ++d; }
            }
        }
        mat::compressed2D<Value> block(m, m, bnnz, bstart.data(), bind.data(), bval.data());

        // Per-block fill-reducing ordering: AMD on the block's symmetric
        // structure A+A^T for non-trivial blocks (KLU's default), natural
        // ordering for tiny blocks where the setup would be pure overhead
        // (singletons are common in circuit matrices). AMD-on-(A+A^T) is used
        // rather than COLAMD-on-(A^T*A) because on indefinite/unsymmetric
        // circuit blocks threshold pivoting deviates from a column ordering and
        // fill explodes; the symmetrized ordering keeps fill near-minimal (#133).
        lu_symbolic sym = (m > 4) ? sparse_lu_symbolic(block, ordering::amd{})
                                  : sparse_lu_symbolic(block);
        result.block_numeric.push_back(
            sparse_lu_numeric(block, sym, threshold));
    }

    return result;
}

/// Refactorize a matrix with the SAME sparsity pattern as a prior native KLU
/// factorization, reusing its BTF permutation, per-block orderings, and per-block
/// pivot sequences -- recomputing only the numeric values. This is the analyze-
/// once / factor-many fast path: BTF, the per-block fill-reducing orderings, the
/// reach DFS, and per-block pivot search are all skipped (each block goes through
/// sparse_lu_refactor). Row scaling is recomputed for the new values.
///
/// Intended for repeated solves of a fixed circuit-matrix pattern with changing
/// values (SPICE transient analysis; the mp-spice mixed-precision study).
///
/// \throws std::invalid_argument if A's dimensions do not match `prev`.
/// \throws std::runtime_error    if a reused pivot is numerically zero for the
///                               new values (prior pivot sequence invalid).
template <typename Value, typename Parameters>
    requires OrderedField<Value>
klu_numeric<Value> native_klu_refactor(
    const mat::compressed2D<Value, Parameters>& A,
    const klu_numeric<Value>& prev)
{
    using std::abs;
    const std::size_t n = prev.num_rows();
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "native_klu_refactor: matrix dimensions do not match prior factorization");
    }

    klu_numeric<Value> result;
    if (n == 0) { result.btf.blocks = {0}; return result; }

    // Reuse the symbolic structure from the prior factorization.
    result.btf      = prev.btf;
    result.block_of = prev.block_of;
    const bool scale = !prev.row_scale.empty();

    // Recompute row scaling for the new values (structural choice unchanged).
    if (scale) {
        const auto& rp  = A.ref_major();
        const auto& dat = A.ref_data();
        result.row_scale.assign(n, Value{1});
        for (std::size_t r = 0; r < n; ++r) {
            Value m = Value{0};
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k) {
                Value a = abs(dat[k]);
                if (a > m) m = a;
            }
            if (m > Value{0}) result.row_scale[r] = Value{1} / m;
        }
    }

    // Rebuild B = P*(R*A)*Q (same pattern, new values).
    auto q_inv = util::invert_permutation(result.btf.col_perm);
    mat::compressed2D<Value> B(n, n);
    {
        const auto& rp  = A.ref_major();
        const auto& ci  = A.ref_minor();
        const auto& dat = A.ref_data();
        mat::inserter<mat::compressed2D<Value>> ins(B);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t orow = result.btf.row_perm[i];
            Value s = scale ? result.row_scale[orow] : Value{1};
            for (std::size_t k = rp[orow]; k < rp[orow + 1]; ++k)
                ins[i][q_inv[ci[k]]] << s * dat[k];
        }
    }
    result.B = std::move(B);

    // Refactor each diagonal block, reusing the prior block's pattern + pivots.
    const auto& Brp  = result.B.ref_major();
    const auto& Bci  = result.B.ref_minor();
    const auto& Bdat = result.B.ref_data();
    result.block_numeric.reserve(result.btf.nblocks());
    std::vector<std::size_t> bstart, bind;
    std::vector<Value>       bval;
    for (std::size_t blk = 0; blk < result.btf.nblocks(); ++blk) {
        const std::size_t r0 = result.btf.blocks[blk];
        const std::size_t r1 = result.btf.blocks[blk + 1];
        const std::size_t m  = r1 - r0;

        bstart.assign(m + 1, 0);
        for (std::size_t i = r0; i < r1; ++i)
            for (std::size_t k = Brp[i]; k < Brp[i + 1]; ++k) {
                std::size_t j = Bci[k];
                if (j >= r0 && j < r1) ++bstart[(i - r0) + 1];
            }
        for (std::size_t t = 0; t < m; ++t) bstart[t + 1] += bstart[t];
        std::size_t bnnz = bstart[m];
        bind.resize(bnnz);
        bval.resize(bnnz);
        for (std::size_t i = r0; i < r1; ++i) {
            std::size_t d = bstart[i - r0];
            for (std::size_t k = Brp[i]; k < Brp[i + 1]; ++k) {
                std::size_t j = Bci[k];
                if (j >= r0 && j < r1) { bind[d] = j - r0; bval[d] = Bdat[k]; ++d; }
            }
        }
        mat::compressed2D<Value> block(m, m, bnnz, bstart.data(), bind.data(), bval.data());

        result.block_numeric.push_back(
            sparse_lu_refactor(block, prev.block_numeric[blk]));
    }

    return result;
}

/// One-shot native KLU solve: factor and solve A*x = b.
template <typename Value, typename Parameters, typename VecX, typename VecB>
    requires OrderedField<Value>
void native_klu_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    Value threshold = Value{1})
{
    auto fac = native_klu_factor(A, threshold);
    fac.solve(x, b);
}

} // namespace mtl::sparse::factorization
