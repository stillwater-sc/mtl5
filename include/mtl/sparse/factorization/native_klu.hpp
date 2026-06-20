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
// Out of scope for v1: row/column scaling and iterative refinement. Per-block
// fill-reducing ordering is natural (identity); COLAMD per block is a follow-up.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/ordering/rcm.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>
#include <mtl/sparse/util/permutation.hpp>

namespace mtl::sparse::factorization {

/// Numeric KLU factorization: BTF permutation, the permuted matrix B = P*A*Q
/// (block upper triangular), the per-new-index block id, and a left-looking LU
/// factorization of each diagonal block.
template <typename Value>
struct klu_numeric {
    ordering::btf_result          btf;            ///< row/col permutations + blocks
    mat::compressed2D<Value>      B;              ///< P*A*Q, block upper triangular
    std::vector<std::size_t>      block_of;       ///< new index -> block id
    std::vector<lu_numeric<Value>> block_numeric; ///< LU of each diagonal block

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

        // c[i] = b[p[i]]; y is updated in place to the final solution per block.
        std::vector<Value> y(n);
        for (std::size_t i = 0; i < n; ++i)
            y[i] = static_cast<Value>(b(btf.row_perm[i]));

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
    Value threshold = Value{1})
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument("native_klu_factor: matrix must be square");
    }
    const std::size_t n = A.num_rows();

    klu_numeric<Value> result;
    if (n == 0) {
        result.btf.blocks = {0};
        return result;
    }

    result.btf = ordering::block_triangular_form(A);
    if (result.btf.structurally_singular) {
        throw std::runtime_error(
            "native_klu_factor: matrix is structurally singular");
    }

    // Build B = P*A*Q. New column of old column c is q_inv[c].
    auto q_inv = util::invert_permutation(result.btf.col_perm);
    mat::compressed2D<Value> B(n, n);
    {
        const auto& rp  = A.ref_major();
        const auto& ci  = A.ref_minor();
        const auto& dat = A.ref_data();
        mat::inserter<mat::compressed2D<Value>> ins(B);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t orow = result.btf.row_perm[i];
            for (std::size_t k = rp[orow]; k < rp[orow + 1]; ++k)
                ins[i][q_inv[ci[k]]] << dat[k];
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
    for (std::size_t blk = 0; blk < result.btf.nblocks(); ++blk) {
        const std::size_t r0 = result.btf.blocks[blk];
        const std::size_t r1 = result.btf.blocks[blk + 1];
        const std::size_t m  = r1 - r0;

        mat::compressed2D<Value> block(m, m);
        {
            mat::inserter<mat::compressed2D<Value>> ins(block);
            for (std::size_t i = r0; i < r1; ++i)
                for (std::size_t k = Brp[i]; k < Brp[i + 1]; ++k) {
                    std::size_t j = Bci[k];
                    if (j >= r0 && j < r1)
                        ins[i - r0][j - r0] << Bdat[k];
                }
        }

        // Per-block fill-reducing ordering: RCM for non-trivial blocks (it is
        // near-linear, O(nnz), and reduces bandwidth/fill), natural ordering
        // for tiny blocks. NOTE: COLAMD/AMD would usually reduce fill more, but
        // MTL5's COLAMD and AMD are currently O(n^2) (#128) and would
        // re-introduce the quadratic blowup this fix removes; switch to COLAMD
        // here once #128 is resolved (#117).
        lu_symbolic sym = (m > 4) ? sparse_lu_symbolic(block, ordering::rcm{})
                                  : sparse_lu_symbolic(block);
        result.block_numeric.push_back(
            sparse_lu_numeric(block, sym, threshold));
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
