#pragma once
// MTL5 -- Block diagonal preconditioner
// Partitions matrix into diagonal blocks, computes LU of each, applies block solves.
#include <mtl/concepts/matrix.hpp>   // FieldMatrix (#505)
#include <algorithm>
#include <cassert>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl::pc {

/// Block diagonal preconditioner: extracts diagonal blocks and uses dense LU.
template <FieldMatrix Matrix>
class block_diagonal {
    using value_type = typename Matrix::value_type;
    using size_type  = typename Matrix::size_type;
public:
    block_diagonal(const Matrix& A, size_type block_size)
        : n_(A.num_rows()), bs_(block_size)
    {
        assert(A.num_rows() == A.num_cols());
        assert(block_size > 0);
        nb_ = (n_ + bs_ - 1) / bs_;  // number of blocks (last may be smaller)
        extract_and_factor(A);
    }

    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        for (size_type blk = 0; blk < nb_; ++blk) {
            size_type start = blk * bs_;
            size_type end   = std::min(start + bs_, n_);
            size_type bsize = end - start;

            // Extract sub-vector of b
            vec::dense_vector<value_type> bsub(bsize);
            vec::dense_vector<value_type> xsub(bsize);
            for (size_type i = 0; i < bsize; ++i)
                bsub(i) = b(start + i);

            // Solve using stored LU
            lu_solve(blocks_[blk], pivots_[blk], xsub, bsub);

            // Write back
            for (size_type i = 0; i < bsize; ++i)
                x(start + i) = xsub(i);
        }
    }

    /// Solve M^H x = b, block by block.
    ///
    /// This used to delegate to solve(), commented "approximate". It is not an
    /// approximation: a diagonal block of a non-symmetric A is itself
    /// non-symmetric, so the wrong operator was applied outright. bicg and qmr
    /// are the only callers and both failed to converge (#394).
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        // M is block DIAGONAL, so M^H is block diagonal with each block
        // conjugate-transposed -- the blocks stay independent and no coupling
        // appears between them.
        for (size_type blk = 0; blk < nb_; ++blk) {
            size_type start = blk * bs_;
            size_type end   = std::min(start + bs_, n_);
            size_type bsize = end - start;

            vec::dense_vector<value_type> bsub(bsize);
            vec::dense_vector<value_type> xsub(bsize);
            for (size_type i = 0; i < bsize; ++i)
                bsub(i) = b(start + i);

            lu_adjoint_solve(blocks_[blk], pivots_[blk], xsub, bsub);

            for (size_type i = 0; i < bsize; ++i)
                x(start + i) = xsub(i);
        }
    }

private:
    void extract_and_factor(const Matrix& A) {
        blocks_.resize(nb_);
        pivots_.resize(nb_);

        for (size_type blk = 0; blk < nb_; ++blk) {
            size_type start = blk * bs_;
            size_type end   = std::min(start + bs_, n_);
            size_type bsize = end - start;

            // Extract diagonal block into dense matrix
            mat::dense2D<value_type> B(bsize, bsize);
            for (size_type i = 0; i < bsize; ++i)
                for (size_type j = 0; j < bsize; ++j)
                    B(i, j) = A(start + i, start + j);

            // Factor in place
            lu_factor(B, pivots_[blk]);
            blocks_[blk] = std::move(B);
        }
    }

    size_type n_;
    size_type bs_;
    size_type nb_;
    std::vector<mat::dense2D<value_type>> blocks_;
    std::vector<std::vector<size_type>> pivots_;
};

} // namespace mtl::itl::pc
