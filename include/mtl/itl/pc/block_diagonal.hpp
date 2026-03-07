#pragma once
// MTL5 — Block diagonal preconditioner
// Partitions matrix into diagonal blocks, computes LU of each, applies block solves.
#include <algorithm>
#include <cassert>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl::pc {

/// Block diagonal preconditioner: extracts diagonal blocks and uses dense LU.
template <typename Matrix>
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

    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);  // approximate
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
