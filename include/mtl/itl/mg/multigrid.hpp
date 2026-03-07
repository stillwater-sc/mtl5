#pragma once
// MTL5 -- Multigrid V-cycle / W-cycle driver
// Template parameters: Smoother, CoarseSolver types.
// Uses restriction/prolongation matrices for inter-grid transfer.
#include <cassert>
#include <cstddef>
#include <vector>
#include <functional>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/mg/prolongation.hpp>

namespace mtl::itl::mg {

/// Multigrid solver/preconditioner.
/// Smoother must have operator()(VecX&, const VecB&) -> VecX& interface.
/// CoarseSolver must have operator()(VecX&, const VecB&) interface.
template <typename Value = double>
class multigrid {
    using matrix_type = mat::compressed2D<Value>;
    using vector_type = vec::dense_vector<Value>;
    using size_type   = std::size_t;
public:
    /// Construct multigrid hierarchy.
    /// @param levels      Vector of system matrices A[0] (finest) ... A[L-1] (coarsest)
    /// @param restrictors Restriction matrices R[0] ... R[L-2]
    /// @param prolongators Prolongation matrices P[0] ... P[L-2]
    /// @param smoother_factory Creates a smoother for a given level matrix
    /// @param coarse_solver  Solves the coarsest level exactly
    /// @param nu_pre   Number of pre-smoothing steps
    /// @param nu_post  Number of post-smoothing steps
    template <typename SmootherFactory, typename CoarseSolver>
    multigrid(const std::vector<matrix_type>& levels,
              const std::vector<matrix_type>& restrictors,
              const std::vector<matrix_type>& prolongators,
              SmootherFactory smoother_factory,
              CoarseSolver coarse_solver,
              int nu_pre = 2, int nu_post = 2)
        : levels_(levels)
        , restrictors_(restrictors)
        , prolongators_(prolongators)
        , nu_pre_(nu_pre)
        , nu_post_(nu_post)
        , n_levels_(levels.size())
    {
        assert(n_levels_ >= 2);
        assert(restrictors.size() == n_levels_ - 1);
        assert(prolongators.size() == n_levels_ - 1);

        // Create smoothers for each level (except coarsest)
        for (size_type l = 0; l < n_levels_ - 1; ++l) {
            auto sm = smoother_factory(levels_[l]);
            smoothers_.push_back(
                [sm](vector_type& x, const vector_type& b) mutable {
                    sm(x, b);
                });
        }

        // Store coarse solver as a function
        coarse_solve_ = [coarse_solver](vector_type& x, const vector_type& b) mutable {
            coarse_solver(x, b);
        };
    }

    /// V-cycle starting from the given level.
    void vcycle(vector_type& x, const vector_type& b, int level = 0) {
        size_type l = static_cast<size_type>(level);

        // Base case: coarsest level -- solve directly
        if (l == n_levels_ - 1) {
            coarse_solve_(x, b);
            return;
        }

        // Pre-smoothing
        for (int s = 0; s < nu_pre_; ++s)
            smoothers_[l](x, b);

        // Compute residual: r = b - A*x
        size_type n = levels_[l].num_rows();
        vector_type r(n);
        auto Ax = levels_[l] * x;
        for (size_type i = 0; i < n; ++i)
            r(i) = b(i) - Ax(i);

        // Restrict residual to coarse grid
        auto r_coarse = mg::restrict(restrictors_[l], r);

        // Recurse on coarse grid
        size_type nc = levels_[l + 1].num_rows();
        vector_type e_coarse(nc, Value{0});
        vcycle(e_coarse, r_coarse, level + 1);

        // Prolongate correction and add to x
        auto e_fine = prolongate(prolongators_[l], e_coarse);
        for (size_type i = 0; i < n; ++i)
            x(i) += e_fine(i);

        // Post-smoothing
        for (int s = 0; s < nu_post_; ++s)
            smoothers_[l](x, b);
    }

    /// W-cycle: like V-cycle but recurses twice at each level.
    void wcycle(vector_type& x, const vector_type& b, int level = 0) {
        size_type l = static_cast<size_type>(level);

        if (l == n_levels_ - 1) {
            coarse_solve_(x, b);
            return;
        }

        // Pre-smoothing
        for (int s = 0; s < nu_pre_; ++s)
            smoothers_[l](x, b);

        // Compute residual
        size_type n = levels_[l].num_rows();
        vector_type r(n);
        auto Ax = levels_[l] * x;
        for (size_type i = 0; i < n; ++i)
            r(i) = b(i) - Ax(i);

        auto r_coarse = mg::restrict(restrictors_[l], r);

        size_type nc = levels_[l + 1].num_rows();
        vector_type e_coarse(nc, Value{0});

        // Recurse TWICE for W-cycle
        wcycle(e_coarse, r_coarse, level + 1);
        wcycle(e_coarse, r_coarse, level + 1);

        auto e_fine = prolongate(prolongators_[l], e_coarse);
        for (size_type i = 0; i < n; ++i)
            x(i) += e_fine(i);

        // Post-smoothing
        for (int s = 0; s < nu_post_; ++s)
            smoothers_[l](x, b);
    }

    /// Apply V-cycle as operator() for use as a stand-alone solver iteration.
    void operator()(vector_type& x, const vector_type& b) {
        vcycle(x, b, 0);
    }

    /// Preconditioner interface: solve() applies one V-cycle to M*x = b.
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        size_type n = levels_[0].num_rows();
        vector_type xx(n, Value{0});
        vector_type bb(n);
        for (size_type i = 0; i < n; ++i)
            bb(i) = b(i);
        const_cast<multigrid*>(this)->vcycle(xx, bb, 0);
        for (size_type i = 0; i < n; ++i)
            x(i) = xx(i);
    }

    /// adjoint_solve: same as solve (symmetric for symmetric smoothers)
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);
    }

private:
    std::vector<matrix_type> levels_;
    std::vector<matrix_type> restrictors_;
    std::vector<matrix_type> prolongators_;
    std::vector<std::function<void(vector_type&, const vector_type&)>> smoothers_;
    std::function<void(vector_type&, const vector_type&)> coarse_solve_;
    int nu_pre_;
    int nu_post_;
    size_type n_levels_;
};

} // namespace mtl::itl::mg
