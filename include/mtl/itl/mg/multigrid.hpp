#pragma once
// MTL5 -- Multigrid V-cycle / W-cycle driver
// Template parameters: Smoother, CoarseSolver types.
// Uses restriction/prolongation matrices for inter-grid transfer.
//
// ACCUMULATOR (#585). A trailing `Accumulator` (default `void`, unchanged)
// reaches every sum of products the cycle itself forms: the residual
// r = b - A x, with b(i) assigned into the accumulator and the products
// subtracted inside it, rounded once; and the restriction and prolongation
// (restrict / prolongate<Accumulator>). The smoothers are the caller's: pass
// accumulator-aware ones (smoother::*<Matrix, Accumulator>, #265) through the
// factory.
#include <cassert>
#include <stdexcept>
#include <string>
#include <mtl/detail/exceptions.hpp>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <functional>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/compressed2D.hpp>
// vcycle/wcycle form the residual with `levels_[l] * x`, whose operator* lives
// in mat/operators.hpp. Without this the header compiled only when the caller
// happened to have included that first, which made it order-dependent (#401).
//
// The failure needs INSTANTIATION to surface: A*x is a dependent expression
// inside a class template, so merely including this header succeeded and hid
// the problem.
#include <mtl/mat/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/mg/prolongation.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl::itl::mg {

/// Multigrid solver/preconditioner.
/// Smoother must have operator()(VecX&, const VecB&) -> VecX& interface.
/// CoarseSolver must have operator()(VecX&, const VecB&) interface.
///
/// Not thread-safe, although vcycle, wcycle, solve and operator() are const:
/// the smoothers and the coarse solver keep scratch (Jacobi's x_new, say)
/// that every cycle writes through std::function. Two threads must not cycle
/// one multigrid object at the same time; give each thread its own (issue
/// 888).
template <typename Value = double, typename Accumulator = void>
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
        // In every build (issue 925): the cycles index the transfers by
        // level, and the transfers' products index x and the residual, so a
        // missing transfer or a mismatched dimension read out of bounds in a
        // release build.
        check_hierarchy();

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
    // The smoothers hold references into levels_: a copy's would point into
    // the source, and dangle once it is gone (issue 867). A move keeps the
    // vector's buffer, and so the references.
    multigrid(const multigrid&) = delete;
    multigrid& operator=(const multigrid&) = delete;
    multigrid(multigrid&&) = default;
    multigrid& operator=(multigrid&&) = default;

    // const: the cycles read the hierarchy and call the stored smoothers and
    // coarse solver, which std::function calls through a const object.
    // solve() const reached vcycle through a const_cast (issue 867).
    void vcycle(vector_type& x, const vector_type& b, int level = 0) const {
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
        vector_type r = residual(levels_[l], x, b);

        // Restrict residual to coarse grid
        auto r_coarse = mg::restrict<Accumulator>(restrictors_[l], r);

        // Recurse on coarse grid
        size_type nc = levels_[l + 1].num_rows();
        vector_type e_coarse(nc, Value{0});
        vcycle(e_coarse, r_coarse, level + 1);

        // Prolongate correction and add to x
        auto e_fine = prolongate<Accumulator>(prolongators_[l], e_coarse);
        for (size_type i = 0; i < n; ++i)
            x(i) += e_fine(i);

        // Post-smoothing
        for (int s = 0; s < nu_post_; ++s)
            smoothers_[l](x, b);
    }

    /// W-cycle: like V-cycle but recurses twice at each level.
    void wcycle(vector_type& x, const vector_type& b, int level = 0) const {
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
        vector_type r = residual(levels_[l], x, b);

        auto r_coarse = mg::restrict<Accumulator>(restrictors_[l], r);

        size_type nc = levels_[l + 1].num_rows();
        vector_type e_coarse(nc, Value{0});

        // Recurse TWICE for W-cycle. Not at the coarsest level: its solver
        // is exact, so a second solve with the same right-hand side gives the
        // same e_coarse and costs a full coarse solve (issue 925).
        wcycle(e_coarse, r_coarse, level + 1);
        if (l + 2 < n_levels_)
            wcycle(e_coarse, r_coarse, level + 1);

        auto e_fine = prolongate<Accumulator>(prolongators_[l], e_coarse);
        for (size_type i = 0; i < n; ++i)
            x(i) += e_fine(i);

        // Post-smoothing
        for (int s = 0; s < nu_post_; ++s)
            smoothers_[l](x, b);
    }

    /// Apply V-cycle as operator() for use as a stand-alone solver iteration.
    void operator()(vector_type& x, const vector_type& b) const {
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
        vcycle(xx, bb, 0);
        for (size_type i = 0; i < n; ++i)
            x(i) = xx(i);
    }

    /// adjoint_solve: same as solve (symmetric for symmetric smoothers)
    template <typename VecX, typename VecB>
    void adjoint_solve(VecX& x, const VecB& b) const {
        solve(x, b);
    }

private:
    void check_hierarchy() const {
        auto fail = [](const std::string& what) {
            ::mtl::detail::throw_or_abort(std::invalid_argument("multigrid: " + what));
        };
        if (n_levels_ < 2) fail("at least two levels are needed");
        if (restrictors_.size() != n_levels_ - 1 || prolongators_.size() != n_levels_ - 1)
            fail("levels - 1 restrictors and prolongators are needed");
        for (size_type l = 0; l < n_levels_; ++l)
            if (levels_[l].num_rows() != levels_[l].num_cols())
                fail("level " + std::to_string(l) + " is not square");
        for (size_type l = 0; l + 1 < n_levels_; ++l) {
            const size_type nf = levels_[l].num_rows(), nc = levels_[l + 1].num_rows();
            if (restrictors_[l].num_rows() != nc || restrictors_[l].num_cols() != nf)
                fail("restrictor " + std::to_string(l) + " must be " + std::to_string(nc) + " x " + std::to_string(nf));
            if (prolongators_[l].num_rows() != nf || prolongators_[l].num_cols() != nc)
                fail("prolongator " + std::to_string(l) + " must be " + std::to_string(nf) + " x " + std::to_string(nc));
        }
    }

    std::vector<matrix_type> levels_;
    /// r = b - A x. With an accumulator, b(i) is assigned and A's row products
    /// subtracted inside it, rounded once; void keeps the original A * x.
    static vector_type residual(const matrix_type& A, const vector_type& x, const vector_type& b) {
        const size_type n = A.num_rows();
        vector_type r(n);
        if constexpr (!std::is_void_v<Accumulator>) {
            using AT = math::accumulator_traits<Accumulator, Value>;
            const auto& starts  = A.ref_major();
            const auto& indices = A.ref_minor();
            const auto& data    = A.ref_data();
            Accumulator acc;
            for (size_type i = 0; i < n; ++i) {
                AT::assign(acc, b(i));
                for (size_type k = starts[i]; k < starts[i + 1]; ++k)
                    AT::add_product(acc, -data[k], x(indices[k]));
                r(i) = AT::template value<Value>(acc);
            }
        } else {
            auto Ax = A * x;
            for (size_type i = 0; i < n; ++i)
                r(i) = b(i) - Ax(i);
        }
        return r;
    }

    std::vector<matrix_type> restrictors_;
    std::vector<matrix_type> prolongators_;
    std::vector<std::function<void(vector_type&, const vector_type&)>> smoothers_;
    std::function<void(vector_type&, const vector_type&)> coarse_solve_;
    int nu_pre_;
    int nu_post_;
    size_type n_levels_;
};

} // namespace mtl::itl::mg
