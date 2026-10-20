// MTL5 -- The Krylov solvers accept a matrix-free operator (#278).
//
// Every solver applies its operator through itl::apply_operator (and bicg and
// qmr their transpose through itl::apply_transpose). A Matrix goes through
// mult<Accumulator> as before; anything else that supports A * x is applied
// that way, and a matrix-free trans(A) supplies the transpose. Checked here:
//
//   compiles  -- each solver takes an operator that is NOT a Matrix and has
//                only operator* (and trans() for bicg / qmr);
//   identical -- the operator computes A * x with the same mult() call the
//                Matrix path makes, so the solve is bit-identical to the one on
//                the stored matrix: same iteration count, same bits in x. The
//                matrix-free path changes the dispatch and nothing else;
//   contract  -- with a non-void Accumulator, the matrix-free operator keeps its
//                own matvec and the accumulator reaches only the solver's dot
//                products: a counting accumulator sees strictly fewer products
//                than on the stored matrix, and more than none.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/krylov/bicg.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/bicgstab_ell.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/cgs.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/krylov/idr_s.hpp>
#include <mtl/itl/krylov/minres.hpp>
#include <mtl/itl/krylov/qmr.hpp>
#include <mtl/itl/krylov/tfqmr.hpp>

using namespace mtl;

namespace matfree {

using smat = mat::compressed2D<double>;
using dvec = vec::dense_vector<double>;

/// A^T applied without forming it: only operator*.
struct transposed_op {
    const smat& S;
    dvec operator*(const dvec& x) const {
        dvec y(S.num_cols());
        mtl::mult(mtl::trans(S), x, y);
        return y;
    }
};

/// A matrix-free operator: no num_rows, no element access, only A * x -- and
/// trans(), found by ADL, for the solvers that need the transpose. It applies
/// the stored matrix with the same mult() call the Matrix path makes.
struct op {
    const smat& S;
    dvec operator*(const dvec& x) const {
        dvec y(S.num_rows());
        mtl::mult(S, x, y);
        return y;
    }
};

inline transposed_op trans(const op& A) { return transposed_op{A.S}; }

} // namespace matfree

static_assert(!Matrix<matfree::op>);
static_assert(LinearOperator<matfree::op, matfree::dvec>);

namespace counting {

// Counts every product it is handed (as in test_eigensolvers.cpp), so a test
// can see which reductions the accumulator reached. No accumulator_merge.
struct count_acc { double s; };

inline std::atomic<std::size_t> products{0};

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }

template <typename V>
    requires std::is_floating_point_v<V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }

template <typename V>
    requires std::is_floating_point_v<V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}

template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

using matfree::dvec;
using matfree::smat;

/// Upwind convection-diffusion on a 1-D grid: 4 on the diagonal, -1 - p below,
/// -1 + p above. p = 0 is symmetric positive definite.
smat tridiag(std::size_t n, double p) {
    smat A(n, n);
    {
        mat::inserter<smat> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0) ins[i][i - 1] << -1.0 - p;
            if (i + 1 < n) ins[i][i + 1] << -1.0 + p;
        }
    }
    return A;
}

dvec rhs(std::size_t n) {
    dvec b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + static_cast<double>(i % 5);
    return b;
}

bool same_bits(const dvec& x, const dvec& y) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::memcmp(&x(i), &y(i), sizeof(double)) != 0) return false;
    return true;
}

struct outcome {
    int code;
    int iterations;
    dvec x;
};

/// Run `solve(A, x, b, M, iter)` on the stored matrix and on the matrix-free
/// operator wrapping it.
template <typename Solve>
void check_matrix_free(const smat& S, Solve solve) {
    const std::size_t n = S.num_rows();
    const dvec b = rhs(n);
    const matfree::op F{S};
    itl::pc::identity<smat> M(S);
    auto run = [&](const auto& A) {
        dvec x(n, 0.0);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        const int code = solve(A, x, b, M, iter);
        return outcome{code, iter.iterations(), x};
    };
    const auto stored = run(S);
    const auto mfree = run(F);
    INFO("stored: code " << stored.code << ", " << stored.iterations << " iterations; matrix-free: code "
         << mfree.code << ", " << mfree.iterations << " iterations");
    CHECK(stored.code == 0);
    CHECK(mfree.code == 0);
    CHECK(mfree.iterations == stored.iterations);
    CHECK(same_bits(mfree.x, stored.x));
}

/// Products a few iterations hand the counting accumulator, on the stored
/// matrix and on the matrix-free operator. The tolerance is unreachable, so
/// both runs take exactly `its` iterations.
template <typename Solve>
void check_accumulator_contract(const smat& S, Solve solve) {
    const std::size_t n = S.num_rows();
    const dvec b = rhs(n);
    const matfree::op F{S};
    itl::pc::identity<smat> M(S);
    auto count = [&](const auto& A) {
        dvec x(n, 0.0);
        itl::basic_iteration<double> iter(b, 3, 0.0);
        counting::products = 0;
        (void)solve(A, x, b, M, iter);
        return counting::products.load();
    };
    const std::size_t stored = count(S);
    const std::size_t mfree = count(F);
    INFO("products: stored " << stored << ", matrix-free " << mfree);
    CHECK(mfree > 0);        // the solver's dot products
    CHECK(stored > mfree);   // plus the matvecs
}

} // namespace

#define MTL5_KRYLOV_ROW(name, symmetric, call)                                                         \
    TEST_CASE("matrix-free operator: " name, "[itl][krylov][matrix-free][278]") {                      \
        const smat S = tridiag(40, (symmetric) ? 0.0 : 0.3);                                         \
        SECTION("bit-identical to the stored matrix") {                                                \
            check_matrix_free(S, [](const auto& A, dvec& x, const dvec& b, const auto& M, auto& it) {  \
                using Acc = void;                                                                      \
                return call;                                                                           \
            });                                                                                        \
        }                                                                                              \
        SECTION("the accumulator reaches only the dot products") {                                     \
            check_accumulator_contract(                                                                \
                S, [](const auto& A, dvec& x, const dvec& b, const auto& M, auto& it) {                \
                    using Acc = counting::count_acc;                                                   \
                    return call;                                                                       \
                });                                                                                    \
        }                                                                                              \
    }

MTL5_KRYLOV_ROW("cg", true, itl::cg<Acc>(A, x, b, M, it))
MTL5_KRYLOV_ROW("minres", true, itl::minres<Acc>(A, x, b, M, it))
MTL5_KRYLOV_ROW("bicg", false, itl::bicg<Acc>(A, x, b, M, it))
MTL5_KRYLOV_ROW("bicgstab", false, itl::bicgstab<Acc>(A, x, b, M, it))
MTL5_KRYLOV_ROW("bicgstab_ell", false, itl::bicgstab_ell<Acc>(A, x, b, M, it, 2))
MTL5_KRYLOV_ROW("cgs", false, itl::cgs<Acc>(A, x, b, M, it))
MTL5_KRYLOV_ROW("gmres", false, itl::gmres<Acc>(A, x, b, M, it, 20))
MTL5_KRYLOV_ROW("idr_s", false, itl::idr_s<Acc>(A, x, b, M, it, 4))
MTL5_KRYLOV_ROW("qmr", false, itl::qmr<Acc>(A, x, b, M, it))
MTL5_KRYLOV_ROW("tfqmr", false, itl::tfqmr<Acc>(A, x, b, M, it))
