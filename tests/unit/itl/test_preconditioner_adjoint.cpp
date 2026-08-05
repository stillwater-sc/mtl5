// Preconditioner adjoint identity (#394).
//
// bicg and qmr are the only solvers that call adjoint_solve; they need M^-H
// applied to the shadow residual. Several preconditioners implemented it by
// delegating to solve(), which is correct only if M is self-adjoint.
//
// Convergence is a poor instrument for this: a wrong adjoint often still
// converges (slowly, or on easy systems), and the failure only shows on a
// non-symmetric problem. The DEFINING property is exact and cheap to check:
//
//     <M^-1 b, c>  ==  <b, M^-H c>       for all b, c
//
// That is what these cases assert. It is what found every entry in the table
// below, including two the issue reporting this had not identified.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/ssor.hpp>

using namespace mtl;

namespace {

using SMat = mat::compressed2D<double>;

/// 2-D Laplacian; `sym == false` breaks the symmetry so the off-diagonals differ.
SMat pc_matrix(std::size_t k, bool sym) {
    const std::size_t n = k * k;
    SMat A(n, n);
    {
        mat::inserter<SMat> ins(A);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 4.0 * (1.0 + static_cast<double>(r % 7) * 0.1);
                if (i)         ins[r][r - k] << -1.0;
                if (i + 1 < k) ins[r][r + k] << (sym ? -1.0 : -0.6);
                if (j)         ins[r][r - 1] << -1.0;
                if (j + 1 < k) ins[r][r + 1] << (sym ? -1.0 : -1.4);
            }
    }
    return A;
}

/// max over random (b, c) of the relative violation of <M^-1 b, c> == <b, M^-H c>.
template <typename Prec>
double adjoint_violation(const Prec& M, std::size_t n) {
    std::mt19937 gen(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    double worst = 0.0;
    for (int t = 0; t < 5; ++t) {
        vec::dense_vector<double> b(n), c(n), Mb(n), MHc(n);
        for (std::size_t i = 0; i < n; ++i) { b[i] = nd(gen); c[i] = nd(gen); }
        M.solve(Mb, b);
        M.adjoint_solve(MHc, c);
        const double lhs = mtl::dot(Mb, c);
        const double rhs = mtl::dot(b, MHc);
        worst = std::max(worst, std::abs(lhs - rhs) / std::max(1.0, std::abs(lhs)));
    }
    return worst;
}

}  // namespace

TEST_CASE("preconditioner adjoint_solve really is the adjoint (#394)",
          "[itl][pc][adjoint][regression]") {
    // Both matrices in one pass -- no SECTIONs, because a SECTION inside a loop
    // re-enters the test case per section and obscures which case failed.
    for (bool sym : {true, false}) {
        const auto A = pc_matrix(12, sym);
        const std::size_t n = A.num_rows();
        const char* what = sym ? "symmetric A" : "NON-symmetric A";

        {
            itl::pc::identity<SMat> M(A);
            const double v = adjoint_violation(M, n);
            INFO("identity, " << what << ", violation = " << v);
            CHECK(v < 1e-12);
        }
        {
            // Has a real implementation: conj(inv_diag) * b.
            itl::pc::diagonal<SMat> M(A);
            const double v = adjoint_violation(M, n);
            INFO("diagonal, " << what << ", violation = " << v);
            CHECK(v < 1e-12);
        }
        {
            // Also #394. A diagonal BLOCK of a non-symmetric A is itself
            // non-symmetric, so delegating to solve() applied the wrong
            // operator; now each block uses lu_adjoint_solve.
            itl::pc::block_diagonal<SMat> M(A, 4);
            const double v = adjoint_violation(M, n);
            INFO("block_diagonal, " << what << ", violation = " << v);
            CHECK(v < 1e-12);
        }
        {
            // The #394 fix. M = L*U is not self-adjoint for a non-symmetric A,
            // so delegating adjoint_solve to solve applied the wrong operator
            // outright -- bicg/qmr did not converge at all. Now U^-H then L^-H,
            // in scatter form over the existing row storage.
            itl::pc::ilu_0<double> M(A);
            const double v = adjoint_violation(M, n);
            INFO("ilu_0, " << what << ", violation = " << v);
            CHECK(v < 1e-12);
        }
    }
}

TEST_CASE("ic_0 adjoint identity (#394)", "[itl][pc][adjoint][regression]") {
    // Correct for a reason worth stating: IC produces L*L^H, which is
    // self-adjoint by construction, so delegating to solve() is right rather
    // than lucky. Same for ildl (L*D*L^T). Only ilu_0, ssor and block_diagonal
    // were ever in question.
    const auto A = pc_matrix(12, true);
    itl::pc::ic_0<double> M(A);
    REQUIRE(adjoint_violation(M, A.num_rows()) < 1e-12);
}

TEST_CASE("ssor adjoint identity (#398)", "[itl][pc][adjoint][ssor][regression]") {
    // ssor's sweeps now start from x = 0, so they apply
    // M = w/(2-w) (D/w + L) D^-1 (D/w + U) exactly. For a SYMMETRIC A that is
    // c X D^-1 X^T -- self-adjoint by construction, which is the whole reason
    // Sheldon (1955) introduced the symmetric variant -- so delegating
    // adjoint_solve to solve is right, not lucky.
    //
    // Before #398 the sweeps started from x = b, adding a G_B G_F term that is
    // not symmetric even when A and M both are (because M^-1 A is not), and
    // this violation measured 1.1e-01.
    //
    // SECTIONs are safe here, unlike in the case above: there is no enclosing
    // loop for Catch2's per-section re-entry to interact with, so each case
    // names itself at the assertion site.
    SECTION("symmetric A is self-adjoint") {
        const auto A = pc_matrix(12, true);
        itl::pc::ssor<SMat> M(A);
        const double v = adjoint_violation(M, A.num_rows());
        INFO("ssor, symmetric A, violation = " << v);
        CHECK(v < 1e-12);
    }
    // Both omegas, since the scalar factor and the D/w terms only separate for
    // w != 1 -- and symmetry must not depend on the relaxation parameter.
    SECTION("symmetric A is self-adjoint at omega = 1.4") {
        const auto A = pc_matrix(12, true);
        itl::pc::ssor<SMat> M(A, 1.4);
        const double v = adjoint_violation(M, A.num_rows());
        INFO("ssor, symmetric A, omega = 1.4, violation = " << v);
        CHECK(v < 1e-12);
    }
    // And the property that SURVIVES the fix, asserted so it is not mistaken
    // for a remaining bug: SSOR of a NON-symmetric A has U != L^T, so M is not
    // symmetric and no rearrangement of the sweeps yields M^-H. bicg/qmr must
    // use ilu_0 there. This is a statement about the method, not the code.
    //
    // Note this asserts a LOWER bound on an error, so it fires if ssor ever
    // gains a true M^-H for non-Hermitian A -- at which point the right move is
    // to delete this section, not to loosen it.
    SECTION("non-Hermitian A is NOT self-adjoint -- a property of the method") {
        const auto A = pc_matrix(12, false);
        itl::pc::ssor<SMat> M(A);
        const double v = adjoint_violation(M, A.num_rows());
        INFO("ssor, NON-symmetric A, violation = " << v << " (expected: large)");
        CHECK(v > 1e-3);
    }
}
