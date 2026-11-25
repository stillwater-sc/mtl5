// Krylov solvers: the iteration-control and robustness findings of issue 924
// (QA sweep #913).
//
//   - bicg and qmr build the shadow sequence with A^H, and converge on complex
//     non-Hermitian systems; tfqmr compiles and converges for complex; every
//     solver has a complex test;
//   - the dot-product solvers are scale-safe in float: b = 1e-30 and 1e30;
//   - a power-of-two scale changes nothing, bit for bit, when it takes effect;
//   - x may alias b;
//   - a NaN or Inf in b stops at once with error 6 (basic_iteration::non_finite);
//   - gmres leaves x finite on a singular inconsistent system, and reports it;
//   - bicgstab_ell rejects ell = 0; cg and bicg report rho == 0;
//   - minres records a zero M-norm residual instead of a stale state;
//   - empty systems; basic_iteration's first(), operator++(int), converged_now.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/itl.hpp>

using namespace mtl;

namespace {

// Every solver, called the same way. `hpd` marks the ones that need a
// Hermitian positive definite (cg) or Hermitian (minres) operator.
template <typename Mat, typename Vec, typename PC, typename Iter>
struct solver {
    std::string name;
    bool hpd;
    std::function<int(const Mat&, Vec&, const Vec&, const PC&, Iter&)> run;
};

template <typename Mat, typename Vec, typename PC, typename Iter>
std::vector<solver<Mat, Vec, PC, Iter>> all_solvers() {
    using S = solver<Mat, Vec, PC, Iter>;
    return {
        S{"cg", true, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::cg(A, x, b, M, it); }},
        S{"minres", true, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::minres(A, x, b, M, it); }},
        S{"bicg", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::bicg(A, x, b, M, it); }},
        S{"bicgstab", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::bicgstab(A, x, b, M, it); }},
        S{"cgs", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::cgs(A, x, b, M, it); }},
        S{"qmr", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::qmr(A, x, b, M, it); }},
        S{"tfqmr", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::tfqmr(A, x, b, M, it); }},
        S{"bicgstab_ell(1)", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::bicgstab_ell(A, x, b, M, it, 1); }},
        S{"bicgstab_ell(2)", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::bicgstab_ell(A, x, b, M, it, 2); }},
        S{"bicgstab_ell(3)", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::bicgstab_ell(A, x, b, M, it, 3); }},
        S{"gmres", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::gmres(A, x, b, M, it, 30); }},
        S{"idr_s", false, [](const Mat& A, Vec& x, const Vec& b, const PC& M, Iter& it) { return itl::idr_s(A, x, b, M, it, 2); }},
    };
}

// ||b - A x|| / ||b||, in double (or its complex), for a dense A.
template <typename Mat, typename Vec>
double dense_relres(const Mat& A, const Vec& x, const Vec& b) {
    using T = typename Vec::value_type;
    using W = std::conditional_t<is_complex_v<T>, std::complex<double>, double>;
    double r2 = 0, b2 = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        W ax(0);
        for (std::size_t j = 0; j < x.size(); ++j) ax += W(A(i, j)) * W(x(j));
        r2 += std::norm(W(b(i)) - ax);
        b2 += std::norm(W(b(i)));
    }
    return std::sqrt(r2 / b2);
}

using C = std::complex<double>;

// A complex non-Hermitian matrix with a dominant diagonal (test_gmres's).
mat::dense2D<C> complex_nonhermitian(std::size_t n) {
    mat::dense2D<C> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? C(4.0 + double(i), 1.0)
                               : C(double((i + 2 * j) % 3) - 1.0, double((2 * i + j) % 5) / 4.0 - 0.5);
    return A;
}

// B^H B + n I: Hermitian positive definite.
mat::dense2D<C> complex_hpd(std::size_t n) {
    const auto B = complex_nonhermitian(n);
    mat::dense2D<C> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            C s(i == j ? double(n) : 0.0);
            for (std::size_t k = 0; k < n; ++k) s += std::conj(B(k, i)) * B(k, j);
            A(i, j) = s;
        }
    return A;
}

vec::dense_vector<C> complex_rhs(std::size_t n) {
    vec::dense_vector<C> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = C(1.0 + double(i), -double(i % 2));
    return b;
}

// tridiag(-1, 4, -1), n x n, in T: symmetric positive definite.
template <typename T>
mat::compressed2D<T> tridiag(std::size_t n) {
    mat::compressed2D<T> A(n, n);
    mat::inserter<mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) ins[i][i - 1] << T(-1);
        ins[i][i] << T(4);
        if (i + 1 < n) ins[i][i + 1] << T(-1);
    }
    return A;
}

template <typename T>
double sparse_relres(const mat::compressed2D<T>& A, const vec::dense_vector<T>& x,
                     const vec::dense_vector<T>& b) {
    const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& d = A.ref_data();
    // In long double, so the squares of b = 1e30 do not overflow.
    long double r2 = 0, b2 = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        long double ax = 0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ax += (long double)d[k] * (long double)x(ci[k]);
        const long double ri = (long double)b(i) - ax;
        r2 += ri * ri;
        b2 += (long double)b(i) * (long double)b(i);
    }
    return double(std::sqrt(r2 / b2));
}

// Swaps the two entries: r^T M^-1 r = 2 r0 r1, which is 0 for r = (1, 0).
struct swap_pc {
    template <typename X, typename B>
    void solve(X& x, const B& b) const { x(0) = b(1); x(1) = b(0); }
    template <typename X, typename B>
    void adjoint_solve(X& x, const B& b) const { solve(x, b); }
};

// M^-1 = 0.
struct zero_pc {
    template <typename X, typename B>
    void solve(X& x, const B&) const { for (std::size_t i = 0; i < x.size(); ++i) x(i) = 0; }
    template <typename X, typename B>
    void adjoint_solve(X& x, const B& b) const { solve(x, b); }
};

// A complex matrix-free operator: A * x, and trans(A) the plain transpose,
// found by ADL. bicg and qmr form A^H x = conj(A^T conj(x)) from it.
struct complex_op {
    const mat::dense2D<std::complex<double>>& A;
    std::size_t num_rows() const { return A.num_rows(); }
    std::size_t num_cols() const { return A.num_cols(); }
    vec::dense_vector<std::complex<double>> operator*(const vec::dense_vector<std::complex<double>>& x) const {
        vec::dense_vector<std::complex<double>> y(A.num_rows(), std::complex<double>(0));
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            for (std::size_t j = 0; j < A.num_cols(); ++j) y(i) += A(i, j) * x(j);
        return y;
    }
};
struct complex_op_t {
    const mat::dense2D<std::complex<double>>& A;
    vec::dense_vector<std::complex<double>> operator*(const vec::dense_vector<std::complex<double>>& x) const {
        vec::dense_vector<std::complex<double>> y(A.num_cols(), std::complex<double>(0));
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            for (std::size_t j = 0; j < A.num_cols(); ++j) y(j) += A(i, j) * x(i);
        return y;
    }
};
inline complex_op_t trans(const complex_op& op) { return complex_op_t{op.A}; }

} // namespace

TEST_CASE("bicg and qmr with a complex matrix-free operator (issue 924)", "[itl][krylov][complex][924]") {
    using M = mat::dense2D<C>;
    using V = vec::dense_vector<C>;
    const std::size_t n = 12;
    const M A = complex_nonhermitian(n);
    const complex_op op{A};
    const V b = complex_rhs(n);
    const itl::pc::identity<M> I(A);
    {
        V x(n, C(0));
        itl::basic_iteration<double> iter(b, 200, 1e-12);
        CHECK(itl::bicg(op, x, b, I, iter) == 0);
        CHECK(dense_relres(A, x, b) < 1e-10);
    }
    {
        V x(n, C(0));
        itl::basic_iteration<double> iter(b, 200, 1e-12);
        CHECK(itl::qmr(op, x, b, I, iter) == 0);
        CHECK(dense_relres(A, x, b) < 1e-10);
    }
}

TEST_CASE("Krylov solvers converge on complex systems (issue 924)", "[itl][krylov][complex][924]") {
    // bicg and qmr used A^T for the shadow sequence while the dots conjugate:
    // on a 3x3 non-Hermitian complex system they stopped at max_iter with a
    // residual of about 2, while cgs and gmres converged in 3 iterations.
    // tfqmr did not compile for a complex value type.
    using M  = mat::dense2D<C>;
    using V  = vec::dense_vector<C>;
    using It = itl::basic_iteration<double>;
    for (const std::size_t n : {3u, 6u, 12u}) {
        const M A  = complex_nonhermitian(n);
        const M H  = complex_hpd(n);
        const V b  = complex_rhs(n);
        SECTION("identity PC, n = " + std::to_string(n)) {
            const itl::pc::identity<M> I(A);
            for (const auto& s : all_solvers<M, V, itl::pc::identity<M>, It>()) {
                INFO(s.name << ", n = " << n);
                const M& Op = s.hpd ? H : A;
                V x(n, C(0));
                It iter(b, 200, 1e-12);
                CHECK(s.run(Op, x, b, I, iter) == 0);
                CHECK(dense_relres(Op, x, b) < 1e-10);
            }
        }
        SECTION("diagonal PC (adjoint_solve conjugates), n = " + std::to_string(n)) {
            for (const auto& s : all_solvers<M, V, itl::pc::diagonal<M>, It>()) {
                INFO(s.name << ", n = " << n);
                const M& Op = s.hpd ? H : A;
                const itl::pc::diagonal<M> D(Op);
                V x(n, C(0));
                It iter(b, 200, 1e-12);
                CHECK(s.run(Op, x, b, D, iter) == 0);
                CHECK(dense_relres(Op, x, b) < 1e-10);
            }
        }
    }
}

TEST_CASE("BiCG and QMR on a complex system take at most n iterations (issue 924)",
          "[itl][krylov][complex][924]") {
    // In exact arithmetic both terminate in n steps; with A^T they did not
    // converge at all.
    using M = mat::dense2D<C>;
    using V = vec::dense_vector<C>;
    const std::size_t n = 3;
    const M A = complex_nonhermitian(n);
    const V b = complex_rhs(n);
    const itl::pc::identity<M> I(A);
    {
        V x(n, C(0));
        itl::basic_iteration<double> iter(b, 200, 1e-10);
        CHECK(itl::bicg(A, x, b, I, iter) == 0);
        CHECK(iter.iterations() <= 2 * int(n));
    }
    {
        V x(n, C(0));
        itl::basic_iteration<double> iter(b, 200, 1e-10);
        CHECK(itl::qmr(A, x, b, I, iter) == 0);
        CHECK(iter.iterations() <= 2 * int(n));
    }
}

TEST_CASE("BiCGSTAB(1) on a complex system tracks BiCGSTAB (issue 924)",
          "[itl][krylov][complex][bicgstab_ell][924]") {
    // The MR part took r_0^H r_j where it needs r_j^H r_0: the conjugate of
    // the minimizing coefficient. It still converged, slowly: 47 iterations at
    // n = 40 where BiCGSTAB, which BiCGSTAB(1) is mathematically, takes 21.
    using M = mat::dense2D<C>;
    using V = vec::dense_vector<C>;
    for (const std::size_t n : {12u, 40u}) {
        INFO("n = " << n);
        M A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A(i, j) = (i == j) ? C(1.0 + 0.1 * double(i), 2.0)
                                   : C(double((i + 2 * j) % 3) - 1.0, double((2 * i + j) % 5) / 4.0 - 0.5);
        const V b = complex_rhs(n);
        const itl::pc::identity<M> I(A);
        V x(n, C(0)), y(n, C(0));
        itl::basic_iteration<double> it_ell(b, 2000, 1e-10), it_ref(b, 2000, 1e-10);
        REQUIRE(itl::bicgstab_ell(A, x, b, I, it_ell, 1) == 0);
        REQUIRE(itl::bicgstab(A, y, b, I, it_ref) == 0);
        // The counts depend on rounding (contraction differs by platform: 2
        // apart on macOS ARM64 at n = 40), so the margin is a quarter of the
        // reference, at least 3. The wrong coefficient took 19 at n = 12 and
        // 47 at n = 40, against bounds of 16 and 26.
        const int margin = std::max(3, it_ref.iterations() / 4);
        CHECK(std::abs(it_ell.iterations() - it_ref.iterations()) <= margin);
    }
}

TEST_CASE("Krylov solvers are scale-safe in float (issue 924)", "[itl][krylov][scaling][924]") {
    // b = 1e-30 ones in float: dot(p, A p) underflowed to 0, so cg and
    // bicgstab reported a breakdown with x = 0. b = 1e20 or 1e30: the dots
    // overflowed and x was NaN after the full max_iter.
    using M  = mat::compressed2D<float>;
    using V  = vec::dense_vector<float>;
    using It = itl::basic_iteration<float>;
    const std::size_t n = 4;
    const M A = tridiag<float>(n);
    const itl::pc::identity<M> I(A);
    for (const float scale : {1e-30f, 1e-20f, 1.0f, 1e20f, 1e30f}) {
        V b(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = scale * float(1 + i);
        for (const auto& s : all_solvers<M, V, itl::pc::identity<M>, It>()) {
            INFO(s.name << ", b = " << scale << " * (1, 2, 3, 4)");
            V x(n, 0.0f);
            It iter(b, 100, 1e-5f);
            CHECK(s.run(A, x, b, I, iter) == 0);
            CHECK(sparse_relres(A, x, b) < 1e-4);
        }
    }
}

TEST_CASE("The power-of-two scaling is exact (issue 924)", "[itl][krylov][scaling][924]") {
    // b 2^-600 in double is outside the band, so it is scaled near 1. Scaling
    // by a power of two rounds nothing, so the solve gives x 2^-600 bit for
    // bit, in the same number of iterations. Without the scaling the dots,
    // near 2^-1200, underflowed.
    using M  = mat::compressed2D<double>;
    using V  = vec::dense_vector<double>;
    using It = itl::basic_iteration<double>;
    const std::size_t n = 16;
    M A(n, n);
    {
        mat::inserter<M> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) ins[i][i - 1] << -1.3;
            ins[i][i] << 4.0;
            if (i + 1 < n) ins[i][i + 1] << -0.7;
        }
    }
    const itl::pc::identity<M> I(A);
    V b(n), bt(n);
    for (std::size_t i = 0; i < n; ++i) {
        b(i)  = 1.0 + double(i % 5) / 7.0;
        bt(i) = std::ldexp(b(i), -600);
    }
    for (const auto& s : all_solvers<M, V, itl::pc::identity<M>, It>()) {
        // A is not symmetric (cg, minres). idr_s takes b's copy without the
        // scaling: it scales its own dots by their largest entry (#874), which
        // is not a power of two, so it promises no bit-identity across scales
        // (it differed under BLIS).
        if (s.hpd || s.name == "idr_s") continue;
        INFO(s.name);
        V x(n, 0.0), xt(n, 0.0);
        It iter(b, 200, 1e-10), itert(bt, 200, 1e-10);
        const int err  = s.run(A, x, b, I, iter);
        const int errt = s.run(A, xt, bt, I, itert);
        CHECK(err == 0);
        CHECK(errt == err);
        CHECK(itert.iterations() == iter.iterations());
        bool same = true;
        for (std::size_t i = 0; i < n; ++i)
            same = same && (xt(i) == std::ldexp(x(i), -600));
        CHECK(same);
    }
}

TEST_CASE("x may alias b (issue 924)", "[itl][krylov][alias][924]") {
    // The solvers re-read b in their true-residual checks after writing x:
    // cg(A, x, x, M, iter) on a 4x4 SPD system returned error 1 and x ~ 6e12.
    using M  = mat::compressed2D<double>;
    using V  = vec::dense_vector<double>;
    using It = itl::basic_iteration<double>;
    const std::size_t n = 4;
    const M A = tridiag<double>(n);
    const itl::pc::identity<M> I(A);
    V b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + double(i);
    for (const auto& s : all_solvers<M, V, itl::pc::identity<M>, It>()) {
        INFO(s.name);
        V x = b;   // the right-hand side, and the starting guess
        It iter(b, 100, 1e-12);
        CHECK(s.run(A, x, x, I, iter) == 0);
        CHECK(sparse_relres(A, x, b) < 1e-10);
    }
}

TEST_CASE("A NaN or Inf in b stops at once (issue 924)", "[itl][krylov][nonfinite][924]") {
    // converged() is false for NaN, so only max_iter stopped the solver: cg
    // with a NaN in b ran all 100000 iterations and returned error 1.
    using M  = mat::compressed2D<double>;
    using V  = vec::dense_vector<double>;
    using It = itl::basic_iteration<double>;
    const std::size_t n = 4;
    const M A = tridiag<double>(n);
    const itl::pc::identity<M> I(A);
    for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        V b(n, 1.0);
        b(1) = bad;
        for (const auto& s : all_solvers<M, V, itl::pc::identity<M>, It>()) {
            INFO(s.name << ", b(1) = " << bad);
            V x(n, 0.0);
            It iter(4.0, 100000, 1e-10);
            CHECK(s.run(A, x, b, I, iter) == It::non_finite);
            CHECK(iter.iterations() <= 1);
        }
    }
}

TEST_CASE("gmres on a singular system leaves x finite (issue 924)", "[itl][krylov][gmres][924]") {
    // A = [[1,1],[1,1]] is singular. With b = (1,0) the system is inconsistent:
    // the lucky-breakdown path divided by H(1,1) = 0 and wrote NaN into x.
    using M = mat::dense2D<double>;
    using V = vec::dense_vector<double>;
    M A(2, 2);
    A(0, 0) = A(0, 1) = A(1, 0) = A(1, 1) = 1.0;
    const itl::pc::identity<M> I(A);
    SECTION("inconsistent: a breakdown, x finite and the least-squares solution") {
        V b(2), x(2, 0.0);
        b(0) = 1.0; b(1) = 0.0;
        itl::basic_iteration<double> iter(b, 100, 1e-10);
        CHECK(itl::gmres(A, x, b, I, iter, 30) == 2);
        CHECK(std::isfinite(x(0)));
        CHECK(std::isfinite(x(1)));
        // The least-squares residual is (1,-1)/2: x(0) + x(1) = 1/2.
        CHECK(std::abs(x(0) + x(1) - 0.5) < 1e-14);
        CHECK(iter.iterations() < 100);
    }
    SECTION("consistent: converges") {
        V b(2, 1.0), x(2, 0.0);
        itl::basic_iteration<double> iter(b, 100, 1e-10);
        CHECK(itl::gmres(A, x, b, I, iter, 30) == 0);
        CHECK(std::abs(x(0) + x(1) - 1.0) < 1e-14);
    }
}

TEST_CASE("bicgstab_ell rejects ell = 0 (issue 924)", "[itl][krylov][bicgstab_ell][924]") {
    // gamma has size 1 at ell = 0, and gamma(1) was read: error 3 and x = 0,
    // silently.
    using M = mat::compressed2D<double>;
    using V = vec::dense_vector<double>;
    const M A = tridiag<double>(4);
    const itl::pc::identity<M> I(A);
    V b(4, 1.0), x(4, 0.0);
    itl::basic_iteration<double> iter(b, 100, 1e-10);
    CHECK_THROWS_AS(itl::bicgstab_ell(A, x, b, I, iter, 0), std::invalid_argument);
}

TEST_CASE("cg and bicg report rho == 0 for an unconverged r (issue 924)", "[itl][krylov][breakdown][924]") {
    // With an indefinite M, rho = r^T M^-1 r can be 0 for r != 0: alpha was 0,
    // the next beta 0 / 0, and x became NaN.
    using M = mat::dense2D<double>;
    using V = vec::dense_vector<double>;
    M A(2, 2);
    A(0, 0) = A(1, 1) = 1.0;
    A(0, 1) = A(1, 0) = 0.0;
    V b(2);
    b(0) = 1.0; b(1) = 0.0;
    const swap_pc P;
    {
        V x(2, 0.0);
        itl::basic_iteration<double> iter(b, 100, 1e-10);
        CHECK(itl::cg(A, x, b, P, iter) == 2);
        CHECK(x(0) == 0.0);
        CHECK(x(1) == 0.0);
    }
    {
        V x(2, 0.0);
        itl::basic_iteration<double> iter(b, 100, 1e-10);
        CHECK(itl::bicg(A, x, b, P, iter) == 2);
        CHECK(std::isfinite(x(0)));
        CHECK(std::isfinite(x(1)));
    }
}

TEST_CASE("minres records a zero M-norm residual (issue 924)", "[itl][krylov][minres][924]") {
    // beta == 0 returned without telling the iteration object, which kept the
    // error and residual of an earlier solve.
    using M = mat::compressed2D<double>;
    using V = vec::dense_vector<double>;
    const M A = tridiag<double>(4);
    V b(4, 1.0);
    SECTION("an exact x0: converged, error 0") {
        // b = 0 and x0 = 0: r0 = 0. The object carries an earlier failure.
        const itl::pc::identity<M> I(A);
        V x(4, 0.0), zero_b(4, 0.0);
        itl::basic_iteration<double> reused(1.0, 100, 1e-12);
        reused.fail(3, "an earlier solve");
        CHECK(itl::minres(A, x, zero_b, I, reused) == 0);
        CHECK(reused.is_finished());
        CHECK(reused.resid() == 0.0);
    }
    SECTION("M^-1 maps r0 to 0: a breakdown") {
        const zero_pc Z;
        V x(4, 0.0);
        itl::basic_iteration<double> iter(b, 100, 1e-12);
        CHECK(itl::minres(A, x, b, Z, iter) == 2);
        CHECK(iter.is_finished());
    }
}

TEST_CASE("Every solver handles an empty system (issue 924)", "[itl][krylov][empty][924]") {
    using M  = mat::compressed2D<double>;
    using V  = vec::dense_vector<double>;
    using It = itl::basic_iteration<double>;
    const M A(0, 0);
    const itl::pc::identity<M> I(A);
    for (const auto& s : all_solvers<M, V, itl::pc::identity<M>, It>()) {
        INFO(s.name);
        V x(0), b(0);
        It iter(b, 100, 1e-10);
        CHECK(s.run(A, x, b, I, iter) == 0);
        CHECK(iter.iterations() == 0);
    }
}

TEST_CASE("basic_iteration: first, post-increment, converged_now, non-finite (issue 924)",
          "[itl][iteration][924]") {
    itl::basic_iteration<double> it(1.0, 3, 1e-2);
    CHECK(it.first());          // i = 0
    const auto old = it++;
    CHECK(old.iterations() == 0);
    CHECK(it.iterations() == 1);
    CHECK(it.first());          // i = 1
    ++it;
    CHECK_FALSE(it.first());    // i = 2

    // converged_now: only the convergence test, not the iteration limit.
    ++it; ++it;                 // i = 4 > max_iter
    CHECK_FALSE(it.converged_now(0.5));
    CHECK_FALSE(it.is_finished());
    CHECK(it.finished(0.5));    // the limit, in finished()
    CHECK(it.error_code() == 1);
    CHECK(it.converged_now(1e-3));
    CHECK(it.error_code() == 0);

    // A NaN or Inf residual finishes with non_finite, before the other tests.
    itl::basic_iteration<double> nan_it(1.0, 100, 1e-2);
    CHECK(nan_it.finished(std::numeric_limits<double>::quiet_NaN()));
    CHECK(nan_it.error_code() == itl::basic_iteration<double>::non_finite);
    nan_it.reset(1.0);
    CHECK(nan_it.finished(std::numeric_limits<double>::infinity()));
    CHECK(int(nan_it) == 6);
    nan_it.reset(1.0);
    CHECK_FALSE(nan_it.finished(std::numeric_limits<double>::max()));

    // cyclic_iteration reports it too.
    std::ostringstream out;
    itl::cyclic_iteration<double> cyc(1.0, 100, 1e-2, 10, 0.0, out);
    CHECK(cyc.finished(std::numeric_limits<double>::quiet_NaN()));
    CHECK(cyc.error_code() == 6);
    CHECK(out.str().find("iteration 0") != std::string::npos);
}
