// MTL5 -- genericity over the number type, through native_shim (#703).
//
// Each entry point runs on float_shim and double_shim, user-defined types
// that a native-only construct cannot accept (tests/support/mtl5_test), and on
// the float or double underneath. Two things are checked:
//
//   - that it compiles: a qualified std::sqrt, an is_floating_point_v gate
//     without a generic branch, or a mixed `x >= 0` on the generic path makes
//     this file fail to build;
//   - that the generic path is right: the shim rounds each operation as its
//     native type does, so the results must be the native ones, bit for bit,
//     wherever MTL5 runs the same code for both.
//
// Bit-identity is not expected where MTL5 gives native types a path of their
// own: the BLAS and LAPACK dispatches, the Highway SIMD kernels, and the
// kernels in simd/ that native types take for the norms, dot, axpy, gemv and
// gemm (and so the Krylov solvers built on them). Those comparisons, and every
// comparison in a BLAS, LAPACK or Highway build, use a tolerance of a few eps
// of the result instead. LU, QR, Cholesky, Householder, Givens, pow2_scaling,
// the SVD, the symmetric eigensolvers, kron and the projections run the same
// code for both and are compared bit for bit.
// The target is built with -ffp-contract=off (tests/unit/CMakeLists.txt), so
// the compiler contracts neither side.
//
// std::complex<native_shim> is not covered: the standard leaves complex<T>
// unspecified for a T that is not a floating-point type.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/mtl.hpp>
#include <mtl5_test/native_shim.hpp>

using namespace mtl;
using mtl5_test::double_shim;
using mtl5_test::float_shim;

#if defined(MTL5_HAS_BLAS) || defined(MTL5_HAS_LAPACK) || defined(MTL5_HAS_HIGHWAY)
constexpr bool native_paths_differ = true;    // native types may take BLAS, LAPACK or SIMD
#else
constexpr bool native_paths_differ = false;
#endif

namespace {

/// The value as its native type: a shim's native(), or a native value itself.
template <typename T> auto native_of(T x) {
    if constexpr (std::is_floating_point_v<T>) return x;
    else return x.native();
}

/// The comparison of one value. `shared` says that MTL5 runs the same code for
/// the shim and the native type, so in a build without native-only paths the
/// two must agree bit for bit; otherwise within tol * scale.
template <typename S, typename N>
void agree(S s, N n, double scale, bool shared, double tol_eps = 64) {
    const double a = static_cast<double>(s), b = static_cast<double>(n);
    CAPTURE(a, b);
    if (std::isnan(b)) { REQUIRE(std::isnan(a)); return; }
    if (shared && !native_paths_differ) {
        REQUIRE(native_of(s) == native_of(n));
    } else {
        const double eps = static_cast<double>(std::numeric_limits<N>::epsilon());
        REQUIRE(std::abs(a - b) <= tol_eps * eps * scale);
    }
}

template <typename S, typename N>
void agree(const vec::dense_vector<S>& s, const vec::dense_vector<N>& n, bool shared, double tol_eps = 64) {
    REQUIRE(s.size() == n.size());
    double scale = 0;
    for (std::size_t i = 0; i < n.size(); ++i) scale = std::max(scale, std::abs(double(n(i))));
    for (std::size_t i = 0; i < n.size(); ++i) {
        CAPTURE(i);
        agree(s(i), n(i), scale, shared, tol_eps);
    }
}

template <typename S, typename N>
void agree(const mat::dense2D<S>& s, const mat::dense2D<N>& n, bool shared, double tol_eps = 64) {
    REQUIRE(s.num_rows() == n.num_rows());
    REQUIRE(s.num_cols() == n.num_cols());
    double scale = 0;
    for (std::size_t i = 0; i < n.num_rows(); ++i)
        for (std::size_t j = 0; j < n.num_cols(); ++j) scale = std::max(scale, std::abs(double(n(i, j))));
    for (std::size_t i = 0; i < n.num_rows(); ++i)
        for (std::size_t j = 0; j < n.num_cols(); ++j) {
            CAPTURE(i, j);
            agree(s(i, j), n(i, j), scale, shared, tol_eps);
        }
}

/// A native and a shim copy of the same data.
template <typename S>
struct pair_data {
    using N = typename S::native_type;
    static vec::dense_vector<S> shim(const vec::dense_vector<N>& x) {
        vec::dense_vector<S> y(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) y(i) = S(x(i));
        return y;
    }
    static mat::dense2D<S> shim(const mat::dense2D<N>& A) {
        mat::dense2D<S> B(A.num_rows(), A.num_cols());
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            for (std::size_t j = 0; j < A.num_cols(); ++j) B(i, j) = S(A(i, j));
        return B;
    }
    static mat::compressed2D<S> shim(const mat::compressed2D<N>& A) {
        mat::compressed2D<S> B(A.num_rows(), A.num_cols());
        {
            mat::inserter<mat::compressed2D<S>> ins(B);
            const auto& starts = A.ref_major();
            const auto& cols = A.ref_minor();
            const auto& vals = A.ref_data();
            for (std::size_t i = 0; i < A.num_rows(); ++i)
                for (std::size_t k = starts[i]; k < starts[i + 1]; ++k) ins[i][cols[k]] << S(vals[k]);
        }
        return B;
    }
    static vec::dense_vector<N> random_vector(std::size_t n, std::mt19937_64& g, double scale = 1.0) {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        vec::dense_vector<N> x(n);
        for (std::size_t i = 0; i < n; ++i) x(i) = static_cast<N>(scale * u(g));
        return x;
    }
    static mat::dense2D<N> random_matrix(std::size_t m, std::size_t n, std::mt19937_64& g) {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        mat::dense2D<N> A(m, n);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<N>(u(g));
        return A;
    }
    /// A symmetric positive definite matrix: B^T B + n I.
    static mat::dense2D<N> spd(std::size_t n, std::mt19937_64& g) {
        const auto B = random_matrix(n, n, g);
        mat::dense2D<N> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                N s = 0;
                for (std::size_t k = 0; k < n; ++k) s += B(k, i) * B(k, j);
                A(i, j) = s + (i == j ? N(n) : N(0));
            }
        return A;
    }
    /// The 1-D Laplacian, 4 on the diagonal: sparse and SPD.
    static mat::compressed2D<N> laplacian(std::size_t n) {
        mat::compressed2D<N> A(n, n);
        {
            mat::inserter<mat::compressed2D<N>> ins(A);
            for (std::size_t i = 0; i < n; ++i) {
                ins[i][i] << N(4);
                if (i > 0) ins[i][i - 1] << N(-1);
                if (i + 1 < n) ins[i][i + 1] << N(-1);
            }
        }
        return A;
    }
};

} // namespace

#define MTL5_SHIMS float_shim, double_shim

TEMPLATE_TEST_CASE("genericity: norms", "[generic][norms]", MTL5_SHIMS) {
    using S = TestType;
    using N = typename S::native_type;
    using P = pair_data<S>;
    std::mt19937_64 g(7031);
    const auto x = P::random_vector(37, g);
    const auto xs = P::shim(x);
    // The norms reduce native types with simd/algorithm.hpp's kernels, which
    // order the sum differently, and the shim with the generic loop.
    agree(two_norm(xs), two_norm(x), double(two_norm(x)), false);
    agree(one_norm(xs), one_norm(x), double(one_norm(x)), false);
    agree(infinity_norm(xs), infinity_norm(x), double(infinity_norm(x)), true);
    const auto A = P::random_matrix(6, 5, g);
    agree(frobenius_norm(P::shim(A)), frobenius_norm(A), double(frobenius_norm(A)), false);
    // The two_norm rescan (#700): entries near max and near min, where the
    // plain sum overflows or underflows. The rescan must reach the shim
    // through ADL frexp and ldexp.
    for (double scale : {double(std::numeric_limits<N>::max()) / 8, double(std::numeric_limits<N>::min()) * 4}) {
        CAPTURE(scale);
        const auto y = P::random_vector(9, g, scale);
        const S r = two_norm(P::shim(y));
        REQUIRE(isfinite(r));
        REQUIRE(r > S(0));
        agree(r, two_norm(y), double(two_norm(y)), false);
    }
}

TEMPLATE_TEST_CASE("genericity: dot, axpy, gemv and gemm", "[generic][blas]", MTL5_SHIMS) {
    using S = TestType;
    using P = pair_data<S>;
    std::mt19937_64 g(7032);
    const auto x = P::random_vector(23, g), y = P::random_vector(23, g);
    const auto xs = P::shim(x), ys = P::shim(y);
    agree(dot(xs, ys), dot(x, y), double(two_norm(x) * two_norm(y)), false);
    {
        auto z = y;
        auto zs = ys;
        axpy(typename S::native_type(0.75), x, z);
        axpy(S(0.75), xs, zs);
        agree(zs, z, false);   // native types take simd::axpy, which may fuse
    }
    const auto A = P::random_matrix(7, 5, g), B = P::random_matrix(5, 6, g);
    const auto As = P::shim(A), Bs = P::shim(B);
    const auto v = P::random_vector(5, g);
    vec::dense_vector<typename S::native_type> w(7);
    vec::dense_vector<S> ws(7);
    mult(A, v, w);
    mult(As, P::shim(v), ws);
    agree(ws, w, false);   // gemv: BLAS for native in a BLAS build
    mat::dense2D<typename S::native_type> C(7, 6);
    mat::dense2D<S> Cs(7, 6);
    mult(A, B, C);
    mult(As, Bs, Cs);
    agree(Cs, C, false);   // gemm: BLAS, or a native-tuned kernel
    const auto K = kron(A, B);
    agree(kron(As, Bs), K, true);
}

TEMPLATE_TEST_CASE("genericity: LU, QR and Cholesky", "[generic][factorization]", MTL5_SHIMS) {
    using S = TestType;
    using N = typename S::native_type;
    using P = pair_data<S>;
    std::mt19937_64 g(7033);
    SECTION("lu_factor and lu_solve") {
        auto A = P::random_matrix(6, 6, g);
        auto As = P::shim(A);
        const auto b = P::random_vector(6, g);
        std::vector<std::size_t> p, ps;
        REQUIRE(lu_factor(A, p) == 0);
        REQUIRE(lu_factor(As, ps) == 0);
        REQUIRE(p == ps);
        agree(As, A, true);
        vec::dense_vector<N> x(6);
        vec::dense_vector<S> xs(6);
        lu_solve(A, p, x, b);
        lu_solve(As, ps, xs, P::shim(b));
        agree(xs, x, true, 1024);
    }
    SECTION("qr_factor, qr_extract_Q and qr_solve") {
        auto A = P::random_matrix(8, 5, g);
        auto As = P::shim(A);
        const auto b = P::random_vector(8, g);
        vec::dense_vector<N> tau;
        vec::dense_vector<S> taus;
        qr_factor(A, tau);
        qr_factor(As, taus);
        agree(As, A, true);
        agree(taus, tau, true);
        agree(qr_extract_Q(As, taus), qr_extract_Q(A, tau), true);
        vec::dense_vector<N> x(5);
        vec::dense_vector<S> xs(5);
        qr_solve(A, tau, x, b);
        qr_solve(As, taus, xs, P::shim(b));
        agree(xs, x, true, 1024);
    }
    SECTION("cholesky_factor and cholesky_solve") {
        auto A = P::spd(6, g);
        auto As = P::shim(A);
        const auto b = P::random_vector(6, g);
        REQUIRE(cholesky_factor(A) == 0);
        REQUIRE(cholesky_factor(As) == 0);
        agree(As, A, true);
        vec::dense_vector<N> x(6);
        vec::dense_vector<S> xs(6);
        cholesky_solve(A, x, b);
        cholesky_solve(As, xs, P::shim(b));
        agree(xs, x, true, 1024);
    }
}

TEMPLATE_TEST_CASE("genericity: Householder, Givens and pow2_scaling", "[generic][rotation]", MTL5_SHIMS) {
    using S = TestType;
    using N = typename S::native_type;
    using P = pair_data<S>;
    std::mt19937_64 g(7034);
    // One code path for both: pow2_scaling's fallback for the shim gives the
    // same exact powers of two as the std::ilogb / std::ldexp specialization.
    for (int t = 0; t < 40; ++t) {
        const double scale = std::ldexp(1.0, (t % 9) * 30 - 120);
        const auto x = P::random_vector(1 + std::size_t(t % 6), g, scale);
        const auto h = householder_reflector(x);
        const auto hs = householder_reflector(P::shim(x));
        CAPTURE(t);
        agree(hs.v, h.v, true);
        agree(hs.tau, h.tau, 2.0, true);
        agree(hs.beta, h.beta, double(std::abs(h.beta)), true);
        const N a = x(0), b = x.size() > 1 ? x(1) : N(0);
        const auto gr = math::givens_rotation(a, b);
        const auto grs = math::givens_rotation(S(a), S(b));
        agree(grs.c, gr.c, 1.0, true);
        agree(grs.s, gr.s, 1.0, true);
        agree(grs.r, gr.r, double(gr.r), true);
        if (a != N(0)) {   // exponent() is defined for a positive finite m
            const N m = std::abs(a);
            REQUIRE(math::pow2_scaling<S>::exponent(S(m)) == math::pow2_scaling<N>::exponent(m));
            REQUIRE(math::pow2_scaling<S>::scale(S(a), 17).native() == math::pow2_scaling<N>::scale(a, 17));
            REQUIRE(math::pow2_scaling<S>::scale(S(a), -40).native() == math::pow2_scaling<N>::scale(a, -40));
        }
    }
}

TEMPLATE_TEST_CASE("genericity: SVD and the symmetric eigensolver", "[generic][spectral]", MTL5_SHIMS) {
    using S = TestType;
    using N = typename S::native_type;
    using P = pair_data<S>;
    std::mt19937_64 g(7035);
    SECTION("svd") {
        const auto A = P::random_matrix(6, 4, g);
        mat::dense2D<N> U, Sv, V;
        mat::dense2D<S> Us, Svs, Vs;
        svd(A, U, Sv, V);
        svd(P::shim(A), Us, Svs, Vs);
        agree(Svs, Sv, true, 256);
    }
    SECTION("eigenvalue_symmetric and eigen_symmetric") {
        const auto A = P::spd(6, g);
        const auto As = P::shim(A);
        agree(eigenvalue_symmetric_generic(As), eigenvalue_symmetric_generic(A), true);
        agree(eigenvalue_symmetric(As), eigenvalue_symmetric(A), true, 256);   // LAPACK for native in a LAPACK build
        const auto [l, V] = eigen_symmetric(A);
        const auto [ls, Vs] = eigen_symmetric(As);
        agree(ls, l, true);
        agree(Vs, V, true);
    }
}

TEMPLATE_TEST_CASE("genericity: Krylov solvers with identity, diagonal and ILU(0) preconditioners",
                   "[generic][itl]", MTL5_SHIMS) {
    using S = TestType;
    using N = typename S::native_type;
    using P = pair_data<S>;
    const std::size_t n = 30;
    const auto A = P::laplacian(n);
    const auto As = P::shim(A);
    vec::dense_vector<N> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = N(1) + N(i % 4);
    const auto bs = P::shim(b);
    const double tol = std::is_same_v<N, float> ? 1e-5 : 1e-10;
    auto run = [&](auto solver, auto make_pc, auto make_pcs) {
        vec::dense_vector<N> x(n, N(0));
        vec::dense_vector<S> xs(n, S(0));
        auto pc = make_pc(A);
        auto pcs = make_pcs(As);
        itl::basic_iteration<N> iter(b, 200, N(tol));
        itl::basic_iteration<S> iters(bs, 200, S(tol));
        REQUIRE(solver(A, x, b, pc, iter) == 0);
        REQUIRE(solver(As, xs, bs, pcs, iters) == 0);
        // dot and two_norm round differently for the two, so the residual
        // estimates, and where they cross the tolerance, can differ a little.
        REQUIRE(std::abs(iters.iterations() - iter.iterations()) <= 2);
        agree(xs, x, false, 1e4);   // dot and two_norm reduce native types their own way
    };
    auto cg = [](auto& M, auto& x, auto& rhs, auto& pc, auto& it) { return itl::cg(M, x, rhs, pc, it); };
    auto bicgstab = [](auto& M, auto& x, auto& rhs, auto& pc, auto& it) { return itl::bicgstab(M, x, rhs, pc, it); };
    auto gmres = [](auto& M, auto& x, auto& rhs, auto& pc, auto& it) { return itl::gmres(M, x, rhs, pc, it, 10); };
    auto minres = [](auto& M, auto& x, auto& rhs, auto& pc, auto& it) { return itl::minres(M, x, rhs, pc, it); };
    auto identity = [](const auto& M) { return itl::pc::identity<std::decay_t<decltype(M)>>(M); };
    auto diagonal = [](const auto& M) { return itl::pc::diagonal<std::decay_t<decltype(M)>>(M); };
    auto ilu = [](const auto& M) {
        return itl::pc::ilu_0<typename std::decay_t<decltype(M)>::value_type>(M);
    };
    SECTION("cg") { run(cg, diagonal, diagonal); }
    SECTION("bicgstab") { run(bicgstab, ilu, ilu); }
    SECTION("gmres") { run(gmres, identity, identity); }
    SECTION("minres") { run(minres, identity, identity); }
}

TEMPLATE_TEST_CASE("genericity: project_onto and embed_into", "[generic][projection]", MTL5_SHIMS) {
    using S = TestType;
    using N = typename S::native_type;
    using P = pair_data<S>;
    std::mt19937_64 g(7036);
    const auto x = P::random_vector(11, g);
    // Native to shim and back is exact: the shim holds the same N.
    const auto xs = project_onto<S>(x);
    agree(xs, x, true);
    agree(embed_into<N>(xs), x, true);
    const auto A = P::random_matrix(3, 4, g);
    agree(project_onto<S>(A), A, true);
}

TEMPLATE_TEST_CASE("genericity: vector properties see the type's own NaN and infinity", "[generic][properties]",
                   MTL5_SHIMS) {
    // A custom type that provides isnan / isinf / isfinite by ADL, as a posit's
    // NaR or an lns NaN does, is classified by them. Before #703 every
    // non-native real was taken to be finite and never NaN.
    using S = TestType;
    using L = std::numeric_limits<S>;
    vec::dense_vector<S> x(4, S(0));
    REQUIRE(is_zero(x));
    REQUIRE(is_finite(x));
    x(1) = S(3);
    x(2) = S(4);
    REQUIRE_FALSE(is_zero(x));
    REQUIRE(is_zero(x, S(4)));
    vec::dense_vector<S> u(4, S(0));
    u(1) = S(0.6);
    u(2) = S(0.8);
    REQUIRE(is_normalized(u));
    vec::dense_vector<S> w(4, S(0));
    w(0) = S(1);
    REQUIRE(is_orthogonal_to(u, w));
    REQUIRE_FALSE(has_nan(x));
    REQUIRE_FALSE(has_inf(x));
    x(3) = L::quiet_NaN();
    REQUIRE(has_nan(x));
    REQUIRE_FALSE(is_finite(x));
    x(3) = -L::infinity();
    REQUIRE(has_inf(x));
    REQUIRE_FALSE(has_nan(x));
    REQUIRE_FALSE(is_finite(x));
}

TEMPLATE_TEST_CASE("genericity: matrix, factorization and spectral properties", "[generic][properties]",
                   MTL5_SHIMS) {
    // Each predicate with its default tolerance, which until #703 was an int
    // literal (tol = 0, or the sentinel -1) that a type without implicit
    // construction from int cannot take.
    using S = TestType;
    using P = pair_data<S>;
    std::mt19937_64 g(7037);
    const auto A = P::shim(P::spd(5, g));
    REQUIRE(is_symmetric(A));
    REQUIRE(is_hermitian(A));
    REQUIRE_FALSE(is_triangular(A));
    REQUIRE_FALSE(is_upper_triangular(A));
    REQUIRE_FALSE(is_lower_triangular(A));
    REQUIRE_FALSE(is_diagonal(A));
    REQUIRE(is_diagonally_dominant(A));
    REQUIRE(is_spd(A));
    REQUIRE(is_positive_definite(A));
    REQUIRE_FALSE(is_singular(A));
    REQUIRE(is_nonsingular(A));
    REQUIRE(is_invertible(A));
    REQUIRE(numerical_rank(A) == 5);
    REQUIRE(nullity(A) == 0);
    const auto in = inertia(A);
    REQUIRE(in.positive == 5);
    REQUIRE_FALSE(is_indefinite(A));
    mat::dense2D<S> I(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) I(i, j) = S(i == j ? 1 : 0);
    REQUIRE(is_diagonal(I));
    REQUIRE(is_orthogonal(I));
    REQUIRE(is_unitary(I));
    REQUIRE(is_normal(I));
}
