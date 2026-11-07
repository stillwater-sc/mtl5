#pragma once
// mtl5_bench -- dense factorizations, eigensolvers, SVD, matrix functions and
// element-wise kernels (#721 milestone M3).
//
// CHECKS. Each is chosen so that a WRONG result cannot pass, which for a
// factorization means checking the factors, not just their shape:
//
//   lu, cholesky, ldlt, ldlt_bk, qr      normwise backward error of a SOLVE with
//                                        the factors: |b - A x| / (|A||x| + |b|)
//   lq                                   L L^T against A A^T (Q is orthogonal),
//                                        each entry against its rows' norms
//   hessenberg                           structure (zero below the subdiagonal)
//                                        and the similarity invariants trace and
//                                        Frobenius norm
//   inv                                  A X = I (64 sampled entries)
//   lu_iterative_refine                  f64 backward error of an f32 solve
//   eigenvalue_symmetric / eigenvalue    trace and Frobenius / trace(A^2):
//                                        sums of eigenvalue powers are traces of
//                                        powers of A + E, well conditioned even
//                                        when single eigenvalues are not
//   eigen_symmetric(_jacobi)             A v = lambda v and V^T V = I (sampled)
//   schur                                Q T Q^T = A, Q^T Q = I, T quasi-upper
//   svd                                  U S V^T = A, U^T U = V^T V = I, sum of
//                                        sigma^2, S nonnegative and descending
//   expm                                 expm(A) expm(-A) = I
//   sylvester                            the residual A X + X B - C
//   householder                          H x = beta e_1
//   exp, log, sin, sqrt, erf             against std:: element by element
//
// Tolerances use gamma_k (common.hpp) with k the problem dimension: a stable
// algorithm sits orders of magnitude inside it, and a wrong one is O(1) outside.
//
// WORK MODEL. Rates use the CONVENTIONAL LAPACK operation counts (2/3 n^3 for
// LU, 4/3 n^3 for QR, 22 n^3 for a full SVD, ...), so a GFLOP/s figure compares
// implementations rather than describing the operations this one performed: a
// one-sided Jacobi SVD does more arithmetic than Golub-Kahan, and its rate is
// stated in Golub-Kahan flops on purpose. The bytes are the compulsory 2 n^2
// (read A, write the factor), so every factorization is compute-bound.
//
// NOT REGISTERED: Givens rotations. The only public Givens function,
// apply_givens_rotation (givens.hpp), is GMRES's internal step on its Hessenberg
// matrix; the rotations are measured where they are used (Krylov solvers, M5).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue_symmetric_jacobi.hpp>
#include <mtl/operation/erf.hpp>
#include <mtl/operation/exp.hpp>
#include <mtl/operation/expm.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/log.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/lu_iterative_refine.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/schur.hpp>
#include <mtl/operation/sin.hpp>
#include <mtl/operation/sqrt.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/sylvester.hpp>

#include <benchmarks/platform/common.hpp>

namespace mtl::bench::platform::dense {

/// Factorization sizes, and the smaller ones for the iterative eigen/SVD
/// methods, whose constant is ~10-20x a factorization's.
inline std::size_t n_fact(size_class c) {
    return c == size_class::small ? 128 : c == size_class::medium ? 512 : 1024;
}
inline std::size_t n_eig(size_class c) {
    return c == size_class::small ? 64 : c == size_class::medium ? 256 : 512;
}
inline std::string sq(std::size_t n) { return std::to_string(n) + "x" + std::to_string(n); }
inline double n3(std::size_t n) { return double(n) * double(n) * double(n); }

// -- inputs -------------------------------------------------------------------

/// Symmetric positive definite, cheaply: a symmetric random matrix (entries in
/// [-1, 1]) plus n*I is diagonally dominant, hence SPD, and well conditioned.
template <typename T>
mat::dense2D<T> spd(std::size_t n, std::uint64_t seed) {
    auto A = symmetric<T>(n, seed);
    for (std::size_t i = 0; i < n; ++i) A(i, i) += T(n);
    return A;
}

/// A + s*I: shifts the spectrum by s.
template <typename T>
mat::dense2D<T> shifted(mat::dense2D<T> A, double s) {
    for (std::size_t i = 0; i < A.num_rows(); ++i) A(i, i) += T(s);
    return A;
}

// -- norms and reductions in long double ---------------------------------------

template <typename M>
long double inf_norm(const M& A) {
    long double best = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        long double s = 0;
        for (std::size_t j = 0; j < A.num_cols(); ++j) s += std::fabs((long double)A(i, j));
        best = std::max(best, s);
    }
    return best;
}
template <typename M>
long double frob(const M& A) {
    long double s = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) s += (long double)A(i, j) * A(i, j);
    return std::sqrt(s);
}
template <typename V>
long double vmax(const V& v) {
    long double m = 0;
    for (std::size_t i = 0; i < v.size(); ++i) m = std::max(m, std::fabs((long double)v(i)));
    return m;
}

/// Normwise backward error of x as a solution of A x = b:
///   |b - A x|_inf / (|A|_inf |x|_inf + |b|_inf).
/// A backward-stable solve gives O(eps); a wrong factor gives O(1).
template <typename T, typename M, typename V>
check_result backward_error(const M& A, const V& x, const V& b, std::size_t n) {
    long double rmax = 0;
    for (std::size_t i = 0; i < n; ++i) {
        long double r = b(i);
        for (std::size_t j = 0; j < n; ++j) r -= (long double)A(i, j) * x(j);
        rmax = std::max(rmax, std::fabs(r));
    }
    error_tracker e(tol<T>(n), "normwise backward error of a solve");
    e.add(rmax, 0.0L, inf_norm(A) * vmax(x) + vmax(b));
    return e.result();
}

/// max over 64 sampled (i, j) of |(X^T Y)(i, j) - delta_ij|, for orthogonality.
template <typename T, typename M>
void sampled_orthogonality(error_tracker& e, const M& Q) {
    const std::size_t n = Q.num_cols();
    for_samples(n, [&](std::size_t i, std::size_t j) {
        long double s = 0;
        for (std::size_t k = 0; k < Q.num_rows(); ++k) s += (long double)Q(k, i) * Q(k, j);
        e.add(s, i == j ? 1.0L : 0.0L, 1.0L);
    });
}

// -- solves through a factorization ----------------------------------------

template <typename T>
problem lu(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; std::vector<std::size_t> piv; vec::dense_vector<T> b, x; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 101), mat::dense2D<T>(n, n), {}, vec_of<T>(n, 102),
                                   vec::dense_vector<T>(n)});
    problem p{sq(n), 2.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; }, [s] { mtl::lu_factor(s->A, s->piv); }, {}};
    p.check = [s, n] {
        mtl::lu_solve(s->A, s->piv, s->x, s->b);
        return backward_error<T>(s->A0, s->x, s->b, n);
    };
    return p;
}

template <typename T>
problem cholesky(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; vec::dense_vector<T> b, x; };
    auto s = std::make_shared<S>(S{spd<T>(n, 103), mat::dense2D<T>(n, n), vec_of<T>(n, 104), vec::dense_vector<T>(n)});
    problem p{sq(n), 1.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; }, [s] { mtl::cholesky_factor(s->A); }, {}};
    p.check = [s, n] {
        mtl::cholesky_solve(s->A, s->x, s->b);
        return backward_error<T>(s->A0, s->x, s->b, n);
    };
    return p;
}

template <typename T>
problem ldlt(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; vec::dense_vector<T> b, x; };
    auto s = std::make_shared<S>(S{spd<T>(n, 105), mat::dense2D<T>(n, n), vec_of<T>(n, 106), vec::dense_vector<T>(n)});
    problem p{sq(n), 1.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; }, [s] { mtl::ldlt_factor(s->A); }, {}};
    p.check = [s, n] {
        mtl::ldlt_solve(s->A, s->x, s->b);
        return backward_error<T>(s->A0, s->x, s->b, n);
    };
    return p;
}

/// Bunch-Kaufman on a symmetric INDEFINITE matrix, where pivoting is the point.
template <typename T>
problem ldlt_bk(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; mtl::bk_pivot_info piv; vec::dense_vector<T> b, x; };
    auto s = std::make_shared<S>(S{symmetric<T>(n, 107), mat::dense2D<T>(n, n), {}, vec_of<T>(n, 108),
                                   vec::dense_vector<T>(n)});
    problem p{sq(n), 1.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; }, [s] { mtl::ldlt_bk_factor(s->A, s->piv); }, {}};
    p.check = [s, n] {
        mtl::ldlt_bk_solve(s->A, s->piv, s->x, s->b);
        return backward_error<T>(s->A0, s->x, s->b, n);
    };
    return p;
}

template <typename T>
problem qr(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; vec::dense_vector<T> tau, b, x; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 109), mat::dense2D<T>(n, n), vec::dense_vector<T>(n),
                                   vec_of<T>(n, 110), vec::dense_vector<T>(n)});
    problem p{sq(n), 4.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; }, [s] { mtl::qr_factor(s->A, s->tau); }, {}};
    p.check = [s, n] {
        mtl::qr_solve(s->A, s->tau, s->x, s->b);
        return backward_error<T>(s->A0, s->x, s->b, n);
    };
    return p;
}

/// LQ has no solve; check L L^T = A A^T, which holds because Q is orthogonal.
/// Each entry is measured against |A(i,:)| |A(j,:)| (Cauchy-Schwarz), the scale
/// a backward-stable factorization's error has.
template <typename T>
problem lq(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; vec::dense_vector<T> tau; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 111), mat::dense2D<T>(n, n), vec::dense_vector<T>(n)});
    problem p{sq(n), 4.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; }, [s] { mtl::lq_factor(s->A, s->tau); }, {}};
    p.check = [s, n] {
        std::vector<long double> rn(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < n; ++k) rn[i] += (long double)s->A0(i, k) * s->A0(i, k);
            rn[i] = std::sqrt(rn[i]);
        }
        error_tracker e(tol<T>(n), "L L^T vs A A^T, per |A(i,:)||A(j,:)| (64 samples)");
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double aat = 0, llt = 0;
            for (std::size_t k = 0; k < n; ++k) aat += (long double)s->A0(i, k) * s->A0(j, k);
            for (std::size_t k = 0; k <= std::min(i, j); ++k) llt += (long double)s->A(i, k) * s->A(j, k);
            e.add(llt, aat, rn[i] * rn[j]);
        });
        return e.result();
    };
    return p;
}

/// H = Q^T A Q: zero below the first subdiagonal, and the orthogonal-similarity
/// invariants trace(H) = trace(A) and |H|_F = |A|_F.
template <typename T>
problem hessenberg(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A, H; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 112), mat::dense2D<T>(1, 1)});
    problem p{sq(n), 10.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { s->H = mtl::hessenberg(s->A); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "trace, |.|_F and structure vs A");
        long double ta = 0, th = 0, below = 0;
        for (std::size_t i = 0; i < n; ++i) {
            ta += s->A(i, i);
            th += s->H(i, i);
            for (std::size_t j = 0; j + 1 < i; ++j) below = std::max(below, std::fabs((long double)s->H(i, j)));
        }
        const long double fa = frob(s->A);
        e.add(th, ta, std::sqrt((long double)n) * fa);
        e.add(frob(s->H), fa, fa);
        e.add(below, 0.0L, fa);
        return e.result();
    };
    return p;
}

template <typename T>
problem inv(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A, X; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 113), mat::dense2D<T>(1, 1)});
    problem p{sq(n), 2.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { s->X = mtl::inv(s->A); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "A X - I, per |A|_inf |X|_inf (64 samples)");
        const long double scale = inf_norm(s->A) * inf_norm(s->X);
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double ax = 0;
            for (std::size_t k = 0; k < n; ++k) ax += (long double)s->A(i, k) * s->X(k, j);
            e.add(ax, i == j ? 1.0L : 0.0L, scale);
        });
        return e.result();
    };
    return p;
}

/// Mixed precision: factor in f32, refine the f64 solution. Judged by the f64
/// backward error it reaches, which is the point of refining.
inline problem lu_iterative_refine_f32_f64(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<double> A; vec::dense_vector<double> b, x; };
    auto s = std::make_shared<S>(S{mat_of<double>(n, n, 114), vec_of<double>(n, 115), vec::dense_vector<double>(n)});
    problem p{sq(n), 2.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(double), [] {},
              [s] { mtl::lu_iterative_refine<float>(s->A, s->b, s->x); }, {}};
    p.check = [s, n] { return backward_error<double>(s->A, s->x, s->b, n); };
    return p;
}

// -- eigenvalues, Schur, SVD -------------------------------------------------

/// Sum of eigenvalues and of their squares against trace(A) and |A|_F^2.
template <typename T, typename V>
void symmetric_invariants(error_tracker& e, const mat::dense2D<T>& A, const V& lambda) {
    const std::size_t n = A.num_rows();
    long double tr = 0, sl = 0, sl2 = 0;
    for (std::size_t i = 0; i < n; ++i) tr += A(i, i);
    for (std::size_t i = 0; i < n; ++i) { sl += lambda(i); sl2 += (long double)lambda(i) * lambda(i); }
    const long double fa = frob(A);
    e.add(sl, tr, std::sqrt((long double)n) * fa);
    e.add(sl2, fa * fa, 2 * fa * fa);
}

template <typename T>
problem eigenvalue_symmetric(size_class c) {
    const std::size_t n = n_eig(c);
    using R = decltype(mtl::eigenvalue_symmetric(std::declval<mat::dense2D<T>>()));
    struct S { mat::dense2D<T> A; R lambda; };
    auto s = std::make_shared<S>(S{symmetric<T>(n, 116), R{}});
    problem p{sq(n), 4.0 / 3.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { s->lambda = mtl::eigenvalue_symmetric(s->A); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "sum lambda vs trace, sum lambda^2 vs |A|_F^2");
        symmetric_invariants(e, s->A, s->lambda);
        return e.result();
    };
    return p;
}

/// Eigenpairs: A v = lambda v on 8 spread columns, V^T V = I sampled, plus the
/// eigenvalue invariants. Jacobi and the tridiagonal QR path share it.
template <typename T, bool Jacobi>
problem eigen_symmetric_pairs(size_class c) {
    const std::size_t n = n_eig(c);
    using R = decltype(mtl::eigen_symmetric(std::declval<mat::dense2D<T>>()));
    struct S { mat::dense2D<T> A; std::unique_ptr<R> r; };
    auto s = std::make_shared<S>(S{symmetric<T>(n, Jacobi ? 117 : 118), nullptr});
    problem p{sq(n), 9.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] {
                  if constexpr (Jacobi) {
                      auto r = mtl::eigen_symmetric_jacobi(s->A);
                      s->r = std::make_unique<R>(R{std::move(r.eigenvalues), std::move(r.eigenvectors)});
                  } else {
                      s->r = std::make_unique<R>(mtl::eigen_symmetric(s->A));
                  }
              }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "A v - lambda v, V^T V - I, eigenvalue invariants");
        const auto& L = s->r->eigenvalues;
        const auto& V = s->r->eigenvectors;
        const long double an = inf_norm(s->A);
        for (std::size_t q = 0; q < 8; ++q) {
            const std::size_t j = q * (n - 1) / 7;
            long double rmax = 0, vm = 0;
            for (std::size_t i = 0; i < n; ++i) {
                long double av = 0;
                for (std::size_t k = 0; k < n; ++k) av += (long double)s->A(i, k) * V(k, j);
                rmax = std::max(rmax, std::fabs(av - (long double)L(j) * V(i, j)));
                vm = std::max(vm, std::fabs((long double)V(i, j)));
            }
            e.add(rmax, 0.0L, an * vm);
        }
        sampled_orthogonality<T>(e, V);
        symmetric_invariants(e, s->A, L);
        return e.result();
    };
    return p;
}

/// General (nonsymmetric) eigenvalues: sum lambda = trace(A) and
/// sum lambda^2 = trace(A^2) = sum_ij A_ij A_ji, both real.
///
/// Called with tol = machine epsilon, NOT eigenvalue()'s default of 1e-10.
/// That default stops the Francis iteration far short of full accuracy -- the
/// check fails it by 5x at n = 64 -- while schur(), the same iteration, uses
/// machine epsilon (schur.hpp). Measuring at full accuracy makes the rate
/// comparable with LAPACK's dgeev and with schur.
template <typename T>
problem eigenvalue_general(size_class c) {
    const std::size_t n = n_eig(c);
    using R = decltype(mtl::eigenvalue(std::declval<mat::dense2D<T>>()));
    struct S { mat::dense2D<T> A; R lambda; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 119), R{}});
    problem p{sq(n), 10.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { s->lambda = mtl::eigenvalue(s->A, std::numeric_limits<T>::epsilon()); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "sum lambda vs trace(A), sum lambda^2 vs trace(A^2)");
        long double tr = 0, tr2 = 0;
        std::complex<long double> sl = 0, sl2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            tr += s->A(i, i);
            for (std::size_t j = 0; j < n; ++j) tr2 += (long double)s->A(i, j) * s->A(j, i);
            const std::complex<long double> l(s->lambda(i).real(), s->lambda(i).imag());
            sl += l;
            sl2 += l * l;
        }
        const long double fa = frob(s->A);
        e.add(sl.real(), tr, std::sqrt((long double)n) * fa);
        e.add(sl.imag(), 0.0L, std::sqrt((long double)n) * fa);
        e.add(sl2.real(), tr2, 2 * fa * fa);
        e.add(sl2.imag(), 0.0L, 2 * fa * fa);
        return e.result();
    };
    return p;
}

/// A = Q T Q^T with Q orthogonal and T quasi-upper-triangular (1x1 and 2x2
/// diagonal blocks, so no two consecutive subdiagonal entries are nonzero).
template <typename T>
problem schur(size_class c) {
    const std::size_t n = n_eig(c);
    using R = decltype(mtl::schur(std::declval<mat::dense2D<T>>()));
    struct S { mat::dense2D<T> A; std::unique_ptr<R> r; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 120), nullptr});
    problem p{sq(n), 25.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { s->r = std::make_unique<R>(mtl::schur(s->A)); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "Q T Q^T - A per |A|_F, Q^T Q - I, quasi-triangularity");
        const auto& Q = s->r->Q;
        const auto& Tm = s->r->T;
        const long double fa = frob(s->A);
        std::vector<long double> w(n);
        for_samples(n, [&](std::size_t i, std::size_t j) {
            for (std::size_t k = 0; k < n; ++k) {          // w = T Q(j,:)^T
                w[k] = 0;
                for (std::size_t l = k > 0 ? k - 1 : 0; l < n; ++l) w[k] += (long double)Tm(k, l) * Q(j, l);
            }
            long double qtq = 0;
            for (std::size_t k = 0; k < n; ++k) qtq += (long double)Q(i, k) * w[k];
            e.add(qtq, s->A(i, j), fa);
        });
        sampled_orthogonality<T>(e, Q);
        long double below = 0;
        bool two_in_a_row = false;
        for (std::size_t i = 1; i < n; ++i) {
            for (std::size_t j = 0; j + 1 < i; ++j) below = std::max(below, std::fabs((long double)Tm(i, j)));
            if (i + 1 < n && Tm(i, i - 1) != T(0) && Tm(i + 1, i) != T(0)) two_in_a_row = true;
        }
        e.add(below + (two_in_a_row ? fa : 0.0L), 0.0L, fa);
        return e.result();
    };
    return p;
}

/// A = U S V^T: reconstruction (sampled), orthogonality of U and V (sampled),
/// sum sigma^2 = |A|_F^2, and S diagonal, nonnegative and descending.
template <typename T>
problem svd(size_class c) {
    const std::size_t n = n_eig(c);
    struct S { mat::dense2D<T> A, U, Sg, V; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 121), mat::dense2D<T>(1, 1), mat::dense2D<T>(1, 1),
                                   mat::dense2D<T>(1, 1)});
    problem p{sq(n), 22.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { mtl::svd(s->A, s->U, s->Sg, s->V); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "U S V^T - A per |A|_F, U^T U, V^T V, sum sigma^2, order");
        const long double fa = frob(s->A);
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double r = 0;
            for (std::size_t k = 0; k < n; ++k) r += (long double)s->U(i, k) * s->Sg(k, k) * s->V(j, k);
            e.add(r, s->A(i, j), fa);
        });
        sampled_orthogonality<T>(e, s->U);
        sampled_orthogonality<T>(e, s->V);
        long double s2 = 0, bad = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const long double sk = s->Sg(k, k);
            s2 += sk * sk;
            if (sk < 0 || (k > 0 && sk > (long double)s->Sg(k - 1, k - 1))) bad = fa;
        }
        e.add(s2, fa * fa, 2 * fa * fa);
        e.add(bad, 0.0L, fa);
        return e.result();
    };
    return p;
}

// -- matrix functions ------------------------------------------------------

/// expm(A) expm(-A) = I. A is scaled to |A| ~ 1, where the exponential is well
/// conditioned, so the check measures the algorithm, not the problem.
template <typename T>
problem expm(size_class c) {
    const std::size_t n = n_eig(c);
    struct S { mat::dense2D<T> A, E; };
    auto A = mat_of<T>(n, n, 122);
    const T scale = T(1.0 / std::sqrt(double(n)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) *= scale;
    auto s = std::make_shared<S>(S{std::move(A), mat::dense2D<T>(1, 1)});
    // ~6 products and one solve of the [13/13] Pade approximant, at |A| ~ 1.
    problem p{sq(n), 15.0 * n3(n), 2.0 * n * n * sizeof(T), [] {},
              [s] { s->E = mtl::expm(s->A); }, {}};
    p.check = [s, n] {
        mat::dense2D<T> mA(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) mA(i, j) = -s->A(i, j);
        const auto F = mtl::expm(mA);
        error_tracker e(tol<T>(n), "expm(A) expm(-A) - I per |E|_inf |F|_inf (64 samples)");
        const long double scale = inf_norm(s->E) * inf_norm(F);
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double ef = 0;
            for (std::size_t k = 0; k < n; ++k) ef += (long double)s->E(i, k) * F(k, j);
            e.add(ef, i == j ? 1.0L : 0.0L, scale);
        });
        return e.result();
    };
    return p;
}

/// A X + X B = C. A and B are shifted by 2 sqrt(n) I, which moves both spectra
/// into the right half plane, so lambda_i(A) + lambda_j(B) stays far from 0 and
/// the equation is well conditioned.
template <typename T>
problem sylvester(size_class c) {
    const std::size_t n = n_eig(c);
    const double sh = 2.0 * std::sqrt(double(n));
    struct S { mat::dense2D<T> A, B, C, X; };
    auto s = std::make_shared<S>(S{shifted(mat_of<T>(n, n, 123), sh), shifted(mat_of<T>(n, n, 124), sh),
                                   mat_of<T>(n, n, 125), mat::dense2D<T>(1, 1)});
    // Bartels-Stewart: two Schur forms (~25 n^3 each), four products, the solve.
    problem p{sq(n), 60.0 * n3(n), 3.0 * n * n * sizeof(T), [] {},
              [s] {
                  auto [X, perturbed] = mtl::sylvester(s->A, s->B, s->C);
                  (void)perturbed;
                  s->X = std::move(X);
              }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "A X + X B - C per (|A|_F + |B|_F)|X|_F + |C|_F (64 samples)");
        const long double scale = (frob(s->A) + frob(s->B)) * frob(s->X) + frob(s->C);
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double r = 0;
            for (std::size_t k = 0; k < n; ++k)
                r += (long double)s->A(i, k) * s->X(k, j) + (long double)s->X(i, k) * s->B(k, j);
            e.add(r, s->C(i, j), scale);
        });
        return e.result();
    };
    return p;
}

/// One Householder reflector built from A's first column and applied to all of
/// A from the left: the rank-1 update at the core of QR, LQ and Hessenberg.
template <typename T>
problem householder(size_class c) {
    const std::size_t n = n_fact(c);
    struct S { mat::dense2D<T> A0, A; vec::dense_vector<T> x; mtl::normalized_reflection<T> h; };
    auto A0 = mat_of<T>(n, n, 126);
    vec::dense_vector<T> x(n);
    for (std::size_t i = 0; i < n; ++i) x(i) = A0(i, 0);
    auto s = std::make_shared<S>(S{A0, mat::dense2D<T>(n, n), x, {}});
    problem p{sq(n), 4.0 * n * n + 3.0 * n, 2.0 * n * n * sizeof(T),
              [s] { s->A = s->A0; },
              [s] {
                  s->h = mtl::householder_reflector(s->x);
                  mtl::apply_householder_left(s->A, s->h.v, s->h.tau, std::size_t(0), std::size_t(0));
              }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "H x - beta e_1 per |x|");
        long double xn = 0;
        for (std::size_t i = 0; i < n; ++i) xn += (long double)s->x(i) * s->x(i);
        xn = std::sqrt(xn);
        e.add(s->A(0, 0), s->h.beta, xn);
        for (std::size_t i = 1; i < n; ++i) e.add(s->A(i, 0), 0.0L, xn);
        e.add(std::fabs((long double)s->h.beta), xn, xn);   // |beta| = |x|
        return e.result();
    };
    return p;
}

// -- element-wise ------------------------------------------------------------

/// One element-wise function over a vector, against std:: per element. The
/// result vector is allocated by the call (these return a new vector), and
/// that allocation is part of what is timed.
template <typename T, int Fn>
problem elementwise(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<T> x, y; };
    auto x = vec_of<T>(n, 127);
    if (Fn == 1 || Fn == 3)   // log, sqrt: positive arguments
        for (std::size_t i = 0; i < n; ++i) x(i) = (x(i) < T(0) ? -x(i) : x(i)) + T(0.5);
    auto s = std::make_shared<S>(S{std::move(x), vec::dense_vector<T>(1)});
    problem p{vshape(n), 1.0 * n, 2.0 * n * sizeof(T), [] {},
              [s] {
                  if constexpr (Fn == 0) s->y = mtl::exp(s->x);
                  else if constexpr (Fn == 1) s->y = mtl::log(s->x);
                  else if constexpr (Fn == 2) s->y = mtl::sin(s->x);
                  else if constexpr (Fn == 3) s->y = mtl::sqrt(s->x);
                  else s->y = mtl::erf(s->x);
              }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(1), "rel err vs std:: per element");
        for (std::size_t i = 0; i < n; ++i) {
            const T v = s->x(i);
            const T r = Fn == 0 ? std::exp(v) : Fn == 1 ? std::log(v) : Fn == 2 ? std::sin(v)
                      : Fn == 3 ? std::sqrt(v) : std::erf(v);
            e.add(s->y(i), r, std::fabs((long double)r) + std::numeric_limits<T>::min());
        }
        return e.result();
    };
    return p;
}

inline void register_all(registry& r) {
    r.add("factor", "lu", "f64", lu<double>);
    r.add("factor", "lu", "f32", lu<float>);
    r.add("factor", "cholesky", "f64", cholesky<double>);
    r.add("factor", "cholesky", "f32", cholesky<float>);
    r.add("factor", "ldlt", "f64", ldlt<double>);
    r.add("factor", "ldlt_bk", "f64", ldlt_bk<double>);
    r.add("factor", "qr", "f64", qr<double>);
    r.add("factor", "lq", "f64", lq<double>);
    r.add("factor", "hessenberg", "f64", hessenberg<double>);
    r.add("factor", "householder", "f64", householder<double>);
    r.add("factor", "inv", "f64", inv<double>);
    r.add("factor", "lu_iterative_refine", "f32>f64", lu_iterative_refine_f32_f64);

    r.add("eigen", "eigenvalue_symmetric", "f64", eigenvalue_symmetric<double>);
    r.add("eigen", "eigen_symmetric", "f64", eigen_symmetric_pairs<double, false>);
    r.add("eigen", "eigen_symmetric_jacobi", "f64", eigen_symmetric_pairs<double, true>);
    r.add("eigen", "eigenvalue", "f64", eigenvalue_general<double>);
    r.add("eigen", "schur", "f64", schur<double>);
    r.add("eigen", "svd", "f64", svd<double>);
    r.add("eigen", "expm", "f64", expm<double>);
    r.add("eigen", "sylvester", "f64", sylvester<double>);

    r.add("elementwise", "exp", "f64", elementwise<double, 0>);
    r.add("elementwise", "log", "f64", elementwise<double, 1>);
    r.add("elementwise", "sin", "f64", elementwise<double, 2>);
    r.add("elementwise", "sqrt", "f64", elementwise<double, 3>);
    r.add("elementwise", "erf", "f64", elementwise<double, 4>);
}

} // namespace mtl::bench::platform::dense
