#pragma once
// mtl5_bench -- sparse kernels, sparse direct solvers and fill-reducing
// orderings (#721 milestone M4). Every input is GENERATED, so nothing is
// downloaded and every machine times the same problem:
//
//   poisson2d(k)     the 5-point Laplacian on a k x k grid: SPD, 5 nnz/row
//   convdiff2d(k)    5-point convection-diffusion, upwind-skewed: unsymmetric,
//                    weakly diagonally dominant and irreducible, so nonsingular
//   scrambled(...)   either one under a random symmetric permutation, so that
//                    the natural order is a BAD order and an ordering has work
//   planted_btf      diagonal blocks with upper coupling, rows and columns
//                    scrambled independently: a known block-triangular form
//
// CHECKS -- each is one a wrong result cannot pass:
//   spmv, spmv_trans, crs_dense    per row, against sum |a_ij||x_j| (gamma_k)
//   spgemm                         C x against A (A x) for a random x
//   lower_solve (level-scheduled)  componentwise backward error per row,
//                                  |L x - b| <= gamma_n |L||x| (Higham): NOT bit
//                                  identity with dense_lower_solve, which
//                                  level_schedule.hpp documents only with FP
//                                  contraction off -- the unit tests build with
//                                  -ffp-contract=off for it, while this binary is
//                                  built as users build, and an FMA in one loop
//                                  and not the other differs in the last bits
//   direct solvers                 normwise backward error of the solve
//   amd, colamd                    a valid permutation, AND at most half the
//                                  Cholesky fill of the scrambled natural order
//                                  (of A for amd, of A^T A for colamd)
//   rcm                            a valid permutation whose bandwidth is within
//                                  4x of the grid's natural bandwidth k
//   block_triangular_form          valid permutations, zero-free diagonal, no
//                                  entry below its block, at least the planted
//                                  number of blocks
//
// WORK MODEL. Sparse kernels count 2 flops per stored nonzero, and bytes as
// values + column indices + row pointers + the vectors, so they sit on the
// bandwidth side of the roofline. Direct solvers have no standard flop count;
// their rates use the CHOLESKY-EQUIVALENT count of the AMD-ordered pattern,
// sum over columns of (nnz in column of L)^2, doubled for the unsymmetric
// solvers (LU, KLU, QR). That makes solvers comparable on a matrix; it is not
// the arithmetic each one performs.
//
// NOT REGISTERED: ordering::minimum_degree and the sparse::analysis helpers
// (elimination tree, postorder, supernodes). They have no user-facing entry
// point of their own: minimum degree is the engine inside amd and colamd, and
// the analyses run inside every symbolic factorization measured here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mtl/generators/laplacian.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/spgemm.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/sparse/factorization/level_schedule.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>
#include <mtl/sparse/ordering/rcm.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>

#include <benchmarks/platform/common.hpp>

namespace mtl::bench::platform::sparse_entries {

using csr = mat::compressed2D<double>;
namespace fac = mtl::sparse::factorization;
namespace ord = mtl::sparse::ordering;

/// Grid sides. Kernels are memory-bound, so their grids are large enough to
/// leave every cache (medium: 512^2 = 262144 rows); direct solvers fill in, so
/// theirs are smaller (medium: 96^2 = 9216 rows).
inline std::size_t grid_kernel(size_class c) {
    return c == size_class::small ? 64 : c == size_class::medium ? 512 : 1024;
}
inline std::size_t grid_direct(size_class c) {
    return c == size_class::small ? 32 : c == size_class::medium ? 96 : 192;
}
inline std::string grid_shape(std::size_t k, const csr& A) {
    return std::to_string(k) + "^2 grid, nnz " + std::to_string(A.nnz());
}

// -- generators ----------------------------------------------------------------

inline csr poisson2d(std::size_t k) { return mtl::generators::laplacian_2d<double>(k, k); }

/// 5-point convection-diffusion with the convection skewing west/south.
inline csr convdiff2d(std::size_t k) {
    const std::size_t n = k * k;
    const double c = 0.4;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t iy = 0; iy < k; ++iy)
            for (std::size_t ix = 0; ix < k; ++ix) {
                const std::size_t r = iy * k + ix;
                ins[r][r] << 4.0;
                if (ix > 0) ins[r][r - 1] << -1.0 - c;
                if (ix + 1 < k) ins[r][r + 1] << -1.0 + c;
                if (iy > 0) ins[r][r - k] << -1.0 - c;
                if (iy + 1 < k) ins[r][r + k] << -1.0 + c;
            }
    }
    return A;
}

inline std::vector<std::size_t> random_perm(std::size_t n, std::uint64_t seed) {
    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    xorshift64 g(seed);
    for (std::size_t i = n; i-- > 1;) {
        const auto j = static_cast<std::size_t>(g.next_double() * double(i + 1));
        std::swap(p[i], p[std::min(j, i)]);
    }
    return p;
}

/// B(i, j) = A(prow[i], pcol[j]): p[new] = old, the library's convention.
inline csr permuted(const csr& A, const std::vector<std::size_t>& prow,
                    const std::vector<std::size_t>& pcol) {
    const std::size_t n = A.num_rows();
    std::vector<std::size_t> rinv(n), cinv(A.num_cols());
    for (std::size_t i = 0; i < n; ++i) rinv[prow[i]] = i;
    for (std::size_t j = 0; j < cinv.size(); ++j) cinv[pcol[j]] = j;
    const auto& st = A.ref_major();
    const auto& ix = A.ref_minor();
    const auto& v = A.ref_data();
    csr B(n, A.num_cols());
    {
        mat::inserter<csr> ins(B);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t q = st[r]; q < st[r + 1]; ++q) ins[rinv[r]][cinv[ix[q]]] << v[q];
    }
    return B;
}

/// `blocks` convdiff blocks of `bk x bk` grids on the diagonal, each coupled to
/// every later block by a few entries above the diagonal, then rows and columns
/// scrambled independently. The block-triangular form has at least `blocks`
/// blocks (the coupling is strictly upper, so no two blocks merge).
inline csr planted_btf(std::size_t blocks, std::size_t bk, std::uint64_t seed) {
    const std::size_t m = bk * bk, n = blocks * m;
    const csr D = convdiff2d(bk);
    xorshift64 g(seed);
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        const auto& st = D.ref_major();
        const auto& ix = D.ref_minor();
        const auto& v = D.ref_data();
        for (std::size_t b = 0; b < blocks; ++b) {
            for (std::size_t r = 0; r < m; ++r)
                for (std::size_t q = st[r]; q < st[r + 1]; ++q) ins[b * m + r][b * m + ix[q]] << v[q];
            for (std::size_t later = b + 1; later < blocks; ++later)
                for (int e = 0; e < 3; ++e) {
                    const auto r = static_cast<std::size_t>(g.next_double() * double(m));
                    const auto c = static_cast<std::size_t>(g.next_double() * double(m));
                    ins[b * m + std::min(r, m - 1)][later * m + std::min(c, m - 1)] << 0.1;
                }
        }
    }
    return permuted(A, random_perm(n, seed + 1), random_perm(n, seed + 2));
}

// -- reference arithmetic on CRS --------------------------------------------------

/// y = A x and its magnitude |A||x|, in long double.
inline void ref_mult(const csr& A, const std::vector<long double>& x, std::vector<long double>& y,
                     std::vector<long double>& ay) {
    const auto& st = A.ref_major();
    const auto& ix = A.ref_minor();
    const auto& v = A.ref_data();
    y.assign(A.num_rows(), 0);
    ay.assign(A.num_rows(), 0);
    for (std::size_t r = 0; r < A.num_rows(); ++r)
        for (std::size_t q = st[r]; q < st[r + 1]; ++q) {
            y[r] += (long double)v[q] * x[ix[q]];
            ay[r] += std::fabs((long double)v[q]) * std::fabs(x[ix[q]]);
        }
}

inline std::size_t max_row_nnz(const csr& A) {
    std::size_t m = 0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) m = std::max(m, A.ref_major()[r + 1] - A.ref_major()[r]);
    return m;
}

inline long double inf_norm(const csr& A) {
    const auto& st = A.ref_major();
    long double best = 0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        long double s = 0;
        for (std::size_t q = st[r]; q < st[r + 1]; ++q) s += std::fabs((long double)A.ref_data()[q]);
        best = std::max(best, s);
    }
    return best;
}

/// Normwise backward error |b - A x|_inf / (|A|_inf |x|_inf + |b|_inf).
/// A nonfinite x fails outright: the maxima below use std::max, which drops a
/// NaN, so an all-NaN solution would otherwise score 0 and pass (#810).
template <typename V>
check_result backward_error(const csr& A, const V& x, const V& b) {
    const std::size_t n = A.num_rows();
    std::vector<long double> xl(n), y, ay;
    long double xm = 0, bm = 0, rm = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x(i))) {
            error_tracker e(tol<double>(n), "normwise backward error of the solve (x not finite)");
            e.add(std::numeric_limits<long double>::quiet_NaN(), 0.0L, 1.0L);
            return e.result();
        }
        xl[i] = x(i);
        xm = std::max(xm, std::fabs(xl[i]));
    }
    ref_mult(A, xl, y, ay);
    for (std::size_t i = 0; i < n; ++i) {
        bm = std::max(bm, std::fabs((long double)b(i)));
        rm = std::max(rm, std::fabs((long double)b(i) - y[i]));
    }
    error_tracker e(tol<double>(n), "normwise backward error of the solve");
    e.add(rm, 0.0L, inf_norm(A) * xm + bm);
    return e.result();
}

/// Cholesky-equivalent work of the AMD-ordered pattern: sum of squared column
/// counts of L.
inline double cholesky_work(const csr& A) {
    const auto sym = fac::sparse_cholesky_symbolic(A, ord::amd{});
    double w = 0;
    for (std::size_t c : sym.col_counts) w += double(c) * double(c);
    return w;
}

/// The natural order, as an ordering functor: the baseline the orderings beat.
struct identity_order {
    template <typename M>
    std::vector<std::size_t> operator()(const M& A) const {
        std::vector<std::size_t> p(A.num_cols());
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = i;
        return p;
    }
};

/// A given permutation, as an ordering functor: lets a check measure the fill of
/// the permutation an entry PRODUCED, rather than of a fresh call to the same
/// ordering (which would test the ordering against itself).
struct fixed_order {
    std::vector<std::size_t> perm;
    template <typename M>
    std::vector<std::size_t> operator()(const M&) const { return perm; }
};

inline double sparse_bytes(const csr& A, std::size_t vectors) {
    return double(A.nnz()) * (sizeof(double) + sizeof(std::size_t)) +
           double(A.num_rows() + 1) * sizeof(std::size_t) +
           double(vectors) * double(A.num_rows()) * sizeof(double);
}

// -- kernels ---------------------------------------------------------------------

inline problem spmv(size_class c) {
    const std::size_t k = grid_kernel(c);
    struct S { csr A; vec::dense_vector<double> x, y; };
    auto A = poisson2d(k);
    const std::size_t n = A.num_rows();
    auto s = std::make_shared<S>(S{std::move(A), vec_of<double>(n, 201), vec::dense_vector<double>(n)});
    problem p{grid_shape(k, s->A), 2.0 * s->A.nnz(), sparse_bytes(s->A, 2), [] {},
              [s] { mtl::mult(s->A, s->x, s->y); }, {}};
    p.check = [s, n] {
        std::vector<long double> x(n), y, ay;
        for (std::size_t i = 0; i < n; ++i) x[i] = s->x(i);
        ref_mult(s->A, x, y, ay);
        error_tracker e(tol<double>(max_row_nnz(s->A)), "per row vs sum |a||x|");
        for (std::size_t i = 0; i < n; ++i) e.add(s->y(i), y[i], ay[i]);
        return e.result();
    };
    return p;
}

/// y = A^T x on an UNSYMMETRIC matrix, so a kernel that ignored the transpose
/// would fail.
inline problem spmv_trans(size_class c) {
    const std::size_t k = grid_kernel(c);
    struct S { csr A; vec::dense_vector<double> x, y; };
    auto A = convdiff2d(k);
    const std::size_t n = A.num_rows();
    auto s = std::make_shared<S>(S{std::move(A), vec_of<double>(n, 202), vec::dense_vector<double>(n)});
    problem p{grid_shape(k, s->A), 2.0 * s->A.nnz(), sparse_bytes(s->A, 2), [] {},
              [s] { mtl::mult(mtl::trans(s->A), s->x, s->y); }, {}};
    p.check = [s, n] {
        std::vector<long double> y(n, 0), ay(n, 0);
        const auto& st = s->A.ref_major();
        const auto& ix = s->A.ref_minor();
        const auto& v = s->A.ref_data();
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t q = st[r]; q < st[r + 1]; ++q) {
                y[ix[q]] += (long double)v[q] * s->x(r);
                ay[ix[q]] += std::fabs((long double)v[q] * s->x(r));
            }
        error_tracker e(tol<double>(max_row_nnz(s->A)), "per entry vs sum |a||x|, A unsymmetric");
        for (std::size_t i = 0; i < n; ++i) e.add(s->y(i), y[i], ay[i]);
        return e.result();
    };
    return p;
}

/// CRS times a dense n x 16 block, the multi-vector form of SpMV.
inline problem crs_dense(size_class c) {
    const std::size_t k = grid_kernel(c), w = 16;
    struct S { csr A; mat::dense2D<double> X, Y; };
    auto A = poisson2d(k);
    const std::size_t n = A.num_rows();
    auto s = std::make_shared<S>(S{std::move(A), mat_of<double>(n, w, 203), mat::dense2D<double>(n, w)});
    problem p{grid_shape(k, s->A) + " x " + std::to_string(w), 2.0 * s->A.nnz() * w,
              double(s->A.nnz()) * (sizeof(double) + sizeof(std::size_t)) + 2.0 * n * w * sizeof(double),
              [] {}, [s] { mtl::mult(s->A, s->X, s->Y); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<double>(max_row_nnz(s->A)), "per entry vs sum |a||x| (64 sampled rows)");
        const auto& st = s->A.ref_major();
        const auto& ix = s->A.ref_minor();
        const auto& v = s->A.ref_data();
        for_samples(n, [&](std::size_t r, std::size_t) {
            for (std::size_t j = 0; j < w; ++j) {
                long double y = 0, ay = 0;
                for (std::size_t q = st[r]; q < st[r + 1]; ++q) {
                    y += (long double)v[q] * s->X(ix[q], j);
                    ay += std::fabs((long double)v[q] * s->X(ix[q], j));
                }
                e.add(s->Y(r, j), y, ay);
            }
        });
        return e.result();
    };
    return p;
}

/// C = A A. Work: 2 * sum over k of (nnz in row k)(nnz in column k), the
/// multiplications a row-by-row SpGEMM performs. Checked through C x = A (A x).
inline problem spgemm(size_class c) {
    const std::size_t k = grid_kernel(c) / 2;
    struct S { csr A, C; };
    auto A = poisson2d(k);
    double flops = 0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        const double rc = double(A.ref_major()[r + 1] - A.ref_major()[r]);
        flops += 2.0 * rc * rc;   // symmetric: column k has the same count as row k
    }
    auto s = std::make_shared<S>(S{std::move(A), csr(1, 1)});
    problem p{grid_shape(k, s->A), flops, sparse_bytes(s->A, 0) * 2, [] {},
              [s] { s->C = mtl::spgemm(s->A, s->A); }, {}};
    p.check = [s] {
        const std::size_t n = s->A.num_rows();
        std::vector<long double> x(n), ax, aax, y1, y2, cx, acx;
        auto xv = vec_of<double>(n, 204);
        for (std::size_t i = 0; i < n; ++i) x[i] = xv(i);
        ref_mult(s->A, x, ax, y1);                // A x and |A||x|
        ref_mult(s->A, ax, aax, y2);              // A (A x)
        ref_mult(s->A, y1, y2, cx);               // cx = |A| (|A||x|): the magnitude bound
        ref_mult(s->C, x, acx, y1);               // C x
        // The scale is cx, the MAGNITUDE |A||A||x| -- not y2, which is the signed
        // product A (|A||x|) and can cancel to near zero. (Measuring against y2
        // passed on x86 Linux, where long double's extra bits kept the difference
        // tiny, and failed on MSVC and Apple ARM64, where long double is double.)
        error_tracker e(tol<double>(2 * max_row_nnz(s->A) * max_row_nnz(s->A)), "C x vs A (A x)");
        for (std::size_t i = 0; i < n; ++i) e.add(acx[i], aax[i], cx[i]);
        return e.result();
    };
    return p;
}

/// The level-scheduled lower solve on a Cholesky factor (rows of a level run in
/// parallel on the pool). Checked by its componentwise backward error, which a
/// triangular solve keeps within gamma_n whatever the summation order or FMA use.
inline problem lower_solve_level(size_class c) {
    const std::size_t k = grid_direct(c) * 2;
    struct S {
        mtl::sparse::util::csc_matrix<double> L;
        fac::lower_solve_schedule sched;
        std::vector<double> b, x;
    };
    const auto A = poisson2d(k);
    const auto sym = fac::sparse_cholesky_symbolic(A, ord::amd{});
    auto num = fac::sparse_cholesky_numeric(A, sym);
    auto s = std::make_shared<S>();
    s->L = num.factor();
    s->sched = fac::build_lower_solve_schedule(s->L);
    const std::size_t n = A.num_rows();
    auto bv = vec_of<double>(n, 205);
    s->b.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) s->b[i] = bv(i);
    const double nnz = double(s->L.nnz());
    problem p{std::to_string(k) + "^2 grid, nnz(L) " + std::to_string(s->L.nnz()), 2.0 * nnz,
              nnz * (sizeof(double) + sizeof(std::size_t)) + 2.0 * n * sizeof(double),
              [s] { s->x = s->b; },
              [s] { fac::level_scheduled_lower_solve(s->L, s->sched, s->x); }, {}};
    p.check = [s] {
        const std::size_t n = s->b.size();
        std::vector<long double> lx(n, 0), alx(n, 0);
        for (std::size_t j = 0; j < n; ++j) {   // CSC: column j scatters into rows i >= j
            for (std::size_t q = s->L.col_ptr[j]; q < s->L.col_ptr[j + 1]; ++q) {
                const long double t = (long double)s->L.values[q] * s->x[j];
                lx[s->L.row_ind[q]] += t;
                alx[s->L.row_ind[q]] += std::fabs(t);
            }
        }
        error_tracker e(tol<double>(n), "per row |L x - b| vs |L||x| + |b|");
        for (std::size_t i = 0; i < n; ++i) e.add(lx[i], s->b[i], alx[i] + std::fabs((long double)s->b[i]));
        return e.result();
    };
    return p;
}

// -- direct solvers --------------------------------------------------------------

/// One-shot solve: symbolic analysis, numeric factorization and the solve, the
/// cost of a solve on a matrix seen for the first time.
template <int Which>
problem direct(size_class c) {
    // sparse_qr gets HALF the grid side (a quarter of the rows): at the common
    // medium grid (96^2) one solve took 4.1 s on an i7-12700K, against 9-73 ms
    // for every other solver on the same matrix, and the large grid would take
    // minutes. The gap is a finding, not a property of QR; its own grid keeps
    // the catalogue's run time bounded meanwhile.
    const std::size_t k = Which == 6 ? grid_direct(c) / 2 : grid_direct(c);
    const bool spd = Which <= 2;   // cholesky, ldlt, supernodal ldlt
    struct S { csr A; vec::dense_vector<double> b, x; };
    auto A = spd ? poisson2d(k) : convdiff2d(k);
    const std::size_t n = A.num_rows();
    const double work = cholesky_work(A) * (spd ? 1.0 : 2.0);
    auto s = std::make_shared<S>(S{std::move(A), vec_of<double>(n, 206 + Which), vec::dense_vector<double>(n)});
    problem p{grid_shape(k, s->A), work, double(s->A.nnz()) * (sizeof(double) + sizeof(std::size_t)), [] {},
              [s] {
                  if constexpr (Which == 0) fac::sparse_cholesky_solve(s->A, s->x, s->b, ord::amd{});
                  else if constexpr (Which == 1) fac::sparse_ldlt_solve(s->A, s->x, s->b, ord::amd{});
                  else if constexpr (Which == 2) fac::supernodal_ldlt_solve(s->A, s->x, s->b, ord::amd{});
                  else if constexpr (Which == 3) fac::sparse_lu_solve(s->A, s->x, s->b, ord::colamd{});
                  else if constexpr (Which == 4) fac::supernodal_lu_solve(s->A, s->x, s->b, ord::colamd{});
                  else if constexpr (Which == 5) fac::native_klu_solve(s->A, s->x, s->b);
                  else fac::sparse_qr_solve(s->A, s->x, s->b);
              }, {}};
    p.check = [s] { return backward_error(s->A, s->x, s->b); };
    return p;
}

// -- orderings ---------------------------------------------------------------------

inline bool valid(const std::vector<std::size_t>& p, std::size_t n) {
    return p.size() == n && mtl::sparse::util::is_valid_permutation(p);
}

/// AMD on a scrambled Laplacian: valid, and at most half the Cholesky fill of
/// the scrambled natural order (in practice it is a small fraction of it).
inline problem amd(size_class c) {
    const std::size_t k = grid_direct(c);
    struct S { csr A; std::vector<std::size_t> perm; };
    auto P = poisson2d(k);
    const auto r = random_perm(P.num_rows(), 207);
    auto s = std::make_shared<S>(S{permuted(P, r, r), {}});
    problem p{grid_shape(k, s->A) + ", scrambled", 0.0, 0.0, [] {},
              [s] { s->perm = ord::amd{}(s->A); }, {}};
    p.check = [s] {
        check_result res;
        res.what = "valid permutation; fill(amd) / fill(scrambled natural) <= 0.5";
        res.tolerance = 0.5;
        if (!valid(s->perm, s->A.num_rows())) { res.error = 1e300; return res; }
        const double f_amd = double(fac::sparse_cholesky_symbolic(s->A, fixed_order{s->perm}).nnz_L);
        const double f_nat = double(fac::sparse_cholesky_symbolic(s->A, identity_order{}).nnz_L);
        res.error = f_amd / f_nat;
        res.ok = res.error <= res.tolerance;
        return res;
    };
    return p;
}

/// COLAMD orders COLUMNS to limit the fill of A^T A (LU and QR). On a
/// column-scrambled convection-diffusion matrix: valid, and at most half the
/// Cholesky fill of A^T A in the scrambled natural column order.
inline problem colamd(size_class c) {
    const std::size_t k = grid_direct(c);
    struct S { csr A; std::vector<std::size_t> perm; };
    auto D = convdiff2d(k);
    const std::size_t n = D.num_rows();
    std::vector<std::size_t> id(n);
    for (std::size_t i = 0; i < n; ++i) id[i] = i;
    auto s = std::make_shared<S>(S{permuted(D, id, random_perm(n, 208)), {}});
    problem p{grid_shape(k, s->A) + ", columns scrambled", 0.0, 0.0, [] {},
              [s] { s->perm = ord::colamd{}(s->A); }, {}};
    p.check = [s, n] {
        check_result res;
        res.what = "valid permutation; fill of (AQ)^T(AQ): colamd / scrambled natural <= 0.5";
        res.tolerance = 0.5;
        if (!valid(s->perm, n)) { res.error = 1e300; return res; }
        // A^T A, as a CRS matrix: the transpose by scatter, then a product.
        const auto& st = s->A.ref_major();
        const auto& ix = s->A.ref_minor();
        csr At(n, n);
        {
            mat::inserter<csr> ins(At);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t q = st[r]; q < st[r + 1]; ++q) ins[ix[q]][r] << 1.0;
        }
        csr Ap(n, n);
        {
            mat::inserter<csr> ins(Ap);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t q = st[r]; q < st[r + 1]; ++q) ins[r][ix[q]] << 1.0;
        }
        const csr AtA = mtl::spgemm(At, Ap);
        const double f_col =
            double(fac::sparse_cholesky_symbolic(permuted(AtA, s->perm, s->perm), identity_order{}).nnz_L);
        const double f_nat = double(fac::sparse_cholesky_symbolic(AtA, identity_order{}).nnz_L);
        res.error = f_col / f_nat;
        res.ok = res.error <= res.tolerance;
        return res;
    };
    return p;
}

inline std::size_t bandwidth(const csr& A) {
    std::size_t b = 0;
    for (std::size_t r = 0; r < A.num_rows(); ++r)
        for (std::size_t q = A.ref_major()[r]; q < A.ref_major()[r + 1]; ++q) {
            const std::size_t cidx = A.ref_minor()[q];
            b = std::max(b, cidx > r ? cidx - r : r - cidx);
        }
    return b;
}

/// RCM on a scrambled Laplacian: valid, and a bandwidth within 4x of the grid
/// side k (the natural order's bandwidth). The scrambled order's is ~n.
inline problem rcm(size_class c) {
    const std::size_t k = grid_direct(c);
    struct S { csr A; std::vector<std::size_t> perm; };
    auto P = poisson2d(k);
    const auto r = random_perm(P.num_rows(), 209);
    auto s = std::make_shared<S>(S{permuted(P, r, r), {}});
    problem p{grid_shape(k, s->A) + ", scrambled", 0.0, 0.0, [] {},
              [s] { s->perm = ord::rcm{}(s->A); }, {}};
    p.check = [s, k] {
        check_result res;
        res.what = "valid permutation; bandwidth(P A P^T) / grid side <= 4";
        res.tolerance = 4.0;
        if (!valid(s->perm, s->A.num_rows())) { res.error = 1e300; return res; }
        res.error = double(bandwidth(permuted(s->A, s->perm, s->perm))) / double(k);
        res.ok = res.error <= res.tolerance;
        return res;
    };
    return p;
}

/// Block-triangular form (Dulmage-Mendelsohn) on a planted, scrambled matrix.
inline problem block_triangular_form(size_class c) {
    const std::size_t blocks = c == size_class::small ? 8 : c == size_class::medium ? 32 : 64;
    const std::size_t bk = c == size_class::small ? 8 : 16;
    struct S { csr A; ord::btf_result btf; };
    auto s = std::make_shared<S>(S{planted_btf(blocks, bk, 210), {}});
    problem p{std::to_string(blocks) + " planted blocks, n " + std::to_string(s->A.num_rows()), 0.0, 0.0,
              [] {}, [s] { s->btf = ord::block_triangular_form(s->A); }, {}};
    p.check = [s, blocks] {
        check_result res;
        res.what = "valid perms, zero-free diagonal, upper block triangular, >= planted blocks";
        res.tolerance = 0.0;
        const std::size_t n = s->A.num_rows();
        const auto& b = s->btf;
        std::size_t faults = 0;
        if (b.structurally_singular || !valid(b.row_perm, n) || !valid(b.col_perm, n)) faults += 1000;
        if (b.nblocks() < blocks) faults += 100;
        if (faults == 0) {
            const csr B = permuted(s->A, b.row_perm, b.col_perm);
            std::vector<std::size_t> blk(n);
            for (std::size_t q = 0; q < b.nblocks(); ++q)
                for (std::size_t i = b.blocks[q]; i < b.blocks[q + 1]; ++i) blk[i] = q;
            for (std::size_t r = 0; r < n; ++r) {
                bool diag = false;
                for (std::size_t q = B.ref_major()[r]; q < B.ref_major()[r + 1]; ++q) {
                    const std::size_t cidx = B.ref_minor()[q];
                    diag |= cidx == r;
                    faults += blk[cidx] < blk[r];   // an entry below its block
                }
                faults += !diag;
            }
        }
        res.error = double(faults);
        res.ok = faults == 0;
        return res;
    };
    return p;
}

inline void register_all(registry& r) {
    r.add("sparse", "spmv", "f64", spmv);
    r.add("sparse", "spmv_trans", "f64", spmv_trans);
    r.add("sparse", "crs_dense", "f64", crs_dense);
    r.add("sparse", "spgemm", "f64", spgemm);
    r.add("sparse", "lower_solve_level", "f64", lower_solve_level);

    r.add("direct", "sparse_cholesky", "f64", direct<0>);
    r.add("direct", "sparse_ldlt", "f64", direct<1>);
    r.add("direct", "supernodal_ldlt", "f64", direct<2>);
    r.add("direct", "sparse_lu", "f64", direct<3>);
    r.add("direct", "supernodal_lu", "f64", direct<4>);
    r.add("direct", "native_klu", "f64", direct<5>);
    r.add("direct", "sparse_qr", "f64", direct<6>);

    r.add("ordering", "amd", "f64", amd);
    r.add("ordering", "colamd", "f64", colamd);
    r.add("ordering", "rcm", "f64", rcm);
    r.add("ordering", "block_triangular_form", "f64", block_triangular_form);
}

} // namespace mtl::bench::platform::sparse_entries
