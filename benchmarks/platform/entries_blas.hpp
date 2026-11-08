#pragma once
// mtl5_bench -- BLAS levels 1, 2 and 3 (#721 milestone M1).
//
// Every entry calls the public mtl:: operation and checks the result against a
// reference computed in long double:
//
//   componentwise error   |got - ref| / (sum of |terms|), the quantity the
//                         standard error bound gamma_k = k*eps/(1 - k*eps)
//                         constrains, so the tolerance is a theorem, not a
//                         guess: tol = 4*k*eps for a reduction of length k.
//                         It holds for every summation order -- blocked,
//                         SIMD, threaded or BLAS -- so a check cannot fail
//                         because a faster path reordered the sum.
//   residual (solves)     |T x - b| / (|T| |x| + |b|), same bound.
//
// Level-3 results are checked on 64 sampled entries of C rather than all n^2:
// the full long-double reference is O(n^3) and would take longer than the
// benchmark. A wrong kernel is wrong on a block of entries, never on one, so
// 64 spread samples catch it.
//
// Work model: flops are the conventional useful counts (2n for dot, 2n^2 for
// gemv, 2n^3 for gemm, n^3 for syrk/trsm/trmm), and bytes are the compulsory
// traffic of each operand once, so a memory-bound op is measured against the
// triad bandwidth and a compute-bound one against the FMA peak.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

// The operation headers this file calls, not the <mtl/mtl.hpp> umbrella: with
// MTL5_HAS_BLAS and MTL5_HAS_SUPERLU both defined, the umbrella pulls in
// SuperLU's headers, whose dgemm_/dtrsm_ prototypes conflict with MTL5's own
// BLAS declarations, and the translation unit does not compile.
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/gemmt.hpp>
#include <mtl/operation/ger.hpp>
#include <mtl/operation/kron.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/sum.hpp>
#include <mtl/operation/sum_dot_k.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/syr2k.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/operation/trsv.hpp>

#include <benchmarks/platform/common.hpp>

namespace mtl::bench::platform::blas {


// -- level 1 -----------------------------------------------------------------

template <typename T>
problem dot(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<T> x, y; T r{}; };
    auto s = std::make_shared<S>(S{nonneg_vec_of<T>(n, 1), nonneg_vec_of<T>(n, 2), T(0)});
    problem p{vshape(n), 2.0 * n, 2.0 * n * sizeof(T), [] {}, [s] { s->r = mtl::dot(s->x, s->y); }, {}};
    p.check = [s, n] {
        long double ref = 0, abs = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const long double t = (long double)s->x(i) * s->y(i);
            ref += t; abs += std::fabs(t);
        }
        error_tracker e(tol<T>(n), "rel err vs sum|x_i y_i|");
        e.add(s->r, ref, abs);
        return e.result();
    };
    return p;
}

template <typename T>
problem axpy(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<T> x, y0, y; };
    auto s = std::make_shared<S>(S{vec_of<T>(n, 1), vec_of<T>(n, 2), vec::dense_vector<T>(n)});
    const T a = T(0.75);
    problem p{vshape(n), 2.0 * n, 3.0 * n * sizeof(T),
              [s] { s->y = s->y0; }, [s, a] { mtl::axpy(a, s->x, s->y); }, {}};
    p.check = [s, n, a] {
        error_tracker e(tol<T>(1), "componentwise rel err");
        for (std::size_t i = 0; i < n; ++i) {
            const long double t = (long double)a * s->x(i);
            e.add(s->y(i), t + s->y0(i), std::fabs(t) + std::fabs((long double)s->y0(i)));
        }
        return e.result();
    };
    return p;
}

template <typename T>
problem scale(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<T> x0, x; };
    auto s = std::make_shared<S>(S{vec_of<T>(n, 1), vec::dense_vector<T>(n)});
    const T a = T(0.75);
    problem p{vshape(n), 1.0 * n, 2.0 * n * sizeof(T),
              [s] { s->x = s->x0; }, [s, a] { mtl::scale(a, s->x); }, {}};
    p.check = [s, n, a] {
        error_tracker e(tol<T>(1), "componentwise rel err");
        for (std::size_t i = 0; i < n; ++i) {
            const long double t = (long double)a * s->x0(i);
            e.add(s->x(i), t, std::fabs(t));
        }
        return e.result();
    };
    return p;
}

/// two_norm, one_norm, infinity_norm, sum: one read of x, one reduction.
template <typename T, int Which>
problem reduction(size_class c) {
    const std::size_t n = n_vec(c);
    struct S { vec::dense_vector<T> x; long double r = 0; };
    // sum is the one reduction here whose reference cancels; give it the same
    // nonnegative inputs as dot so a zero result cannot pass.
    auto s = std::make_shared<S>(S{Which == 3 ? nonneg_vec_of<T>(n, 3) : vec_of<T>(n, 3)});
    const double flops = Which == 0 ? 2.0 * n : 1.0 * n;
    problem p{vshape(n), flops, 1.0 * n * sizeof(T), [] {},
              [s] {
                  if constexpr (Which == 0) s->r = mtl::two_norm(s->x);
                  else if constexpr (Which == 1) s->r = mtl::one_norm(s->x);
                  else if constexpr (Which == 2) s->r = mtl::infinity_norm(s->x);
                  else s->r = mtl::sum(s->x);
              }, {}};
    p.check = [s, n] {
        long double ref = 0, abs = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const long double v = s->x(i);
            if constexpr (Which == 0) { ref += v * v; }
            else if constexpr (Which == 1) { ref += std::fabs(v); }
            else if constexpr (Which == 2) { ref = std::max(ref, std::fabs(v)); }
            else { ref += v; abs += std::fabs(v); }
        }
        if constexpr (Which == 0) ref = std::sqrt(ref);
        if constexpr (Which != 3) abs = ref;
        error_tracker e(tol<T>(n), "rel err vs reference");
        e.add(s->r, ref, abs);
        return e.result();
    };
    return p;
}

/// DotK with K = 3 (sum_dot_k.hpp): the dot product as if in triple precision,
/// on a dot product built to be ill conditioned, so that the accuracy it is for
/// is what the check sees.
///
/// INPUTS. Integers, so the exact answer is an int64 sum and needs no extended
/// reference. In the first half x_i, y_i are in [2^(b-1), 2^b); the second
/// half repeats x and sets y_(h+i) = -y_i + delta_i, delta_i in {1, 2}. The halves
/// cancel to s = sum x_i delta_i ~ n 2^b, while S = sum |x_i y_i| ~ n 2^(2b): a
/// condition number S/|s| ~ 2^b. b is chosen so S < 2^62 (exact in int64) and
/// every product is exact in double; the running sum of the first half reaches
/// ~2^61, far past 2^53, so a plain dot rounds and is wrong by many ulps of s.
///
/// CHECK. Ogita, Rump and Oishi, Proposition 5.11: |res - s| <= (u + 2 g^2)|s|
/// + g^3 S, with g = gamma_(4n-2). The reported error is |res - s| over that
/// bound, so it must be <= 1. A plain dot misses it by a factor of ~10^7
/// (error ~ n u S). Dot2 meets it on this input, and that is a limit of the
/// test, not of the bound: K = 2 and K = 3 differ only once the condition number
/// passes ~1/u^2 = 2^106, and an int64-exact reference stops at ~2^62.
///
/// WORK. (12K - 14) n = 22 n flops (the header's cost model), which is what the
/// rate is in; 2 n of them are the dot product the user asked for.
inline problem dot_k3(size_class c) {
    const std::size_t n = n_vec(c) & ~std::size_t(1);
    int lg = 0;
    while ((std::size_t(1) << lg) < n) ++lg;
    const int b = (62 - lg) / 2;            // n 2^(2b) <= 2^62
    struct S { vec::dense_vector<double> x, y; double r = 0; };
    auto s = std::make_shared<S>(S{vec::dense_vector<double>(n), vec::dense_vector<double>(n)});
    xorshift64 g(0xD07C3);
    const double lo = std::ldexp(1.0, b - 1);
    const std::size_t h = n / 2;
    for (std::size_t i = 0; i < h; ++i) {
        const double xi = lo + std::floor(g.next_double() * lo);
        const double yi = lo + std::floor(g.next_double() * lo);
        s->x(i) = xi;      s->y(i) = yi;
        s->x(h + i) = xi;  s->y(h + i) = -yi + (g.next_double() < 0.5 ? 1.0 : 2.0);
    }
    problem p{vshape(n), 22.0 * n, 2.0 * n * sizeof(double), [] {},
              [s] { s->r = mtl::dot_k<3>(s->x, s->y); }, {}};
    p.check = [s, n] {
        std::int64_t exact = 0, mag = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto t = static_cast<std::int64_t>(s->x(i)) * static_cast<std::int64_t>(s->y(i));
            exact += t;
            mag += t < 0 ? -t : t;
        }
        const long double u = eps<double>() / 2;
        const long double ku = (4.0L * n - 2) * u;
        const long double gm = ku / (1 - ku);
        const long double bound = (u + 2 * gm * gm) * std::fabs((long double)exact)
                                + gm * gm * gm * (long double)mag;
        check_result res;
        res.what = "|res - s| / Ogita-Rump-Oishi DotK bound (exact int64 s, cond ~2^b)";
        res.tolerance = 1.0;
        const long double err = std::fabs((long double)s->r - (long double)exact);
        res.error = std::isfinite(s->r) ? static_cast<double>(err / bound)
                                        : std::numeric_limits<double>::infinity();
        res.ok = res.error <= res.tolerance;
        return res;
    };
    return p;
}

// -- level 2 -----------------------------------------------------------------

template <typename T>
problem gemv(size_class c) {
    const std::size_t n = n_mv(c);
    struct S { mat::dense2D<T> A; vec::dense_vector<T> x, y; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 4), vec_of<T>(n, 5), vec::dense_vector<T>(n)});
    problem p{vshape(n), 2.0 * n * n, (1.0 * n * n + 2.0 * n) * sizeof(T), [] {},
              [s] { mtl::mult(s->A, s->x, s->y); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "componentwise rel err vs |A||x|");
        for (std::size_t i = 0; i < n; ++i) {
            long double ref = 0, abs = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const long double t = (long double)s->A(i, j) * s->x(j);
                ref += t; abs += std::fabs(t);
            }
            e.add(s->y(i), ref, abs);
        }
        return e.result();
    };
    return p;
}

template <typename T>
problem symv(size_class c) {
    const std::size_t n = n_mv(c);
    struct S { mat::dense2D<T> A; vec::dense_vector<T> x, y; };
    auto s = std::make_shared<S>(S{symmetric<T>(n, 6), vec_of<T>(n, 7), vec::dense_vector<T>(n)});
    problem p{vshape(n), 2.0 * n * n, (1.0 * n * n + 2.0 * n) * sizeof(T), [] {},
              [s] { mtl::symv(T(1), s->A, s->x, T(0), s->y); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "componentwise rel err vs |A||x|");
        for (std::size_t i = 0; i < n; ++i) {
            long double ref = 0, abs = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const long double t = (long double)s->A(i, j) * s->x(j);
                ref += t; abs += std::fabs(t);
            }
            e.add(s->y(i), ref, abs);
        }
        return e.result();
    };
    return p;
}

template <typename T>
problem trmv(size_class c) {
    const std::size_t n = n_mv(c);
    struct S { mat::dense2D<T> L; vec::dense_vector<T> x0, x; };
    auto s = std::make_shared<S>(S{lower_tri<T>(n, 8), vec_of<T>(n, 9), vec::dense_vector<T>(n)});
    problem p{vshape(n), 1.0 * n * n, (0.5 * n * n + 2.0 * n) * sizeof(T),
              [s] { s->x = s->x0; }, [s] { mtl::trmv(s->L, s->x, /*upper=*/false); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "componentwise rel err vs |L||x|");
        for (std::size_t i = 0; i < n; ++i) {
            long double ref = 0, abs = 0;
            for (std::size_t j = 0; j <= i; ++j) {
                const long double t = (long double)s->L(i, j) * s->x0(j);
                ref += t; abs += std::fabs(t);
            }
            e.add(s->x(i), ref, abs);
        }
        return e.result();
    };
    return p;
}

template <typename T>
problem trsv(size_class c) {
    const std::size_t n = n_mv(c);
    struct S { mat::dense2D<T> L; vec::dense_vector<T> b, x; };
    auto s = std::make_shared<S>(S{lower_tri<T>(n, 10), vec_of<T>(n, 11), vec::dense_vector<T>(n)});
    problem p{vshape(n), 1.0 * n * n, (0.5 * n * n + 2.0 * n) * sizeof(T), [] {},
              [s] { mtl::trsv(s->L, s->x, s->b, /*upper=*/false); }, {}};
    p.check = [s, n] {
        error_tracker e(tol<T>(n), "residual |Lx-b| / (|L||x|+|b|)");
        for (std::size_t i = 0; i < n; ++i) {
            long double lx = 0, abs = 0;
            for (std::size_t j = 0; j <= i; ++j) {
                const long double t = (long double)s->L(i, j) * s->x(j);
                lx += t; abs += std::fabs(t);
            }
            e.add(lx, s->b(i), abs + std::fabs((long double)s->b(i)));
        }
        return e.result();
    };
    return p;
}

template <typename T>
problem ger(size_class c) {
    const std::size_t n = n_mv(c);
    struct S { mat::dense2D<T> A0, A; vec::dense_vector<T> x, y; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 12), mat::dense2D<T>(n, n), vec_of<T>(n, 13), vec_of<T>(n, 14)});
    const T a = T(0.5);
    problem p{vshape(n), 2.0 * n * n, (2.0 * n * n + 2.0 * n) * sizeof(T),
              [s] { s->A = s->A0; }, [s, a] { mtl::ger(a, s->x, s->y, s->A); }, {}};
    p.check = [s, n, a] {
        error_tracker e(tol<T>(2), "componentwise rel err");
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const long double t = (long double)a * s->x(i) * s->y(j);
                e.add(s->A(i, j), s->A0(i, j) + t, std::fabs((long double)s->A0(i, j)) + std::fabs(t));
            }
        return e.result();
    };
    return p;
}

// -- level 3 -----------------------------------------------------------------

inline std::string mshape(std::size_t n) {
    return std::to_string(n) + "x" + std::to_string(n) + "x" + std::to_string(n);
}

/// Sampled check of C(i, j) against sum_k f(i, k, j), for any product form.
template <typename T, typename M, typename Term>
check_result sampled(const M& C, std::size_t n, std::size_t k, Term term, const char* what,
                     bool lower_only = false) {
    error_tracker e(tol<T>(k), what);
    for_samples(n, [&](std::size_t i, std::size_t j) {
        if (lower_only && j > i) std::swap(i, j);
        long double ref = 0, abs = 0;
        for (std::size_t p = 0; p < k; ++p) {
            const long double t = term(i, p, j);
            ref += t; abs += std::fabs(t);
        }
        e.add(C(i, j), ref, abs);
    });
    return e.result();
}

template <typename T>
problem gemm(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> A, B, C; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 15), mat_of<T>(n, n, 16), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 2.0 * n * n * n, 3.0 * n * n * sizeof(T), [] {},
              [s] { mtl::mult(s->A, s->B, s->C); }, {}};
    p.check = [s, n] {
        return sampled<T>(s->C, n, n, [&](std::size_t i, std::size_t k, std::size_t j) {
            return (long double)s->A(i, k) * s->B(k, j); }, "rel err vs sum|a_ik b_kj| (64 samples)");
    };
    return p;
}

template <typename T>
problem gemmt(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> A, B, C; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 17), mat_of<T>(n, n, 18), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 1.0 * n * n * n, 3.0 * n * n * sizeof(T), [] {},
              [s] { mtl::gemmt(T(1), s->A, s->B, T(0), s->C); }, {}};
    p.check = [s, n] {
        return sampled<T>(s->C, n, n, [&](std::size_t i, std::size_t k, std::size_t j) {
            return (long double)s->A(i, k) * s->B(k, j); },
            "rel err, lower triangle (64 samples)", /*lower_only=*/true);
    };
    return p;
}

template <typename T>
problem symm(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> A, B, C; };
    auto s = std::make_shared<S>(S{symmetric<T>(n, 19), mat_of<T>(n, n, 20), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 2.0 * n * n * n, 3.0 * n * n * sizeof(T), [] {},
              [s] { mtl::symm(T(1), s->A, s->B, T(0), s->C); }, {}};
    p.check = [s, n] {
        return sampled<T>(s->C, n, n, [&](std::size_t i, std::size_t k, std::size_t j) {
            return (long double)s->A(i, k) * s->B(k, j); }, "rel err (64 samples)");
    };
    return p;
}

template <typename T>
problem syrk(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> A, C; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 21), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 1.0 * n * n * n, 2.0 * n * n * sizeof(T), [] {},
              [s] { mtl::syrk(T(1), s->A, T(0), s->C); }, {}};
    p.check = [s, n] {
        return sampled<T>(s->C, n, n, [&](std::size_t i, std::size_t k, std::size_t j) {
            return (long double)s->A(i, k) * s->A(j, k); }, "rel err vs A A^T (64 samples)");
    };
    return p;
}

template <typename T>
problem syr2k(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> A, B, C; };
    auto s = std::make_shared<S>(S{mat_of<T>(n, n, 22), mat_of<T>(n, n, 23), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 2.0 * n * n * n, 3.0 * n * n * sizeof(T), [] {},
              [s] { mtl::syr2k(T(1), s->A, s->B, T(0), s->C); }, {}};
    p.check = [s, n] {
        // C = A B^T + B A^T: two terms per k, as one 2n-long sum.
        return sampled<T>(s->C, n, 2 * n, [&](std::size_t i, std::size_t q, std::size_t j) {
            const std::size_t k = q / 2;
            return q % 2 == 0 ? (long double)s->A(i, k) * s->B(j, k)
                              : (long double)s->B(i, k) * s->A(j, k); },
            "rel err vs A B^T + B A^T (64 samples)");
    };
    return p;
}

template <typename T>
problem trmm(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> L, B0, B; };
    auto s = std::make_shared<S>(S{lower_tri<T>(n, 24), mat_of<T>(n, n, 25), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 1.0 * n * n * n, (0.5 * n * n + 2.0 * n * n) * sizeof(T),
              [s] { s->B = s->B0; }, [s] { mtl::trmm(T(1), s->L, s->B, /*upper=*/false); }, {}};
    p.check = [s, n] {
        return sampled<T>(s->B, n, n, [&](std::size_t i, std::size_t k, std::size_t j) {
            return k <= i ? (long double)s->L(i, k) * s->B0(k, j) : 0.0L; }, "rel err vs L B (64 samples)");
    };
    return p;
}

template <typename T>
problem trsm(size_class c) {
    const std::size_t n = n_mm(c);
    struct S { mat::dense2D<T> L, B0, X; };
    auto s = std::make_shared<S>(S{lower_tri<T>(n, 26), mat_of<T>(n, n, 27), mat::dense2D<T>(n, n)});
    problem p{mshape(n), 1.0 * n * n * n, (0.5 * n * n + 2.0 * n * n) * sizeof(T),
              [s] { s->X = s->B0; }, [s] { mtl::trsm(T(1), s->L, s->X, /*upper=*/false); }, {}};
    p.check = [s, n] {
        // Residual on sampled entries: (L X)(i, j) against B0(i, j).
        error_tracker e(tol<T>(n), "residual |LX-B| / (|L||X|+|B|) (64 samples)");
        for_samples(n, [&](std::size_t i, std::size_t j) {
            long double lx = 0, abs = 0;
            for (std::size_t k = 0; k <= i; ++k) {
                const long double t = (long double)s->L(i, k) * s->X(k, j);
                lx += t; abs += std::fabs(t);
            }
            e.add(lx, s->B0(i, j), abs + std::fabs((long double)s->B0(i, j)));
        });
        return e.result();
    };
    return p;
}

template <typename T>
problem kron(size_class c) {
    const std::size_t m = c == size_class::small ? 24 : c == size_class::medium ? 48 : 64;
    const std::size_t N = m * m;
    struct S { mat::dense2D<T> A, B, K; };
    auto s = std::make_shared<S>(S{mat_of<T>(m, m, 28), mat_of<T>(m, m, 29), mat::dense2D<T>(1, 1)});
    // One multiply per output entry; the N x N output write dominates, so this
    // is a store-bandwidth test. The result matrix is allocated inside the call
    // (kron returns it), and that allocation is part of what is timed.
    problem p{std::to_string(m) + "x" + std::to_string(m) + " (x) " + std::to_string(m) + "x" + std::to_string(m),
              1.0 * N * N, 1.0 * N * N * sizeof(T), [] {},
              [s] { s->K = mtl::kron(s->A, s->B); }, {}};
    p.check = [s, m, N] {
        error_tracker e(tol<T>(1), "rel err (64 samples)");
        for_samples(N, [&](std::size_t r, std::size_t q) {
            const long double t = (long double)s->A(r / m, q / m) * s->B(r % m, q % m);
            e.add(s->K(r, q), t, std::fabs(t));
        });
        return e.result();
    };
    return p;
}

/// Register every BLAS entry. f64 throughout; f32 for the operations where
/// single precision is a different kernel or a different roofline (the
/// reductions and the products), so the card shows both peaks being reached.
inline void register_all(registry& r) {
    r.add("blas1", "dot",  "f64", dot<double>);
    r.add("blas1", "dot",  "f32", dot<float>);
    r.add("blas1", "axpy", "f64", axpy<double>);
    r.add("blas1", "axpy", "f32", axpy<float>);
    r.add("blas1", "scale", "f64", scale<double>);
    r.add("blas1", "two_norm", "f64", reduction<double, 0>);
    r.add("blas1", "one_norm", "f64", reduction<double, 1>);
    r.add("blas1", "infinity_norm", "f64", reduction<double, 2>);
    r.add("blas1", "sum", "f64", reduction<double, 3>);
    r.add("blas1", "dot_k3", "f64", dot_k3);

    r.add("blas2", "gemv", "f64", gemv<double>);
    r.add("blas2", "gemv", "f32", gemv<float>);
    r.add("blas2", "symv", "f64", symv<double>);
    r.add("blas2", "trmv", "f64", trmv<double>);
    r.add("blas2", "trsv", "f64", trsv<double>);
    r.add("blas2", "ger",  "f64", ger<double>);

    r.add("blas3", "gemm",  "f64", gemm<double>);
    r.add("blas3", "gemm",  "f32", gemm<float>);
    r.add("blas3", "gemmt", "f64", gemmt<double>);
    r.add("blas3", "symm",  "f64", symm<double>);
    r.add("blas3", "syrk",  "f64", syrk<double>);
    r.add("blas3", "syr2k", "f64", syr2k<double>);
    r.add("blas3", "trmm",  "f64", trmm<double>);
    r.add("blas3", "trsm",  "f64", trsm<double>);
    r.add("blas3", "trsm",  "f32", trsm<float>);
    r.add("blas3", "kron",  "f64", kron<double>);
}

} // namespace mtl::bench::platform::blas
