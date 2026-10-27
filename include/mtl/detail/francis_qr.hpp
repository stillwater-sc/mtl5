#pragma once
// MTL5 -- Francis implicit double-shift QR on an upper Hessenberg matrix, shared
// by eigenvalue() (eigenvalues only, EISPACK hqr) and schur() (the real Schur
// form T and its Schur vectors Z, EISPACK hqr2 with LAPACK's dlanv2 for the 2x2
// blocks). Generic in the element type: it uses + - * /, comparisons, and abs
// and sqrt found by ADL, so posits, cfloats, LNS and double-double run it as
// they are. Real element types only.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/detail/unit_roundoff.hpp>

namespace mtl::detail {

/// Constants for lanv2(): the relative precision and the power-of-two scale
/// window of LAPACK's dlanv2. The field names are dlanv2's, so the port can be
/// read side by side with the Fortran:
///
/// - `eps`: the machine epsilon of T (dlanv2's EPS, from dlamch('P')).
/// - `safmn2`: a "safe minimum". LAPACK's SAFMIN, dlamch('S'), is the safe
///   minimum, the smallest positive number whose reciprocal does not overflow.
///   SAFMN2 (the 2 sets it apart from SAFMIN) is derived from it: the power of
///   two nearest sqrt(SAFMIN / EPS) from below. Its square is still a normal
///   number, with a margin of 1/EPS, so entries that small can be multiplied
///   without underflow. dlanv2 rescales a block whose entries fall below it.
/// - `safmx2`: the matching "safe maximum", 1 / safmn2. dlanv2 rescales a
///   block whose entries rise above it.
/// - `have_range`: false for a type without numeric_limits. That type then
///   skips the rescaling loop, which only guards against overflow and
///   underflow in tau.
template <typename T>
struct lanv2_constants {
    T eps{};
    T safmn2{};
    T safmx2{};
    bool have_range = false;
};

template <typename T>
lanv2_constants<T> make_lanv2_constants() {
    lanv2_constants<T> c;
    c.eps = machine_epsilon<T>();
    if constexpr (std::numeric_limits<T>::is_specialized) {
        // safmn2 = the power of two nearest sqrt(safmin / eps) from below, as
        // dlanv2 computes it with logarithms; halving keeps it exact without log.
        const T target = std::numeric_limits<T>::min() / c.eps;
        if (target > T(0)) {
            T p = T(1);
            while (p * p > target) p = p * T(0.5);
            c.safmn2 = p;
            c.safmx2 = T(1) / p;
            c.have_range = c.safmx2 > c.safmn2;
        }
    }
    return c;
}

/// sqrt(x^2 + y^2) without avoidable overflow or underflow. This is LAPACK's
/// dlapy2: "la" marks a LAPACK auxiliary routine, "py" Pythagoras, and "2" the
/// two arguments (dlapy3 takes three). std::hypot computes the same thing but
/// only for the built-in floating-point types, so this generic version is
/// used for user-defined types too.
template <typename T>
T lapy2(T x, T y) {
    using std::abs;
    using std::sqrt;
    const T xa = abs(x), ya = abs(y);
    const T w = xa < ya ? ya : xa;
    const T z = xa < ya ? xa : ya;
    if (z == T(0) || !(w == w)) return w;
    const T q = z / w;
    return w * sqrt(T(1) + q * q);
}

/// |a| with the sign of b, where b == 0 counts as positive. This is Fortran's
/// intrinsic SIGN(a, b), which the EISPACK and LAPACK code ported here uses.
/// It is not std::copysign. That function exists only for the built-in
/// floating-point types, and it gives -0.0 its negative sign: copysign(1, -0.0)
/// is -1, while with_sign_of(1, -0.0) is 1. eigenvalue() has always treated
/// zero as positive, and keeping that keeps its results bit-identical.
template <typename T>
T with_sign_of(T a, T b) {
    using std::abs;
    return (b >= T(0)) ? abs(a) : -abs(a);
}

/// The Schur factorization of a real 2x2 block in standard form,
///
///   [a b]   [cs -sn] [aa bb] [ cs sn]
///   [c d] = [sn  cs] [cc dd] [-sn cs]
///
/// where cs and sn are the cosine and sine of a plane rotation. On return
/// (a, b, c, d) holds (aa, bb, cc, dd), in one of two forms:
///
/// - cc == 0: the block is upper triangular, and its real eigenvalues are
///   aa and dd.
/// - aa == dd and bb*cc < 0: the eigenvalues are the complex pair
///   aa +- i sqrt(-bb*cc).
///
/// That is the form LAPACK's gees returns for each 2x2 block, and the form
/// Higham's real square root (sqrtm) relies on.
///
/// The eigenvalues are returned as roots of the block's characteristic
/// polynomial, under LAPACK's names: rt1r and rt1i are the real and imaginary
/// parts of root 1, and rt2r and rt2i those of root 2. For a complex pair,
/// rt1i > 0 and rt2i = -rt1i.
///
/// **Name and origin.** This is a port of LAPACK's dlanv2. In LAPACK's naming,
/// the first letter is the arithmetic: s for float, d for double, and c or z
/// for the complex types. "la" marks an auxiliary routine. LAPACK does not
/// expand the remaining letters; the routine handles the nonsymmetric 2x2
/// case. Here the arithmetic is the template parameter T instead of a letter,
/// so the routine is named lanv2: lanv2<float> is slanv2, lanv2<double> is
/// dlanv2, and lanv2<posit<32,2>> is the same algorithm in posit arithmetic.
/// The local names follow dlanv2 so the two can be compared line by line.
/// They are explained where they are defined.
///
/// One change from dlanv2: the scale window comes from lanv2_constants, so a
/// type without numeric_limits skips the rescaling. Fortran's SIGN is
/// with_sign_of().
template <typename T>
void lanv2(T& a, T& b, T& c, T& d, T& rt1r, T& rt1i, T& rt2r, T& rt2i,
           T& cs, T& sn, const lanv2_constants<T>& k) {
    using std::abs;
    using std::sqrt;
    // multpl: dlanv2's MULTPL, the multiple of eps below which the
    // discriminant is treated as zero.
    const T zero(0), one(1), half(0.5), multpl(4);
    if (c == zero) {
        cs = one; sn = zero;
    } else if (b == zero) {
        // Swap rows and columns.
        cs = zero; sn = one;
        const T temp = d;
        d = a; a = temp;
        b = -c; c = zero;
    } else if ((a - d) == zero && with_sign_of(one, b) != with_sign_of(one, c)) {
        cs = one; sn = zero;
    } else {
        T temp = a - d;
        T p = half * temp;
        // b*c enters only as (bcmax / scale) * bcmis, so it is never formed
        // unscaled and cannot overflow. bcmax = max(|b|, |c|), and bcmis
        // (read as "min with sign") is min(|b|, |c|) carrying the sign of b*c.
        const T bcmax = abs(b) < abs(c) ? abs(c) : abs(b);
        const T bcmis = (abs(b) < abs(c) ? abs(b) : abs(c)) * with_sign_of(one, b) * with_sign_of(one, c);
        T scale = abs(p) < bcmax ? bcmax : abs(p);
        T z = (p / scale) * p + (bcmax / scale) * bcmis;
        if (z >= multpl * k.eps) {
            // Real eigenvalues: compute a and d.
            z = p + with_sign_of(sqrt(scale) * sqrt(z), p);
            a = d + z;
            d = d - (bcmax / z) * bcmis;
            const T tau = lapy2(c, z);
            cs = z / tau;
            sn = c / tau;
            b = b - c;
            c = zero;
        } else {
            // Complex eigenvalues, or real (almost) equal ones: make the
            // diagonal elements equal.
            T sigma = b + c;
            for (int count = 0; k.have_range && count < 20; ++count) {
                scale = abs(temp) < abs(sigma) ? abs(sigma) : abs(temp);
                if (scale >= k.safmx2) {
                    sigma = sigma * k.safmn2; temp = temp * k.safmn2;
                } else if (scale <= k.safmn2) {
                    sigma = sigma * k.safmx2; temp = temp * k.safmx2;
                } else {
                    break;
                }
            }
            p = half * temp;
            T tau = lapy2(sigma, temp);
            cs = sqrt(half * (one + abs(sigma) / tau));
            sn = -(p / (tau * cs)) * with_sign_of(one, sigma);

            // [aa bb; cc dd] = [a b; c d] [cs -sn; sn cs]
            const T aa =  a * cs + b * sn;
            const T bb = -a * sn + b * cs;
            const T cc =  c * cs + d * sn;
            const T dd = -c * sn + d * cs;
            // [a b; c d] = [cs sn; -sn cs] [aa bb; cc dd]
            a =  aa * cs + cc * sn;
            b =  bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;

            temp = half * (a + d);
            a = temp;
            d = temp;
            if (c != zero) {
                if (b != zero) {
                    if (with_sign_of(one, b) == with_sign_of(one, c)) {
                        // Real eigenvalues: reduce to upper triangular form.
                        const T sab = sqrt(abs(b));   // sab = sqrt of absolute b
                        const T sac = sqrt(abs(c));   // sac = sqrt of absolute c
                        p = with_sign_of(sab * sac, c);
                        tau = one / sqrt(abs(b + c));
                        a = temp + p;
                        d = temp - p;
                        b = b - c;
                        c = zero;
                        // (cs1, sn1): cosine and sine of a second rotation,
                        // the one that makes this block upper triangular. It
                        // is folded into the first, (cs, sn), so the caller
                        // applies a single rotation.
                        const T cs1 = sab * tau;
                        const T sn1 = sac * tau;
                        temp = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = temp;
                    }
                } else {
                    b = -c;
                    c = zero;
                    temp = cs;
                    cs = -sn;
                    sn = temp;
                }
            }
        }
    }
    rt1r = a;
    rt2r = d;
    if (c == zero) {
        rt1i = zero;
        rt2i = zero;
    } else {
        rt1i = sqrt(abs(b)) * sqrt(abs(c));
        rt2i = -rt1i;
    }
}

/// What francis_qr() computes.
enum class francis_output {
    eigenvalues,    ///< eigenvalues only (EISPACK hqr); H is left as working storage
    schur_form      ///< the real Schur form T, and its Schur vectors when asked (EISPACK hqr2)
};

/// Options for francis_qr(). `deflation_floor` and `lanv2` are used only for
/// francis_output::schur_form.
template <typename T>
struct francis_options {
    T tol{};                    ///< relative deflation tolerance
    int max_its = 100;          ///< sweeps per eigenvalue before giving up
    T deflation_floor{};        ///< absolute floor for a negligible subdiagonal
    lanv2_constants<T> lanv2{};
    std::string_view who = "mtl::eigenvalue";  ///< prefix of the error message
};

/// Francis implicit double-shift QR on the upper Hessenberg H (EISPACK hqr /
/// hqr2). Fills wr, wi (size n) with the eigenvalues.
///
/// francis_output::eigenvalues: eigenvalues only. Each sweep touches only the active block,
/// and H is left as working storage. This is the loop eigenvalue() has always
/// run, operation for operation.
///
/// francis_output::schur_form: H is reduced to the real Schur form T. Every transformation
/// is applied to the whole of T and, when Z != nullptr, accumulated into Z's
/// columns, so A = Z T Z^T holds for the Z and H that came in as A = Z H Z^T.
/// Each 2x2 block is put into standard form by lanv2(), the exceptional shift is
/// added back to the diagonal as each block converges, and an exceptional
/// shift is taken every ten sweeps. Entries below the subdiagonal are left for
/// the caller to clear.
///
/// Throws std::runtime_error when an eigenvalue takes more than max_its sweeps.
template <francis_output Output, typename T>
void francis_qr(mat::dense2D<T>& H, mat::dense2D<T>* Z,
                vec::dense_vector<T>& wr, vec::dense_vector<T>& wi,
                const francis_options<T>& opt) {
    constexpr bool want_schur = Output == francis_output::schur_form;
    using size_type = typename mat::dense2D<T>::size_type;
    using std::abs;
    using std::sqrt;
    using idx = long long;
    using value_type = T;
    const size_type n = H.num_rows();
    const idx last = static_cast<idx>(n) - 1;
    const value_type tol = opt.tol;
    const int max_its = opt.max_its;
    wr = vec::dense_vector<value_type>(n, value_type(0));
    wi = vec::dense_vector<value_type>(n, value_type(0));
    if (n == 0) return;

    // Rotation [cs sn; -sn cs] of rows/columns k, k+1 that lanv2 computed for
    // the block at k: the rest of T, then Z.
    auto rotate_block = [&](idx k, value_type cs, value_type sn) {
        for (idx j = k + 2; j <= last; ++j) {
            const value_type x = H(k, j), y = H(k + 1, j);
            H(k, j)     = cs * x + sn * y;
            H(k + 1, j) = cs * y - sn * x;
        }
        for (idx i = 0; i < k; ++i) {
            const value_type x = H(i, k), y = H(i, k + 1);
            H(i, k)     = cs * x + sn * y;
            H(i, k + 1) = cs * y - sn * x;
        }
        if (Z != nullptr)
            for (size_type i = 0; i < n; ++i) {
                const value_type x = (*Z)(i, k), y = (*Z)(i, k + 1);
                (*Z)(i, k)     = cs * x + sn * y;
                (*Z)(i, k + 1) = cs * y - sn * x;
            }
    };

    // Norm of the Hessenberg matrix (fallback scale for the deflation test).
    value_type anorm = value_type(0);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = (i >= 1 ? i - 1 : size_type(0)); j < n; ++j)
            anorm += abs(H(i, j));

    idx nn = last;                     // bottom index of the active block
    value_type t = value_type(0);      // accumulated exceptional shift

    while (nn >= 0) {
        int its = 0;
        idx l;
        do {
            // Look for a single small subdiagonal element.
            for (l = nn; l >= 1; --l) {
                value_type s = abs(H(l - 1, l - 1)) + abs(H(l, l));
                if (s == value_type(0)) s = anorm;
                if constexpr (want_schur) {
                    const value_type h = abs(H(l, l - 1));
                    if (h <= tol * s || h <= opt.deflation_floor) { H(l, l - 1) = value_type(0); break; }
                } else {
                    if (abs(H(l, l - 1)) <= tol * s) { H(l, l - 1) = value_type(0); break; }
                }
            }
            if (l < 0) l = 0;

            value_type x = H(nn, nn);
            if (l == nn) {
                // One real root.
                wr[static_cast<size_type>(nn)] = x + t;
                wi[static_cast<size_type>(nn)] = value_type(0);
                if constexpr (want_schur) H(nn, nn) = x + t;
                nn -= 1;
            } else {
                value_type y = H(nn - 1, nn - 1);
                value_type w = H(nn, nn - 1) * H(nn - 1, nn);
                if (l == nn - 1) {
                    if constexpr (want_schur) {
                        // Two roots: restore the shift, then put the block in
                        // standard form and carry its rotation through T and Z.
                        const idx k = nn - 1;
                        H(nn, nn) = x + t;
                        H(k, k)   = y + t;
                        value_type cs, sn, rt1r, rt1i, rt2r, rt2i;
                        lanv2(H(k, k), H(k, nn), H(nn, k), H(nn, nn),
                              rt1r, rt1i, rt2r, rt2i, cs, sn, opt.lanv2);
                        rotate_block(k, cs, sn);
                        wr[static_cast<size_type>(k)]  = rt1r;
                        wi[static_cast<size_type>(k)]  = rt1i;
                        wr[static_cast<size_type>(nn)] = rt2r;
                        wi[static_cast<size_type>(nn)] = rt2i;
                    } else {
                        // Two roots: eigenvalues of the trailing 2x2 block.
                        value_type p = value_type(0.5) * (y - x);
                        value_type q = p * p + w;
                        value_type z = sqrt(abs(q));
                        x += t;
                        if (q >= value_type(0)) {
                            z = p + with_sign_of(z, p);
                            wr[static_cast<size_type>(nn - 1)] = wr[static_cast<size_type>(nn)] = x + z;
                            if (z != value_type(0)) wr[static_cast<size_type>(nn)] = x - w / z;
                            wi[static_cast<size_type>(nn - 1)] = wi[static_cast<size_type>(nn)] = value_type(0);
                        } else {
                            wr[static_cast<size_type>(nn - 1)] = wr[static_cast<size_type>(nn)] = x + p;
                            wi[static_cast<size_type>(nn)]     = z;
                            wi[static_cast<size_type>(nn - 1)] = -z;
                        }
                    }
                    nn -= 2;
                } else {
                    // No root yet: perform a Francis double-shift QR sweep.
                    if (its == max_its)
                        throw std::runtime_error(
                            std::string(opt.who) + ": QR iteration failed to converge");
                    value_type p = value_type(0), q = value_type(0), r = value_type(0);
                    value_type z = value_type(0);
                    bool exceptional;
                    if constexpr (want_schur) exceptional = its > 0 && its % 10 == 0;
                    else                 exceptional = its == 10 || its == 20;
                    if (exceptional) {
                        // Exceptional (ad-hoc Wilkinson) shift to break stagnation.
                        t += x;
                        for (idx i = 0; i <= nn; ++i) H(i, i) -= x;
                        value_type s = abs(H(nn, nn - 1)) + abs(H(nn - 1, nn - 2));
                        y = x = value_type(0.75) * s;
                        w = value_type(-0.4375) * s * s;
                    }
                    ++its;

                    // Look for two consecutive small subdiagonal elements.
                    idx m;
                    for (m = nn - 2; m >= l; --m) {
                        z = H(m, m);
                        r = x - z;
                        value_type s = y - z;
                        p = (r * s - w) / H(m + 1, m) + H(m, m + 1);
                        q = H(m + 1, m + 1) - z - r - s;
                        r = H(m + 2, m + 1);
                        s = abs(p) + abs(q) + abs(r);
                        p /= s; q /= s; r /= s;
                        if (m == l) break;
                        value_type u = abs(H(m, m - 1)) * (abs(q) + abs(r));
                        value_type v = abs(p) * (abs(H(m - 1, m - 1)) + abs(z) + abs(H(m + 1, m + 1)));
                        if (u <= tol * v) break;
                    }

                    for (idx i = m + 2; i <= nn; ++i) {
                        H(i, i - 2) = value_type(0);
                        if (i != m + 2) H(i, i - 3) = value_type(0);
                    }

                    // The Schur form updates the whole of T; eigenvalues alone
                    // need only the active block l..nn.
                    const idx jlast  = want_schur ? last : nn;
                    const idx ifirst = want_schur ? idx(0) : l;

                    // Chase the bulge with Householder(3)/(2) transforms.
                    for (idx k = m; k <= nn - 1; ++k) {
                        if (k != m) {
                            p = H(k, k - 1);
                            q = H(k + 1, k - 1);
                            r = value_type(0);
                            if (k != nn - 1) r = H(k + 2, k - 1);
                            x = abs(p) + abs(q) + abs(r);
                            if (x != value_type(0)) { p /= x; q /= x; r /= x; }
                        }
                        value_type s = with_sign_of(sqrt(p * p + q * q + r * r), p);
                        if (s != value_type(0)) {
                            if (k == m) {
                                if (l != m) H(k, k - 1) = -H(k, k - 1);
                            } else {
                                H(k, k - 1) = -s * x;
                            }
                            p += s;
                            x = p / s; y = q / s; z = r / s;
                            q /= p; r /= p;
                            // Row transformation.
                            for (idx j = k; j <= jlast; ++j) {
                                p = H(k, j) + q * H(k + 1, j);
                                if (k != nn - 1) {
                                    p += r * H(k + 2, j);
                                    H(k + 2, j) -= p * z;
                                }
                                H(k + 1, j) -= p * y;
                                H(k, j)     -= p * x;
                            }
                            idx mmin = (nn < k + 3) ? nn : k + 3;
                            // Column transformation.
                            for (idx i = ifirst; i <= mmin; ++i) {
                                p = x * H(i, k) + y * H(i, k + 1);
                                if (k != nn - 1) {
                                    p += z * H(i, k + 2);
                                    H(i, k + 2) -= p * r;
                                }
                                H(i, k + 1) -= p * q;
                                H(i, k)     -= p;
                            }
                            // Accumulate the transformation into Z.
                            if constexpr (want_schur) {
                                if (Z != nullptr)
                                    for (size_type i = 0; i < n; ++i) {
                                        p = x * (*Z)(i, k) + y * (*Z)(i, k + 1);
                                        if (k != nn - 1) {
                                            p += z * (*Z)(i, k + 2);
                                            (*Z)(i, k + 2) -= p * r;
                                        }
                                        (*Z)(i, k + 1) -= p * q;
                                        (*Z)(i, k)     -= p;
                                    }
                            }
                        }
                    }
                }
            }
        } while (l < nn - 1);
    }
}

} // namespace mtl::detail
