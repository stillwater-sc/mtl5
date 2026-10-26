#!/usr/bin/env python3
"""Generate the theta_m tables of include/mtl/operation/expm.hpp.

expm scales A by 2^-s until the backward error of the [m/m] Pade approximant
r_m of exp is below the unit roundoff u. Higham (2005) writes that backward
error as a power series: with h_{2m+1}(x) = log(e^-x r_m(x)) = sum_k c_k x^k,
a matrix with norm (or alpha_p, Al-Mohy & Higham 2009) at most theta has a
relative backward error of at most

    htilde(theta) / theta,   htilde(x) = sum_k |c_k| x^k,

and theta_m is the largest theta for which that bound is at most u. Higham's
tables are for u = 2^-53. This script solves the same equation for any u, so
expm can select the degree for float, double-double, quad and the rest.

It also prints, for each table:
  * |c_{2m+1}|, the leading coefficient of h_{2m+1}, which Al-Mohy & Higham's
    ell(A, m) correction uses;
  * an upper bound on the condition number of the Pade denominator q_m(A)
    over all A with ||A|| <= theta_m, from the roots z_i of q_m:
        kappa(q_m(A)) <= prod_i (1 + theta/|z_i|) / (1 - theta/|z_i|).
    expm solves q_m(A) X = p_m(A), so this bounds how much that solve
    amplifies rounding errors. A large u makes theta_m large, and q_m(A)
    then becomes ill-conditioned: at u = 2^-24 the bound for m = 13 is 2.7e7.
    Al-Mohy & Higham (2009, Sec. 5) already lower double's theta_13 from 5.37
    to 4.25 for this reason, to condition q_13 better. This script turns that
    choice into a rule for every u: each theta_m is capped where its bound
    reaches the bound at theta_13 = 4.25. For double the cap changes only
    theta_13, to 4.25; it also binds for u >= 2^-24.

The script first checks itself against the double-precision values Higham
published (SIMAX 26(4), 2005, Table 2.3), and stops if they disagree.

Usage:
    python3 tools/expm_theta.py            # check, then print the C++ table
    python3 tools/expm_theta.py --check    # check only

Requires mpmath.
"""

import argparse
import sys

import mpmath as mp

DEGREES = (3, 5, 7, 9, 13)

# Unit roundoffs the header carries a table for, as powers of two: u = 2^-t.
# bfloat16, half, float, double, x87 extended, double-double, binary128,
# quad-double, binary256.
TABLES = (
    (8, "bfloat16"),
    (11, "binary16"),
    (24, "binary32"),
    (53, "binary64"),
    (64, "x87 extended"),
    (106, "double-double"),
    (113, "binary128"),
    (212, "quad-double"),
    (237, "binary256"),
)

# Higham (2005), Table 2.3: theta_m for u = 2^-53.
HIGHAM_2005 = {
    3: mp.mpf("1.495585217958292e-2"),
    5: mp.mpf("2.539398330063230e-1"),
    7: mp.mpf("9.504178996162932e-1"),
    9: mp.mpf("2.097847961257068e0"),
    13: mp.mpf("5.371920351148152e0"),
}

# Al-Mohy & Higham (2009), Sec. 5: theta_13 for double, lowered from 5.37 so
# that q_13(A) is better conditioned.
ALHI_2009_THETA13 = mp.mpf("4.25")

# Terms of the backward-error series kept. The series converges for x below
# the modulus of the nearest zero of p_m or q_m, and theta_m stays well inside
# it, so 400 terms leave the truncation far below 2^-237.
TERMS = 400


def pade_coefficients(m):
    """Coefficients of p_m(x) = sum_j b_j x^j, normalized so b_0 = 1."""
    b = [mp.mpf(1)]
    for j in range(1, m + 1):
        b.append(b[-1] * (m - j + 1) / (j * (2 * m - j + 1)))
    return b


def series_div(num, den, n):
    """Power series num/den to n terms (den[0] != 0)."""
    out = []
    for k in range(n):
        acc = num[k] if k < len(num) else mp.mpf(0)
        for j in range(1, min(k, len(den) - 1) + 1):
            acc -= den[j] * out[k - j]
        out.append(acc / den[0])
    return out


def series_mul(a, b, n):
    return [mp.fsum(a[j] * b[k - j] for j in range(max(0, k - len(b) + 1), min(k, len(a) - 1) + 1))
            for k in range(n)]


def series_log(y, n):
    """log(y) for a series with y[0] == 1, from (log y)' = y'/y."""
    dy = [(k + 1) * y[k + 1] for k in range(n - 1)]
    q = series_div(dy, y, n - 1)
    return [mp.mpf(0)] + [q[k] / (k + 1) for k in range(n - 1)]


def backward_error_coefficients(m, n=TERMS):
    """c_k of h_{2m+1}(x) = log(e^-x r_m(x)), k = 0 .. n-1."""
    b = pade_coefficients(m)
    p = b
    q = [(-1) ** j * bj for j, bj in enumerate(b)]
    r = series_div(p, q, n)
    emx = [mp.mpf(-1) ** k / mp.factorial(k) for k in range(n)]
    c = series_log(series_mul(emx, r, n), n)
    # h_{2m+1} starts at x^{2m+1}; everything below is rounding noise.
    lead = 2 * m + 1
    scale = abs(c[lead])
    for k in range(lead):
        assert abs(c[k]) < scale * mp.mpf(10) ** (-mp.mp.dps // 2), (m, k, c[k])
        c[k] = mp.mpf(0)
    return c


def relative_bound(c, x):
    """htilde(x) / x."""
    return mp.fsum(abs(c[k]) * x ** (k - 1) for k in range(1, len(c)))


def theta(c, u):
    """Largest x with htilde(x)/x <= u (the bound increases with x)."""
    lo, hi = mp.mpf(0), mp.mpf(1)
    while relative_bound(c, hi) <= u:
        lo, hi = hi, 2 * hi
    for _ in range(mp.mp.prec + 20):
        mid = (lo + hi) / 2
        if relative_bound(c, mid) <= u:
            lo = mid
        else:
            hi = mid
    return lo


def kappa_bound(m, x):
    """Upper bound on kappa(q_m(A)) over ||A|| <= x; inf if x reaches a root."""
    b = pade_coefficients(m)
    q = [(-1) ** j * bj for j, bj in enumerate(b)]
    roots = mp.polyroots(list(reversed(q)), maxsteps=200, extraprec=200)
    k = mp.mpf(1)
    for z in roots:
        t = x / abs(z)
        if t >= 1:
            return mp.inf
        k *= (1 + t) / (1 - t)
    return k


def theta_cap(m, kappa_max):
    """Largest x with kappa_bound(m, x) <= kappa_max (the bound increases with x)."""
    lo, hi = mp.mpf(0), mp.mpf(1)
    while kappa_bound(m, hi) <= kappa_max:
        lo, hi = hi, 2 * hi
    for _ in range(mp.mp.prec + 20):
        mid = (lo + hi) / 2
        if kappa_bound(m, mid) <= kappa_max:
            lo = mid
        else:
            hi = mid
    return lo


def truncation_ok(c, x, u):
    """The last kept term of htilde(x)/x is negligible against u."""
    n = len(c) - 1
    return abs(c[n]) * x ** (n - 1) < u * mp.mpf(10) ** -20


def literal(x, round_up=False, digits=17):
    """x as a C++ literal of `digits` significant digits, rounded toward zero
    (or away from it), so that the rounded value is on the conservative side."""
    e = int(mp.floor(mp.log10(x)))
    mant = x / mp.mpf(10) ** e
    scale = mp.mpf(10) ** (digits - 1)
    q = mp.ceil(mant * scale) if round_up else mp.floor(mant * scale)
    if q >= 10 * scale:
        q, e = q / 10, e + 1
    d = str(int(q))
    return f"{d[0]}.{d[1:]}e{e:+03d}"


def check():
    worst = mp.mpf(0)
    for m in DEGREES:
        c = backward_error_coefficients(m)
        t = theta(c, mp.mpf(2) ** -53)
        rel = abs(t - HIGHAM_2005[m]) / HIGHAM_2005[m]
        worst = max(worst, rel)
        print(f"  m={m:2d}  theta={mp.nstr(t, 16):>24}  Higham 2005 {mp.nstr(HIGHAM_2005[m], 16):>24}"
              f"  rel diff {mp.nstr(rel, 3)}")
    # Higham prints 16 significant digits.
    if worst > mp.mpf("1e-15"):
        print("FAILED: does not reproduce Higham (2005), Table 2.3", file=sys.stderr)
        return False
    print("  reproduces Higham (2005), Table 2.3, to the 16 digits printed there")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only check against Higham (2005)")
    ap.add_argument("--dps", type=int, default=160, help="mpmath decimal digits")
    args = ap.parse_args()
    mp.mp.dps = args.dps

    print("check, u = 2^-53:")
    if not check():
        return 1
    if args.check:
        return 0

    coeffs = {m: backward_error_coefficients(m) for m in DEGREES}
    kappa_max = kappa_bound(13, ALHI_2009_THETA13)
    caps = {m: theta_cap(m, kappa_max) for m in DEGREES}
    print()
    print(f"kappa cap (bound at Al-Mohy & Higham's theta_13 = 4.25): {mp.nstr(kappa_max, 6)}")
    print("|c_{2m+1}|:")
    for m in DEGREES:
        print(f"  m={m:2d}  {mp.nstr(abs(coeffs[m][2 * m + 1]), 17)}   theta cap {mp.nstr(caps[m], 17)}")
    print()
    table = []
    for t, name in TABLES:
        u = mp.mpf(2) ** -t
        row = []
        print(f"u = 2^-{t} ({name}):")
        for m in DEGREES:
            th = theta(coeffs[m], u)
            used = min(th, caps[m])
            if not truncation_ok(coeffs[m], used, u):
                print(f"FAILED: series not converged for m={m}, u=2^-{t}", file=sys.stderr)
                return 1
            print(f"  m={m:2d}  theta={mp.nstr(th, 17):>24}  kappa(q_m) <= {mp.nstr(kappa_bound(m, th), 4):>10}"
                  f"  used {mp.nstr(used, 17)}{'  (capped)' if used < th else ''}")
            row.append(used)
        table.append((t, name, row))
    print()
    print("// C++ table, generated by tools/expm_theta.py (values rounded toward zero)")
    for t, name, row in table:
        vals = ", ".join(literal(v) for v in row)
        print(f"    {{{t:3d}, {{{vals}}}}},  // {name}")
    print("// |c_{2m+1}|, rounded up")
    print("    " + ", ".join(literal(abs(coeffs[m][2 * m + 1]), round_up=True) for m in DEGREES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
