#!/usr/bin/env python3
"""Generate the animated SVGs of docs/algorithms/givens-householder-pictures.md.

Four figures, written to docs/img/ (or the directory given with --out):

  gh-plane.svg  a vector carried onto the x axis by a Givens rotation and by
                a Householder reflection
  gh-qr-fill.svg Givens and Householder QR of the same sparse matrix, one
                step at a time, with the fill and the operation counts
  gh-bulge.svg  one Golub-Kahan SVD step on an upper bidiagonal matrix: the
                bulge chased down the band by alternating rotations
  gh-drift.svg  the unit circle after k computed rotations in fp8 (E4M3),
                bfloat16, binary16 and posit<16,2>, and the radius error
                against k

It also writes docs-site/tests/gh-vectors.json: rounding, square-root,
rotation, reflector and drift results in ten narrow formats, computed here in
exact arithmetic, against which the JavaScript arithmetic of the interactive
page (docs/site/algorithms/givens-householder-explorer.mdx) is tested.

The animations use SVG's own SMIL elements (<animate>, <animateTransform>),
not script, so they play wherever the SVG is shown, including through an
<img> tag. The output depends only on this file: rerunning it reproduces the
committed SVGs byte for byte.

The drift figure needs arithmetic in narrow formats. Each operation is done
exactly in rational arithmetic (fractions.Fraction) and the exact result is
rounded once to the target format, so every operation is correctly rounded
and no double-rounding argument is needed. Square roots are rounded from an
integer square root carried well past the format's precision, with a sticky
bit. Only the standard library is used.
"""

import argparse
import math
import os
import sys
from fractions import Fraction


# --------------------------------------------------------------------------
# Number formats
# --------------------------------------------------------------------------

def floor_log2(a):
    """floor(log2(a)) for a positive Fraction, exactly."""
    n, d = a.numerator, a.denominator
    s = n.bit_length() - d.bit_length()
    # 2^s <= a  <=>  n * 2^-s >= d
    if s >= 0:
        if n < (d << s):
            s -= 1
    else:
        if (n << -s) < d:
            s -= 1
    return s


def round_scaled(a, qexp):
    """Round the positive Fraction a to a multiple of 2^qexp, ties to even.
    Returns the integer multiple."""
    scaled = a / Fraction(2) ** qexp
    m = scaled.numerator // scaled.denominator
    rem = scaled - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m % 2 == 1):
        m += 1
    return m


class IEEELike:
    """A binary floating-point format with p significand bits (hidden bit
    included) and exponents emin .. emax, with emin = 1 - emax unless given.
    OCP E4M3 is the exception: bias 7 gives emin = -6, but it keeps exponent
    field 15 for normals, so emax = 8."""

    def __init__(self, name, p, emax, subnormals=True, saturate=False,
                 maxval=None, emin=None):
        self.name = name
        self.p = p
        self.emax = emax
        self.emin = 1 - emax if emin is None else emin
        self.subnormals = subnormals
        self.saturate = saturate
        self.maxval = (Fraction(maxval) if maxval is not None else
                       (2 - Fraction(2) ** (1 - p)) * Fraction(2) ** emax)

    def round(self, q):
        if q == 0:
            return Fraction(0)
        sign = -1 if q < 0 else 1
        a = abs(q)
        s = floor_log2(a)
        if s < self.emin and not self.subnormals:
            # Flush: the nearest of 0 and the smallest normal.
            tiny = Fraction(2) ** self.emin
            r = tiny if a >= tiny / 2 else Fraction(0)
            return sign * r
        qexp = s - self.p + 1
        if self.subnormals:
            qexp = max(qexp, self.emin - self.p + 1)
        r = round_scaled(a, qexp) * Fraction(2) ** qexp
        if r > self.maxval:
            if self.saturate:
                r = self.maxval
            else:
                return sign * math.inf
        return sign * r


class Posit:
    """posit<n, es>, rounded in its encoding as the posit standard requires:
    the exact value's bit string is rounded to n-1 bits, ties to even, and
    never to zero or NaR."""

    def __init__(self, name, n, es):
        self.name = name
        self.n = n
        self.es = es

    def _decode(self, bits):
        w = self.n - 1
        s = format(bits, '0%db' % w)
        r0 = s[0]
        run = len(s) - len(s.lstrip(r0))
        k = run - 1 if r0 == '1' else -run
        rest = s[run + 1:]
        ebits = (rest[:self.es]).ljust(self.es, '0')
        e = int(ebits, 2) if self.es else 0
        fbits = rest[self.es:]
        f = Fraction(int(fbits, 2), 2 ** len(fbits)) if fbits else Fraction(0)
        return Fraction(2) ** (k * 2 ** self.es + e) * (1 + f)

    def round(self, q):
        if q == 0:
            return Fraction(0)
        sign = -1 if q < 0 else 1
        a = abs(q)
        s = floor_log2(a)
        k = s >> self.es
        e = s - (k << self.es)
        regime = ('1' * (k + 1) + '0') if k >= 0 else ('0' * (-k) + '1')
        head = int(regime + (format(e, '0%db' % self.es) if self.es else ''), 2)
        L = len(regime) + self.es
        w = self.n - 1
        T = max(w - L, 0) + 2
        frac = (a / Fraction(2) ** s - 1) * 2 ** T
        fint = frac.numerator // frac.denominator
        sticky = frac != fint
        full = (head << T) | fint
        shift = L + T - w
        kept = full >> shift
        rest = full & ((1 << shift) - 1)
        guard = (rest >> (shift - 1)) & 1
        sticky = sticky or (rest & ((1 << (shift - 1)) - 1)) != 0
        if guard and (sticky or kept & 1):
            kept += 1
        kept = min(max(kept, 1), (1 << w) - 1)
        return sign * self._decode(kept)


def to_fraction(x):
    return x if isinstance(x, Fraction) else Fraction(x)


def rnd(fmt, q):
    return fmt.round(to_fraction(q))


def rsqrt_exact(fmt, q):
    """The format's correct rounding of sqrt(q), q a nonnegative Fraction."""
    if q == 0:
        return Fraction(0)
    K = 200
    scaled = q * Fraction(4) ** K
    n = scaled.numerator // scaled.denominator
    t = math.isqrt(n)
    exact = t * t == n and scaled.denominator == 1
    # A value strictly between t/2^K and (t+1)/2^K stands in for an inexact
    # root: no rounding boundary of a format this narrow lies in that gap.
    v = Fraction(t, 2 ** K) if exact else Fraction(2 * t + 1, 2 ** (K + 1))
    return fmt.round(v)


FORMATS = [
    IEEELike('fp8 E4M3 (OCP)', p=4, emax=8, emin=-6, saturate=True,
             maxval=448),
    IEEELike('bfloat16', p=8, emax=127),
    IEEELike('binary16', p=11, emax=15),
    Posit('posit<16,2>', n=16, es=2),
]


# --------------------------------------------------------------------------
# SVG helpers
# --------------------------------------------------------------------------

STYLE = """<style>
  .bg    { fill: #fbfbfd; stroke: #d0d4dc; }
  .ink   { fill: #1f2430; }
  .muted { fill: #6b7280; }
  .line  { stroke: #1f2430; fill: none; }
  .axis  { stroke: #9aa1ad; fill: none; }
  .guide { stroke: #9aa1ad; fill: none; stroke-dasharray: 4 4; }
  .vec   { stroke: #2563eb; fill: none; }
  .vecf  { fill: #2563eb; }
  .ghost { stroke: #93b4f5; fill: none; }
  .mirror{ stroke: #c2410c; fill: none; stroke-dasharray: 6 4; }
  .accent{ fill: #c2410c; }
  .accs  { stroke: #c2410c; fill: none; }
  .nz    { fill: #3b6fd8; }
  .fill  { fill: #e8892c; }
  .gone  { fill: #f3c1c1; }
  .zero  { fill: #ffffff; stroke: #d7dbe3; }
  .band  { fill: #fde68a; }
  .s0 { stroke: #2563eb; } .s1 { stroke: #c2410c; }
  .s2 { stroke: #059669; } .s3 { stroke: #7c3aed; }
  .f0 { fill: #2563eb; } .f1 { fill: #c2410c; }
  .f2 { fill: #059669; } .f3 { fill: #7c3aed; }
  text { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; }
  @media (prefers-color-scheme: dark) {
    .bg    { fill: #171a21; stroke: #343a46; }
    .ink   { fill: #e6e8ee; }
    .muted { fill: #9aa1ad; }
    .line  { stroke: #e6e8ee; }
    .axis  { stroke: #5b6270; }
    .guide { stroke: #5b6270; }
    .vec   { stroke: #6ea0ff; }
    .vecf  { fill: #6ea0ff; }
    .ghost { stroke: #3c5a99; }
    .mirror{ stroke: #f08a4b; }
    .accent{ fill: #f08a4b; }
    .accs  { stroke: #f08a4b; }
    .nz    { fill: #5b8cf0; }
    .fill  { fill: #f0a050; }
    .gone  { fill: #6b3a3a; }
    .zero  { fill: #1f232c; stroke: #343a46; }
    .band  { fill: #5c4a12; }
    .s0 { stroke: #6ea0ff; } .s1 { stroke: #f08a4b; }
    .s2 { stroke: #34d399; } .s3 { stroke: #b28cff; }
    .f0 { fill: #6ea0ff; } .f1 { fill: #f08a4b; }
    .f2 { fill: #34d399; } .f3 { fill: #b28cff; }
  }
</style>"""


def f(x):
    """A coordinate, printed compactly and deterministically."""
    s = '%.2f' % x
    s = s.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def ft(x):
    """A keyTime fraction. Two decimals, as f() prints, would make equal
    slots unequal: with 59 slots some states would show twice as long."""
    s = ('%.6f' % x).rstrip('0').rstrip('.')
    return '0' if s == '' else s


def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;'))


def svg_open(w, h, title, desc):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
            'width="%d" height="%d" role="img">' % (w, h, w, h),
            '<title>%s</title>' % esc(title),
            '<desc>%s</desc>' % esc(desc),
            STYLE,
            '<rect class="bg" x="0.5" y="0.5" width="%d" height="%d" rx="8"/>'
            % (w - 1, h - 1)]


def text(x, y, s, cls='ink', size=13, anchor='start', extra=''):
    return ('<text class="%s" x="%s" y="%s" font-size="%d" '
            'text-anchor="%s"%s>%s</text>'
            % (cls, f(x), f(y), size, anchor, extra, esc(s)))


def arrow(x1, y1, x2, y2, cls='vec', width=2.5):
    """A line with a filled triangular head, in screen coordinates."""
    ang = math.atan2(y2 - y1, x2 - x1)
    hl, hw = 11, 5
    bx, by = x2 - hl * math.cos(ang), y2 - hl * math.sin(ang)
    px, py = -math.sin(ang) * hw, math.cos(ang) * hw
    head = 'M%s,%s L%s,%s L%s,%s Z' % (f(x2), f(y2), f(bx + px), f(by + py),
                                        f(bx - px), f(by - py))
    fcls = {'vec': 'vecf', 'accs': 'accent', 'ghost': 'vecf'}.get(cls, 'vecf')
    op = ' opacity="0.45"' if cls == 'ghost' else ''
    return ('<line class="%s" x1="%s" y1="%s" x2="%s" y2="%s" '
            'stroke-width="%s"%s/><path class="%s" d="%s"%s/>'
            % (cls, f(x1), f(y1), f(bx), f(by), width, op, fcls, head, op))


def discrete(attr, values, dur, extra=''):
    """A discrete SMIL animation stepping through values in equal slots."""
    n = len(values)
    times = ';'.join(ft(i / n) for i in range(n))
    return ('<animate attributeName="%s" calcMode="discrete" dur="%ss" '
            'repeatCount="indefinite" values="%s" keyTimes="%s"%s/>'
            % (attr, f(dur), ';'.join(values), times, extra))


def state_cell(x, y, w, h, rx, states, dur):
    """A cell whose class steps through states. SMIL cannot portably animate
    the class attribute, and an animated fill would lose to the stylesheet,
    so each state the cell takes gets its own rect and only its opacity is
    animated."""
    out = []
    for st in sorted(set(states), key=states.index):
        vis = ['1' if v == st else '0' for v in states]
        anim = discrete_sparse('opacity', vis, dur) or ''
        out.append('<rect class="%s" x="%s" y="%s" width="%s" height="%s" '
                   'rx="%s" opacity="%s">%s</rect>'
                   % (st, f(x), f(y), f(w), f(h), f(rx), vis[0], anim))
    return ''.join(out)


def compress(values):
    """values with runs of equal entries merged, as (value, start) pairs."""
    out = []
    for i, v in enumerate(values):
        if not out or out[-1][0] != v:
            out.append((v, i))
    return out


def discrete_sparse(attr, values, dur):
    """discrete() for a value list with long constant runs."""
    n = len(values)
    runs = compress(values)
    if len(runs) == 1:
        return None
    vals = ';'.join(v for v, _ in runs)
    times = ';'.join(ft(i / n) for _, i in runs)
    return ('<animate attributeName="%s" calcMode="discrete" dur="%ss" '
            'repeatCount="indefinite" values="%s" keyTimes="%s"/>'
            % (attr, f(dur), vals, times))


# --------------------------------------------------------------------------
# Figure 1: a rotation and a reflection in the plane
# --------------------------------------------------------------------------

def fig_plane():
    W, H = 760, 380
    R = 100                         # pixels per unit
    theta = 50                      # angle of x, in degrees
    out = svg_open(W, H, 'A Givens rotation and a Householder reflection',
                   'Left: the vector x is rotated clockwise through its angle '
                   'theta onto the positive x axis. Right: the same x is '
                   'reflected across the line that bisects x and the x axis, '
                   'landing on the same point.')
    dur = 8
    # keyTimes: hold, move, hold, return
    kt = '0;0.15;0.5;0.85;1'

    for panel, (ox, label) in enumerate(((190, 'Givens rotation'),
                                         (570, 'Householder reflection'))):
        oy = 195
        out.append(text(ox, 34, label, size=16, anchor='middle',
                        extra=' font-weight="600"'))
        out.append('<line class="axis" x1="%s" y1="%s" x2="%s" y2="%s"/>'
                   % (f(ox - 140), f(oy), f(ox + 150), f(oy)))
        out.append('<line class="axis" x1="%s" y1="%s" x2="%s" y2="%s"/>'
                   % (f(ox), f(oy + 30), f(ox), f(oy - 135)))
        out.append('<circle class="guide" cx="%s" cy="%s" r="%d"/>'
                   % (f(ox), f(oy), R))
        tx = ox + R * math.cos(math.radians(theta))
        ty = oy - R * math.sin(math.radians(theta))
        out.append(arrow(ox, oy, tx, ty, cls='ghost'))
        out.append(text(tx + 6, ty - 6, 'x = (a, b)', cls='muted', size=12))
        out.append(text(ox + R + 6, oy + 18, '(r, 0)', cls='muted', size=12))

        if panel == 0:
            # The swept angle.
            ar = 42
            ex = ox + ar * math.cos(math.radians(theta))
            ey = oy - ar * math.sin(math.radians(theta))
            out.append('<path class="accs" stroke-width="1.5" '
                       'd="M%s,%s A%d,%d 0 0 1 %s,%s"/>'
                       % (f(ex), f(ey), ar, ar, f(ox + ar), f(oy)))
            out.append(text(ox + 48, oy - 16, 'θ', cls='accent', size=14))
            out.append('<g transform="translate(%s,%s)"><g>' % (f(ox), f(oy)))
            out.append('<animateTransform attributeName="transform" '
                       'type="rotate" dur="%ds" repeatCount="indefinite" '
                       'values="0;0;%d;%d;0" keyTimes="%s"/>'
                       % (dur, theta, theta, kt))
            ax = R * math.cos(math.radians(theta))
            ay = -R * math.sin(math.radians(theta))
            out.append(arrow(0, 0, ax, ay))
            out.append('</g></g>')
            out.append(text(ox, 322, 'G = [ c  s ; −s  c ],   c = a/r,  s = b/r',
                            size=13, anchor='middle'))
            out.append(text(ox, 344, 'r = √(a² + b²): one entry zeroed, '
                            'two rows touched', cls='muted', size=12,
                            anchor='middle'))
        else:
            half = theta / 2
            # The mirror: the line through the origin at angle theta/2.
            mx = 150 * math.cos(math.radians(half))
            my = 150 * math.sin(math.radians(half))
            out.append('<line class="mirror" stroke-width="1.5" x1="%s" '
                       'y1="%s" x2="%s" y2="%s"/>'
                       % (f(ox - mx * 0.35), f(oy + my * 0.35), f(ox + mx),
                          f(oy - my)))
            # v = x - r e1, drawn from r e1 to x.
            out.append('<line class="guide" x1="%s" y1="%s" x2="%s" y2="%s"/>'
                       % (f(ox + R), f(oy), f(tx), f(ty)))
            # Reflect: rotate the mirror onto the axis, scale y from 1 to
            # -1, rotate back. Screen angles are negated math angles.
            out.append('<g transform="translate(%s,%s) rotate(%s)"><g>'
                       % (f(ox), f(oy), f(-half)))
            out.append('<animateTransform attributeName="transform" '
                       'type="scale" dur="%ds" repeatCount="indefinite" '
                       'values="1 1;1 1;1 -1;1 -1;1 1" keyTimes="%s"/>'
                       % (dur, kt))
            out.append('<g transform="rotate(%s)">' % f(half))
            ax = R * math.cos(math.radians(theta))
            ay = -R * math.sin(math.radians(theta))
            out.append(arrow(0, 0, ax, ay))
            out.append('</g></g></g>')
            out.append(text(ox, 322, 'H = I − τ v vᵀ,   τ = 2 / vᵀv',
                            size=13, anchor='middle'))
            out.append(text(ox, 344, 'one reflector zeroes a whole column '
                            'below the diagonal', cls='muted', size=12,
                            anchor='middle'))
            out.append(text(ox, 362, 'grey dashes: v = x − r e₁;  orange '
                            'dashes: the mirror, ⟂ v', cls='muted', size=11,
                            anchor='middle'))
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------
# Figure 2: Givens and Householder QR on a sparse matrix
# --------------------------------------------------------------------------

# A dense first column and one other entry per row: the reflector for
# column 1 touches every row at once, the rotations two rows at a time.
PATTERN = [
    'XX..',
    'X.X.',
    'X..X',
    'XX..',
    'X.X.',
    'X..X',
    'X...',
]

ZERO, NZ, FILL, GONE = 0, 1, 2, 3


def qr_frames(method):
    """Symbolic QR of PATTERN. Returns a list of frames; each frame is
    (state grid, active rows, caption, flops so far, fill so far)."""
    m, n = len(PATTERN), len(PATTERN[0])
    A = [[NZ if c == 'X' else ZERO for c in row] for row in PATTERN]
    frames = [([r[:] for r in A], (), 'start', 0, 0)]
    flops = 0
    fill = 0

    def settle(G):
        return [[ZERO if v == GONE else v for v in r] for r in G]

    def nonzero(v):
        return v in (NZ, FILL)

    for j in range(n):
        if method == 'givens':
            # Rotate each row below j that has an entry in column j into
            # row j (the row-at-a-time scheme of sparse Givens QR).
            for i in range(m - 1, j, -1):
                if not nonzero(A[i][j]):
                    continue
                cols = [c for c in range(j, n)
                        if nonzero(A[j][c]) or nonzero(A[i][c])]
                # 4 mul + 2 add per column right of j; about 6 for c, s, r.
                flops += 6 * len([c for c in cols if c > j]) + 6
                for c in cols:
                    for r in (j, i):
                        if not nonzero(A[r][c]):
                            A[r][c] = FILL
                            fill += 1
                A[i][j] = GONE
                frames.append(([r[:] for r in A], (j, i),
                               'rotate rows %d, %d: zero (%d, %d)'
                               % (j + 1, i + 1, i + 1, j + 1), flops, fill))
                A = settle(A)
        else:
            support = [i for i in range(j, m) if nonzero(A[i][j])]
            if not support or support == [j]:
                continue
            cols = [c for c in range(j + 1, n)
                    if any(nonzero(A[i][c]) for i in support)]
            flops += 3 * len(support) + 4 * len(support) * len(cols)
            for c in cols:
                for i in support:
                    if not nonzero(A[i][c]):
                        A[i][c] = FILL
                        fill += 1
            if not nonzero(A[j][j]):
                A[j][j] = FILL
                fill += 1
            for i in support:
                if i != j:
                    A[i][j] = GONE
            frames.append(([r[:] for r in A], tuple(support),
                           'reflect %d rows: zero column %d below the diagonal'
                           % (len(support), j + 1),
                           flops, fill))
            A = settle(A)
    frames.append((A, (), 'R', flops, fill))
    return frames


def fig_qr():
    gfr = qr_frames('givens')
    hfr = qr_frames('householder')
    nframes = max(len(gfr), len(hfr)) + 2      # hold the end
    step = 1.2
    dur = nframes * step
    m, n = len(PATTERN), len(PATTERN[0])
    cs = 30
    W, H = 760, 440
    out = svg_open(W, H, 'Givens and Householder QR of a sparse matrix',
                   'The same 7 by 4 sparse matrix is reduced to triangular '
                   'form by Givens rotations, one entry at a time, and by '
                   'Householder reflectors, one column at a time. Blue cells '
                   'are original nonzeros, orange cells are fill, and the '
                   'counters show rotations or reflectors, operations and '
                   'fill.')
    cls_of = {ZERO: 'zero', NZ: 'nz', FILL: 'fill', GONE: 'gone'}
    for panel, (frames, label, ox) in enumerate(
            ((gfr, 'Givens: one entry at a time', 60),
             (hfr, 'Householder: one column at a time', 440))):
        frames = frames + [frames[-1]] * (nframes - len(frames))
        out.append(text(ox + 130, 34, label, size=15, anchor='middle',
                        extra=' font-weight="600"'))
        oy = 60
        # Row highlight bands.
        for i in range(m):
            vals = ['1' if i in fr[1] else '0' for fr in frames]
            anim = discrete_sparse('opacity', vals, dur)
            y = oy + i * cs
            out.append('<rect class="band" x="%s" y="%s" width="%s" '
                       'height="%s" opacity="%s">%s</rect>'
                       % (f(ox - 6), f(y), f(n * cs + 12), f(cs), vals[0],
                          anim or ''))
        for i in range(m):
            for j in range(n):
                vals = [cls_of[fr[0][i][j]] for fr in frames]
                out.append(state_cell(ox + j * cs + 2, oy + i * cs + 2,
                                      cs - 4, cs - 4, 3, vals, dur))
        for i in range(m):
            out.append(text(ox - 14, oy + i * cs + 20, str(i + 1), cls='muted',
                            size=11, anchor='end'))
        # Captions and counters, one text per distinct frame.
        tx = ox + n * cs + 24
        kind = 'rotations' if panel == 0 else 'reflectors'
        for k, fr in enumerate(frames):
            if k > 0 and fr is frames[k - 1]:
                continue
            last = k
            while last + 1 < nframes and frames[last + 1] is fr:
                last += 1
            vals = ['1' if k <= q <= last else '0' for q in range(nframes)]
            anim = discrete_sparse('opacity', vals, dur) or ''
            nsteps = len([x for x in frames[1:last + 1] if x[1]])
            stored = [sum(v != ZERO for r in x[0] for v in r)
                      for x in frames[:last + 1]]
            block = [
                text(tx, oy + 16, '%s: %d' % (kind, nsteps), size=13),
                text(tx, oy + 38, 'operations: %d' % fr[3], size=13),
                text(tx, oy + 60, 'fill so far: %d' % fr[4], size=13),
                text(tx, oy + 82, 'stored: %d (peak %d)'
                     % (stored[-1], max(stored)), size=13),
            ]
            out.append('<g opacity="%s">%s%s</g>' % (vals[0], anim,
                                                     ''.join(block)))
            cap = fr[2] if fr[2] not in ('start', 'R') else (
                'the sparse matrix A' if fr[2] == 'start' else 'R: done')
            out.append('<g opacity="%s">%s%s</g>'
                       % (vals[0], anim,
                          text(ox + 130, oy + m * cs + 30, cap, cls='muted',
                               size=12, anchor='middle')))
    # Legend.
    ly = 360
    for k, (cls, lab) in enumerate((('nz', 'nonzero of A'),
                                    ('fill', 'fill (new nonzero)'),
                                    ('gone', 'zeroed this step'),
                                    ('band', 'rows touched'))):
        lx = 90 + k * 160
        out.append('<rect class="%s" x="%d" y="%d" width="14" height="14" '
                   'rx="3"/>' % (cls, lx, ly))
        out.append(text(lx + 20, ly + 12, lab, cls='muted', size=12))
    out.append(text(380, 400, 'operations: multiplies and adds on nonzero '
                    'columns only (6 per rotated column pair, about 4 per '
                    'reflected entry)', cls='muted', size=11, anchor='middle'))
    out.append(text(380, 420, 'Both end with the same R: its pattern is that '
                    'of the Cholesky factor of AᵀA.', cls='muted', size=11,
                    anchor='middle'))
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------
# Figure 3: chasing the bulge in a Golub-Kahan SVD step
# --------------------------------------------------------------------------

def bulge_frames(n=5):
    """Upper bidiagonal n x n. Frames: (nonzero set, bulge cell, zeroed cell,
    band (kind, a, b), caption)."""
    base = {(i, i) for i in range(n)} | {(i, i + 1) for i in range(n - 1)}
    frames = [(base, None, None, None, 'upper bidiagonal B')]
    bulge = None
    for k in range(n - 1):
        # Right rotation on columns k, k+1.
        zeroed = bulge
        bulge = (k + 1, k)
        cap = ('right rotation, columns %d and %d: shift enters, bulge at '
               '(%d, %d)' % (k + 1, k + 2, k + 2, k + 1) if k == 0 else
               'right rotation, columns %d and %d: zero (%d, %d), bulge at '
               '(%d, %d)' % (k + 1, k + 2, zeroed[0] + 1, zeroed[1] + 1,
                             k + 2, k + 1))
        frames.append((base, bulge, zeroed, ('col', k, k + 1), cap))
        # Left rotation on rows k, k+1.
        zeroed = bulge
        bulge = (k, k + 2) if k + 2 < n else None
        if bulge:
            cap = ('left rotation, rows %d and %d: zero (%d, %d), bulge at '
                   '(%d, %d)' % (k + 1, k + 2, zeroed[0] + 1, zeroed[1] + 1,
                                 k + 1, k + 3))
        else:
            cap = ('left rotation, rows %d and %d: zero (%d, %d), bidiagonal '
                   'again' % (k + 1, k + 2, zeroed[0] + 1, zeroed[1] + 1))
        frames.append((base, bulge, zeroed, ('row', k, k + 1), cap))
    frames.append((base, None, None, None,
                   'one step done: the last superdiagonal entry shrinks'))
    return frames


def fig_bulge():
    n = 5
    frames = bulge_frames(n)
    frames = frames + [frames[-1]]
    nf = len(frames)
    step = 1.6
    dur = nf * step
    cs = 42
    W, H = 760, 380
    ox, oy = 70, 70
    out = svg_open(W, H, 'Chasing the bulge in a Golub-Kahan SVD step',
                   'A 5 by 5 upper bidiagonal matrix. Alternating right and '
                   'left Givens rotations push a single nonzero, the bulge, '
                   'down and to the right until it falls off the matrix and '
                   'the matrix is bidiagonal again.')
    out.append(text(ox + n * cs / 2, 40, 'One implicit-shift SVD step',
                    size=15, anchor='middle', extra=' font-weight="600"'))
    # Bands.
    for kind in ('row', 'col'):
        for a in range(n):
            vals = ['1' if fr[3] and fr[3][0] == kind and a in fr[3][1:]
                    else '0' for fr in frames]
            anim = discrete_sparse('opacity', vals, dur)
            if anim is None and vals[0] == '0':
                continue
            if kind == 'row':
                geom = (ox - 6, oy + a * cs, n * cs + 12, cs)
            else:
                geom = (ox + a * cs, oy - 6, cs, n * cs + 12)
            out.append('<rect class="band" x="%s" y="%s" width="%s" '
                       'height="%s" opacity="%s">%s</rect>'
                       % (f(geom[0]), f(geom[1]), f(geom[2]), f(geom[3]),
                          vals[0], anim or ''))
    for i in range(n):
        for j in range(n):
            vals = []
            for base, bulge, zeroed, _, _ in frames:
                if (i, j) == bulge:
                    vals.append('fill')
                elif (i, j) == zeroed:
                    vals.append('gone')
                elif (i, j) in base:
                    vals.append('nz')
                else:
                    vals.append('zero')
            out.append(state_cell(ox + j * cs + 3, oy + i * cs + 3,
                                  cs - 6, cs - 6, 4, vals, dur))
    tx = ox + n * cs + 40
    for k, fr in enumerate(frames):
        if k > 0 and fr is frames[k - 1]:
            continue
        last = k
        while last + 1 < nf and frames[last + 1] is fr:
            last += 1
        vals = ['1' if k <= q <= last else '0' for q in range(nf)]
        anim = discrete_sparse('opacity', vals, dur) or ''
        nrot = len([x for x in frames[1:last + 1] if x[3]])
        out.append('<g opacity="%s">%s%s%s</g>'
                   % (vals[0], anim,
                      text(tx, oy + 30, fr[4], size=13),
                      text(tx, oy + 56, 'rotations so far: %d' % nrot,
                           cls='muted', size=12)))
    notes = [
        'Each rotation zeroes the bulge the previous one made',
        'and makes a new one one place further down.',
        'An n x n step costs 2(n − 1) rotations, O(n) work.',
        'The Francis step on a Hessenberg matrix chases its',
        'bulge the same way, with 3 x 3 reflectors.',
    ]
    for k, line in enumerate(notes):
        out.append(text(tx, oy + 120 + k * 20, line, cls='muted', size=12))
    ly = oy + n * cs + 40
    for k, (cls, lab) in enumerate((('nz', 'band entry'),
                                    ('fill', 'bulge'),
                                    ('gone', 'just zeroed'),
                                    ('band', 'rows or columns rotated'))):
        lx = 70 + k * 160
        out.append('<rect class="%s" x="%d" y="%d" width="14" height="14" '
                   'rx="3"/>' % (cls, lx, ly))
        out.append(text(lx + 20, ly + 12, lab, cls='muted', size=12))
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------
# Figure 4: orthogonality drift under repeated rotation
# --------------------------------------------------------------------------

NPTS = 48
KMAX = 400
KSTEP = 8
# One radian: not a rational multiple of pi, and not chosen to favour any
# format. How far c^2 + s^2 strays from 1 depends on where cos and sin of the
# angle happen to fall between representable values, so another angle ranks
# binary16 and posit<16,2> differently.
THETA = 1.0


def drift_series(fmt):
    """For k = 0, KSTEP, ..., KMAX: the points G^k x computed in fmt and
    rotated back exactly by -k*theta, and max | |G^k x| - 1 |."""
    c = rnd(fmt, math.cos(THETA))
    s = rnd(fmt, math.sin(THETA))
    pts = [(rnd(fmt, math.cos(2 * math.pi * i / NPTS)),
            rnd(fmt, math.sin(2 * math.pi * i / NPTS))) for i in range(NPTS)]
    out = []
    for k in range(KMAX + 1):
        if k % KSTEP == 0:
            back = []
            err = 0.0
            for x, y in pts:
                xf, yf = float(x), float(y)
                ca, sa = math.cos(-k * THETA), math.sin(-k * THETA)
                back.append((ca * xf - sa * yf, sa * xf + ca * yf))
                err = max(err, abs(math.hypot(xf, yf) - 1))
            out.append((k, back, err))
        nxt = []
        for x, y in pts:
            x1 = rnd(fmt, rnd(fmt, c * x) - rnd(fmt, s * y))
            y1 = rnd(fmt, rnd(fmt, s * x) + rnd(fmt, c * y))
            nxt.append((x1, y1))
        pts = nxt
    return float(c), float(s), out


def fig_drift():
    series = [(fmt,) + drift_series(fmt) for fmt in FORMATS]
    nf = len(series[0][3])
    step = 0.25
    hold = 8
    dur = (nf + hold) * step
    W, H = 760, 560
    out = svg_open(W, H, 'Orthogonality drift under repeated rotation',
                   'Forty-eight points on the unit circle are rotated '
                   'repeatedly by the same Givens rotation, computed in '
                   'fp8, bfloat16, binary16 and posit<16,2>, and rotated '
                   'back exactly so that only the error remains. The chart '
                   'below shows the largest radius error against the '
                   'number of rotations.')
    out.append(text(W / 2, 30, 'The unit circle after k computed rotations '
                    '(rotated back exactly)', size=15, anchor='middle',
                    extra=' font-weight="600"'))
    R = 62
    for p, (fmt, c, s, data) in enumerate(series):
        cx = 95 + p * 190
        cy = 140
        out.append('<circle class="guide" cx="%s" cy="%s" r="%d"/>'
                   % (f(cx), f(cy), R))
        frames = []
        for k, back, err in data:
            frames.append(' '.join('%s,%s' % (f(cx + R * x), f(cy - R * y))
                                   for x, y in back))
        frames += [frames[-1]] * hold
        anim = discrete('points', frames, dur)
        out.append('<polygon class="s%d" fill="none" stroke-width="1.6" '
                   'points="%s">%s</polygon>' % (p, frames[0], anim))
        out.append(text(cx, cy + R + 26, fmt.name, size=13, anchor='middle',
                        extra=' font-weight="600"'))
        out.append(text(cx, cy + R + 44, 'c² + s² − 1 = %.1e'
                        % (c * c + s * s - 1), cls='muted', size=11,
                        anchor='middle'))
    # k counter.
    kvals = [str(k) for k, _, _ in series[0][3]] + [str(KMAX)] * hold
    for idx, k in enumerate(sorted(set(int(v) for v in kvals))):
        vis = ['1' if int(v) == k else '0' for v in kvals]
        anim = discrete_sparse('opacity', vis, dur) or ''
        out.append('<g opacity="%s">%s%s</g>'
                   % (vis[0], anim, text(W / 2, 270, 'k = %d' % k,
                                         size=14, anchor='middle')))
    # Error chart, log scale.
    gx, gy, gw, gh = 90, 300, 570, 200
    lo, hi = -5, 1                      # decades
    out.append('<rect class="zero" x="%d" y="%d" width="%d" height="%d"/>'
               % (gx, gy, gw, gh))
    for d in range(lo, hi + 1):
        y = gy + gh - (d - lo) / (hi - lo) * gh
        out.append('<line class="axis" stroke-width="0.6" x1="%d" y1="%s" '
                   'x2="%d" y2="%s"/>' % (gx, f(y), gx + gw, f(y)))
        out.append(text(gx - 8, y + 4, '1e%d' % d, cls='muted', size=11,
                        anchor='end'))
    for k in range(0, KMAX + 1, 100):
        x = gx + k / KMAX * gw
        out.append(text(x, gy + gh + 18, str(k), cls='muted', size=11,
                        anchor='middle'))
    out.append(text(gx + gw / 2, gy + gh + 38, 'rotations applied, k',
                    cls='muted', size=12, anchor='middle'))
    out.append(text(gx - 56, gy + gh / 2, 'max | |Gᵏx| − 1 |', cls='muted',
                    size=12, anchor='middle',
                    extra=' transform="rotate(-90 %s %s)"'
                    % (f(gx - 56), f(gy + gh / 2))))

    def ymap(err):
        e = max(err, 10.0 ** lo)
        d = min(max(math.log10(e), lo), hi)
        return gy + gh - (d - lo) / (hi - lo) * gh

    ends = []
    for p, (fmt, c, s, data) in enumerate(series):
        pts = ' '.join('%s,%s' % (f(gx + k / KMAX * gw), f(ymap(err)))
                       for k, _, err in data)
        out.append('<polyline class="s%d" fill="none" stroke-width="1.8" '
                   'points="%s"/>' % (p, pts))
        ends.append([ymap(data[-1][2]) + 4, p, fmt.name.split(' ')[0]])
    # End labels, pushed apart where the lines finish close together.
    ends.sort()
    for a in range(1, len(ends)):
        ends[a][0] = max(ends[a][0], ends[a - 1][0] + 13)
    for y, p, name in ends:
        out.append(text(gx + gw + 6, y, name, cls='f%d' % p, size=11))
    # Moving cursor.
    xs = [f(gx + k / KMAX * gw) for k, _, _ in series[0][3]]
    xs += [xs[-1]] * hold
    out.append('<line class="accs" stroke-width="1" x1="%s" y1="%d" x2="%s" '
               'y2="%d">%s%s</line>'
               % (xs[0], gy, xs[0], gy + gh, discrete('x1', xs, dur),
                  discrete('x2', xs, dur)))
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------

def self_test():
    """Spot checks of the rounding against known values."""
    h = IEEELike('binary16', p=11, emax=15)
    assert h.round(Fraction(1) + Fraction(1, 2048)) == 1          # tie to even
    assert h.round(Fraction(1) + Fraction(3, 2048)) == 1 + Fraction(2, 1024)
    assert h.round(Fraction(65520)) == math.inf                 # overflow
    assert h.round(Fraction(65519)) == 65504
    assert h.round(Fraction(2) ** -24) == Fraction(2) ** -24     # subnormal
    assert h.round(Fraction(2) ** -26) == 0
    e4 = IEEELike('e4m3', p=4, emax=8, emin=-6, saturate=True, maxval=448)
    assert e4.round(Fraction(1000)) == 448
    assert e4.round(Fraction(2) ** -9) == Fraction(2) ** -9      # min subnormal
    assert e4.round(Fraction(2) ** -10) == 0                     # tie to even
    assert e4.round(Fraction(2) ** -6 - Fraction(2) ** -9) == \
        Fraction(2) ** -6 - Fraction(2) ** -9                    # max subnormal
    assert e4.round(Fraction(2) ** -7 + Fraction(2) ** -10) == \
        Fraction(2) ** -7                                        # spacing 2^-9
    p8 = Posit('p8', n=8, es=0)
    assert p8.round(Fraction(1)) == 1
    assert p8.round(Fraction(1000)) == 64                        # maxpos
    assert p8.round(Fraction(1, 1000)) == Fraction(1, 64)        # minpos
    p16 = Posit('p16', n=16, es=2)
    # Near 1, posit<16,2> has 11 fraction bits.
    assert p16.round(Fraction(1) + Fraction(1, 2048)) == 1 + Fraction(1, 2048)
    assert p16.round(Fraction(1) + Fraction(1, 4096)) == 1       # tie to even
    assert p16.round(Fraction(1) + Fraction(3, 4096)) == 1 + Fraction(2, 2048)
    assert p16.round(Fraction(2) ** 60) == Fraction(2) ** 56     # maxpos
    assert rsqrt_exact(h, Fraction(2)) == h.round(Fraction(1448, 1024))
    for fmt in FORMATS:
        for v in (Fraction(3, 7), Fraction(-5, 3), Fraction(1, 3) ** 9):
            r = fmt.round(v)
            assert fmt.round(r) == r, (fmt.name, v)


# --------------------------------------------------------------------------
# Test vectors for the interactive page
# --------------------------------------------------------------------------
#
# docs-site/src/components/gh/numformats.js rounds in JavaScript, from a
# double: each operation is done in double and the result rounded once to the
# format. For formats this narrow that is the same as rounding the exact
# result (Figueroa: double rounding is innocuous for +, -, *, / and sqrt when
# the wide format has at least 2p + 2 bits). These vectors, rounded here in
# exact arithmetic, are what docs-site/tests/numformats.test.mjs checks the
# port against.

VECTOR_FORMATS = [
    {'kind': 'ieee', 'p': 4, 'emax': 8, 'emin': -6, 'saturate': True,
     'maxval': 448},                                         # OCP E4M3
    {'kind': 'ieee', 'p': 3, 'emax': 15},                    # OCP E5M2
    {'kind': 'ieee', 'p': 8, 'emax': 127},                   # bfloat16
    {'kind': 'ieee', 'p': 11, 'emax': 15},                   # binary16
    {'kind': 'ieee', 'p': 5, 'emax': 7, 'subnormals': False},
    {'kind': 'posit', 'n': 8, 'es': 0},
    {'kind': 'posit', 'n': 8, 'es': 2},
    {'kind': 'posit', 'n': 16, 'es': 1},
    {'kind': 'posit', 'n': 16, 'es': 2},
    {'kind': 'posit', 'n': 16, 'es': 3},
]


def vector_format(spec):
    if spec['kind'] == 'posit':
        return Posit('posit', spec['n'], spec['es'])
    return IEEELike('ieee', spec['p'], spec['emax'],
                    subnormals=spec.get('subnormals', True),
                    saturate=spec.get('saturate', False),
                    maxval=spec.get('maxval'), emin=spec.get('emin'))


def representable(fmt):
    """The format's positive finite values, ascending."""
    if isinstance(fmt, Posit):
        return [fmt._decode(b) for b in range(1, 1 << (fmt.n - 1))]
    out = []
    if fmt.subnormals:
        q = Fraction(2) ** (fmt.emin - fmt.p + 1)
        out += [m * q for m in range(1, 1 << (fmt.p - 1))]
    for e in range(fmt.emin, fmt.emax + 1):
        q = Fraction(2) ** (e - fmt.p + 1)
        out += [m * q for m in range(1 << (fmt.p - 1), 1 << fmt.p)
                if m * q <= fmt.maxval]
    return out


def jnum(q):
    """A rounded value for JSON: a float, or a string for inf and NaN."""
    if isinstance(q, float) and not math.isfinite(q):
        return 'nan' if math.isnan(q) else ('inf' if q > 0 else '-inf')
    return float(q)


def ops_for(fmt):
    """Correctly rounded +, *, / and sqrt. An overflow to infinity (only the
    non-saturating IEEE formats have one) continues in IEEE float rules, as
    it does in JavaScript."""
    def special(*xs):
        return any(isinstance(x, float) and not math.isfinite(x) for x in xs)

    def mul(a, b):
        if special(a, b):
            return float(a) * float(b)
        return rnd(fmt, a * b)

    def add(a, b):
        if special(a, b):
            return float(a) + float(b)
        return rnd(fmt, a + b)

    def div(a, b):
        if special(a, b):
            fa, fb = float(a), float(b)
            return math.nan if math.isinf(fa) and math.isinf(fb) else fa / fb
        return rnd(fmt, Fraction(a) / b)

    def sqrt(a):
        if special(a):
            return math.sqrt(a) if a > 0 else math.nan
        return rsqrt_exact(fmt, a)
    return mul, add, div, sqrt


def givens_ref(fmt, a, b, scaled):
    """The rotation as docs-site/src/components/gh/kernels.js computes it."""
    mul, add, div, sqrt = ops_for(fmt)
    if scaled:
        sc = max(abs(a), abs(b))
        if sc == 0:
            return 1, 0, 0
        a1, b1 = div(a, sc), div(b, sc)
        r = mul(sc, sqrt(add(mul(a1, a1), mul(b1, b1))))
    else:
        r = sqrt(add(mul(a, a), mul(b, b)))
    if r == 0:
        return 1, 0, 0
    return div(a, r), div(b, r), r


def householder_ref(fmt, a, b):
    """MTL5's real householder() for n = 2: (v1, tau), or None for H = I."""
    mul, add, div, sqrt = ops_for(fmt)
    sc = max(abs(a), abs(b))
    if sc == 0:
        return None
    x1 = div(b, sc)
    sigma = mul(x1, x1)
    if sigma == 0:
        return None
    x0 = div(a, sc)
    norm = sqrt(add(mul(x0, x0), sigma))
    if x0 <= 0:
        v0 = add(x0, -norm)
    else:
        if add(x0, sqrt(sigma)) == x0:
            return None
        v0 = -div(sigma, add(x0, norm))
    tau = -div(v0, norm)
    if tau == 0:
        return None
    v1 = div(x1, v0)
    c = add(tau, mul(mul(tau, v1), v1))
    dev = mul(abs(add(2, -c)), c)
    if not dev <= div(sqrt(sigma), norm):
        return None
    return v1, tau


def make_vectors():
    import json
    import random
    rng = random.Random(665)
    out = {'note': 'generated by tools/givens_householder_svg.py; '
                   'do not edit', 'formats': []}
    for spec in VECTOR_FORMATS:
        fmt = vector_format(spec)
        vals = representable(fmt)
        lo, hi = float(vals[0]), float(vals[-1])
        xs = [0.0, lo / 4, lo / 2, lo, hi, hi * 1.0625, hi * 2, 1e30,
              float(vals[0] + vals[1]) / 2]
        for _ in range(60):
            xs.append(math.exp(rng.uniform(math.log(lo) - 2,
                                           math.log(hi) + 2)))
        for _ in range(30):
            i = rng.randrange(len(vals) - 1)
            mid = (vals[i] + vals[i + 1]) / 2
            xs.append(float(mid))
            xs.append(math.nextafter(float(mid), -math.inf))
            xs.append(math.nextafter(float(mid), math.inf))
        rows = []
        for j, x in enumerate(xs):
            x = -x if j % 2 else x
            rows.append([x, jnum(fmt.round(Fraction(x)))])
        sq = []
        for _ in range(20):
            x = float(vals[rng.randrange(len(vals))])
            sq.append([x, jnum(rsqrt_exact(fmt, Fraction(x)))])
        giv, hh = [], []
        for _ in range(24):
            a = rng.choice((-1, 1)) * vals[rng.randrange(len(vals))]
            b = rng.choice((-1, 1)) * vals[rng.randrange(len(vals))]
            if rng.random() < 0.3:
                b = b / 2 ** rng.randrange(4, 12)
                b = fmt.round(b) if b else b
            for scaled in (False, True):
                c, s, r = givens_ref(fmt, a, b, scaled)
                giv.append([float(a), float(b), scaled,
                            [jnum(x) for x in (c, s, r)]])
            ref = householder_ref(fmt, a, b)
            hh.append([float(a), float(b),
                       None if ref is None else [jnum(v) for v in ref]])
        c = rnd(fmt, math.cos(THETA))
        s = rnd(fmt, math.sin(THETA))
        pts = [(rnd(fmt, math.cos(2 * math.pi * i / NPTS)),
                rnd(fmt, math.sin(2 * math.pi * i / NPTS)))
               for i in range(0, NPTS, 6)]
        start = [[float(x), float(y)] for x, y in pts]
        for _ in range(KSTEP):
            pts = [(rnd(fmt, rnd(fmt, c * x) - rnd(fmt, s * y)),
                    rnd(fmt, rnd(fmt, s * x) + rnd(fmt, c * y)))
                   for x, y in pts]
        out['formats'].append({
            'spec': spec, 'count': len(vals), 'round': rows, 'sqrt': sq,
            'givens': giv, 'householder': hh,
            'drift': {'c': float(c), 's': float(s), 'k': KSTEP,
                      'start': start,
                      'end': [[jnum(x), jnum(y)] for x, y in pts]}})
    return json.dumps(out, indent=None, separators=(',', ':')) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--out', default=os.path.join(here, '..', 'docs', 'img'),
                    help='output directory (default: docs/img)')
    ap.add_argument('--vectors',
                    default=os.path.join(here, '..', 'docs-site', 'tests',
                                         'gh-vectors.json'),
                    help='test vectors for the interactive page '
                         '(default: docs-site/tests/gh-vectors.json)')
    args = ap.parse_args()
    self_test()
    figs = {
        'gh-plane.svg': fig_plane,
        'gh-qr-fill.svg': fig_qr,
        'gh-bulge.svg': fig_bulge,
        'gh-drift.svg': fig_drift,
    }
    outputs = [(os.path.join(args.out, name), fn)
               for name, fn in figs.items()]
    outputs.append((args.vectors, make_vectors))
    for path, fn in outputs:
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(fn())
        print('wrote %s (%d bytes)' % (os.path.normpath(path),
                                       os.path.getsize(path)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
