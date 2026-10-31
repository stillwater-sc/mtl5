// The 2 x 2 Givens rotation and Householder reflector, computed in a narrow
// format with the same operation sequences MTL5 uses. `ops` is arith(F) from
// numformats.js. The Python reference (givens_ref and householder_ref in
// tools/givens_householder_svg.py) follows the same sequences.

/**
 * The rotation G = [c s; -s c] with G (a, b) = (r, 0).
 *
 * scaled = false is MTL5's apply_givens_rotation (operation/givens.hpp):
 * r = sqrt(a*a + b*b), which overflows or underflows when a or b is far
 * from 1. scaled = true divides by max(|a|, |b|) first, as lapy2 and
 * MTL5's householder() do.
 */
export function givens(ops, a, b, scaled) {
  let r;
  if (scaled) {
    const sc = Math.max(Math.abs(a), Math.abs(b));
    if (sc === 0) return { c: 1, s: 0, r: 0 };
    const a1 = ops.div(a, sc);
    const b1 = ops.div(b, sc);
    r = ops.mul(sc, ops.sqrt(ops.add(ops.mul(a1, a1), ops.mul(b1, b1))));
  } else {
    r = ops.sqrt(ops.add(ops.mul(a, a), ops.mul(b, b)));
  }
  if (r === 0) return { c: 1, s: 0, r: 0 };
  return { c: ops.div(a, r), s: ops.div(b, r), r };
}

/** G applied to (u, w), rounded as givens.hpp does. */
export function applyGivens(ops, { c, s }, [u, w]) {
  return [ops.add(ops.mul(c, u), ops.mul(s, w)), ops.sub(ops.mul(c, w), ops.mul(s, u))];
}

/**
 * MTL5's real householder() (operation/householder.hpp) for n = 2:
 * v = (1, v1) and tau with H = I - tau v v^T and H (a, b) = (||x||, 0) for
 * a > 0 (Parlett's form). Returns {identity: true, reason} where MTL5
 * returns H = I.
 */
export function householder(ops, a, b) {
  const sc = Math.max(Math.abs(a), Math.abs(b));
  if (sc === 0) return { identity: true, reason: 'x is the zero vector' };
  const x1 = ops.div(b, sc);
  const sigma = ops.mul(x1, x1);
  if (sigma === 0) return { identity: true, reason: 'the scaled tail squares to zero' };
  const x0 = ops.div(a, sc);
  const norm = ops.sqrt(ops.add(ops.mul(x0, x0), sigma));
  let v0;
  if (x0 <= 0) {
    v0 = ops.sub(x0, norm);
  } else {
    if (ops.add(x0, ops.sqrt(sigma)) === x0)
      return { identity: true, reason: 'b is below half an ulp of a' };
    v0 = -ops.div(sigma, ops.add(x0, norm));
  }
  const tau = -ops.div(v0, norm);
  if (tau === 0) return { identity: true, reason: 'tau rounds to zero' };
  const v1 = ops.div(x1, v0);
  // MTL5 keeps the reflector only if it is closer to orthogonal than the
  // identity is to correct: |2 - c| c <= ||tail|| / ||x||, c = tau v^T v.
  const c = ops.add(tau, ops.mul(ops.mul(tau, v1), v1));
  const dev = ops.mul(Math.abs(ops.sub(2, c)), c);
  const tail = ops.div(ops.sqrt(sigma), norm);
  if (!(dev <= tail))
    return {
      identity: true,
      reason: `tau vᵀv = ${fmtNum(c)} is too far from 2: |2 − c|·c = ${fmtNum(dev)} exceeds |b|/‖x‖ = ${fmtNum(tail)}`,
    };
  return { identity: false, v1, tau, check: c };
}

/**
 * H applied to (u, w) the way qr_factor applies it (apply_householder_left):
 * d = v^T y, then y(i) -= (tau * v(i)) * d. With v0 = 1 the first entry is
 * u - tau d; the second rounds tau v1 before multiplying by d.
 */
export function applyHouseholder(ops, h, [u, w]) {
  if (h.identity) return [u, w];
  const d = ops.add(u, ops.mul(h.v1, w));
  return [ops.sub(u, ops.mul(h.tau, d)), ops.sub(w, ops.mul(ops.mul(h.tau, h.v1), d))];
}

/** ||Q^T Q - I||_2 of a 2 x 2 matrix with entries m = [m00, m01, m10, m11]. */
export function orthogonalityError([m00, m01, m10, m11]) {
  // Q^T Q - I is symmetric; its 2-norm is its largest |eigenvalue|.
  const p = m00 * m00 + m10 * m10 - 1;
  const q = m00 * m01 + m10 * m11;
  const r = m01 * m01 + m11 * m11 - 1;
  const mid = (p + r) / 2;
  const rad = Math.hypot((p - r) / 2, q);
  return Math.max(Math.abs(mid + rad), Math.abs(mid - rad));
}

/** Compact display of a number. */
export function fmtNum(x, digits = 4) {
  if (Number.isNaN(x)) return 'NaN';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
  if (x === 0) return '0';
  const a = Math.abs(x);
  // Integers that fit are shown whole: binary16's largest value is 65504.
  const s = Number.isInteger(x) && a < 1e7 ? String(x)
    : a >= 1e-3 && a < 1e5 ? String(Number(x.toPrecision(digits)))
    : x.toExponential(digits - 1);
  return s.replace('-', '−');
}
