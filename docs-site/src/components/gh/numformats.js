// Narrow number formats for the Givens/Householder explorer.
//
// Every value is held as a JavaScript double. An operation is done in double
// and its result rounded once to the format. For formats of 16 bits or fewer
// this equals rounding the exact result: double rounding is innocuous for
// +, -, *, / and sqrt when the wide format has at least 2p + 2 significand
// bits (S. A. Figueroa, "When is double rounding innocuous?", SIGNUM
// Newsletter 30(3), 1995), and a 16-bit posit has at most 14. Rounding a
// double to the format is itself exact: scaling by a power of two and
// splitting off an integer part lose nothing.
//
// The algorithms follow IEEELike and Posit in tools/givens_householder_svg.py,
// which rounds in exact rational arithmetic and writes the test vectors in
// docs-site/tests/gh-vectors.json that this file is checked against.

/** floor(log2(a)) for a positive finite double, exactly. */
export function floorLog2(a) {
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1;
  else if (2 ** (e + 1) <= a) e += 1;
  return e;
}

/** Round the nonnegative double m to an integer, ties to even. */
function roundHalfEven(m) {
  const fl = Math.floor(m);
  const rem = m - fl;
  if (rem > 0.5 || (rem === 0.5 && fl % 2 === 1)) return fl + 1;
  return fl;
}

/**
 * A binary floating-point format with p significand bits (hidden bit
 * included) and exponents emin .. emax, emin = 1 - emax unless given.
 * Overflow gives infinity, or maxval when saturate is set.
 */
export function ieeeLike({ p, emax, emin, subnormals = true, saturate = false, maxval }) {
  const lo = emin ?? 1 - emax;
  const top = maxval ?? (2 - 2 ** (1 - p)) * 2 ** emax;
  return {
    kind: 'ieee', p, emax, emin: lo, subnormals, saturate, maxval: top,
    minval: subnormals ? 2 ** (lo - p + 1) : 2 ** lo,
    round(x) {
      if (x === 0 || Number.isNaN(x)) return x;
      const sign = x < 0 ? -1 : 1;
      if (!Number.isFinite(x)) return saturate ? sign * top : x;
      const a = Math.abs(x);
      const s = floorLog2(a);
      if (s < lo && !subnormals) {
        const tiny = 2 ** lo;
        return sign * (a >= tiny / 2 ? tiny : 0);
      }
      let qexp = s - p + 1;
      if (subnormals) qexp = Math.max(qexp, lo - p + 1);
      const r = roundHalfEven(a / 2 ** qexp) * 2 ** qexp;
      if (r > top) return saturate ? sign * top : sign * Infinity;
      return sign * r;
    },
    /** The positive finite values, ascending. */
    values() {
      const out = [];
      if (subnormals) {
        const q = 2 ** (lo - p + 1);
        for (let m = 1; m < 2 ** (p - 1); ++m) out.push(m * q);
      }
      for (let e = lo; e <= emax; ++e) {
        const q = 2 ** (e - p + 1);
        for (let m = 2 ** (p - 1); m < 2 ** p; ++m) if (m * q <= top) out.push(m * q);
      }
      return out;
    },
  };
}

/**
 * posit<n, es>, rounded in its encoding as the posit standard requires: the
 * exact value's bit string is rounded to n - 1 bits, ties to even, and never
 * to zero or NaR. Out-of-range values go to minpos or maxpos.
 */
export function posit({ n, es }) {
  const useed = 2 ** es;
  const maxpos = 2 ** ((n - 2) * useed);
  const minpos = 1 / maxpos;
  const w = n - 1;

  function decode(bits) {
    const s = bits.toString(2).padStart(w, '0');
    const r0 = s[0];
    let run = 0;
    while (run < s.length && s[run] === r0) ++run;
    const k = r0 === '1' ? run - 1 : -run;
    const rest = s.slice(run + 1);
    const ebits = rest.slice(0, es).padEnd(es, '0');
    const e = es ? parseInt(ebits, 2) : 0;
    const fbits = rest.slice(es);
    const f = fbits.length ? parseInt(fbits, 2) / 2 ** fbits.length : 0;
    return 2 ** (k * useed + e) * (1 + f);
  }

  return {
    kind: 'posit', n, es, maxval: maxpos, minval: minpos,
    round(x) {
      if (x === 0 || Number.isNaN(x)) return x;
      const sign = x < 0 ? -1 : 1;
      const a = Math.abs(x);
      if (a >= maxpos) return sign * maxpos;
      if (a <= minpos) return sign * minpos;
      const s = floorLog2(a);
      const k = Math.floor(s / useed);
      const e = s - k * useed;
      const regime = k >= 0 ? '1'.repeat(k + 1) + '0' : '0'.repeat(-k) + '1';
      const head = BigInt('0b' + regime + (es ? e.toString(2).padStart(es, '0') : ''));
      const L = regime.length + es;
      const T = Math.max(w - L, 0) + 2;
      const frac = (a / 2 ** s - 1) * 2 ** T;
      const fint = Math.floor(frac);
      let sticky = frac !== fint;
      const full = (head << BigInt(T)) | BigInt(fint);
      const shift = BigInt(L + T - w);
      let kept = full >> shift;
      const rest = full & ((1n << shift) - 1n);
      const guard = (rest >> (shift - 1n)) & 1n;
      sticky = sticky || (rest & ((1n << (shift - 1n)) - 1n)) !== 0n;
      if (guard && (sticky || (kept & 1n))) kept += 1n;
      const lim = (1n << BigInt(w)) - 1n;
      if (kept < 1n) kept = 1n;
      if (kept > lim) kept = lim;
      return sign * decode(Number(kept));
    },
    values() {
      const out = [];
      for (let b = 1; b < 2 ** w; ++b) out.push(decode(b));
      return out;
    },
  };
}

/** double itself, as the reference. */
export function binary64() {
  return {
    kind: 'double', maxval: Number.MAX_VALUE, minval: Number.MIN_VALUE,
    round: (x) => x,
    values: () => null,
  };
}

/** A format from a spec: {kind: 'ieee' | 'posit' | 'double', ...}. */
export function makeFormat(spec) {
  if (spec.kind === 'posit') return posit(spec);
  if (spec.kind === 'double') return binary64();
  return ieeeLike(spec);
}

/** Correctly rounded arithmetic in format F. */
export function arith(F) {
  const r = F.round;
  return {
    F,
    rnd: r,
    add: (a, b) => r(a + b),
    sub: (a, b) => r(a - b),
    mul: (a, b) => r(a * b),
    div: (a, b) => r(a / b),
    sqrt: (a) => r(Math.sqrt(a)),
  };
}
