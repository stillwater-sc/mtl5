// Plot helpers for the example animations (#933). The functions above the
// DOM section are pure, so docs-site/tests/ can check them under node; the
// DOM section builds SVG elements in the browser.

/** A linear map from [d0, d1] to [r0, r1]. */
export function scaleLinear([d0, d1], [r0, r1]) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = (x) => r0 + (x - d0) * k;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

/**
 * A base-10 log map from [d0, d1] (both > 0) to [r0, r1]. Values at or
 * below zero, and non-finite values, map to the bottom of the range, so a
 * residual that reaches exactly 0 is drawn on the axis instead of vanishing.
 */
export function scaleLog([d0, d1], [r0, r1]) {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const k = l1 === l0 ? 0 : (r1 - r0) / (l1 - l0);
  const f = (x) => {
    if (!(x > 0) || !Number.isFinite(x)) return r0;
    return r0 + (Math.log10(x) - l0) * k;
  };
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

/** Decades spanning [lo, hi]: the powers of ten from floor to ceil. */
export function logDomain(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v > 0 && Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) return [0.1, 1];
  const a = 10 ** Math.floor(Math.log10(lo));
  let b = 10 ** Math.ceil(Math.log10(hi));
  if (b === a) b = a * 10;
  return [a, b];
}

/** The powers of ten in [d0, d1], thinned to at most `max` ticks. */
export function logTicks([d0, d1], max = 7) {
  const e0 = Math.ceil(Math.log10(d0) - 1e-9);
  const e1 = Math.floor(Math.log10(d1) + 1e-9);
  const count = e1 - e0 + 1;
  const stride = Math.max(1, Math.ceil(count / max));
  const ticks = [];
  for (let e = e0; e <= e1; e += stride) ticks.push(10 ** e);
  return ticks;
}

/** About `count` round ticks (1, 2 or 5 times a power of ten) over [d0, d1]. */
export function niceTicks([d0, d1], count = 5) {
  if (d1 === d0) return [d0];
  const span = d1 - d0;
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const first = Math.ceil(d0 / step - 1e-9);
  const last = Math.floor(d1 / step + 1e-9);
  const ticks = [];
  // Multiply rather than accumulate, and round off the binary noise, so the
  // ticks are 0.6 and not 0.6000000000000001.
  for (let i = first; i <= last; ++i) ticks.push(Number((i * step).toPrecision(12)) || 0);
  return ticks;
}

/** The smallest and largest finite values, padded when they are equal. */
export function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  return [lo, hi];
}

/** An SVG path through the points (xs[i], ys[i]) under the two scales. */
export function linePath(xs, ys, sx, sy) {
  let d = '';
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; ++i) {
    const x = sx(xs[i]);
    const y = sy(ys[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    d += `${d ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }
  return d;
}

/** Linear interpolation, and the same for two equal-length arrays. */
export const lerp = (a, b, t) => a + (b - a) * t;
export function lerpArray(a, b, t) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; ++i) out[i] = lerp(a[i], b[i] ?? a[i], t);
  return out;
}

/** Smooth start and stop for tweens: 3t^2 - 2t^3. */
export const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

/** A number for a label: fixed for ordinary sizes, otherwise 1.2e-5 style. */
export function fmt(x, digits = 3) {
  if (x === 0) return '0';
  if (!Number.isFinite(x)) return String(x);
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e5) return Number(x.toPrecision(digits)).toString();
  const [m, e] = x.toExponential(Math.max(0, digits - 1)).split('e');
  return `${Number(m)}e${Number(e)}`;
}

/** 10^e as a label: 1, 10, 100, 1e-3 ... */
export function fmtPow10(x) {
  const e = Math.round(Math.log10(x));
  if (e >= 0 && e <= 3) return String(10 ** e);
  return `1e${e}`;
}

// ── DOM ──────────────────────────────────────────────────────────────────

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element with attributes and optional text. */
export function svg(tag, attrs = {}, text) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) el.setAttribute(k, String(v));
  }
  if (text !== undefined) el.textContent = String(text);
  return el;
}

/**
 * Draw x and y axes with ticks into group `g` for the box
 * {left, right, top, bottom}. `xTicks`/`yTicks` are values; `xFmt`/`yFmt`
 * label them.
 */
export function drawAxes(g, box, sx, sy, { xTicks = [], yTicks = [], xFmt = fmt, yFmt = fmt, xLabel, yLabel } = {}) {
  const { left, right, top, bottom } = box;
  g.append(svg('line', { x1: left, y1: bottom, x2: right, y2: bottom, class: 'ex-axis' }));
  g.append(svg('line', { x1: left, y1: top, x2: left, y2: bottom, class: 'ex-axis' }));
  for (const t of xTicks) {
    const x = sx(t);
    g.append(svg('line', { x1: x, y1: bottom, x2: x, y2: bottom + 4, class: 'ex-axis' }));
    g.append(svg('text', { x, y: bottom + 15, 'text-anchor': 'middle', class: 'ex-tick' }, xFmt(t)));
  }
  for (const t of yTicks) {
    const y = sy(t);
    g.append(svg('line', { x1: left - 4, y1: y, x2: left, y2: y, class: 'ex-axis' }));
    g.append(svg('line', { x1: left, y1: y, x2: right, y2: y, class: 'ex-grid' }));
    g.append(svg('text', { x: left - 6, y: y + 3.5, 'text-anchor': 'end', class: 'ex-tick' }, yFmt(t)));
  }
  if (xLabel) {
    g.append(svg('text', { x: (left + right) / 2, y: bottom + 30, 'text-anchor': 'middle', class: 'ex-label' }, xLabel));
  }
  if (yLabel) {
    g.append(svg('text', { x: left, y: top - 8, 'text-anchor': 'start', class: 'ex-label' }, yLabel));
  }
}

/** Remove every child of an element. */
export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}
