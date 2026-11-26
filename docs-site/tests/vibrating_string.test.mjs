// 6A: the modes and spectra the animation draws, against vibrating_string.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { stringModel, relativeFrequency, skewAnalytic } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('vibrating_string');
const model = stringModel(trace);
const n = trace.n;

test('the computed eigenvalues are the analytic 4 sin^2(k pi / 2(n+1))', () => {
  for (let k = 0; k < n; ++k) {
    // stored to 12 significant digits
    assert.ok(Math.abs(trace.eigenvalues[k] - 4 * Math.sin(((k + 1) * Math.PI) / (2 * (n + 1))) ** 2) < 1e-10);
    assert.ok(Math.abs(trace.eigenvalues[k] - trace.analytic[k]) < 1e-10);
  }
});

test('each mode is a unit eigenvector of T: T v = lambda v', () => {
  trace.modes.forEach((v, k) => {
    const norm = Math.hypot(...v);
    assert.ok(Math.abs(norm - 1) < 1e-5, `mode ${k}`);
    for (let i = 0; i < n; ++i) {
      const Tv = 2 * v[i] - (v[i - 1] ?? 0) - (v[i + 1] ?? 0);
      assert.ok(Math.abs(Tv - trace.eigenvalues[k] * v[i]) < 1e-5, `mode ${k}, entry ${i}`);
    }
  });
});

test('mode k has k - 1 sign changes, and its frequency is sqrt(lambda_k / lambda_1) times the fundamental', () => {
  trace.modes.forEach((v, k) => {
    // Count sign changes between the entries that are not nodes (exact zeros).
    const signs = v.filter((x) => Math.abs(x) > 1e-9).map(Math.sign);
    let changes = 0;
    for (let i = 1; i < signs.length; ++i) if (signs[i] !== signs[i - 1]) ++changes;
    assert.equal(changes, k);
  });
  assert.equal(relativeFrequency(model, 0), 1);
  assert.ok(relativeFrequency(model, n - 1) > relativeFrequency(model, 1));
});

test('the membrane eigenvalues are the sums lambda_i + lambda_j of the 4-point string', () => {
  const m = trace.n2d;
  const l1 = Array.from({ length: m }, (_, k) => 4 * Math.sin(((k + 1) * Math.PI) / (2 * (m + 1))) ** 2);
  const sums = l1.flatMap((a) => l1.map((b) => a + b)).sort((a, b) => a - b);
  trace.membrane_eigenvalues.forEach((l, k) => assert.ok(Math.abs(l - sums[k]) < 1e-9, `${k}`));
});

test('the lopsided sweep: real for d <= 1, complex pairs on the formula beyond', () => {
  for (const s of trace.skew_sweep) {
    const order = (a, b) => a[0] - b[0] || a[1] - b[1];
    const exact = skewAnalytic(n, s.d).map(([re, im]) => [re, Math.abs(im)]).sort(order);
    const got = s.re.map((re, i) => [re, Math.abs(s.im[i])]).sort(order);
    got.forEach(([re, im], i) => {
      assert.ok(Math.abs(re - exact[i][0]) < 1e-6 && Math.abs(im - exact[i][1]) < 1e-6, `d = ${s.d}, eigenvalue ${i}`);
    });
    const im = Math.max(...s.im.map(Math.abs));
    if (s.d <= 1) assert.equal(im, 0, `d = ${s.d}`);
    else assert.ok(im > 0, `d = ${s.d}`);
  }
});

test('frames: string modes, membrane modes, then the sweep', () => {
  assert.equal(model.count, n + trace.n2d ** 2 + trace.skew_sweep.length);
  assert.equal(model.frames[n].kind, 'membrane');
});
