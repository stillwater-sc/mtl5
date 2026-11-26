// 5A: the fits the animation draws, against least_squares_qr.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('least_squares_qr');
const fits = trace.fits;

test('one fit per degree, 1 .. m - 2, each curve on the fine grid', () => {
  assert.deepEqual(fits.map((f) => f.degree), Array.from({ length: trace.m - 2 }, (_, i) => i + 1));
  for (const f of fits) {
    assert.equal(f.curve_qr.length, trace.x_fine.length);
    if (f.cholesky_info === 0) assert.equal(f.curve_normal.length, trace.x_fine.length);
    else assert.equal(f.curve_normal, null);
  }
});

test('QR residual never rises with degree (nested least-squares problems)', () => {
  for (let i = 1; i < fits.length; ++i) assert.ok(fits[i].residual_qr <= fits[i - 1].residual_qr * (1 + 1e-6), `degree ${fits[i].degree}`);
});

test('the normal equations match QR at low degree, fall behind later, and Cholesky fails at the end', () => {
  for (const f of fits.slice(0, 5)) assert.ok(Math.abs(f.residual_normal - f.residual_qr) <= 1e-6 * f.residual_qr);
  const worse = fits.filter((f) => f.residual_normal !== null && f.residual_normal > 2 * f.residual_qr);
  assert.ok(worse.length > 0);
  assert.notEqual(fits[fits.length - 1].cholesky_info, 0);
});

test('cond(V) grows with degree; the eigenvalue estimate of cond(VtV) saturates below cond(V)^2', () => {
  for (let i = 1; i < fits.length; ++i) assert.ok(fits[i].cond_V_svd > fits[i - 1].cond_V_svd);
  // Where cond(V)^2 is well inside double range the estimate tracks it ...
  const f3 = fits.find((f) => f.degree === 3);
  assert.ok(Math.abs(f3.cond_VtV_eig / f3.cond_V_svd ** 2 - 1) < 0.01);
  // ... and at the top it falls short by orders of magnitude.
  const top = fits[fits.length - 1];
  assert.ok(top.cond_VtV_eig < 1e-6 * top.cond_V_svd ** 2);
});

test('the QR curve passes near the data at the data points', () => {
  const f = fits.find((g) => g.degree === 8);
  // x_fine runs 0..pi in 120 steps, and the data sits at multiples of pi/19,
  // so compare at the endpoints, which both grids contain.
  assert.ok(Math.abs(f.curve_qr[0] - trace.y_data[0]) < 0.05);
  assert.ok(Math.abs(f.curve_qr[f.curve_qr.length - 1] - trace.y_data[trace.m - 1]) < 0.05);
});
