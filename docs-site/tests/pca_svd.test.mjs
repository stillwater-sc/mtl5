// 6B: the projections the animation draws, against pca_svd.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pcaModel } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('pca_svd');
const model = pcaModel(trace);
const close = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

test('the data is centered and V is orthonormal', () => {
  for (let j = 0; j < trace.features; ++j) assert.ok(Math.abs(trace.centered.reduce((a, r) => a + r[j], 0)) < 1e-9);
  for (let a = 0; a < trace.features; ++a)
    for (let b = 0; b < trace.features; ++b) {
      const d = trace.V.reduce((s, r) => s + r[a] * r[b], 0);
      assert.ok(Math.abs(d - (a === b ? 1 : 0)) < 1e-7);
    }
});

test('the scores are Xc V, and their column norms are the singular values', () => {
  trace.centered.forEach((row, i) => {
    for (let k = 0; k < trace.features; ++k) assert.ok(close(row.reduce((s, x, j) => s + x * trace.V[j][k], 0), trace.scores[i][k]));
  });
  trace.sigma.forEach((s, k) => assert.ok(close(Math.hypot(...trace.scores.map((r) => r[k])), s)));
});

test('Eckart-Young: the rank-r error is the norm of the dropped singular values', () => {
  for (const r of trace.reconstructions) {
    const tail = Math.hypot(...trace.sigma.slice(r.rank));
    assert.ok(Math.abs(r.error_fro - tail) < 1e-6, `rank ${r.rank}`);
  }
});

test('the rank-1 reconstruction lies on the first principal direction', () => {
  const r1 = trace.reconstructions.find((r) => r.rank === 1);
  const v = trace.V.map((row) => row[0]);
  for (const x of r1.data) {
    const t = x.reduce((s, xi, j) => s + xi * v[j], 0);
    x.forEach((xi, j) => assert.ok(Math.abs(xi - t * v[j]) < 1e-6));
  }
});

test('frames and the variance shares', () => {
  assert.deepEqual(model.frames.map((f) => f.kind), ['data', 'scores', 'recon', 'recon']);
  assert.deepEqual(model.frames.slice(2).map((f) => f.rank), [2, 1]);
  assert.ok(Math.abs(model.explained.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(model.explained[0] > 0.9);
});
