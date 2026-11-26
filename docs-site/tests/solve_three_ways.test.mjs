// 5B: the elimination states the animation draws, against solve_three_ways.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { factorModel, factorFrame } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('solve_three_ways');
const model = factorModel(trace);
const n = trace.n;
const get = (part, name) => part.methods.find((m) => m.method === name);
const close = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

function matmul(A, B) {
  return A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
}

for (const part of trace.parts) {
  test(`${part.name}: every method starts from the matrix`, () => {
    for (const m of part.methods) {
      const s0 = m.states[0];
      for (let i = 0; i < n; ++i)
        for (let j = 0; j < n; ++j) if (m.method !== 'Cholesky' || j <= i) assert.ok(close(s0[i][j], part.matrix[i][j]), `${m.method} ${i},${j}`);
    }
  });

  test(`${part.name}: the final LU state is a factorization of the row-swapped matrix`, () => {
    const lu = get(part, 'LU');
    const F = lu.states[n];
    const L = F.map((r, i) => r.map((v, j) => (j < i ? v : j === i ? 1 : 0)));
    const U = F.map((r, i) => r.map((v, j) => (j >= i ? v : 0)));
    const P = part.matrix.map((r) => [...r]);
    lu.pivot.forEach((p, k) => ([P[k], P[p]] = [P[p], P[k]]));
    const LU = matmul(L, U);
    for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) assert.ok(close(LU[i][j], P[i][j]), `${i},${j}`);
  });

  test(`${part.name}: QR states zero one more column per step`, () => {
    const qr = get(part, 'QR');
    for (let k = 1; k < n; ++k) {
      const s = qr.states[k];
      for (let j = 0; j < k; ++j) for (let i = j + 1; i < n; ++i) assert.ok(Math.abs(s[i][j]) < 1e-9, `step ${k}, ${i},${j}`);
    }
  });
}

test('SPD: L L^T reproduces the matrix, and all three factor it', () => {
  const spd = trace.parts[0];
  const ch = get(spd, 'Cholesky');
  assert.equal(ch.info, 0);
  const L = ch.states[n].map((r, i) => r.map((v, j) => (j <= i ? v : 0)));
  const LLt = matmul(L, L.map((_, i) => L.map((r) => r[i])));
  for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) assert.ok(close(LLt[i][j], spd.matrix[i][j]));
  for (const m of spd.methods) assert.equal(m.info, 0, m.method);
});

test('indefinite: Cholesky fails at its non-positive pivot; LU and QR finish', () => {
  const ind = trace.parts[1];
  const ch = get(ind, 'Cholesky');
  assert.notEqual(ch.info, 0);
  assert.ok(ch.failed_pivot <= 0);
  const k = ch.info - 1;
  assert.ok(close(ch.states[k][k][k], ch.failed_pivot));
  assert.equal(get(ind, 'LU').info, 0);
  assert.equal(get(ind, 'QR').info, 0);
  // The frames show it failed from the step it could not take.
  const f = model.frames.findIndex((fr) => fr.part === ind && fr.k === ch.info);
  assert.equal(factorFrame(model, f).methods.find((m) => m.method === 'Cholesky').failed, true);
});

test('flop counts never fall (LU\'s last step costs 0) and approach 2 : 1 : 4 for large n', () => {
  for (const m of trace.parts[0].methods) for (let k = 1; k < m.flops.length; ++k) assert.ok(m.flops[k] >= m.flops[k - 1]);
  const big = trace.flops_by_n[trace.flops_by_n.length - 1];
  assert.ok(Math.abs(big.lu / big.cholesky - 2) < 0.01);
  assert.ok(Math.abs(big.qr / big.cholesky - 4) < 0.01);
});
