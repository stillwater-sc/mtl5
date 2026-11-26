// 5B: the elimination states the animation draws, against solve_three_ways.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { factorModel, factorFrame, hilbertModel } from '../src/components/examples/lib/models.js';
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

// ── Part 3, the Hilbert sweep (issue 937) ─────────────────────────────────

const eps = 2.220446049250313e-16;
const hilbert = trace.hilbert;
const of = (r, name) => r.methods.find((m) => m.method === name);

test('the matrices are the generators the caption names', () => {
  const [spd, ind] = trace.parts;
  // lehmer(4): (min + 1) / (max + 1)
  for (let i = 0; i < n; ++i)
    for (let j = 0; j < n; ++j) assert.ok(close(spd.matrix[i][j], (Math.min(i, j) + 1) / (Math.max(i, j) + 1)), `lehmer ${i},${j}`);
  // wilkinson(4): |i - 3/2| on the diagonal, 1 beside it
  for (let i = 0; i < n; ++i)
    for (let j = 0; j < n; ++j) {
      const w = i === j ? Math.abs(i - 1.5) : Math.abs(i - j) === 1 ? 1 : 0;
      assert.ok(close(ind.matrix[i][j], w), `wilkinson ${i},${j}`);
    }
});

test('Hilbert: one row per order 4 .. 15, every method listed', () => {
  assert.deepEqual(hilbert.map((r) => r.n), Array.from({ length: 12 }, (_, i) => i + 4));
  for (const r of hilbert) assert.deepEqual(r.methods.map((m) => m.method), ['LU', 'Cholesky', 'QR', 'inv(H) * b']);
});

test('Hilbert: the factorizations are backward stable at every order', () => {
  for (const r of hilbert)
    for (const name of ['LU', 'Cholesky', 'QR']) {
      const b = of(r, name).backward;
      if (b !== null) assert.ok(b <= 10 * eps, `${name} at n = ${r.n}: ${b}`);
    }
});

test('Hilbert: the inverse is not backward stable, and its error grows by orders of magnitude', () => {
  const inv = hilbert.map((r) => of(r, 'inv(H) * b').backward);
  assert.ok(Math.max(...inv) > 1e6 * inv[0], 'grows by more than 1e6');
  assert.ok(Math.max(...inv) > 1e8 * 10 * eps, 'far from eps');
  for (const r of hilbert) assert.ok(of(r, 'inv(H) * b').backward > of(r, 'LU').backward, `n = ${r.n}`);
});

test('Hilbert: forward errors grow with cond(H); LU stays below cond(H) eps where cond is resolved; the inverse is worse', () => {
  const resolved = hilbert.filter((r) => r.cond < 1e16);
  assert.ok(resolved.length >= 8);
  for (const r of resolved) assert.ok(of(r, 'LU').forward <= r.cond * eps, `n = ${r.n}`);
  for (let i = 1; i < resolved.length; ++i) assert.ok(resolved[i].cond > resolved[i - 1].cond);
  for (const r of hilbert) assert.ok(of(r, 'inv(H) * b').forward > of(r, 'LU').forward, `n = ${r.n}`);
});

test('Hilbert: Cholesky succeeds below its first failure and fails from it on, past cond = 1/eps', () => {
  const first = trace.hilbert_cholesky_first_fail;
  assert.ok(first > 4 && first <= 15);
  for (const r of hilbert) {
    if (r.n < first) {
      assert.equal(r.cholesky_info, 0, `n = ${r.n}`);
      assert.notEqual(of(r, 'Cholesky').backward, null);
    } else {
      assert.notEqual(r.cholesky_info, 0, `n = ${r.n}`);
      assert.equal(of(r, 'Cholesky').backward, null);
    }
  }
  // The caption says it still succeeds one order earlier, past 1/eps.
  assert.ok(hilbert.find((r) => r.n === first - 1).cond > 1 / eps);
});

test('Hilbert model: series and references line up with the rows', () => {
  const m = hilbertModel(trace);
  assert.equal(m.count, hilbert.length);
  assert.equal(m.series.length, 4);
  for (const s of m.series) {
    assert.equal(s.backward.length, hilbert.length);
    assert.equal(s.forward.length, hilbert.length);
  }
  hilbert.forEach((r, i) => assert.ok(close(m.condEps[i], r.cond * eps)));
});
