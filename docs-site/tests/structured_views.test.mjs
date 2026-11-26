// 7B: the entries each view reads, against structured_views.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { viewsModel, entriesRead } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('structured_views');
const model = viewsModel(trace);
const n = trace.n;
const view = (name) => trace.views.find((v) => v.name === name);

test('every view value is the stored entry it reads, or 0, or 1 for the identity', () => {
  for (const v of trace.views) {
    const base = v.base === 'A_upper' ? trace.A_upper : trace.A_full;
    for (let i = 0; i < v.rows; ++i)
      for (let j = 0; j < v.cols; ++j) {
        const s = v.sources[i * v.cols + j];
        const want = s ? base[s[0]][s[1]] : v.name === 'identity' ? (i === j ? 1 : 0) : 0;
        assert.equal(v.values[i][j], want, `${v.name} ${i},${j}`);
      }
  }
});

test('the Hermitian view reads only the upper triangle, the mirrored entries twice', () => {
  const h = view('Hermitian');
  for (const s of h.sources) assert.ok(s[0] <= s[1]);
  assert.equal(entriesRead(h), (n * (n + 1)) / 2);
  for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) assert.equal(h.values[i][j], h.values[j][i]);
});

test('entries read per view', () => {
  assert.equal(entriesRead(view('tridiagonal band')), 3 * n - 2);
  assert.equal(entriesRead(view('diagonal')), n);
  assert.equal(entriesRead(view('sub-matrix')), 9);
  assert.equal(entriesRead(view('rows reversed')), n * n);
  assert.equal(entriesRead(view('identity')), 0);
});

test('CG on the Hermitian view converges, from 1 to below 1e-10', () => {
  const r = trace.cg.residual;
  assert.equal(r[0], 1);
  assert.equal(trace.cg.code, 0);
  assert.ok(r[r.length - 1] < 1e-10);
  assert.equal(trace.cg.iterates.length, trace.cg.iterations + 1);
  assert.equal(model.count, trace.views.length + trace.cg.iterations + 1);
});
