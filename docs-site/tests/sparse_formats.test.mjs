// 7A: the memory each format reads, against sparse_formats.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatsModel, formatsFrame } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('sparse_formats');
const model = formatsModel(trace);
const n = trace.n;

test('the three formats hold the same entries', () => {
  const key = (r, c, v) => `${r},${c},${v}`;
  const coo = trace.coo.rows.map((r, q) => key(r, trace.coo.cols[q], trace.coo.values[q])).sort();
  const crs = [];
  for (let r = 0; r < n; ++r) for (let q = trace.crs.starts[r]; q < trace.crs.starts[r + 1]; ++q) crs.push(key(r, trace.crs.cols[q], trace.crs.values[q]));
  const ell = [];
  const w = trace.ell.width;
  for (let r = 0; r < n; ++r) for (let q = r * w; q < (r + 1) * w; ++q) if (trace.ell.cols[q] >= 0) ell.push(key(r, trace.ell.cols[q], trace.ell.values[q]));
  assert.deepEqual(crs.sort(), coo);
  assert.deepEqual(ell.sort(), coo);
});

test('stored values match the example\'s table', () => {
  const nnz = trace.crs.values.length;
  assert.equal(trace.stored_values.dense, n * n);
  assert.equal(trace.stored_values.coo, 3 * nnz);
  assert.equal(trace.stored_values.crs, 2 * nnz + n + 1);
  assert.equal(trace.stored_values.ell, 2 * n * trace.ell.width);
});

test('each row reads the same nonzeros in every format; ELL reads its padding too', () => {
  let padding = 0;
  for (let r = 0; r < n; ++r) {
    const f = formatsFrame(model, r);
    assert.equal(f.coo.length, f.crs.length);
    assert.equal(f.ell.length, f.crs.length);
    assert.equal(f.ell.length + f.padding.length, trace.ell.width);
    // sorted COO keeps a row's triplets together
    for (let q = 1; q < f.coo.length; ++q) assert.equal(f.coo[q], f.coo[q - 1] + 1);
    padding += f.padding.length;
  }
  assert.equal(padding, trace.ell.cols.filter((c) => c < 0).length);
});
