// 1B: the stamps the animation draws, against kirchhoff_network.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { kirchhoffModel, changedCells, rowSums } from '../src/components/examples/lib/models.js';
import { loadTrace, close } from './_traces.mjs';

const trace = loadTrace('kirchhoff_network');
const model = kirchhoffModel(trace);

test('frames: empty, one per stamp, CRS, balanced, amended', () => {
  assert.equal(model.count, trace.resistors.length + 4);
  assert.deepEqual(model.frames.map((f) => f.kind), ['empty', 'stamp', 'stamp', 'stamp', 'stamp', 'stamp', 'crs', 'balanced', 'amended']);
});

test('each stamp changes exactly its four entries by +g and -g', () => {
  model.frames.filter((f) => f.kind === 'stamp').forEach((f, k) => {
    const r = trace.resistors[k];
    const want = [[r.a, r.a], [r.a, r.b], [r.b, r.a], [r.b, r.b]].map(String).sort();
    assert.deepEqual(f.changed.map(String).sort(), want, r.name);
    const before = trace.after_stamp[k];
    assert.ok(close(f.matrix[r.a][r.a] - before[r.a][r.a], r.siemens, 1e-6));
    assert.ok(close(f.matrix[r.a][r.b] - before[r.a][r.b], -r.siemens, 1e-6));
  });
});

test('every snapshot is symmetric and its rows sum to zero (Kirchhoff)', () => {
  // The trace keeps 8 significant digits, so "zero" is relative to the row.
  for (const m of [...trace.after_stamp, trace.unbalanced.matrix]) {
    rowSums(m).forEach((s, i) => assert.ok(Math.abs(s) <= 1e-7 * Math.max(...m[i].map(Math.abs)), `row ${i}`));
    for (let i = 0; i < m.length; ++i) for (let j = 0; j < m.length; ++j) assert.equal(m[i][j], m[j][i]);
  }
});

test('the CRS arrays hold exactly the nonzeros of the final matrix', () => {
  const full = trace.after_stamp[trace.after_stamp.length - 1];
  const { starts, columns, values } = trace.crs;
  assert.equal(starts.length, full.length + 1);
  assert.equal(starts[starts.length - 1], values.length);
  let stored = 0;
  for (let i = 0; i < full.length; ++i) {
    for (let k = starts[i]; k < starts[i + 1]; ++k) {
      assert.ok(close(values[k], full[i][columns[k]], 1e-7));
      if (k > starts[i]) assert.ok(columns[k] > columns[k - 1], 'columns sorted within a row');
    }
    stored += full[i].filter((v) => v !== 0).length;
  }
  assert.equal(stored, values.length);
});

test('balanced: no galvanometer current; amended: only R4 entries change', () => {
  assert.ok(close(trace.balanced.v1, trace.balanced.v2, 1e-9));
  assert.ok(Math.abs(trace.balanced.galvanometer_amps) < 1e-12);
  const amended = model.frames[model.count - 1];
  const r4 = trace.resistors[3];
  assert.deepEqual(amended.changed.map(String).sort(), [[r4.a, r4.a], [r4.a, r4.b], [r4.b, r4.a], [r4.b, r4.b]].map(String).sort());
  const i5 = (trace.unbalanced.v1 - trace.unbalanced.v2) / trace.resistors[4].ohms;
  assert.ok(close(i5, trace.unbalanced.galvanometer_amps, 1e-6));
});

test('changedCells finds differing entries only', () => {
  assert.deepEqual(changedCells([[1, 2], [3, 4]], [[1, 0], [3, 4]]), [[0, 1]]);
});

test('the mesh stores about five entries per row', () => {
  for (const row of trace.mesh) {
    assert.ok(row.nnz <= 5 * row.nodes);
    assert.ok(row.crs_bytes < row.dense_bytes || row.nodes < 8);
  }
});
