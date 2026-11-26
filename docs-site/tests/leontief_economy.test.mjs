// 1A: the rounds the animation draws, against leontief_economy.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { leontiefModel, leontiefFrame } from '../src/components/examples/lib/models.js';
import { loadTrace, close } from './_traces.mjs';

const trace = loadTrace('leontief_economy');
const model = leontiefModel(trace);
const n = trace.sectors.length;

test('round 0 is the final demand', () => {
  assert.deepEqual(trace.rounds[0], trace.demand);
  assert.deepEqual(leontiefFrame(model, 0).output, trace.demand);
});

test('each round is A times the previous one', () => {
  for (let k = 1; k < trace.rounds.length; ++k) {
    for (let i = 0; i < n; ++i) {
      let s = 0;
      for (let j = 0; j < n; ++j) s += trace.A[i][j] * trace.rounds[k - 1][j];
      assert.ok(close(s, trace.rounds[k][i], 1e-6), `round ${k}, sector ${i}`);
    }
  }
});

test('the last frame is the converged output, which solves (I - A) x = d', () => {
  const last = model.cumulative[model.count - 1];
  for (let i = 0; i < n; ++i) {
    assert.ok(close(last[i], trace.output[i], 1e-7));
    let ax = 0;
    for (let j = 0; j < n; ++j) ax += trace.A[i][j] * trace.output[j];
    assert.ok(close(trace.output[i] - ax, trace.demand[i], 1e-8), `sector ${i}`);
  }
});

test('the residual falls every round and ends below the tolerance', () => {
  for (let k = 1; k < trace.residual.length; ++k) assert.ok(trace.residual[k] < trace.residual[k - 1]);
  assert.ok(trace.residual[trace.residual.length - 1] < trace.tolerance);
  // Geometric decay: the ratio is below the largest column sum, which bounds it.
  const ratio = trace.residual[2] / trace.residual[1];
  assert.ok(ratio < trace.max_column_sum);
});

test('a tween runs from the previous round to this one', () => {
  const half = leontiefFrame(model, 3, 0.5).output;
  for (let i = 0; i < n; ++i) assert.ok(close(half[i], (model.cumulative[2][i] + model.cumulative[3][i]) / 2));
  assert.deepEqual(leontiefFrame(model, 3, 1).output, model.cumulative[3]);
});

test('the multiplier is total output over total demand', () => {
  const sum = (v) => v.reduce((a, b) => a + b, 0);
  assert.ok(close(trace.multiplier, sum(trace.output) / sum(trace.demand), 1e-5));
});
