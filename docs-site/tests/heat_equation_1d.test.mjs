// 3A: the CG steps the animation draws, against heat_equation_1d.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { heatModel, heatFrame } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('heat_equation_1d');
const model = heatModel(trace);
const [plain, jacobi] = trace.runs;

test('one iterate and one residual per step, from x0 = 0', () => {
  for (const r of trace.runs) {
    assert.equal(r.iterates.length, r.iterations + 1, r.name);
    assert.equal(r.residual.length, r.iterations + 1, r.name);
    assert.ok(r.iterates[0].every((v) => v === 0), r.name);
    assert.equal(r.residual[0], 1, r.name);
    for (const u of r.iterates) assert.equal(u.length, trace.n);
  }
});

test('both runs end below the tolerance, the preconditioned one sooner', () => {
  assert.ok(jacobi.iterations < plain.iterations);
  for (const r of trace.runs) assert.ok(r.residual[r.iterations] <= trace.tolerance, r.name);
  // Until its last step each run is above the tolerance, or it would have stopped.
  for (const r of trace.runs) assert.ok(r.residual[r.iterations - 1] > trace.tolerance, r.name);
});

test('the two final profiles agree', () => {
  const a = plain.iterates[plain.iterations];
  const b = jacobi.iterates[jacobi.iterations];
  for (let i = 0; i < trace.n; ++i) assert.ok(Math.abs(a[i] - b[i]) <= 1e-3 * Math.max(...b));
});

test('the frame count covers the slower run; a converged run rests', () => {
  assert.equal(model.count, plain.iterations + 1);
  const end = heatFrame(model, model.count - 1);
  assert.equal(end[1].step, jacobi.iterations);
  assert.equal(end[1].converged, true);
  assert.deepEqual(end[1].u, jacobi.iterates[jacobi.iterations]);
});

test('a tween runs from step k - 1 to step k', () => {
  const [s] = heatFrame(model, 5, 0.25);
  for (let i = 0; i < trace.n; ++i) {
    const want = plain.iterates[4][i] + (plain.iterates[5][i] - plain.iterates[4][i]) * 0.25;
    assert.ok(Math.abs(s.u[i] - want) < 1e-15);
  }
});

test('the preconditioned residual reaches the tolerance at step n', () => {
  assert.equal(jacobi.iterations, trace.n);
});
