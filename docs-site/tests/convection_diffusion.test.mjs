// 3B: the runs the animation draws, against convection_diffusion.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { convectionModel, convectionFrame } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('convection_diffusion');
const model = convectionModel(trace);

test('one run per epsilon, each with CG and BiCGSTAB from x0 = 0', () => {
  assert.ok(model.runs.length >= 2);
  for (const r of model.runs) {
    assert.equal(r.cg.name, 'cg');
    assert.equal(r.bicgstab.name, 'bicgstab');
    for (const s of [r.cg, r.bicgstab]) {
      assert.ok(s.iterates[0].every((v) => v === 0));
      assert.equal(s.residual[0], 1);
      assert.equal(s.iterates.length, s.residual.length);
    }
  }
});

test('BiCGSTAB converges at every epsilon; CG reaches its 500-step cap', () => {
  for (const r of model.runs) {
    assert.equal(r.bicgstab.code, 0);
    assert.ok(r.bicgstab.residual[r.bicgstab.residual.length - 1] <= trace.tolerance, `eps ${r.epsilon}`);
    assert.equal(r.cg.code, 1);
    assert.equal(r.cg.iterations, 500);
  }
});

test('the caption: CG nearly converges at the largest epsilon and diverges at the smallest', () => {
  const first = model.runs[0];
  const last = model.runs[model.runs.length - 1];
  assert.ok(first.cg.final_residual < 1e-6);
  assert.ok(last.cg.final_residual > 1);
});

test('cell Peclet h / (2 eps) passes 1 only at the smallest epsilon, where the discrete solution overshoots', () => {
  const h = trace.x[0];
  for (const r of model.runs) assert.ok(Math.abs(r.cell_peclet - h / (2 * r.epsilon)) < 1e-5 * r.cell_peclet);   // x is stored to 6 digits
  const last = model.runs[model.runs.length - 1];
  assert.ok(last.cell_peclet > 1);
  const u = last.bicgstab.iterates[last.bicgstab.iterates.length - 1];
  assert.ok(Math.max(...u) > Math.max(...last.exact_u));
  for (const r of model.runs.slice(0, -1)) assert.ok(r.cell_peclet < 1);
});

test('the exact solution satisfies the boundary conditions', () => {
  for (const r of model.runs) {
    assert.ok(Math.abs(r.exact_u[0]) < 1e-9);
    assert.ok(Math.abs(r.exact_u[r.exact_u.length - 1]) < 1e-4);
  }
});

test('frames: BiCGSTAB rests once converged, and a tween runs from k - 1', () => {
  const r = model.runs[0];
  const end = convectionFrame(r, r.count - 1);
  assert.equal(end[1].step, r.bicgstab.iterates.length - 1);
  assert.equal(end[1].done, true);
  const [cg] = convectionFrame(r, 3, 0.5);
  for (let i = 0; i < trace.n; ++i) assert.ok(Math.abs(cg.u[i] - (r.cg.iterates[2][i] + r.cg.iterates[3][i]) / 2) < 1e-12);
});
