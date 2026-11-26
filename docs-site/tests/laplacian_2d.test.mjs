// 4A: the assembly and solves the animation draws, against laplacian_2d.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { laplacianModel, laplacianFrame } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('laplacian_2d');
const model = laplacianModel(trace);

test('the 5-point stencil: nnz = 5N - 4n, symmetric, rows in order', () => {
  assert.equal(trace.nonzeros.length, trace.nnz);
  assert.equal(trace.nnz, 5 * trace.N - 4 * trace.n);
  const set = new Set(trace.nonzeros.map(([r, c]) => `${r},${c}`));
  for (const [r, c] of trace.nonzeros) assert.ok(set.has(`${c},${r}`));
  for (let q = 1; q < trace.nonzeros.length; ++q) assert.ok(trace.nonzeros[q][0] >= trace.nonzeros[q - 1][0]);
});

test('assembly frames fill n matrix rows each, ending with every nonzero', () => {
  assert.equal(laplacianFrame(model, 0).rows, trace.n);
  assert.equal(laplacianFrame(model, model.assembly - 1).filled, trace.nnz);
});

test('every solver converges; GMRES with the larger restart takes no more steps', () => {
  for (const s of trace.solvers) {
    assert.equal(s.code, 0, s.name);
    assert.ok(s.residual[s.iterations] <= trace.tolerance, s.name);
    assert.equal(s.iterates.length, s.iterations + 1);
  }
  const [g5, g20] = trace.solvers;
  assert.ok(g20.iterations < g5.iterations);
});

test('the two GMRES runs agree until the first restart, then GMRES(5) falls behind', () => {
  const [g5, g20] = trace.solvers;
  for (let k = 0; k <= g5.restart; ++k) assert.ok(Math.abs(g5.residual[k] - g20.residual[k]) <= 1e-4 * g20.residual[k], `step ${k}`);
  assert.ok(g5.residual[g5.restart + 1] > g20.residual[g5.restart + 1]);
});

test('solve frames: step k after the assembly frames; converged runs rest', () => {
  const s = laplacianFrame(model, model.count - 1);
  assert.equal(s.phase, 'solve');
  assert.equal(s.step, model.steps - 1);
  assert.ok(s.solvers.every((v) => v.done));
});
