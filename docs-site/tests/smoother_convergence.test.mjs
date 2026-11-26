// 4B: the sweeps the animation draws, against smoother_convergence.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { smootherModel, smootherFrame, modeAmplitude } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('smoother_convergence');
const model = smootherModel(trace);
const [lo, hi] = trace.smoothing.modes;

test('a residual per sweep for every method, starting at 1', () => {
  for (const h of trace.residuals) {
    assert.equal(h.residual.length, trace.sweeps + 1, h.name);
    assert.equal(h.residual[0], 1, h.name);
  }
});

test('Jacobi is slowest, SOR at omega_opt fastest, as the caption says', () => {
  const at = (h) => h.residual[trace.sweeps];
  const [jac, gs] = trace.residuals;
  const opt = trace.residuals[trace.residuals.length - 1];
  assert.ok(Math.abs(opt.omega - trace.omega_opt) < 1e-4);   // stored to 6 digits
  assert.ok(at(jac) > at(gs));
  assert.ok(at(gs) > at(opt));
  const h = trace.x[0];
  assert.ok(Math.abs(trace.omega_opt - 2 / (1 + Math.sin(Math.PI * h))) < 1e-5);   // x is stored to 6 digits
});

test('the omega sweep\'s best residual is at omega_opt', () => {
  const i = trace.omega_sweep_residual.indexOf(Math.min(...trace.omega_sweep_residual));
  assert.ok(Math.abs(trace.omega_sweep[i] - trace.best_omega) < 1e-5);
  assert.ok(Math.abs(trace.best_omega - trace.omega_opt) < 1e-5);
});

test('the recorded mode amplitudes are the projections of the recorded errors', () => {
  for (const s of [trace.smoothing.jacobi, trace.smoothing.gauss_seidel]) {
    for (const k of [0, 1, 5, 20]) {
      assert.ok(Math.abs(modeAmplitude(s.errors[k], lo) - s.low_mode[k]) < 2e-4, `low ${k}`);
      assert.ok(Math.abs(modeAmplitude(s.errors[k], hi) - s.high_mode[k]) < 2e-4, `high ${k}`);
    }
  }
});

test('smoothing: Gauss-Seidel kills the oscillatory mode, the smooth one survives', () => {
  const g = trace.smoothing.gauss_seidel;
  const j = trace.smoothing.jacobi;
  assert.ok(Math.abs(g.high_mode[10]) < 1e-3 * Math.abs(g.high_mode[0]));
  assert.ok(g.low_mode[10] > 0.9 * g.low_mode[0]);
  // Undamped Jacobi flips the oscillatory mode's sign each sweep, and is slower on it.
  for (let k = 1; k < 6; ++k) assert.ok(Math.sign(j.high_mode[k]) === -Math.sign(j.high_mode[k - 1]));
  assert.ok(Math.abs(j.high_mode[10]) > Math.abs(g.high_mode[10]));
});

test('a tween runs from sweep k - 1', () => {
  const f = smootherFrame(model, 4, 0.5);
  const s = trace.smoothing.jacobi;
  for (let i = 0; i < trace.n; ++i) assert.ok(Math.abs(f.jacobi[i] - (s.errors[3][i] + s.errors[4][i]) / 2) < 1e-12);
});
