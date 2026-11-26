// What each example animation draws, derived from its trace (#933). Pure
// functions of the JSON the example program wrote, so docs-site/tests/ can
// check them against the trace's own numbers under node. The figures'
// scripts only draw what these return.

import { lerpArray } from './plot.js';

// ── 1A Leontief ───────────────────────────────────────────────────────────

/**
 * Frame k shows the gross output after rounds 0..k: demand (round 0), the
 * indirect output of rounds 1..k, and the residual of (I - A) x = d.
 */
export function leontiefModel(trace) {
  const n = trace.sectors.length;
  const cumulative = [];
  let total = new Array(n).fill(0);
  for (const r of trace.rounds) {
    total = total.map((v, i) => v + r[i]);
    cumulative.push(total);
  }
  const maxOutput = Math.max(...cumulative[cumulative.length - 1]);
  return {
    count: trace.rounds.length,
    sectors: trace.sectors,
    demand: trace.demand,
    cumulative,
    final: trace.output,
    residual: trace.residual,
    maxOutput,
  };
}

/** Totals for frame k, tweened from frame k - 1 by t. */
export function leontiefFrame(model, k, t = 1) {
  const now = model.cumulative[k];
  const prev = model.cumulative[Math.max(0, k - 1)];
  return { output: k === 0 ? now : lerpArray(prev, now, t), residual: model.residual[k] };
}

// ── 1B Kirchhoff ──────────────────────────────────────────────────────────

/** The cells whose value differs between two n x n snapshots. */
export function changedCells(before, after) {
  const cells = [];
  for (let i = 0; i < after.length; ++i)
    for (let j = 0; j < after[i].length; ++j) if (before[i][j] !== after[i][j]) cells.push([i, j]);
  return cells;
}

/** Row sums of a dense snapshot. */
export const rowSums = (m) => m.map((r) => r.reduce((a, b) => a + b, 0));

/**
 * Frames: the empty matrix, one per stamp, the CRS arrays, the balanced
 * bridge's voltages, and the bridge after R4 is amended.
 */
export function kirchhoffModel(trace) {
  const snaps = trace.after_stamp;
  const frames = [{ kind: 'empty', matrix: snaps[0], changed: [], resistor: -1 }];
  for (let k = 1; k < snaps.length; ++k) {
    frames.push({ kind: 'stamp', matrix: snaps[k], changed: changedCells(snaps[k - 1], snaps[k]), resistor: k - 1 });
  }
  const full = snaps[snaps.length - 1];
  frames.push({ kind: 'crs', matrix: full, changed: [], resistor: -1 });
  frames.push({ kind: 'balanced', matrix: full, changed: [], resistor: -1, solution: trace.balanced });
  frames.push({
    kind: 'amended',
    matrix: trace.unbalanced.matrix,
    changed: changedCells(full, trace.unbalanced.matrix),
    resistor: 3,
    solution: trace.unbalanced,
  });
  return { count: frames.length, frames, resistors: trace.resistors, crs: trace.crs, volts: trace.source_volts };
}

// ── 2A Dürer ──────────────────────────────────────────────────────────────

/** Where each value sits: value -> [row, col]. */
export function positions(square) {
  const at = new Map();
  square.forEach((row, i) => row.forEach((v, j) => at.set(v, [i, j])));
  return at;
}

/** The magic test, recomputed: every row, column and both diagonals. */
export function isMagic(m, target) {
  const n = m.length;
  let d = 0;
  let a = 0;
  for (let i = 0; i < n; ++i) {
    let r = 0;
    let c = 0;
    for (let j = 0; j < n; ++j) {
      r += m[i][j];
      c += m[j][i];
    }
    if (r !== target || c !== target) return false;
    d += m[i][i];
    a += m[i][n - 1 - i];
  }
  return d === target && a === target;
}

/**
 * One frame per group of cells (rows, columns, diagonals, blocks), then one
 * per rearrangement, which moves every value from its place in the square
 * to its place in the result.
 */
export function durerModel(trace) {
  const home = positions(trace.square);
  const frames = trace.groups.map((g) => ({ kind: 'group', name: g.name, view: g.view, cells: g.cells, sum: g.sum }));
  for (const r of trace.rearrangements) {
    const to = positions(r.result);
    const moves = [...home.keys()].map((v) => ({ value: v, from: home.get(v), to: to.get(v) }));
    frames.push({ kind: 'move', name: r.name, view: r.how, result: r.result, magic: r.magic, moves });
  }
  return { count: frames.length, frames, square: trace.square, target: trace.magic_constant };
}

// ── 2B bandwidth reduction ────────────────────────────────────────────────

/** max |i - j| over the nonzeros once each index k is renumbered index_of[k]. */
export function bandwidthOf(nonzeros, indexOf) {
  let bw = 0;
  for (const [i, j] of nonzeros) bw = Math.max(bw, Math.abs(indexOf[i] - indexOf[j]));
  return bw;
}

/**
 * Frame per ordering. Each nonzero (i, j) of the natural matrix is drawn at
 * (index_of[i], index_of[j]); a tween slides it between two orderings.
 */
export function bandwidthModel(trace) {
  return {
    count: trace.orderings.length,
    m: trace.m,
    n: trace.n,
    nonzeros: trace.natural_nonzeros,
    orderings: trace.orderings,
  };
}

/** Nonzero positions for frame k, tweened from frame k - 1 by t. */
export function bandwidthFrame(model, k, t = 1) {
  const now = model.orderings[k].index_of;
  const prev = model.orderings[Math.max(0, k - 1)].index_of;
  const idx = k === 0 ? now : lerpArray(prev, now, t);
  return { indexOf: idx, ordering: model.orderings[k] };
}

// ── 3A heat equation ──────────────────────────────────────────────────────

/** Frame k is CG step k for both runs; a run that has converged rests. */
export function heatModel(trace) {
  const count = Math.max(...trace.runs.map((r) => r.iterations)) + 1;
  return { count, x: trace.x, runs: trace.runs, tolerance: trace.tolerance };
}

/** Each run's iterate and residual at step k, tweened from k - 1 by t. */
export function heatFrame(model, k, t = 1) {
  return model.runs.map((r) => {
    const step = Math.min(k, r.iterations);
    const moving = k <= r.iterations && k > 0;
    const u = moving ? lerpArray(r.iterates[step - 1], r.iterates[step], t) : r.iterates[step];
    return { name: r.name, step, u, residual: r.residual[step], converged: k >= r.iterations };
  });
}

// ── 3B convection-diffusion ───────────────────────────────────────────────

/**
 * One run per epsilon. Frame k is step k of both solvers: CG is replayed for
 * as many steps as the trace holds (it runs to the 500-step cap), BiCGSTAB
 * until it converges, then rests.
 */
export function convectionModel(trace) {
  const runs = trace.runs.map((r) => {
    const [cg, bicgstab] = r.solvers;
    return { ...r, cg, bicgstab, count: cg.iterates.length };
  });
  return { x: trace.x, tolerance: trace.tolerance, runs };
}

/** Each solver's iterate and residual at step k of a run, tweened from k - 1. */
export function convectionFrame(run, k, t = 1) {
  return [run.cg, run.bicgstab].map((s) => {
    const last = s.iterates.length - 1;
    const step = Math.min(k, last);
    const moving = k > 0 && k <= last;
    const u = moving ? lerpArray(s.iterates[step - 1], s.iterates[step], t) : s.iterates[step];
    return { name: s.name, step, u, residual: s.residual[step], done: k >= last };
  });
}

// ── 4A 2D Laplacian ───────────────────────────────────────────────────────

/**
 * Frames: the inserter filling the matrix one grid row (n matrix rows) at a
 * time, then the three solvers step by step, each resting once converged.
 */
export function laplacianModel(trace) {
  const assembly = trace.n;   // one frame per grid row
  const steps = Math.max(...trace.solvers.map((s) => s.iterations)) + 1;
  return { ...trace, assembly, steps, count: assembly + steps };
}

/** The frame's phase: assembly (rows filled so far) or solve (step k). */
export function laplacianFrame(model, f, t = 1) {
  if (f < model.assembly) {
    const rows = (f + 1) * model.n;
    const filled = model.nonzeros.filter(([r]) => r < rows).length;
    return { phase: 'assembly', rows, filled };
  }
  const k = f - model.assembly;
  const solvers = model.solvers.map((s) => {
    const step = Math.min(k, s.iterations);
    const moving = k > 0 && k <= s.iterations;
    const u = moving ? lerpArray(s.iterates[step - 1], s.iterates[step], t) : s.iterates[step];
    return { name: s.name, step, u, residual: s.residual[step], done: k >= s.iterations };
  });
  return { phase: 'solve', step: k, solvers };
}

// ── 4B smoothers ──────────────────────────────────────────────────────────

/** Frame k is sweep k of every method and of the smoothing experiment. */
export function smootherModel(trace) {
  return { ...trace, count: trace.sweeps + 1 };
}

/** The amplitude of sine mode `mode` in a grid function e on x = (i+1)h. */
export function modeAmplitude(e, mode) {
  const n = e.length;
  let a = 0;
  for (let i = 0; i < n; ++i) a += e[i] * Math.sin((mode * Math.PI * (i + 1)) / (n + 1));
  return (2 * a) / (n + 1);
}

/** Error profiles at sweep k, tweened from k - 1. */
export function smootherFrame(model, k, t = 1) {
  const pick = (s) => (k > 0 ? lerpArray(s.errors[k - 1], s.errors[k], t) : s.errors[0]);
  return { jacobi: pick(model.smoothing.jacobi), gaussSeidel: pick(model.smoothing.gauss_seidel) };
}

// ── 5A least squares ──────────────────────────────────────────────────────

/** Frame per polynomial degree. */
export function leastSquaresModel(trace) {
  return { ...trace, count: trace.fits.length };
}

// ── 5B, the Hilbert sweep ─────────────────────────────────────────────────

/**
 * One frame per Hilbert order n. Each method's backward and forward errors
 * as arrays over n, with null where it failed (Cholesky past its first
 * failure), and cond(H) * eps, the forward error the problem itself allows.
 */
export function hilbertModel(trace) {
  const rows = trace.hilbert;
  const eps = 2.220446049250313e-16;
  const names = rows[0].methods.map((m) => m.method);
  const series = names.map((name) => ({
    method: name,
    backward: rows.map((r) => r.methods.find((m) => m.method === name).backward),
    forward: rows.map((r) => r.methods.find((m) => m.method === name).forward),
  }));
  return {
    n: rows.map((r) => r.n),
    cond: rows.map((r) => r.cond),
    condEps: rows.map((r) => r.cond * eps),
    choleskyInfo: rows.map((r) => r.cholesky_info),
    firstFail: trace.hilbert_cholesky_first_fail,
    series,
    eps,
    count: rows.length,
  };
}

// ── 5B three factorizations ───────────────────────────────────────────────

/**
 * Frames: for each part (SPD, then indefinite), the states after 0..n steps.
 * A method that stopped early (Cholesky on the indefinite matrix) holds its
 * last state and is marked failed from the step it could not take.
 */
export function factorModel(trace) {
  const frames = [];
  for (const part of trace.parts) {
    for (let k = 0; k <= trace.n; ++k) frames.push({ part, k });
  }
  return { n: trace.n, frames, count: frames.length, flopsByN: trace.flops_by_n };
}

/** Each method's state, cost and status at a frame. */
export function factorFrame(model, f) {
  const { part, k } = model.frames[f];
  const methods = part.methods.map((m) => {
    const last = m.states.length - 1;
    const step = Math.min(k, last);
    const failed = m.info !== 0 && k > last;
    return { method: m.method, step, state: m.states[step], flops: m.flops[step], failed, failedPivot: m.failed_pivot, pivot: m.pivot };
  });
  return { part: part.name, k, matrix: part.matrix, methods };
}

// ── 6A vibrating string ───────────────────────────────────────────────────

/**
 * Frames: the string's n modes, then the membrane's n2d^2 modes, then the
 * lopsided operator's eigenvalues for each d of the sweep.
 */
export function stringModel(trace) {
  const frames = [
    ...trace.modes.map((v, k) => ({ kind: 'string', k })),
    ...trace.membrane_modes.map((v, k) => ({ kind: 'membrane', k })),
    ...trace.skew_sweep.map((s, k) => ({ kind: 'skew', k })),
  ];
  return { ...trace, frames, count: frames.length };
}

/** Relative frequency of string mode k against the fundamental: sqrt(lambda_k / lambda_1). */
export const relativeFrequency = (model, k) => Math.sqrt(model.eigenvalues[k] / model.eigenvalues[0]);

/** The analytic eigenvalues of the lopsided Laplacian, 2 - 2 sqrt(1 - d^2) cos(k pi / (n+1)), as [re, im]. */
export function skewAnalytic(n, d) {
  const out = [];
  for (let k = 1; k <= n; ++k) {
    const c = 2 * Math.cos((k * Math.PI) / (n + 1));
    const q = 1 - d * d;
    out.push(q >= 0 ? [2 - c * Math.sqrt(q), 0] : [2, -c * Math.sqrt(-q)]);
  }
  return out;
}

// ── 6B PCA ────────────────────────────────────────────────────────────────

/**
 * Frames, each a set of 2D point positions: the centered data in the plane of
 * features `fx` and `fy`, the principal scores (PC1, PC2), and the rank-2
 * and rank-1 reconstructions back in the feature plane.
 */
export function pcaModel(trace, fx = 0, fy = 2) {
  const plane = (rows) => rows.map((r) => [r[fx], r[fy]]);
  const frames = [
    { kind: 'data', label: 'centered data', points: plane(trace.centered) },
    { kind: 'scores', label: 'principal coordinates', points: trace.scores.map((r) => [r[0], r[1]]) },
    ...trace.reconstructions
      .filter((r) => r.rank < trace.features)
      .sort((a, b) => b.rank - a.rank)
      .map((r) => ({ kind: 'recon', rank: r.rank, error: r.error_fro, label: `rank-${r.rank} reconstruction`, points: plane(r.data) })),
  ];
  const total = trace.sigma.reduce((a, s) => a + s * s, 0);
  return { ...trace, fx, fy, frames, count: frames.length, explained: trace.sigma.map((s) => (s * s) / total) };
}

// ── 7A sparse formats ─────────────────────────────────────────────────────

/** Frame r is row r of a matvec: which slots each format reads for it. */
export function formatsModel(trace) {
  return { ...trace, count: trace.n };
}

export function formatsFrame(model, r) {
  const coo = [];
  model.coo.rows.forEach((row, q) => row === r && coo.push(q));
  const crs = [];
  for (let q = model.crs.starts[r]; q < model.crs.starts[r + 1]; ++q) crs.push(q);
  const w = model.ell.width;
  const ell = [];
  const padding = [];
  for (let q = r * w; q < (r + 1) * w; ++q) (model.ell.cols[q] < 0 ? padding : ell).push(q);
  const cols = crs.map((q) => model.crs.cols[q]);
  return { row: r, coo, crs, crsStarts: [r, r + 1], ell, padding, cols };
}

// ── 7B structured views ───────────────────────────────────────────────────

/** Frames: one per view, then CG on the Hermitian view step by step. */
export function viewsModel(trace) {
  const frames = [...trace.views.map((v, i) => ({ kind: 'view', i })), ...trace.cg.iterates.map((x, k) => ({ kind: 'cg', k }))];
  return { ...trace, frames, count: frames.length };
}

/** The distinct stored entries a view reads. */
export function entriesRead(view) {
  return new Set(view.sources.filter((s) => s !== null).map(([i, j]) => `${i},${j}`)).size;
}
