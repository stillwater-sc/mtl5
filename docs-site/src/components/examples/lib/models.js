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
