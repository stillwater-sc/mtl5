// 2A: the views and moves the animation draws, against durer_magic_square.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { durerModel, isMagic, positions } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('durer_magic_square');
const model = durerModel(trace);
const M = trace.square;

test('the square is Durer\'s, with the date in the bottom row', () => {
  assert.deepEqual(M[3].slice(1, 3), [15, 14]);
  assert.ok(isMagic(M, trace.magic_constant));
});

test('each group sums its cells to what the view reported', () => {
  for (const g of trace.groups) {
    const s = g.cells.reduce((a, [i, j]) => a + M[i][j], 0);
    assert.equal(s, g.sum, g.name);
    assert.equal(g.sum, trace.magic_constant, g.name);
  }
});

test('the column groups are the columns, the anti-diagonal runs corner to corner', () => {
  const col1 = trace.groups.find((g) => g.name === 'column 1');
  assert.deepEqual(col1.cells, [[0, 1], [1, 1], [2, 1], [3, 1]]);
  const anti = trace.groups.find((g) => g.name === 'anti-diagonal');
  assert.deepEqual(anti.cells.map(([i, j]) => i + j), [3, 3, 3, 3]);
});

test('every rearrangement moves the same sixteen values, and its magic flag holds', () => {
  for (const r of trace.rearrangements) {
    assert.deepEqual(r.result.flat().sort((a, b) => a - b), M.flat().sort((a, b) => a - b), r.name);
    assert.equal(isMagic(r.result, trace.magic_constant), r.magic, r.name);
  }
});

test('the transpose moves each value from (i, j) to (j, i)', () => {
  const t = model.frames.find((f) => f.kind === 'move' && f.name === 'transpose');
  for (const m of t.moves) assert.deepEqual(m.to, [m.from[1], m.from[0]]);
});

test('the one non-magic move keeps rows and columns but breaks the diagonals', () => {
  const bad = trace.rearrangements.filter((r) => !r.magic);
  assert.equal(bad.length, 1);
  const m = bad[0].result;
  const c = trace.magic_constant;
  for (let i = 0; i < 4; ++i) {
    assert.equal(m[i].reduce((a, b) => a + b, 0), c);
    assert.equal(m.reduce((a, r) => a + r[i], 0), c);
  }
  assert.notEqual(m.reduce((a, r, i) => a + r[i], 0), c);
  assert.notEqual(m.reduce((a, r, i) => a + r[3 - i], 0), c);
});

test('positions maps each value to its cell', () => {
  assert.deepEqual(positions(M).get(16), [0, 0]);
  assert.deepEqual(positions(M).get(1), [3, 3]);
});
