// 2B: the orderings the animation draws, against bandwidth_reduction.json (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bandwidthModel, bandwidthFrame, bandwidthOf } from '../src/components/examples/lib/models.js';
import { loadTrace } from './_traces.mjs';

const trace = loadTrace('bandwidth_reduction');
const model = bandwidthModel(trace);

test('each ordering is a permutation of 0 .. n-1', () => {
  for (const o of trace.orderings) {
    assert.deepEqual([...o.index_of].sort((a, b) => a - b), Array.from({ length: trace.n }, (_, i) => i), o.name);
  }
});

test('the bandwidth drawn is the bandwidth MTL5 measured', () => {
  for (const o of trace.orderings) assert.equal(bandwidthOf(trace.natural_nonzeros, o.index_of), o.bandwidth, o.name);
});

test('the grid matrix: 5-point stencil, natural bandwidth m', () => {
  const m = trace.m;
  assert.equal(trace.natural_nonzeros.length, 5 * m * m - 4 * m);
  assert.equal(trace.orderings[0].bandwidth, m);
});

test('RCM undoes the shuffle: bandwidth and profile come back down', () => {
  const by = Object.fromEntries(trace.orderings.map((o) => [o.name, o]));
  assert.ok(by.shuffled.bandwidth > 5 * by.rcm.bandwidth);
  assert.ok(by.rcm.bandwidth <= by.natural.bandwidth);
  assert.ok(by.rcm.profile < by.shuffled.profile);
  assert.equal(trace.views_agree, true);
});

test('a tween slides each index from the previous ordering', () => {
  const half = bandwidthFrame(model, 2, 0.5).indexOf;
  const a = trace.orderings[1].index_of;
  const b = trace.orderings[2].index_of;
  for (let k = 0; k < trace.n; ++k) assert.equal(half[k], (a[k] + b[k]) / 2);
  assert.deepEqual(bandwidthFrame(model, 0, 0.3).indexOf, trace.orderings[0].index_of);
});
