// The animation shell's timeline and plot helpers (#933).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { advance, clampFrame, resumeFrom } from '../src/components/examples/lib/player.js';
import { scaleLinear, scaleLog, logDomain, logTicks, niceTicks, linePath, lerpArray, ease, fmt } from '../src/components/examples/lib/plot.js';

const opts = { count: 5, frameMs: 100, tweenMs: 40 };

test('clampFrame keeps the index in range', () => {
  assert.equal(clampFrame(-3, 5), 0);
  assert.equal(clampFrame(9, 5), 4);
  assert.equal(clampFrame(2.4, 5), 2);
  assert.equal(clampFrame(3, 0), 0);
});

test('advance holds each frame for frameMs and tweens over tweenMs', () => {
  let s = advance({ frame: 0, elapsed: 0 }, 20, opts);
  assert.deepEqual([s.frame, s.t, s.done], [0, 0.5, false]);
  s = advance({ frame: 0, elapsed: 0 }, 120, opts);
  assert.equal(s.frame, 1);
  assert.equal(s.elapsed, 20);
  assert.equal(s.t, 0.5);
  s = advance({ frame: 1, elapsed: 20 }, 30, opts);
  assert.equal(s.t, 1);
});

test('advance stops on the last frame once its tween is done', () => {
  let s = advance({ frame: 3, elapsed: 90 }, 15, opts);
  assert.equal(s.frame, 4);
  assert.equal(s.done, false);
  s = advance(s, 40, opts);
  assert.equal(s.frame, 4);
  assert.equal(s.done, true);
  // A large dt never runs past the end.
  s = advance({ frame: 0, elapsed: 0 }, 1e6, opts);
  assert.equal(s.frame, 4);
  assert.equal(s.done, true);
});

test('advance without a tween settles at once', () => {
  const s = advance({ frame: 0, elapsed: 0 }, 1, { ...opts, tweenMs: 0 });
  assert.equal(s.t, 1);
});

test('scales map their domain onto their range', () => {
  const sx = scaleLinear([0, 10], [100, 200]);
  assert.equal(sx(0), 100);
  assert.equal(sx(10), 200);
  assert.equal(sx(2.5), 125);
  const sy = scaleLog([1e-10, 1], [300, 0]);
  assert.equal(sy(1), 0);
  assert.equal(sy(1e-10), 300);
  assert.ok(Math.abs(sy(1e-5) - 150) < 1e-9);
  // zero and NaN sit on the axis instead of vanishing
  assert.equal(sy(0), 300);
  assert.equal(sy(NaN), 300);
});

test('logDomain spans whole decades and ignores zeros', () => {
  assert.deepEqual(logDomain([3e-11, 0, 9.8, NaN]), [1e-11, 10]);
  assert.deepEqual(logDomain([1, 1]), [1, 10]);
  assert.deepEqual(logDomain([0, -1]), [0.1, 1]);
});

test('logTicks are powers of ten, thinned to the limit', () => {
  assert.deepEqual(logTicks([1e-3, 10]), [1e-3, 1e-2, 1e-1, 1, 10]);
  const many = logTicks([1e-12, 10], 5);
  assert.ok(many.length <= 5);
  assert.equal(many[0], 1e-12);
});

test('niceTicks are round numbers inside the domain', () => {
  assert.deepEqual(niceTicks([0, 1], 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepEqual(niceTicks([0, 145], 5), [0, 50, 100]);
  assert.deepEqual(niceTicks([-0.3, 0.3], 3), [-0.2, 0, 0.2]);
});

test('linePath skips points that do not map to a number', () => {
  const id = (x) => x;
  assert.equal(linePath([0, 1, 2], [0, NaN, 2], id, id), 'M0.00,0.00L2.00,2.00');
});

test('tweening helpers', () => {
  assert.deepEqual(lerpArray([0, 10], [10, 20], 0.5), [5, 15]);
  assert.equal(ease(0), 0);
  assert.equal(ease(1), 1);
  assert.equal(ease(0.5), 0.5);
  assert.equal(fmt(1.30e-12, 2), '1.3e-12');
  assert.equal(fmt(774.3412, 4), '774.3');
  assert.equal(fmt(0), '0');
});

test('Play resumes a paused tween where it stopped', () => {
  const o = { count: 3, tweenMs: 1600 };
  // Paused 400 ms into a 1600 ms tween: resuming keeps the 400 ms ...
  assert.deepEqual(resumeFrom({ frame: 1, elapsed: 400 }, o), { frame: 1, elapsed: 400, restarted: false });
  // ... so the next tick is still mid-tween, not settled.
  const s = advance({ frame: 1, elapsed: 400 }, 16, { ...o, frameMs: 2600 });
  assert.ok(s.t < 1);
});

test('Play at the last frame replays from frame 0, settled', () => {
  assert.deepEqual(resumeFrom({ frame: 2, elapsed: 9999 }, { count: 3, tweenMs: 1600 }), { frame: 0, elapsed: 1600, restarted: true });
});

test('Play paused during the last frame\'s tween finishes it instead of restarting', () => {
  const o = { count: 3, tweenMs: 1600 };
  assert.deepEqual(resumeFrom({ frame: 2, elapsed: 400 }, o), { frame: 2, elapsed: 400, restarted: false });
  // Settled exactly at the end counts as finished.
  assert.equal(resumeFrom({ frame: 2, elapsed: 1600 }, o).restarted, true);
});
