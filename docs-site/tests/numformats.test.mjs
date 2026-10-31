// Checks the explorer's JavaScript arithmetic against gh-vectors.json, which
// tools/givens_householder_svg.py computes in exact rational arithmetic.
// Run with `npm test` in docs-site/.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeFormat, arith } from '../src/components/gh/numformats.js';
import { givens, householder, applyHouseholder } from '../src/components/gh/kernels.js';

const vectors = JSON.parse(readFileSync(new URL('./gh-vectors.json', import.meta.url), 'utf-8'));

const num = (v) => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v === 'nan' ? NaN : v);
const label = (spec) => JSON.stringify(spec);

function same(got, want, what) {
  want = num(want);
  // Object.is tells 0 from -0; the formats here only need values to agree.
  const ok = Number.isNaN(want) ? Number.isNaN(got) : got === want;
  assert.ok(ok, `${what}: got ${got}, want ${want}`);
}

for (const fv of vectors.formats) {
  const F = makeFormat(fv.spec);
  const ops = arith(F);
  const name = label(fv.spec);

  test(`${name}: enumerates ${fv.count} positive values`, () => {
    const vals = F.values();
    assert.equal(vals.length, fv.count);
    for (let i = 1; i < vals.length; ++i) assert.ok(vals[i] > vals[i - 1]);
    for (const v of vals) assert.equal(F.round(v), v);
  });

  test(`${name}: rounding`, () => {
    for (const [x, want] of fv.round) same(F.round(x), want, `round(${x})`);
  });

  test(`${name}: square root`, () => {
    for (const [x, want] of fv.sqrt) same(ops.sqrt(x), want, `sqrt(${x})`);
  });

  test(`${name}: Givens rotation`, () => {
    for (const [a, b, scaled, [c, s, r]] of fv.givens) {
      const g = givens(ops, a, b, scaled);
      const what = `givens(${a}, ${b}, ${scaled ? 'scaled' : 'unscaled'})`;
      same(g.r, r, `${what}.r`);
      same(g.c, c, `${what}.c`);
      same(g.s, s, `${what}.s`);
    }
  });

  test(`${name}: Householder reflector`, () => {
    for (const [a, b, want] of fv.householder) {
      const h = householder(ops, a, b);
      const what = `householder(${a}, ${b})`;
      if (want === null) {
        assert.ok(h.identity, `${what}: want the identity`);
      } else {
        assert.ok(!h.identity, `${what}: got the identity (${h.reason})`);
        same(h.v1, want[0], `${what}.v1`);
        same(h.tau, want[1], `${what}.tau`);
      }
    }
  });

  test(`${name}: Householder reflector applied`, () => {
    for (const [a, b, u, w, [y0, y1]] of fv.apply_householder) {
      const [g0, g1] = applyHouseholder(ops, householder(ops, a, b), [u, w]);
      const what = `householder(${a}, ${b}) applied to (${u}, ${w})`;
      same(g0, y0, `${what}[0]`);
      same(g1, y1, `${what}[1]`);
    }
  });

  test(`${name}: ${fv.drift.k} rotations of the drift figure`, () => {
    const { c, s, k, start, end } = fv.drift;
    start.forEach(([x, y], i) => {
      for (let j = 0; j < k; ++j)
        [x, y] = [ops.sub(ops.mul(c, x), ops.mul(s, y)), ops.add(ops.mul(s, x), ops.mul(c, y))];
      same(x, end[i][0], `point ${i} x`);
      same(y, end[i][1], `point ${i} y`);
    });
  });
}
