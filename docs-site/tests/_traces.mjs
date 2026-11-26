// Shared by the example animation tests (#933): load a committed trace.
import { readFileSync } from 'node:fs';

export function loadTrace(name) {
  return JSON.parse(readFileSync(new URL(`../src/data/traces/${name}.json`, import.meta.url), 'utf-8'));
}

/** |a - b| <= tol * max(1, |a|, |b|) */
export function close(a, b, tol = 1e-9) {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}
