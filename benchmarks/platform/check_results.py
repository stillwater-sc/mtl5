#!/usr/bin/env python3
"""Check the fields of mtl5_bench result files (#928, #929).

    check_results.py FILE.json [FILE.json ...] [--require-samples]

For every result of every file:
  - `version` is an integer >= 1 (#928);
  - a timed result (median_s > 0) has p10_s, p25_s, p75_s, p90_s and max_s,
    in order: min_s <= p10_s <= p25_s <= median_s <= p75_s <= p90_s <= max_s,
    and p10_s == min_s when samples < 10 (#929);
  - where `samples_s` is written (mtl5_bench --samples), it has `samples`
    values, sorted, and each quantile is the sample the documented rule picks:
    sorted[min(n - 1, floor(q n))], with min_s the first and max_s the last;
  - no (area, op, dtype, shape) appears twice.
--require-samples also requires `samples_s` on every timed result.

Standard library only. Exit 0 when every file passes, 1 when one does not,
2 on bad usage or an unreadable file.
"""
import argparse
import json
import math
import sys
from pathlib import Path

QUANTILES = (("p10_s", 0.10), ("p25_s", 0.25), ("median_s", 0.50), ("p75_s", 0.75), ("p90_s", 0.90))


def finite_number(x):
    """A finite int or float, not a bool. Python's json reads NaN and Infinity,
    and every ordering comparison with NaN is false, so the order checks below
    would pass one (#973 review)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def quantile_index(n, q):
    """The index of the q-quantile of n sorted samples: min(n - 1, floor(q n))."""
    return min(n - 1, int(math.floor(q * n)))


def check_result(r, require_samples):
    """Problems with one result, as strings."""
    out = []
    v = r.get("version")
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        out.append(f"version must be an integer >= 1, got {v!r}")
    med = r.get("median_s")
    if med is None or (finite_number(med) and med <= 0):
        return out   # not timed (failed, or threw): nothing more to check
    if not finite_number(med):
        out.append(f"median_s is not a finite number ({med!r})")
        return out
    names = ["min_s"] + [k for k, _ in QUANTILES[:2]] + ["median_s"] + [k for k, _ in QUANTILES[3:]] + ["max_s"]
    vals = []
    for k in names:
        x = r.get(k)
        if not finite_number(x):
            out.append(f"{k} missing or not a finite number ({x!r})")
            return out
        vals.append(x)
    for (a, xa), (b, xb) in zip(zip(names, vals), zip(names[1:], vals[1:])):
        if xa > xb:
            out.append(f"{a} = {xa} > {b} = {xb}")
    n = r.get("samples")
    if not isinstance(n, int) or n < 1:
        out.append(f"samples must be a positive integer, got {n!r}")
        return out
    if n < 10 and r["p10_s"] != r["min_s"]:
        out.append(f"samples = {n} < 10 but p10_s {r['p10_s']} != min_s {r['min_s']}")
    s = r.get("samples_s")
    if s is None:
        if require_samples:
            out.append("samples_s missing (--require-samples)")
        return out
    if not isinstance(s, list) or len(s) != n:
        out.append(f"samples_s has {len(s) if isinstance(s, list) else '?'} values, samples = {n}")
        return out
    # Every value is checked before any is compared: a string among them made
    # the sort check raise TypeError, and the remaining files went unchecked.
    bad = [i for i, x in enumerate(s) if not finite_number(x)]
    if bad:
        out.append(f"samples_s[{bad[0]}] is not a finite number ({s[bad[0]]!r})"
                   + (f", and {len(bad) - 1} more" if len(bad) > 1 else ""))
        return out
    if any(a > b for a, b in zip(s, s[1:])):
        out.append("samples_s is not sorted")
    if r["min_s"] != s[0]:
        out.append(f"min_s {r['min_s']} != samples_s[0] {s[0]}")
    if r["max_s"] != s[-1]:
        out.append(f"max_s {r['max_s']} != samples_s[-1] {s[-1]}")
    for k, q in QUANTILES:
        i = quantile_index(n, q)
        if r[k] != s[i]:
            out.append(f"{k} {r[k]} != samples_s[{i}] {s[i]} (n = {n})")
    return out


def check_file(path, require_samples):
    """(problems, number of results) for one file; raises on an unreadable one."""
    data = json.loads(Path(path).read_text())
    problems = []
    if data.get("schema") != "mtl5-bench/1":
        problems.append(f"schema is {data.get('schema')!r}, not 'mtl5-bench/1'")
    seen = set()
    results = data.get("results", [])
    for r in results:
        key = (r.get("area"), r.get("op"), r.get("dtype"), r.get("shape"))
        label = "/".join(str(k) for k in key[:3]) + f" {key[3]}"
        if key in seen:
            problems.append(f"{label}: listed twice")
        seen.add(key)
        problems += [f"{label}: {p}" for p in check_result(r, require_samples)]
    return problems, len(results)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--require-samples", action="store_true",
                    help="require samples_s on every timed result")
    a = ap.parse_args(argv)
    bad = 0
    for f in a.files:
        try:
            problems, n = check_file(f, a.require_samples)
        except (OSError, ValueError) as e:
            print(f"check_results: cannot read {f}: {e}", file=sys.stderr)
            return 2
        if problems:
            bad += 1
            print(f"{f}: {len(problems)} problem(s)")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"{f}: OK ({n} results)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
