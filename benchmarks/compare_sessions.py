#!/usr/bin/env python3
"""Repeatability of an mtl5_bench session: compare two sessions entry by entry (#737).

#737 runs every machine twice, on different days or after a cold boot, and
publishes the first session with its repeatability band: the per-entry ratio
of session-2 to session-1 medians. An entry outside the band (+-5% by default)
is marked noisy on the card.

Usage:
    benchmarks/compare_sessions.py SESSION1 SESSION2 [--band 5] [--markdown FILE]

Each SESSION is a platform.sh results directory (or a published
benchmarks/data/<machine>/mtl5_bench/<session>/). Runs are matched by file
name (<variant>-t<T>.json), entries by (area, op, dtype, shape). The two
sessions must be the same commit and class; a mismatch is an error, because
then the ratio measures the change, not the noise.

Prints, per run, the median and the spread of the ratios and the entries
outside the band, and a one-line band for the card page. --markdown also
writes that report as a Markdown file. Exit 0 when the comparison ran (noisy
entries are a finding, not a failure), 1 when the sessions don't match, 2 on
bad usage.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def sidecar(path):
    """The key=value sidecar of a run, as a dict (first value of each key)."""
    out = {}
    p = Path(str(path) + ".sysinfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out.setdefault(k.strip(), v.strip())
    return out


def entries(path):
    data = json.loads(Path(path).read_text())
    res = {}
    for r in data.get("results", []):
        key = (r.get("area"), r.get("op"), r.get("dtype"), r.get("shape"))
        if r.get("median_s"):
            res[key] = r
    return data, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session1")
    ap.add_argument("session2")
    ap.add_argument("--band", type=float, default=5.0, help="noise band in percent (default 5)")
    ap.add_argument("--markdown", help="also write the report to this Markdown file")
    a = ap.parse_args()
    s1, s2 = Path(a.session1), Path(a.session2)
    for s in (s1, s2):
        if not s.is_dir():
            print(f"compare_sessions: {s} is not a directory", file=sys.stderr)
            return 2

    runs = sorted(p.name for p in s1.glob("*-t*.json") if (s2 / p.name).exists())
    if not runs:
        print("compare_sessions: the sessions share no <variant>-t<T>.json runs", file=sys.stderr)
        return 1

    lo, hi = 1 - a.band / 100, 1 + a.band / 100
    lines = [f"# Repeatability: `{s2.name}` against `{s1.name}`", "",
             f"Ratio of session-2 to session-1 median time per entry; band +-{a.band:g}%.", "",
             "| run | entries | median ratio | min | max | outside band |",
             "|---|---|---|---|---|---|"]
    noisy_all = []
    all_ratios = []
    for name in runs:
        d1, e1 = entries(s1 / name)
        d2, e2 = entries(s2 / name)
        c1, c2 = sidecar(s1 / name), sidecar(s2 / name)
        if c1.get("git_commit") != c2.get("git_commit") or d1.get("class") != d2.get("class"):
            print(f"compare_sessions: {name}: commit {c1.get('git_commit')} class {d1.get('class')} "
                  f"against commit {c2.get('git_commit')} class {d2.get('class')} -- not the same "
                  "build, so the ratio would measure the change, not the noise", file=sys.stderr)
            return 1
        ratios = []
        for key, r1 in e1.items():
            r2 = e2.get(key)
            if r2 is None:
                continue
            q = r2["median_s"] / r1["median_s"]
            ratios.append(q)
            if not lo <= q <= hi:
                noisy_all.append((name, "/".join(str(k) for k in key[:3]), key[3], q))
        all_ratios += ratios
        out = sum(1 for q in ratios if not lo <= q <= hi)
        lines.append(f"| `{name[:-5]}` | {len(ratios)} | {statistics.median(ratios):.3f} | "
                     f"{min(ratios):.3f} | {max(ratios):.3f} | {out} |")

    within = sum(1 for q in all_ratios if lo <= q <= hi)
    band_line = (f"Repeatability: {within} of {len(all_ratios)} entries within +-{a.band:g}% "
                 f"between sessions `{s1.name}` and `{s2.name}` (median ratio "
                 f"{statistics.median(all_ratios):.3f}).")
    lines += ["", band_line, ""]
    if noisy_all:
        lines += ["Entries outside the band (noisy):", "", "| run | entry | shape | ratio |", "|---|---|---|---|"]
        lines += [f"| `{r[:-5]}` | `{e}` | {sh} | {q:.3f} |" for r, e, sh, q in sorted(noisy_all, key=lambda t: -abs(t[3] - 1))]
    report = "\n".join(lines) + "\n"
    print(report)
    if a.markdown:
        Path(a.markdown).write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
