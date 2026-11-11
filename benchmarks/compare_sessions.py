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
outside the band, and a one-line band for the card page. Entries that failed
their check or have no timing in either session, or exist in only one, are
listed and not counted. --markdown also writes that report as a Markdown file.
Exit 0 when the comparison ran (noisy entries are a finding, not a failure);
1 when the sessions don't match: a run in only one of them, or a commit or
class that differs or isn't recorded; 2 on bad usage.
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
    """The run's entries by (area, op, dtype, shape): (median_s or None, ok)."""
    data = json.loads(Path(path).read_text())
    res = {}
    for r in data.get("results", []):
        key = (r.get("area"), r.get("op"), r.get("dtype"), r.get("shape"))
        check = r.get("check") or {}
        ok = bool(check.get("ok")) and not r.get("error")
        res[key] = (r.get("median_s") or None, ok)
    return data, res


def label(key):
    return "/".join(str(k) for k in key[:3]) + f" {key[3]}"


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

    runs1 = {p.name for p in s1.glob("*-t*.json")}
    runs2 = {p.name for p in s2.glob("*-t*.json")}
    only = sorted((runs1 - runs2) | (runs2 - runs1))
    runs = sorted(runs1 & runs2)
    if only:
        # A partial session would give a complete-looking band over fewer runs.
        for n in only:
            print(f"compare_sessions: {n} is in only one session "
                  f"({s1 if n in runs1 else s2})", file=sys.stderr)
        return 1
    if not runs:
        print("compare_sessions: the sessions have no <variant>-t<T>.json runs", file=sys.stderr)
        return 1

    lo, hi = 1 - a.band / 100, 1 + a.band / 100
    lines = [f"# Repeatability: `{s2.name}` against `{s1.name}`", "",
             f"Ratio of session-2 to session-1 median time per entry; band +-{a.band:g}%. "
             "Entries that failed their check, or have no timing, in either session are "
             "listed separately and not counted.", "",
             "| run | compared | median ratio | min | max | outside band | not compared |",
             "|---|---|---|---|---|---|---|"]
    noisy_all, uncompared, all_ratios = [], [], []
    for name in runs:
        d1, e1 = entries(s1 / name)
        d2, e2 = entries(s2 / name)
        c1, c2 = sidecar(s1 / name), sidecar(s2 / name)
        meta = (c1.get("git_commit"), d1.get("class"), c2.get("git_commit"), d2.get("class"))
        if not all(meta):
            print(f"compare_sessions: {name}: commit or class not recorded in one of the sessions "
                  f"({meta}) -- the provenance cannot be matched", file=sys.stderr)
            return 1
        if meta[0] != meta[2] or meta[1] != meta[3]:
            print(f"compare_sessions: {name}: commit {meta[0]} class {meta[1]} against commit "
                  f"{meta[2]} class {meta[3]} -- not the same build, so the ratio would measure "
                  "the change, not the noise", file=sys.stderr)
            return 1
        ratios = []
        n_unc = 0
        for key in sorted(set(e1) | set(e2), key=str):
            m1, ok1 = e1.get(key, (None, False))
            m2, ok2 = e2.get(key, (None, False))
            why = []
            if key not in e1 or key not in e2:
                why.append("missing in session " + ("1" if key not in e1 else "2"))
            if not (m1 and m2):
                why.append("no timing")
            if key in e1 and key in e2 and not (ok1 and ok2):
                why.append("check failed in session " + ",".join(
                    str(i) for i, ok in ((1, ok1), (2, ok2)) if not ok))
            if why:
                uncompared.append((name, label(key), "; ".join(why)))
                n_unc += 1
                continue
            q = m2 / m1
            ratios.append(q)
            if not lo <= q <= hi:
                noisy_all.append((name, label(key), q))
        all_ratios += ratios
        if ratios:
            out = sum(1 for q in ratios if not lo <= q <= hi)
            lines.append(f"| `{name[:-5]}` | {len(ratios)} | {statistics.median(ratios):.3f} | "
                         f"{min(ratios):.3f} | {max(ratios):.3f} | {out} | {n_unc} |")
        else:
            lines.append(f"| `{name[:-5]}` | 0 | -- | -- | -- | -- | {n_unc} |")

    within = sum(1 for q in all_ratios if lo <= q <= hi)
    median = f"{statistics.median(all_ratios):.3f}" if all_ratios else "--"
    band_line = (f"Repeatability: {within} of {len(all_ratios)} compared entries within "
                 f"+-{a.band:g}% between sessions `{s1.name}` and `{s2.name}` (median ratio "
                 f"{median}); {len(uncompared)} not compared.")
    lines += ["", band_line, ""]
    if noisy_all:
        lines += ["Entries outside the band (noisy):", "", "| run | entry | ratio |", "|---|---|---|"]
        lines += [f"| `{r[:-5]}` | `{e}` | {q:.3f} |" for r, e, q in sorted(noisy_all, key=lambda t: -abs(t[2] - 1))]
        lines.append("")
    if uncompared:
        lines += ["Entries not compared:", "", "| run | entry | why |", "|---|---|---|"]
        lines += [f"| `{r[:-5]}` | `{e}` | {w} |" for r, e, w in uncompared]
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if a.markdown:
        Path(a.markdown).write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
