#!/usr/bin/env python3
"""Wall-clock tracking: compare a machine's session with its previous one, across commits (#944).

The per-PR gate (benchmarks/regression/) counts instructions, which are
deterministic but blind to caches, SIMD width and threading. This is the
other half: the platform benchmark, timed on the benchmark machines, compared
with the same machine's last published session. It reports; it never gates.

    track_sessions.py compare BASELINE NEW [--run-band 2] [--entry-band 10]
                                           [--markdown FILE] [--json FILE]
    track_sessions.py page [--data benchmarks/data] [--out docs/benchmarks/tracking.md]

compare: BASELINE and NEW are session directories (platform.sh results, or
published benchmarks/data/<machine>/mtl5_bench/<session>/) of the same machine
and class, normally at different commits. Runs are matched by file name
(<variant>-t<T>.json), entries by (area, op, dtype, shape), as in
compare_sessions.py, which measures the noise between two sessions of ONE
commit. For each run:
  - the ratio NEW/BASELINE of each entry's median time, for the entries whose
    checks passed in both;
  - the median of those ratios, the rule of analyze_gate.py (#327): one entry's
    ratio moves several percent between sessions of the same commit, while the
    median of a run's ratios repeats to about 0.5% (the cards' repeatability
    notes: median ratio 0.998 to 1.005 per run). A run whose median is outside
    +-run-band % is marked "slower" or "faster", else "steady";
  - the entries outside +-entry-band %, slowest first. About 5-10% of entries
    move more than 5% between sessions of one commit, so the default band is
    10%: an entry listed here is worth a look, not yet a finding.
It also lists what changed in the build and the machine between the sessions
(compiler, flags, kernel, pin list, clock policy, ...): a shift that comes with
a new compiler is not a regression in MTL5.

--json also writes FILE.sysinfo, the report's provenance (the two sessions),
since every JSON under benchmarks/data carries a sidecar (check_sidecars.sh).

page: collects the --json reports under DATA/<machine>/tracking/ and writes
the tracking page: per machine, the latest report's runs, and its history.

Exit: 0 when the report was written, whatever it found (slower runs are a
finding, not a failure); 1 when the sessions don't match (another machine or
class, no run in common, an unreadable session); 2 on bad usage.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

# What to show when it differs between the sessions: the build's provenance and
# the machine's state. Keys missing from a sidecar read as "<not recorded>".
ENV_KEYS = (
    "compiler", "compiler_version", "cxx_flags", "build_isa", "simd_backend",
    "blas_vendor", "os_version", "preflight_kernel", "pool_size", "bench_cpus",
    "clock_policy", "governor", "turbo", "power_mode",
)


def sidecar(path):
    """The key=value sidecar of a run, as a dict (first value of each key)."""
    out = {}
    p = Path(str(path) + ".sysinfo")
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out.setdefault(k.strip(), v.strip())
    return out


def entries(path):
    """The run's JSON, and its entries by (area, op, dtype, shape): (median_s or None, ok)."""
    data = json.loads(Path(path).read_text())
    res = {}
    for r in data.get("results", []):
        key = (r.get("area"), r.get("op"), r.get("dtype"), r.get("shape"))
        check = r.get("check") or {}
        ok = bool(check.get("ok")) and not r.get("error")
        t = r.get("median_s")
        res[key] = (t if isinstance(t, (int, float)) and t > 0 else None, ok)
    return data, res


def label(key):
    return "/".join(str(k) for k in key[:3]) + f" {key[3]}"


def session_info(sdir):
    """(machine, class, commit) of a session, from its runs; None where they disagree."""
    machines, classes, commits = set(), set(), set()
    for p in sorted(sdir.glob("*-t*.json")):
        data = json.loads(p.read_text())
        sc = sidecar(p)
        machines.add(data.get("machine") or sc.get("machine"))
        classes.add(data.get("class") or sc.get("class"))
        commits.add(sc.get("git_commit"))
    one = lambda s: next(iter(s)) if len(s) == 1 else None  # noqa: E731
    return one(machines), one(classes), one(commits)


def compare_run(base_path, new_path, entry_band):
    """One run's comparison: the ratios, their median, and the entries outside the band."""
    _, base = entries(base_path)
    _, new = entries(new_path)
    ratios, outside, skipped = [], [], []
    for key in sorted(set(base) | set(new), key=label):
        if key not in base or key not in new:
            skipped.append((label(key), "only in the new session" if key in new else "only in the baseline"))
            continue
        (tb, okb), (tn, okn) = base[key], new[key]
        if not (okb and okn):
            skipped.append((label(key), "failed its check"))
            continue
        if tb is None or tn is None:
            skipped.append((label(key), "no timing"))
            continue
        r = tn / tb
        ratios.append(r)
        if abs(r - 1.0) * 100.0 > entry_band:
            outside.append({"entry": label(key), "baseline_s": tb, "new_s": tn, "ratio": r})
    outside.sort(key=lambda e: -e["ratio"])
    med = statistics.median(ratios) if ratios else None
    return {"ratios": ratios, "median": med, "outside": outside, "skipped": skipped}


def env_changes(base_dir, new_dir, runs):
    """The ENV_KEYS that differ between the sessions, per key: {run: (baseline, new)}."""
    changes = {}
    for run in runs:
        b, n = sidecar(base_dir / run), sidecar(new_dir / run)
        for k in ENV_KEYS:
            vb, vn = b.get(k, "<not recorded>"), n.get(k, "<not recorded>")
            if vb != vn:
                changes.setdefault(k, {})[run] = (vb, vn)
    return changes


def verdict(median, run_band):
    if median is None:
        return "no entries"
    dev = (median - 1.0) * 100.0
    if dev > run_band:
        return "slower"
    if dev < -run_band:
        return "faster"
    return "steady"


def cmd_compare(a):
    base_dir, new_dir = Path(a.baseline), Path(a.new)
    for s in (base_dir, new_dir):
        if not s.is_dir():
            print(f"track_sessions: {s} is not a directory", file=sys.stderr)
            return 2
    try:
        bm, bc, bcommit = session_info(base_dir)
        nm, nc, ncommit = session_info(new_dir)
    except (OSError, ValueError) as e:
        print(f"track_sessions: cannot read a session: {e}", file=sys.stderr)
        return 1
    if bm is None or nm is None or bm != nm:
        print(f"track_sessions: the sessions are not of one machine (baseline {bm}, new {nm})", file=sys.stderr)
        return 1
    if bc is None or nc is None or bc != nc:
        # Another class measures other sizes: the ratio would compare problems, not commits.
        print(f"track_sessions: the sessions are not of one class (baseline {bc}, new {nc})", file=sys.stderr)
        return 1

    runs_b = {p.name for p in base_dir.glob("*-t*.json")}
    runs_n = {p.name for p in new_dir.glob("*-t*.json")}
    runs = sorted(runs_b & runs_n)
    if not runs:
        print("track_sessions: the sessions have no run in common", file=sys.stderr)
        return 1
    only = sorted((runs_b - runs_n) | (runs_n - runs_b))

    report = {
        "schema": "mtl5-track/1",
        "machine": nm,
        "class": nc,
        "baseline": {"session": base_dir.name, "commit": bcommit},
        "new": {"session": new_dir.name, "commit": ncommit},
        "run_band_pct": a.run_band,
        "entry_band_pct": a.entry_band,
        "runs": [],
        "runs_not_compared": only,
        "env_changes": {k: {r: list(v) for r, v in d.items()} for k, d in env_changes(base_dir, new_dir, runs).items()},
    }
    for run in runs:
        c = compare_run(base_dir / run, new_dir / run, a.entry_band)
        rs = c["ratios"]
        report["runs"].append({
            "run": run[:-len(".json")],
            "entries": len(rs),
            "median_ratio": c["median"],
            "min_ratio": min(rs) if rs else None,
            "max_ratio": max(rs) if rs else None,
            "verdict": verdict(c["median"], a.run_band),
            "outside": c["outside"],
            "skipped": len(c["skipped"]),
        })

    md = render_report(report)
    print(md, end="")
    if a.markdown:
        Path(a.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown).write_text(md)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
        # Every committed JSON under benchmarks/data carries a sidecar (#477,
        # check_sidecars.sh). A report is derived data: its provenance is the two
        # sessions it compares, whose own sidecars hold the build and machine.
        Path(a.json + ".sysinfo").write_text("".join(f"{k}={v}\n" for k, v in (
            ("derived_by", "benchmarks/tracking/track_sessions.py compare"),
            ("schema", report["schema"]),
            ("machine", nm),
            ("class", nc),
            ("baseline_session", base_dir.name),
            ("baseline_commit", bcommit),
            ("new_session", new_dir.name),
            ("new_commit", ncommit),
            ("run_band_pct", f"{a.run_band:g}"),
            ("entry_band_pct", f"{a.entry_band:g}"),
        )))
    return 0


def fmt_ratio(r):
    return "—" if r is None else f"{r:.3f}"


def render_report(rep):
    b, n = rep["baseline"], rep["new"]
    lines = [
        f"# Wall-clock tracking: {rep['machine']}, session {n['session']}",
        "",
        f"Session `{n['session']}` (commit `{n['commit']}`) against the previous session "
        f"`{b['session']}` (commit `{b['commit']}`), class {rep['class']}. "
        f"A ratio is new time / baseline time, so above 1 is slower. A run is marked "
        f"slower or faster when the median of its entries' ratios is outside "
        f"±{rep['run_band_pct']:g}%; entries outside ±{rep['entry_band_pct']:g}% are listed. "
        "Report only (#944): nothing here gates a merge.",
        "",
        "| run | entries compared | not compared | median ratio | min | max | outside the entry band | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rep["runs"]:
        lines.append(f"| {r['run']} | {r['entries']} | {r['skipped']} | {fmt_ratio(r['median_ratio'])} | "
                     f"{fmt_ratio(r['min_ratio'])} | {fmt_ratio(r['max_ratio'])} | "
                     f"{len(r['outside'])} | {r['verdict']} |")
    lines += ["", "An entry is not compared when it is in only one session (added or removed "
              "between the commits), failed its check in either, or has no timing.", ""]
    if rep["runs_not_compared"]:
        lines += ["Runs in only one session, not compared: "
                  + ", ".join(f"`{x}`" for x in rep["runs_not_compared"]) + ".", ""]
    if rep["env_changes"]:
        lines += ["## What changed besides the commit", "",
                  "A shift that comes with one of these is not necessarily MTL5's.", "",
                  "| key | runs | baseline | new |", "|---|---|---|---|"]
        for k, d in sorted(rep["env_changes"].items()):
            # Group the runs that changed the same way, so one compiler upgrade is one row.
            groups = {}
            for run, (vb, vn) in sorted(d.items()):
                groups.setdefault((vb, vn), []).append(run[:-len(".json")])
            for (vb, vn), rs in groups.items():
                lines.append(f"| `{k}` | {', '.join(rs)} | `{esc(vb)}` | `{esc(vn)}` |")
        lines.append("")
    else:
        lines += ["The build and the machine state are the same in both sessions, as far as "
                  "the sidecars record them.", ""]
    for r in rep["runs"]:
        if not r["outside"]:
            continue
        lines += [f"## {r['run']}: entries outside ±{rep['entry_band_pct']:g}%", "",
                  "| entry | baseline (s) | new (s) | ratio |", "|---|---:|---:|---:|"]
        for e in r["outside"]:
            lines.append(f"| {esc(e['entry'])} | {e['baseline_s']:.4g} | {e['new_s']:.4g} | {e['ratio']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def esc(s):
    return str(s).replace("|", "\\|")


def cmd_page(a):
    data = Path(a.data)
    if not data.is_dir():
        print(f"track_sessions: {data} is not a directory", file=sys.stderr)
        return 2
    reports = {}
    for p in sorted(data.glob("*/tracking/*.json")):
        try:
            rep = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            print(f"track_sessions: cannot read {p}: {e}", file=sys.stderr)
            return 1
        if rep.get("schema") != "mtl5-track/1":
            continue
        reports.setdefault(rep["machine"], []).append((p, rep))
    out = Path(a.out)
    # Links go to GitHub, as the cards' raw-data links do: the site serves docs/, not benchmarks/.
    root = data.resolve().parent.parent
    blob = "https://github.com/stillwater-sc/mtl5/blob/main/"
    lines = [
        "# Wall-clock tracking",
        "",
        "When the lab machines are on, each runs `mtl5_bench` on `main` (on demand), and each session",
        "is compared with that machine's previous one (#944). The instruction-count gate on",
        "every PR (`bench-regression.yml`) catches extra work. This page catches what",
        "instruction counts cannot see: caches, vectorization and threading. It is a report and",
        "gates nothing.",
        "",
        "A ratio is new time / previous time, so above 1 is slower. A run's ratio is the median",
        "over its entries, which repeats to about 0.5% between two sessions of one commit (the",
        "cards' repeatability notes). A run is marked *slower* or *faster* outside its report's",
        "run band. How the sessions are run and compared is in",
        f"[`benchmarks/tracking/README.md`]({blob}benchmarks/tracking/README.md).",
        "",
    ]
    if not reports:
        lines += ["No tracking session has been published yet.", ""]
    for machine in sorted(reports):
        reps = sorted(reports[machine], key=lambda t: t[1]["new"]["session"])
        p, last = reps[-1]
        rel = p.with_suffix(".md").resolve().relative_to(root)
        b, n = last["baseline"], last["new"]
        lines += [
            f"## {machine}",
            "",
            f"Latest: session `{n['session']}` (commit `{n['commit']}`) against `{b['session']}` "
            f"(commit `{b['commit']}`), class {last['class']}. [Full report]({blob}{rel.as_posix()}).",
            "",
            "| run | median ratio | outside ±{:g}% | verdict |".format(last["entry_band_pct"]),
            "|---|---:|---:|---|",
        ]
        for r in last["runs"]:
            lines.append(f"| {r['run']} | {fmt_ratio(r['median_ratio'])} | {len(r['outside'])} | {r['verdict']} |")
        lines.append("")
        if last.get("env_changes"):
            lines += ["Changed besides the commit: "
                      + ", ".join(f"`{k}`" for k in sorted(last["env_changes"])) + ".", ""]
        if len(reps) > 1:
            lines += ["History, the median ratio of each run:", "",
                      "| session | commit | " + " | ".join(r["run"] for r in last["runs"]) + " |",
                      "|---|---|" + "---:|" * len(last["runs"])]
            for _, rep in reversed(reps):
                by = {r["run"]: r for r in rep["runs"]}
                cells = [fmt_ratio(by[r["run"]]["median_ratio"]) if r["run"] in by else "—" for r in last["runs"]]
                lines.append(f"| `{rep['new']['session']}` | `{rep['new']['commit']}` | " + " | ".join(cells) + " |")
            lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"track_sessions: wrote {out} ({sum(len(v) for v in reports.values())} reports, {len(reports)} machines)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="compare a session with the machine's previous one")
    c.add_argument("baseline")
    c.add_argument("new")
    c.add_argument("--run-band", type=float, default=2.0, help="run median band in percent (default 2)")
    c.add_argument("--entry-band", type=float, default=10.0, help="per-entry band in percent (default 10)")
    c.add_argument("--markdown", help="also write the report to this Markdown file")
    c.add_argument("--json", help="also write the report as JSON (what `page` reads)")
    p = sub.add_parser("page", help="write the tracking page from the JSON reports")
    p.add_argument("--data", default="benchmarks/data")
    p.add_argument("--out", default="docs/benchmarks/tracking.md")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        if a.run_band <= 0 or a.entry_band <= 0:
            ap.error("the bands must be positive")
        return cmd_compare(a)
    return cmd_page(a)


if __name__ == "__main__":
    sys.exit(main())
