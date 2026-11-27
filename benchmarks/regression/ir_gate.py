#!/usr/bin/env python3
"""The instruction-count regression gate (#944).

Two commands, standard library only:

  ir_gate.py count   --bench build/benchmarks/mtl5_bench --kernels kernels.txt
                     --config native --out counts.json [--valgrind valgrind]
      Runs `mtl5_bench --count ID` under callgrind for each kernel, counting
      only the call to mtl5_bench_counted_region (--toggle-collect), and writes
      the instruction counts (Ir) as JSON. Fails if a kernel's correctness
      check fails or valgrind does not report a count.

  ir_gate.py compare --baseline main.json --current counts.json
                     --accepted accepted.json [--threshold 0.02] [--summary FILE]
      Compares a run against main's baseline. A kernel whose count rose by more
      than the threshold is a regression, and fails the gate, unless
      accepted.json names it with a max_ir at or above the new count. The table
      is printed, and appended to --summary (the job summary) when given. A
      missing baseline file (the first run, or an expired artifact) passes,
      with a note.

Instruction counts, unlike wall-clock times, repeat exactly from run to run on
one binary, and do not depend on the runner's load. They do not see cache,
SIMD-width or threading effects; the wall-clock tracking on the benchmark
machines does (benchmarks/regression/README.md).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REGION = "mtl5_bench_counted_region"

# The environment every count runs in. One thread, so that no count depends on
# a scheduler. The GEMM blocking (kc, mc) and the nc model are derived from the
# detected caches, which differ between runners (valgrind passes the host's
# cache description through), so they are pinned: kc and mc through
# MTL5_ENABLE_BLOCKING_OVERRIDE, which the build must define, and nc by the
# model that does not read the L3. The counts then measure the kernels' code,
# not the runner. How the blocking responds to a cache is a wall-clock question.
PINNED_ENV = {
    "MTL5_NUM_THREADS": "1",
    "MTL5_BLOCKING_KC": "256",
    "MTL5_BLOCKING_MC": "96",
    "MTL5_NC_MODEL": "m0_default",
}


# -- count ---------------------------------------------------------------------

def read_kernels(path):
    """Kernel ids from a kernels.txt: one per line, '#' comments, blank lines."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: a kernel is listed twice")
    return ids


def parse_callgrind_totals(text):
    """The Ir total from a callgrind output file: its `totals:` line (or
    `summary:`, which older versions write), whose first number is Ir."""
    events, value = None, None
    for line in text.splitlines():
        if line.startswith("events:"):
            events = line.split()[1:]
        elif line.startswith("totals:") or (value is None and line.startswith("summary:")):
            value = int(line.split()[1])
    if events is not None and (not events or events[0] != "Ir"):
        raise ValueError(f"callgrind output's first event is {events[:1]}, not Ir")
    if value is None:
        raise ValueError("callgrind output has no totals line")
    return value


def count_one(valgrind, bench, kernel, cls):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "callgrind.out")
        cmd = valgrind + ["--tool=callgrind", f"--toggle-collect={REGION}",
                          f"--callgrind-out-file={out}", bench, "--count", kernel, "--class", cls]
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **PINNED_ENV})
        if p.returncode != 0:
            raise RuntimeError(f"{kernel}: exit {p.returncode}\n{p.stdout}{p.stderr}")
        if "blocking_override=on" not in p.stdout:
            raise RuntimeError(f"{kernel}: mtl5_bench was built without "
                               "-DMTL5_ENABLE_BLOCKING_OVERRIDE, so the GEMM blocking "
                               "would follow this machine's caches")
        with open(out, encoding="utf-8") as f:
            ir = parse_callgrind_totals(f.read())
        if ir <= 0:
            raise RuntimeError(f"{kernel}: callgrind counted nothing in {REGION}")
        return ir, p.stdout.strip()


def cmd_count(a):
    kernels = read_kernels(a.kernels)
    valgrind = a.valgrind.split()
    version = subprocess.run(valgrind + ["--version"], capture_output=True, text=True).stdout.strip()
    counts, failed = {}, []
    for k in kernels:
        try:
            ir, line = count_one(valgrind, a.bench, k, a.cls)
            counts[k] = ir
            print(f"{ir:>14,}  {line}", flush=True)
        except (RuntimeError, ValueError, OSError) as e:
            failed.append(k)
            print(f"error: {e}", file=sys.stderr, flush=True)
    doc = {"config": a.config, "class": a.cls, "valgrind": version, "env": PINNED_ENV,
           "kernels": counts}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    if failed:
        print(f"count: {len(failed)} kernel(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


# -- compare -------------------------------------------------------------------

def validate_accepted(accepted):
    """Problems with an accepted.json document, as strings."""
    problems = []
    kernels = accepted.get("kernels", {})
    if not isinstance(kernels, dict):
        return ["accepted.json: 'kernels' is not an object"]
    for k, v in kernels.items():
        if not isinstance(v, dict):
            problems.append(f"accepted.json: {k}: not an object")
            continue
        if not isinstance(v.get("max_ir"), int) or v["max_ir"] <= 0:
            problems.append(f"accepted.json: {k}: max_ir must be a positive integer")
        if not str(v.get("reason", "")).strip():
            problems.append(f"accepted.json: {k}: a reason is required")
    return problems


def compare(baseline, current, accepted, threshold):
    """Rows (kernel, base, cur, ratio, status), notes, and whether the gate
    fails. `baseline` is None when there is none. Statuses: ok, faster,
    REGRESSION, accepted, new, removed."""
    notes, failed = [], False
    problems = validate_accepted(accepted)
    if problems:
        notes += problems
        failed = True
    acc = accepted.get("kernels", {}) if not problems else {}

    if baseline is None:
        notes.append("No baseline from main: nothing to compare (the first run, or the "
                     "artifact expired). The counts are reported, and the gate passes.")
        rows = [(k, None, v, None, "new") for k, v in sorted(current["kernels"].items())]
        return rows, notes, failed

    for key in ("config", "class", "env"):
        if baseline.get(key) != current.get(key):
            notes.append(f"The baseline's {key} is {baseline.get(key)!r}, this run's is "
                         f"{current.get(key)!r}: the counts are not comparable.")
            failed = True
    if baseline.get("valgrind") != current.get("valgrind"):
        notes.append(f"valgrind differs: baseline {baseline.get('valgrind')!r}, this run "
                     f"{current.get('valgrind')!r}. Counts are compared anyway.")

    rows = []
    base_k, cur_k = baseline.get("kernels", {}), current.get("kernels", {})
    for k in sorted(set(base_k) | set(cur_k)):
        b, c = base_k.get(k), cur_k.get(k)
        if b is None:
            rows.append((k, None, c, None, "new"))
            continue
        if c is None:
            rows.append((k, b, None, None, "removed"))
            continue
        ratio = c / b if b > 0 else float("inf")
        if ratio > 1.0 + threshold:
            a = acc.get(k)
            if a is not None and c <= a["max_ir"]:
                status = "accepted"
            else:
                status = "REGRESSION"
                failed = True
                if a is not None:
                    notes.append(f"{k}: {c:,} is above its accepted max_ir {a['max_ir']:,}.")
        elif ratio < 1.0 - threshold:
            status = "faster"
        else:
            status = "ok"
        rows.append((k, b, c, ratio, status))

    for k, a in sorted(acc.items()):
        b = base_k.get(k)
        if b is not None and a["max_ir"] <= b * (1.0 + threshold):
            notes.append(f"{k}: the acceptance (max_ir {a['max_ir']:,}) is stale: main's "
                         f"baseline {b:,} already carries it. Remove it from accepted.json.")
        elif k not in cur_k:
            notes.append(f"{k}: accepted, but not measured by this run.")
    return rows, notes, failed


def render(rows, notes, failed, threshold, config):
    def num(x):
        return "" if x is None else f"{x:,}"

    def pct(r):
        return "" if r is None else f"{(r - 1.0) * 100.0:+.2f}%"

    out = [f"### Instruction counts: `{config}`, threshold {threshold * 100:g}%", "",
           "| kernel | main | this PR | change | status |", "|---|---:|---:|---:|---|"]
    for k, b, c, r, s in rows:
        shown = f"**{s}**" if s == "REGRESSION" else s
        out.append(f"| `{k}` | {num(b)} | {num(c)} | {pct(r)} | {shown} |")
    out.append("")
    for n in notes:
        out.append(f"- {n}")
    if notes:
        out.append("")
    if failed:
        out.append("**The gate fails.** A deliberate increase is accepted by adding the kernel to "
                   "`benchmarks/regression/accepted.json` in this PR, with this run's count as "
                   "`max_ir` and the reason.")
    else:
        out.append("The gate passes.")
    out.append("")
    return "\n".join(out)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_compare(a):
    current = load_json(a.current)
    baseline = load_json(a.baseline) if a.baseline and os.path.exists(a.baseline) else None
    accepted = load_json(a.accepted) if a.accepted else {"kernels": {}}
    rows, notes, failed = compare(baseline, current, accepted, a.threshold)
    text = render(rows, notes, failed, a.threshold, current.get("config", "?"))
    print(text)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("count", help="count each kernel's instructions under callgrind")
    c.add_argument("--bench", required=True)
    c.add_argument("--kernels", required=True)
    c.add_argument("--config", required=True, help="the build configuration's name, e.g. native")
    c.add_argument("--out", required=True)
    c.add_argument("--class", dest="cls", default="small", choices=["small", "medium", "large"])
    c.add_argument("--valgrind", default="valgrind", help="the valgrind command (may carry arguments)")
    c.set_defaults(func=cmd_count)
    m = sub.add_parser("compare", help="compare a run against main's baseline")
    m.add_argument("--baseline", help="main's counts; a missing file means no baseline")
    m.add_argument("--current", required=True)
    m.add_argument("--accepted")
    m.add_argument("--threshold", type=float, default=0.02)
    m.add_argument("--summary", help="append the markdown table to this file")
    m.set_defaults(func=cmd_compare)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
