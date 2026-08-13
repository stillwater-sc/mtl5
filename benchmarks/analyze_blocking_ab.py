#!/usr/bin/env python3
"""Compare the two arms of the cache-blocking A/B (#426, #432).

Reads the CSVs written by run_blocking_ab.sh and reports, per (shape, threads),
the best of each arm and the ratio. Deliberately conservative about what counts
as a difference:

  * per point the arms are compared on their MINIMUM over all rounds, which is
    the statistic run_blocking_ab.sh is built to produce;
  * a point is only called a win or a loss when the arms' [min, max] ranges over
    rounds do NOT overlap AND the difference clears a noise floor -- otherwise it
    is reported as noise, however tempting the mean looks;
  * if both arms compiled to the SAME blocking parameters, the run is a null
    experiment by construction and no point is called at all. That is not a
    hypothetical: on a machine whose L1/L2 already match the compile-time
    defaults (e.g. xeon-e5-2420v2) detection changes nothing, and an early
    version of this script duly reported a 4.8% "win" between two identical
    configurations. Structural controls beat statistical ones;
  * a checksum mismatch is a hard error, not a footnote: the arms must compute
    the same result, and if they do not, no timing from that run means anything.
"""
import csv
import sys
from collections import defaultdict

# Below this relative difference, a non-overlap is not worth believing.
NOISE_FLOOR = 0.02
# Fewer rounds than this and [min,max] is not a range, it is two samples.
MIN_ROUNDS = 3


def load(path):
    rows = defaultdict(list)          # (dtype,m,n,k,threads) -> [row, ...]
    params = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["dtype"], int(r["m"]), int(r["n"]), int(r["k"]), int(r["threads"]))
            rows[key].append(r)
            params[r["dtype"]] = (r["mr"], r["nr"], r["kc"], r["mc"], r["nc"])
    return rows, params


def main(argv):
    if len(argv) != 3:
        print("usage: analyze_blocking_ab.py <detected.csv> <default.csv>", file=sys.stderr)
        return 2
    det, det_p = load(argv[1])
    dfl, dfl_p = load(argv[2])

    for dt in sorted(set(det_p) | set(dfl_p)):
        print(f"blocking ({dt}):  detected mr,nr,kc,mc,nc = {det_p.get(dt)}"
              f"   default = {dfl_p.get(dt)}")

    # Structural control: identical parameters means the arms are the same
    # program. Any timing difference is measurement noise, by construction.
    null_run = all(det_p.get(dt) == dfl_p.get(dt) for dt in set(det_p) | set(dfl_p))
    if null_run:
        print("\n*** NULL RUN: both arms compiled to identical blocking parameters.")
        print("*** Detection changes nothing on this machine (its L1/L2 already match")
        print("*** the compile-time defaults), so every difference below is noise and")
        print("*** no point is called. This is a useful harness self-test, not a result.")
    print()

    # Integrity gate. An A/B is only an A/B if both arms measured the same points
    # the same number of times; comparing the intersection, or a 5-round arm
    # against a 2-round one, yields confident verdicts from a partial run. Fail
    # loudly rather than analysing what happens to be present.
    only_det = sorted(set(det) - set(dfl))
    only_dfl = sorted(set(dfl) - set(det))
    uneven = [(k, len(det[k]), len(dfl[k])) for k in sorted(set(det) & set(dfl))
              if len(det[k]) != len(dfl[k])]
    if only_det or only_dfl or uneven:
        print("INCOMPLETE A/B -- refusing to compare.", file=sys.stderr)
        for k in only_det:
            print(f"  only in detected: {k}", file=sys.stderr)
        for k in only_dfl:
            print(f"  only in default:  {k}", file=sys.stderr)
        for k, na, nb in uneven:
            print(f"  unequal rounds:   {k} detected={na} default={nb}", file=sys.stderr)
        print("Re-run both arms over the same shapes and rounds.", file=sys.stderr)
        return 2

    hdr = f"{'shape':>22} {'T':>3} {'default':>10} {'detected':>10} {'ratio':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))

    def gflops(m, n, k, secs):
        return 2.0 * m * n * k / secs / 1e9

    bad_checksum = []
    wins = losses = noise = 0
    for key in sorted(set(det) & set(dfl)):
        dt, m, n, k, t = key
        a, b = det[key], dfl[key]

        ca = {round(float(r["checksum"]), 6) for r in a}
        cb = {round(float(r["checksum"]), 6) for r in b}
        if ca != cb:
            bad_checksum.append((key, ca, cb))

        a_s = sorted(float(r["min_s"]) for r in a)
        b_s = sorted(float(r["min_s"]) for r in b)
        a_best, b_best = a_s[0], b_s[0]
        ratio = b_best / a_best            # >1 => detected is faster

        # A point is called only when the arms differ structurally, their ranges
        # do not overlap, AND the effect clears the noise floor.
        rounds = len(a_s)                  # == len(b_s), enforced by the gate above
        if null_run:
            verdict = "null (identical blocking)"
            noise += 1
        elif abs(ratio - 1.0) < NOISE_FLOOR:
            verdict = f"noise (<{NOISE_FLOOR:.0%})"
            noise += 1
        elif rounds < MIN_ROUNDS:
            verdict = f"unresolved (only {rounds} rounds)"
            noise += 1
        elif a_s[-1] < b_s[0]:
            verdict = "detected faster"
            wins += 1
        elif b_s[-1] < a_s[0]:
            verdict = "DEFAULT faster"
            losses += 1
        else:
            verdict = "noise (ranges overlap)"
            noise += 1

        print(f"{m}x{n}x{k:>6}".rjust(22)
              + f" {t:>3} {gflops(m, n, k, b_best):>10.2f}"
              + f" {gflops(m, n, k, a_best):>10.2f} {ratio:>7.3f}  {verdict}")

    print()
    print(f"{wins} faster, {losses} slower, {noise} indistinguishable "
          f"(GFLOP/s columns; ratio > 1 means detection helped; "
          f"differences under {NOISE_FLOOR:.0%} are not called)")

    if bad_checksum:
        print("\nCHECKSUM MISMATCH -- the arms did not compute the same result. "
              "Timings from this run are meaningless until this is explained:",
              file=sys.stderr)
        for key, ca, cb in bad_checksum:
            print(f"  {key}: detected={ca} default={cb}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
