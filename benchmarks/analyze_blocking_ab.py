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
    """rows, blocking params, and the ARM NAME the data carries.

    The name comes from the file rather than from this script's assumptions:
    the harness now runs four arms (default / detected / kconly / mconly) and
    any pair of them can be compared, so a hardcoded "detected vs default"
    heading would mislabel three comparisons out of four."""
    rows = defaultdict(list)          # (dtype,m,n,k,threads) -> [row, ...]
    params = {}
    arm = None
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            arm = arm or r.get("arm")
            key = (r["dtype"], int(r["m"]), int(r["n"]), int(r["k"]), int(r["threads"]))
            rows[key].append(r)
            params[r["dtype"]] = (r["mr"], r["nr"], r["kc"], r["mc"], r["nc"])
    return rows, params, (arm or "test")


def main(argv):
    if len(argv) != 3:
        print("usage: analyze_blocking_ab.py <detected.csv> <default.csv>", file=sys.stderr)
        return 2
    det, det_p, det_name = load(argv[1])
    dfl, dfl_p, dfl_name = load(argv[2])

    for dt in sorted(set(det_p) | set(dfl_p)):
        print(f"blocking ({dt}):  {det_name} mr,nr,kc,mc,nc = {det_p.get(dt)}"
              f"   {dfl_name} = {dfl_p.get(dt)}")

    # The CONFIGURED mc above is not the one the loops step by: it is capped to
    # fill the thread budget (plan_gemm_grid) and then rounded so the partition
    # divides evenly (balanced_mc). Reading the configured value as the value
    # under test is wrong for BOTH arms -- a default arm configured mc=32 runs at
    # 30 serially and 29 on six threads -- and it varies with the SHAPE, so it
    # belongs per point rather than in a summary line. Collected here, printed in
    # the table below.
    def plan_of(rows):
        out = {}
        for key, rs in rows.items():
            for r in rs:
                if r.get("mc_used"):
                    out[key] = (r["mc_used"], f'{r.get("ic_nt","?")}x{r.get("jc_nt","?")}')
        return out
    det_plan, dfl_plan = plan_of(det), plan_of(dfl)
    if not det_plan and not dfl_plan:
        print("\nNOTE: this CSV predates the mc_used column, so the mc above is the"
              "\n      CONFIGURED bound and not the step any loop used (#430).")

    # A request larger than the machine is silently clamped: thread_pool caps
    # MTL5_NUM_THREADS at hardware_concurrency. Every grid calculation depends on
    # the CLAMPED value, so say so loudly rather than leaving it to be inferred
    # from a speedup that looks low. `pool` is absent from CSVs written before
    # this column existed, which is exactly when it bit.
    clamped = sorted({(int(r["threads"]), int(r["pool"]))
                      for rows in (det, dfl) for rs in rows.values() for r in rs
                      if r.get("pool") and int(r["pool"]) != int(r["threads"])})
    for asked, got in clamped:
        print(f"NOTE: threads={asked} was clamped to a pool of {got} "
              f"(hardware_concurrency); grids are bounded by {got}, not {asked}.")
    if any(not r.get("pool") for rows in (det, dfl) for rs in rows.values() for r in rs):
        print("NOTE: this CSV predates the `pool` column -- the effective thread "
              "count was not recorded and may be lower than `threads`.")

    # Structural control: identical parameters means the arms are the same
    # program. Any timing difference is measurement noise, by construction.
    null_run = all(det_p.get(dt) == dfl_p.get(dt) for dt in set(det_p) | set(dfl_p))
    if null_run:
        print("\n*** NULL RUN: both arms compiled to identical blocking parameters.")
        print(f"*** {det_name} and {dfl_name} derive the same kc/mc on this machine")
        print("*** (its caches already match the compile-time model), so every difference")
        print("*** below is noise and no point is called. A useful harness self-test.")
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

    hdr = (f"{'shape':>22} {'T':>3} {dfl_name:>10} {det_name:>10} {'ratio':>7}"
           f" {'mc base/test':>13} {'grid':>6}  verdict")
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
            verdict = f"{det_name} faster"
            wins += 1
        elif b_s[-1] < a_s[0]:
            verdict = f"{dfl_name.upper()} faster"
            losses += 1
        else:
            verdict = "noise (ranges overlap)"
            noise += 1

        # mc as RUN, per arm, and the grid (identical for both arms by
        # construction -- it depends on m, n and the budget, not on the blocking).
        pa, pb = dfl_plan.get(key), det_plan.get(key)
        mc_col = f"{pa[0] if pa else '?'}/{pb[0] if pb else '?'}"
        grid_col = (pb or pa or ("", "?"))[1]
        print(f"{m}x{n}x{k:>6}".rjust(22)
              + f" {t:>3} {gflops(m, n, k, b_best):>10.2f}"
              + f" {gflops(m, n, k, a_best):>10.2f} {ratio:>7.3f}"
              + f" {mc_col:>13} {grid_col:>6}  {verdict}")

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
