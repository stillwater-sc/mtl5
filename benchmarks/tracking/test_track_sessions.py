#!/usr/bin/env python3
"""Tests for track_sessions.py (#944). Standard library only; run directly:

    python3 benchmarks/tracking/test_track_sessions.py
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_sessions  # noqa: E402


def write_run(sdir, run, times, machine="m1", cls="medium", commit="aaaa", sc_extra=None, failed=(),
              versions=None):
    """A <run>.json with one entry per name in times (median_s), and its sidecar.
    versions: {name: version}; a name not in it has no version field (before #928)."""
    sdir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, t in times.items():
        r = {"area": "blas1", "op": name, "dtype": "f64", "shape": "n=8",
             "median_s": t, "check": {"ok": name not in failed}}
        if versions and name in versions:
            r["version"] = versions[name]
        results.append(r)
    (sdir / f"{run}.json").write_text(json.dumps({
        "schema": "mtl5-bench/1", "label": run.split("-t")[0], "machine": machine,
        "class": cls, "results": results,
    }))
    sc = {"machine": machine, "class": cls, "git_commit": commit, "compiler": "gcc", "compiler_version": "13.3.0"}
    sc.update(sc_extra or {})
    (sdir / f"{run}.json.sysinfo").write_text("".join(f"{k}={v}\n" for k, v in sc.items()))


def run_compare(base, new, *extra):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = track_sessions.main(["compare", str(base), str(new), *extra])
    return rc, out.getvalue(), err.getvalue()


TIMES = {f"op{i}": 1.0 + 0.1 * i for i in range(9)}


class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.base = self.d / "data" / "m1" / "mtl5_bench" / "20260101T000000Z"
        self.new = self.d / "data" / "m1" / "mtl5_bench" / "20260102T000000Z"

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, *extra):
        j = self.d / "r.json"
        rc, out, err = run_compare(self.base, self.new, "--json", str(j), *extra)
        self.assertEqual(rc, 0, err)
        return json.loads(j.read_text()), out

    def test_equal_times_are_steady(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", TIMES, commit="bbbb")
        rep, out = self.report()
        r = rep["runs"][0]
        self.assertEqual(r["verdict"], "steady")
        self.assertAlmostEqual(r["median_ratio"], 1.0)
        self.assertEqual(r["outside"], [])
        self.assertEqual(rep["baseline"]["commit"], "aaaa")
        self.assertEqual(rep["new"]["commit"], "bbbb")
        self.assertIn("| native-t1 | 9 | 0 | 1.000 |", out)
        # The JSON carries its sidecar, as check_sidecars.sh requires under benchmarks/data.
        sc = (self.d / "r.json.sysinfo").read_text()
        self.assertIn("baseline_session=20260101T000000Z\n", sc)
        self.assertIn("new_commit=bbbb\n", sc)

    def test_doubled_times_are_slower_and_every_entry_listed(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", {k: 2 * v for k, v in TIMES.items()}, commit="bbbb")
        rep, _ = self.report()
        r = rep["runs"][0]
        self.assertEqual(r["verdict"], "slower")
        self.assertAlmostEqual(r["median_ratio"], 2.0)
        self.assertEqual(len(r["outside"]), 9)

    def test_halved_times_are_faster(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", {k: v / 2 for k, v in TIMES.items()}, commit="bbbb")
        rep, _ = self.report()
        self.assertEqual(rep["runs"][0]["verdict"], "faster")

    def test_the_median_not_one_entry_decides_the_run(self):
        # One entry 3x slower, the rest unchanged: the run is steady, the entry is listed.
        t = dict(TIMES)
        t["op4"] *= 3
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", t, commit="bbbb")
        rep, _ = self.report()
        r = rep["runs"][0]
        self.assertEqual(r["verdict"], "steady")
        self.assertEqual([e["entry"] for e in r["outside"]], ["blas1/op4/f64 n=8"])

    def test_run_band_edges(self):
        # Every entry 3% slower: slower at the default 2% band, steady at 5%.
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", {k: 1.03 * v for k, v in TIMES.items()}, commit="bbbb")
        self.assertEqual(self.report()[0]["runs"][0]["verdict"], "slower")
        self.assertEqual(self.report("--run-band", "5")[0]["runs"][0]["verdict"], "steady")

    def test_entries_in_one_session_or_failed_are_not_compared(self):
        t = dict(TIMES)
        t["added"] = 5.0
        write_run(self.base, "native-t1", TIMES, failed=("op0",))
        write_run(self.new, "native-t1", t, commit="bbbb")
        r = self.report()[0]["runs"][0]
        self.assertEqual(r["entries"], 8)
        self.assertEqual(r["skipped"], 2)

    def test_a_changed_definition_starts_a_new_series(self):
        # #928: op3 is version 1 in the baseline and 2 now, 3x slower. It is
        # not compared, so it neither shows as a regression nor moves the median.
        t = dict(TIMES)
        t["op3"] *= 3
        v1 = {k: 1 for k in TIMES}
        v2 = dict(v1, op3=2)
        write_run(self.base, "native-t1", TIMES, versions=v1)
        write_run(self.new, "native-t1", t, commit="bbbb", versions=v2)
        rep, out = self.report()
        r = rep["runs"][0]
        self.assertEqual(r["entries"], 8)
        self.assertEqual(r["skipped"], 1)
        self.assertEqual(r["outside"], [])
        self.assertEqual(r["redefined"], [{"entry": "blas1/op3/f64 n=8", "from": 1, "to": 2}])
        self.assertIn("Entries whose definition changed", out)

    def test_a_session_without_versions_is_compared(self):
        # A baseline from before #928 records no version: compared as before.
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", TIMES, commit="bbbb", versions={k: 2 for k in TIMES})
        r = self.report()[0]["runs"][0]
        self.assertEqual(r["entries"], 9)
        self.assertEqual(r["redefined"], [])

    def test_runs_in_one_session_are_reported_not_compared(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", TIMES, commit="bbbb")
        write_run(self.new, "native-t8", TIMES, commit="bbbb")
        rep, out = self.report()
        self.assertEqual([r["run"] for r in rep["runs"]], ["native-t1"])
        self.assertEqual(rep["runs_not_compared"], ["native-t8.json"])
        self.assertIn("`native-t8.json`", out)

    def test_environment_changes_are_listed(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", TIMES, commit="bbbb", sc_extra={"compiler_version": "14.2.0"})
        rep, out = self.report()
        self.assertEqual(rep["env_changes"], {"compiler_version": {"native-t1.json": ["13.3.0", "14.2.0"]}})
        self.assertIn("What changed besides the commit", out)

    def test_another_class_is_refused(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", TIMES, cls="large")
        rc, _, err = run_compare(self.base, self.new)
        self.assertEqual(rc, 1)
        self.assertIn("class", err)

    def test_another_machine_is_refused(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t1", TIMES, machine="m2")
        self.assertEqual(run_compare(self.base, self.new)[0], 1)

    def test_no_common_run_is_refused(self):
        write_run(self.base, "native-t1", TIMES)
        write_run(self.new, "native-t8", TIMES)
        self.assertEqual(run_compare(self.base, self.new)[0], 1)

    def test_missing_directory_is_usage(self):
        write_run(self.base, "native-t1", TIMES)
        self.assertEqual(run_compare(self.base, self.d / "nope")[0], 2)


class Page(unittest.TestCase):
    def test_page_shows_latest_report_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            data = d / "benchmarks" / "data"
            s = [data / "m1" / "mtl5_bench" / f"2026010{i}T000000Z" for i in (1, 2, 3)]
            write_run(s[0], "native-t1", TIMES)
            write_run(s[1], "native-t1", TIMES, commit="bbbb")
            write_run(s[2], "native-t1", {k: 2 * v for k, v in TIMES.items()}, commit="cccc")
            for a, b in ((0, 1), (1, 2)):
                tr = data / "m1" / "tracking" / s[b].name
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = track_sessions.main(["compare", str(s[a]), str(s[b]),
                                              "--json", f"{tr}.json", "--markdown", f"{tr}.md"])
                self.assertEqual(rc, 0)
            out = d / "docs" / "benchmarks" / "tracking.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = track_sessions.main(["page", "--data", str(data), "--out", str(out)])
            self.assertEqual(rc, 0)
            page = out.read_text()
            self.assertIn("## m1", page)
            self.assertIn("Latest: session `20260103T000000Z` (commit `cccc`)", page)
            self.assertIn("| native-t1 | 2.000 | 9 | slower |", page)
            self.assertIn("blob/main/benchmarks/data/m1/tracking/20260103T000000Z.md", page)
            # History, newest first.
            self.assertLess(page.index("`20260103T000000Z` | `cccc`"), page.index("`20260102T000000Z` | `bbbb`"))

    def test_page_without_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "benchmarks" / "data").mkdir(parents=True)
            out = d / "tracking.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = track_sessions.main(["page", "--data", str(d / "benchmarks" / "data"), "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn("No tracking session has been published yet.", out.read_text())


if __name__ == "__main__":
    unittest.main()
