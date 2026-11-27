#!/usr/bin/env python3
"""Tests for ir_gate.py (#944): python3 benchmarks/regression/test_ir_gate.py"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ir_gate  # noqa: E402

T = 0.02


def run(kernels, config="native", cls="small"):
    return {"config": config, "class": cls, "valgrind": "valgrind-3.26.0",
            "env": dict(ir_gate.PINNED_ENV), "kernels": kernels}


def statuses(rows):
    return {k: s for k, _, _, _, s in rows}


class Compare(unittest.TestCase):
    def test_equal_counts_pass(self):
        rows, _, failed = ir_gate.compare(run({"a": 1000, "b": 5}), run({"a": 1000, "b": 5}), {}, T)
        self.assertFalse(failed)
        self.assertEqual(statuses(rows), {"a": "ok", "b": "ok"})

    def test_doubled_count_fails(self):
        rows, _, failed = ir_gate.compare(run({"a": 1000}), run({"a": 2000}), {}, T)
        self.assertTrue(failed)
        self.assertEqual(statuses(rows)["a"], "REGRESSION")

    def test_threshold_edges(self):
        _, _, failed = ir_gate.compare(run({"a": 100000}), run({"a": 102000}), {}, T)
        self.assertFalse(failed)            # +2.00%: at the threshold
        _, _, failed = ir_gate.compare(run({"a": 100000}), run({"a": 102001}), {}, T)
        self.assertTrue(failed)             # just above it

    def test_faster_passes(self):
        rows, _, failed = ir_gate.compare(run({"a": 1000}), run({"a": 500}), {}, T)
        self.assertFalse(failed)
        self.assertEqual(statuses(rows)["a"], "faster")

    def test_accepted_increase_passes_up_to_max_ir(self):
        acc = {"kernels": {"a": {"max_ir": {"native": 2000}, "reason": "#1: packs twice"}}}
        rows, _, failed = ir_gate.compare(run({"a": 1000}), run({"a": 2000}), acc, T)
        self.assertFalse(failed)
        self.assertEqual(statuses(rows)["a"], "accepted")
        _, notes, failed = ir_gate.compare(run({"a": 1000}), run({"a": 2001}), acc, T)
        self.assertTrue(failed)
        self.assertTrue(any("above its accepted max_ir" in n for n in notes))

    def test_acceptance_needs_a_reason_and_a_count(self):
        for bad in ({"max_ir": {"native": 2000}}, {"max_ir": {"native": 2000}, "reason": "  "},
                    {"reason": "x"}, {"max_ir": {"native": -1}, "reason": "x"},
                    {"max_ir": {"native": 1.5}, "reason": "x"}, {"max_ir": {"native": True}, "reason": "x"},
                    {"max_ir": {}, "reason": "x"},
                    # A bare count would accept the change in every configuration at once.
                    {"max_ir": 2000, "reason": "x"}):
            _, _, failed = ir_gate.compare(run({"a": 1000}), run({"a": 1000}), {"kernels": {"a": bad}}, T)
            self.assertTrue(failed, bad)

    def test_acceptance_is_per_configuration(self):
        # Accepted for native only: the same rise in native-fast-arm64 still fails.
        acc = {"kernels": {"a": {"max_ir": {"native": 2000}, "reason": "#1"}}}
        for cfg, want in (("native", False), ("native-fast-arm64", True)):
            rows, _, failed = ir_gate.compare(run({"a": 1000}, config=cfg), run({"a": 2000}, config=cfg), acc, T)
            self.assertEqual(failed, want, cfg)
            self.assertEqual(statuses(rows)["a"], "REGRESSION" if want else "accepted")

    def test_stale_acceptance_is_reported_not_failed(self):
        acc = {"kernels": {"a": {"max_ir": {"native": 2000}, "reason": "#1"}}}
        _, notes, failed = ir_gate.compare(run({"a": 2000}), run({"a": 2000}), acc, T)
        self.assertFalse(failed)
        self.assertTrue(any("stale" in n for n in notes))

    def test_new_and_removed_kernels_pass(self):
        rows, _, failed = ir_gate.compare(run({"a": 1, "old": 7}), run({"a": 1, "new": 9}), {}, T)
        self.assertFalse(failed)
        self.assertEqual(statuses(rows), {"a": "ok", "new": "new", "old": "removed"})

    def test_no_baseline_passes(self):
        rows, notes, failed = ir_gate.compare(None, run({"a": 1}), {}, T)
        self.assertFalse(failed)
        self.assertEqual(statuses(rows), {"a": "new"})
        self.assertTrue(notes)

    def test_other_config_or_class_fails(self):
        _, _, failed = ir_gate.compare(run({"a": 1}, config="native-fast"), run({"a": 1}), {}, T)
        self.assertTrue(failed)
        _, _, failed = ir_gate.compare(run({"a": 1}, cls="medium"), run({"a": 1}), {}, T)
        self.assertTrue(failed)
        other_env = run({"a": 1})
        other_env["env"] = {**other_env["env"], "MTL5_BLOCKING_KC": "512"}
        _, _, failed = ir_gate.compare(other_env, run({"a": 1}), {}, T)
        self.assertTrue(failed)

    def test_render_marks_the_regression(self):
        rows, notes, failed = ir_gate.compare(run({"a": 1000}), run({"a": 2000}), {}, T)
        text = ir_gate.render(rows, notes, failed, T, "native")
        self.assertIn("**REGRESSION**", text)
        self.assertIn("+100.00%", text)
        self.assertIn("The gate fails", text)


class Parsing(unittest.TestCase):
    def test_callgrind_totals(self):
        text = "version: 1\ncmd: x\nevents: Ir\nfn=f\n0 3\nsummary: 120\ntotals: 123\n"
        self.assertEqual(ir_gate.parse_callgrind_totals(text), 123)
        self.assertEqual(ir_gate.parse_callgrind_totals("events: Ir\nsummary: 77\n"), 77)
        with self.assertRaises(ValueError):
            ir_gate.parse_callgrind_totals("events: Ir\n")
        with self.assertRaises(ValueError):
            ir_gate.parse_callgrind_totals("events: Dr Ir\ntotals: 1 2\n")

    def test_kernels_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.txt")
            with open(p, "w") as f:
                f.write("# comment\n\nblas1/dot/f64   # trailing\nfactor/lu/f64\n")
            self.assertEqual(ir_gate.read_kernels(p), ["blas1/dot/f64", "factor/lu/f64"])
            with open(p, "w") as f:
                f.write("a\na\n")
            with self.assertRaises(ValueError):
                ir_gate.read_kernels(p)

    def test_repository_files_are_valid(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "accepted.json")) as f:
            self.assertEqual(ir_gate.validate_accepted(json.load(f)), [])
        self.assertTrue(ir_gate.read_kernels(os.path.join(here, "kernels.txt")))


class CommandLine(unittest.TestCase):
    def test_compare_exit_status_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            def put(name, doc):
                p = os.path.join(d, name)
                with open(p, "w") as f:
                    json.dump(doc, f)
                return p
            base, same, double = put("b.json", run({"a": 10})), put("s.json", run({"a": 10})), put("d.json", run({"a": 20}))
            summary = os.path.join(d, "summary.md")
            self.assertEqual(ir_gate.main(["compare", "--baseline", base, "--current", same, "--summary", summary]), 0)
            self.assertEqual(ir_gate.main(["compare", "--baseline", base, "--current", double]), 1)
            missing = os.path.join(d, "none.json")
            self.assertEqual(ir_gate.main(["compare", "--baseline", missing, "--current", double]), 0)
            with open(summary) as f:
                self.assertIn("The gate passes", f.read())


if __name__ == "__main__":
    unittest.main()
