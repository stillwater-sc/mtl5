#!/usr/bin/env python3
"""Tests for check_results.py (#928, #929). Standard library only:

    python3 benchmarks/platform/test_check_results.py
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_results as cr  # noqa: E402


def result(samples, version=1, with_samples=False, **over):
    """A result as mtl5_bench writes it for these sorted samples."""
    s = sorted(samples)
    n = len(s)
    q = lambda p: s[cr.quantile_index(n, p)]  # noqa: E731
    r = {"area": "blas1", "op": "dot", "dtype": "f64", "shape": "n=8", "version": version,
         "median_s": q(0.5), "min_s": s[0], "samples": n,
         "p10_s": q(0.10), "p25_s": q(0.25), "p75_s": q(0.75), "p90_s": q(0.90), "max_s": s[-1]}
    if with_samples:
        r["samples_s"] = s
    r.update(over)
    return r


class QuantileRule(unittest.TestCase):
    def test_index(self):
        # n = 10: p10 is the 2nd sample, the median the 6th (sorted[n / 2]), p90 the 10th.
        self.assertEqual([cr.quantile_index(10, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)], [1, 2, 5, 7, 9])
        # n < 10: p10 is the first, min_s.
        self.assertEqual(cr.quantile_index(9, 0.1), 0)
        self.assertEqual(cr.quantile_index(3, 0.5), 1)
        self.assertEqual(cr.quantile_index(1, 0.9), 0)
        # The median is sorted[n // 2], as mtl5_bench has always written it.
        for n in range(1, 50):
            self.assertEqual(cr.quantile_index(n, 0.5), n // 2)


class Results(unittest.TestCase):
    def test_a_well_formed_result_passes(self):
        for n in (1, 3, 9, 10, 11, 100):
            r = result([0.001 * (i + 1) for i in range(n)], with_samples=True)
            self.assertEqual(cr.check_result(r, True), [], n)

    def test_version(self):
        for bad in (0, -1, 1.0, "1", None, True):
            r = result([1, 2, 3], version=bad)
            self.assertTrue(any("version" in p for p in cr.check_result(r, False)), bad)
        self.assertEqual(cr.check_result(result([1, 2, 3], version=2), False), [])

    def test_missing_or_unordered_quantile(self):
        r = result([1, 2, 3, 4, 5])
        del r["p75_s"]
        self.assertTrue(cr.check_result(r, False))
        r = result([1, 2, 3, 4, 5], p90_s=0.5)
        self.assertTrue(any(">" in p for p in cr.check_result(r, False)))

    def test_p10_is_min_below_ten_samples(self):
        r = result([1, 2, 3, 4, 5], p10_s=2)
        self.assertTrue(any("p10_s" in p for p in cr.check_result(r, False)))

    def test_samples_must_match_the_rule(self):
        r = result([0.1 * i for i in range(1, 21)], with_samples=True)
        r["p25_s"] = r["samples_s"][4]   # the rule picks index 5 for n = 20
        self.assertTrue(any("p25_s" in p for p in cr.check_result(r, False)))
        r = result([1, 2, 3], with_samples=True)
        r["samples_s"] = [1, 3, 2]
        self.assertTrue(any("sorted" in p for p in cr.check_result(r, False)))
        r = result([1, 2, 3], with_samples=True)
        r["samples_s"] = [1, 2]
        self.assertTrue(cr.check_result(r, False))

    def test_require_samples(self):
        r = result([1, 2, 3])
        self.assertEqual(cr.check_result(r, False), [])
        self.assertTrue(cr.check_result(r, True))

    def test_untimed_result_needs_only_a_version(self):
        r = {"area": "a", "op": "b", "dtype": "f64", "shape": "s", "version": 1, "median_s": None}
        self.assertEqual(cr.check_result(r, True), [])


class Files(unittest.TestCase):
    def run_main(self, docs, *extra):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, doc in enumerate(docs):
                p = Path(d) / f"r{i}.json"
                p.write_text(json.dumps(doc))
                paths.append(str(p))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return cr.main(paths + list(extra))

    def test_exit_codes(self):
        good = {"schema": "mtl5-bench/1", "results": [result([1, 2, 3])]}
        self.assertEqual(self.run_main([good]), 0)
        twice = {"schema": "mtl5-bench/1", "results": [result([1, 2, 3]), result([1, 2, 3])]}
        self.assertEqual(self.run_main([twice]), 1)
        old = {"schema": "mtl5-bench/1", "results": [{k: v for k, v in result([1, 2, 3]).items() if k != "version"}]}
        self.assertEqual(self.run_main([old]), 1)
        self.assertEqual(self.run_main([{"schema": "other", "results": []}]), 1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cr.main(["/nonexistent/x.json"]), 2)


if __name__ == "__main__":
    unittest.main()
