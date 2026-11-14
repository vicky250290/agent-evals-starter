import csv
import os
import tempfile
import unittest

from agent_evals import ToolCall, Trace, load_traces
from agent_evals import graders as g
from agent_evals.error_analysis import count_categories, export_sample
from agent_evals.flakiness import measure_flakiness

HERE = os.path.dirname(__file__)


def trace(*calls, cost=0.0):
    return Trace(id="x", input="i", tool_calls=[ToolCall(n, a) for n, a in calls], cost_usd=cost)


class GradersTest(unittest.TestCase):
    def test_called_and_args(self):
        t = trace(("get_orders", {"customer_id": "C1"}), ("search_kb", {}))
        self.assertTrue(g.called(t, "search_kb"))
        self.assertFalse(g.called(t, "issue_refund"))
        self.assertEqual(g.args_of(t, "get_orders"), [{"customer_id": "C1"}])

    def test_checks(self):
        t = trace(("a", {}), ("b", {}), ("c", {}), cost=0.06)
        self.assertFalse(g.step_budget(t, 2).passed)
        self.assertTrue(g.step_budget(t, 3).passed)
        self.assertFalse(g.cost_budget(t, 0.05).passed)
        self.assertFalse(g.not_called(t, "b").passed)
        self.assertTrue(g.must_call(t, "c").passed)

    def test_final_state_ignores_order(self):
        self.assertTrue(g.final_state("s", [("B", 1), ("A", 2)], [("A", 2), ("B", 1)]).passed)
        self.assertFalse(g.final_state("s", [("A", 2)], [("A", 3)]).passed)

    def test_passed_and_failures(self):
        rs = [g.Result("ok", True), g.Result("bad", False)]
        self.assertFalse(g.passed(rs))
        self.assertEqual([r.name for r in g.failures(rs)], ["bad"])


class TracesAndAnalysisTest(unittest.TestCase):
    def test_load_sample_traces(self):
        traces = load_traces(os.path.join(HERE, "..", "examples", "sample_traces.jsonl"))
        self.assertEqual(len(traces), 4)
        self.assertEqual(traces[0].tool_path, ("get_orders", "issue_refund"))

    def test_export_and_count(self):
        traces = [Trace(id=str(i), input="q") for i in range(10)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ea.csv")
            self.assertEqual(export_sample(traces, path, n=4, seed=1), 4)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            for row, cat in zip(rows, ["wrong id", "no kb", "wrong id", ""]):
                row["category"] = cat
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            self.assertEqual(count_categories(path), [("wrong id", 2), ("no kb", 1)])

    def test_flakiness(self):
        outcomes = iter([True, False, True])

        def run(task):
            ok = next(outcomes)
            return Trace(id="t", input=task, success=ok,
                         tool_calls=[ToolCall("a" if ok else "b")])

        [r] = measure_flakiness(run, ["task"], n=3)
        self.assertEqual((r.passes, r.distinct_paths, r.flaky), (2, 2, True))


if __name__ == "__main__":
    unittest.main()
