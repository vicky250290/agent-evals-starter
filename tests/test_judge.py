import unittest

from agent_evals.judge import calibrate, make_judge


class JudgeTest(unittest.TestCase):
    def test_make_judge_parses_verdict(self):
        # Fake model: passes only when the trace shows a knowledge-base result.
        judge = make_judge(lambda p: "REASONING: cites the KB.\nVERDICT: PASS"
                           if "KB_RESULT" in p else "REASONING: memory.\nVERDICT: FAIL")
        self.assertTrue(judge("search_kb -> KB_RESULT: 30 days"))
        self.assertFalse(judge("no tools"))

    def test_calibrate(self):
        # Judge says PASS for everything: catches no failures (TNR 0).
        tpr, tnr = calibrate(lambda t: True, [("a", True), ("b", False), ("c", False)])
        self.assertEqual((tpr, tnr), (1.0, 0.0))
        tpr, tnr = calibrate(lambda t: t == "a", [("a", True), ("b", False), ("c", True)])
        self.assertEqual((tpr, tnr), (0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
