import unittest

from agent_evals.metrics import flips, pass_at_k, pass_hat_k, wilson_interval


class MetricsTest(unittest.TestCase):
    def test_pass_at_k_bounds(self):
        self.assertEqual(pass_at_k(10, 0, 3), 0.0)
        self.assertEqual(pass_at_k(10, 10, 3), 1.0)
        self.assertEqual(pass_at_k(10, 8, 3), 1.0)  # only 2 failures, can't fill 3

    def test_pass_at_k_value(self):
        # 1 - C(5,2)/C(10,2) = 1 - 10/45
        self.assertAlmostEqual(pass_at_k(10, 5, 2), 1 - 10 / 45)

    def test_pass_hat_k(self):
        self.assertEqual(pass_hat_k(10, 2, 3), 0.0)
        self.assertAlmostEqual(pass_hat_k(10, 9, 2), 36 / 45)
        self.assertEqual(pass_hat_k(5, 5, 5), 1.0)

    def test_reliability_gap_from_book(self):
        # 90% per trial: pass^10 is about 35%.
        self.assertAlmostEqual(0.9**10, 0.3487, places=3)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            pass_at_k(5, 6, 1)
        with self.assertRaises(ValueError):
            pass_hat_k(5, 2, 0)

    def test_wilson(self):
        lo, hi = wilson_interval(25, 50)
        self.assertAlmostEqual(lo, 0.366, places=2)
        self.assertAlmostEqual(hi, 0.634, places=2)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_flips(self):
        old = {"a": True, "b": False, "c": True}
        new = {"a": False, "b": True, "c": True}
        self.assertEqual(flips(old, new), {"fixed": ["b"], "broken": ["a"]})


if __name__ == "__main__":
    unittest.main()
