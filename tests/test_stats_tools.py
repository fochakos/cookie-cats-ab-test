"""Checks the hand-written statistics against SciPy and against simulation.

Run from the project root:  python -m unittest discover tests   (or: pytest)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stats_tools import (beta_posterior, bootstrap_mean_diff, bootstrap_prop_diff, holm,  # noqa: E402
                         mde_two_prop, n_per_arm, power_two_prop, srm_test, two_prop_test)


class TestSRM(unittest.TestCase):
    def test_matches_scipy_chisquare(self):
        ref = stats.chisquare([44700, 45489])
        got = srm_test([44700, 45489])
        self.assertAlmostEqual(got["chi2"], ref.statistic, places=8)
        self.assertAlmostEqual(got["p_value"], ref.pvalue, places=10)

    def test_uneven_design(self):
        ref = stats.chisquare([900, 100], f_exp=[900, 100])
        self.assertAlmostEqual(srm_test([900, 100], [0.9, 0.1])["p_value"], ref.pvalue)


class TestTwoProp(unittest.TestCase):
    def test_z_squared_equals_chi2_without_correction(self):
        x_a, n_a, x_b, n_b = 8502, 44700, 8279, 45489
        table = [[x_a, n_a - x_a], [x_b, n_b - x_b]]
        chi2 = stats.chi2_contingency(table, correction=False)
        t = two_prop_test(x_a, n_a, x_b, n_b)
        self.assertAlmostEqual(t.z ** 2, chi2.statistic, places=6)
        self.assertAlmostEqual(t.p_value, chi2.pvalue, places=10)

    def test_ci_coverage_is_about_95_percent(self):
        rng = np.random.default_rng(0)
        p_a, p_b, n, sims = 0.19, 0.18, 5_000, 2_000
        xa, xb = rng.binomial(n, p_a, sims), rng.binomial(n, p_b, sims)
        hits = sum(two_prop_test(a, n, b, n).ci_low <= p_b - p_a <= two_prop_test(a, n, b, n).ci_high
                   for a, b in zip(xa, xb))
        self.assertTrue(0.935 < hits / sims < 0.965, hits / sims)

    def test_false_positive_rate_is_about_alpha(self):
        rng = np.random.default_rng(1)
        n, sims = 5_000, 4_000
        xa, xb = rng.binomial(n, 0.2, sims), rng.binomial(n, 0.2, sims)
        rate = np.mean([two_prop_test(a, n, b, n).p_value < 0.05 for a, b in zip(xa, xb)])
        self.assertTrue(0.04 < rate < 0.06, rate)


class TestHolm(unittest.TestCase):
    def test_known_example(self):
        got = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(got["a"], 0.03)
        self.assertAlmostEqual(got["c"], 0.06)
        self.assertAlmostEqual(got["b"], 0.06)   # stays monotone

    def test_capped_at_one(self):
        self.assertEqual(holm({"a": 0.6, "b": 0.9})["b"], 1.0)


class TestResampling(unittest.TestCase):
    def test_bootstrap_sd_matches_standard_error(self):
        x_a, n_a, x_b, n_b = 8502, 44700, 8279, 45489
        boot = bootstrap_prop_diff(x_a, n_a, x_b, n_b, n_boot=40_000)
        p_a, p_b = x_a / n_a, x_b / n_b
        se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        self.assertAlmostEqual(boot.std() / se, 1.0, delta=0.02)

    def test_bootstrap_mean_diff_centres_on_observed(self):
        rng = np.random.default_rng(2)
        a, b = rng.exponential(50, 3_000), rng.exponential(55, 3_000)
        boot = bootstrap_mean_diff(a, b, n_boot=1_000)
        self.assertAlmostEqual(boot.mean(), b.mean() - a.mean(), delta=0.3)

    def test_beta_posterior_is_symmetric_for_equal_data(self):
        r = beta_posterior(500, 1000, 500, 1000)
        self.assertAlmostEqual(r["prob_b_better"], 0.5, delta=0.01)


class TestPower(unittest.TestCase):
    def test_mde_gives_80_percent_power(self):
        n = 44_700
        mde = mde_two_prop(0.19, n)
        self.assertAlmostEqual(power_two_prop(0.19, 0.19 + mde, n), 0.8, delta=0.01)

    def test_sample_size_gives_80_percent_power(self):
        n = n_per_arm(0.45, -0.01)
        self.assertAlmostEqual(power_two_prop(0.45, 0.44, n), 0.8, delta=0.005)

    def test_power_by_simulation(self):
        rng = np.random.default_rng(3)
        n, sims = 4_000, 3_000
        xa, xb = rng.binomial(n, 0.20, sims), rng.binomial(n, 0.22, sims)
        sim = np.mean([two_prop_test(a, n, b, n).p_value < 0.05 for a, b in zip(xa, xb)])
        self.assertAlmostEqual(sim, power_two_prop(0.20, 0.22, n), delta=0.03)


if __name__ == "__main__":
    unittest.main()
