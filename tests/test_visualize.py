import unittest

from src.features.visualize import bootstrap_mean_ci, holm_bonferroni


class VisualizeHelpersTest(unittest.TestCase):
    def test_holm_bonferroni_reorders_unsorted_input_correctly(self):
        adjusted = holm_bonferroni([0.04, 0.002, 0.03, 0.01])
        expected = [0.06, 0.008, 0.06, 0.03]
        for actual, target in zip(adjusted, expected):
            self.assertAlmostEqual(actual, target)

    def test_bootstrap_mean_ci_is_exact_for_constant_sample(self):
        lower, upper = bootstrap_mean_ci([0.5, 0.5, 0.5], n_boot=100, seed=7)

        self.assertEqual(lower, 0.5)
        self.assertEqual(upper, 0.5)


if __name__ == "__main__":
    unittest.main()
