import math
import unittest

import pandas as pd

from src.analysis.stats import (
    benjamini_hochberg,
    cramers_v,
    dominant_emotion_result,
    numeric_test_result,
    rank_biserial_correlation,
)


class StatsHelpersTest(unittest.TestCase):
    def test_rank_biserial_is_one_when_first_sample_dominates(self):
        self.assertEqual(rank_biserial_correlation(4.0, 2, 2), 1.0)

    def test_benjamini_hochberg_is_monotonic_after_sorting(self):
        adjusted = benjamini_hochberg([0.001, 0.01, 0.03, 0.2])
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))

    def test_cramers_v_returns_zero_for_balanced_table(self):
        contingency = pd.DataFrame({"covid": [10, 10], "climate": [10, 10]})
        self.assertEqual(cramers_v(0.0, contingency), 0.0)

    def test_dominant_emotion_result_reports_counts(self):
        covid = pd.DataFrame(
            {
                "anger": [0.7, 0.1],
                "disgust": [0.1, 0.1],
                "fear": [0.1, 0.1],
                "joy": [0.05, 0.6],
                "neutral": [0.02, 0.05],
                "sadness": [0.02, 0.03],
                "surprise": [0.01, 0.02],
            }
        )
        climate = pd.DataFrame(
            {
                "anger": [0.1, 0.2],
                "disgust": [0.1, 0.1],
                "fear": [0.6, 0.5],
                "joy": [0.05, 0.05],
                "neutral": [0.1, 0.05],
                "sadness": [0.03, 0.05],
                "surprise": [0.02, 0.05],
            }
        )
        covid["dominant_emotion"] = covid[
            ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        ].idxmax(axis=1)
        climate["dominant_emotion"] = climate[
            ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        ].idxmax(axis=1)

        result, payload = dominant_emotion_result(covid, climate)

        self.assertEqual(result["feature"], "dominant_emotion")
        self.assertEqual(payload["covid"]["counts"]["anger"], 1)
        self.assertEqual(payload["covid"]["counts"]["joy"], 1)
        self.assertEqual(payload["climate"]["counts"]["fear"], 2)
        self.assertTrue(math.isfinite(result["effect_size"]))

    def test_numeric_test_result_reports_mwu_metadata_and_filters_nans(self):
        result = numeric_test_result(
            "sadness",
            "emotion",
            [3.0, float("nan"), 4.0],
            [1.0, 2.0, float("nan")],
        )

        self.assertEqual(result["test"], "mann_whitney_u")
        self.assertEqual(result["statistic_name"], "u")
        self.assertEqual(result["effect_size_name"], "rank_biserial_correlation")
        self.assertEqual(result["covid_n"], 2)
        self.assertEqual(result["climate_n"], 2)
        self.assertEqual(result["covid_median"], 3.5)
        self.assertEqual(result["climate_median"], 1.5)
        self.assertEqual(result["median_difference"], 2.0)
        self.assertTrue(math.isfinite(result["p_value"]))

    def test_numeric_test_result_handles_identical_samples_with_ties(self):
        result = numeric_test_result("fear", "emotion", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

        self.assertAlmostEqual(result["effect_size"], 0.0)
        self.assertAlmostEqual(result["p_value"], 1.0)

    def test_numeric_test_result_handles_empty_after_nan_filtering(self):
        result = numeric_test_result(
            "joy",
            "emotion",
            [float("nan"), float("nan")],
            [1.0, 2.0],
        )

        self.assertEqual(result["covid_n"], 0)
        self.assertTrue(math.isnan(result["statistic"]))
        self.assertTrue(math.isnan(result["p_value"]))
        self.assertEqual(result["notes"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
