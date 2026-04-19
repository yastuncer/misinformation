import math
import unittest

import pandas as pd

from src.analysis.stats import benjamini_hochberg, cohens_d, cramers_v, dominant_emotion_result


class StatsHelpersTest(unittest.TestCase):
    def test_cohens_d_is_zero_for_identical_samples(self):
        self.assertEqual(cohens_d([1, 2, 3], [1, 2, 3]), 0.0)

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


if __name__ == "__main__":
    unittest.main()
