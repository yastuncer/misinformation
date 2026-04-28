import unittest

import numpy as np
import pandas as pd

from src.analysis.tfidf_context import _ensure_vader_columns, build_term_context_records


class FakeVectorizer:
    def __init__(self, features):
        self._features = np.array(features)

    def get_feature_names_out(self):
        return self._features


class TfidfContextTest(unittest.TestCase):
    def test_ensure_vader_columns_replaces_partial_existing_columns(self):
        df = pd.DataFrame(
            {
                "text": ["covid claim", "climate claim"],
                "vader_compound": [-0.1, 0.2],
            }
        )

        enriched = _ensure_vader_columns(df)

        self.assertEqual(list(enriched.filter(regex=r"^vader_").columns), [
            "vader_neg",
            "vader_neu",
            "vader_pos",
            "vader_compound",
        ])
        self.assertEqual(enriched.filter(regex=r"^vader_").shape[1], 4)

    def test_build_term_context_records_summarizes_matching_documents(self):
        df = pd.DataFrame(
            {
                "text": [
                    "covid claims spread online",
                    "another covid claim goes viral",
                    "climate misinformation sample",
                ],
                "text_lemma": [
                    "covid claim spread online",
                    "another covid claim go viral",
                    "climate misinformation sample",
                ],
                "dataset": ["covid_a", "covid_b", "climate_a"],
                "source": ["cdl", "princeton", "cdl"],
                "vader_neg": [0.2, 0.1, 0.3],
                "vader_neu": [0.7, 0.8, 0.5],
                "vader_pos": [0.1, 0.1, 0.2],
                "vader_compound": [-0.4, -0.1, 0.2],
                "anger": [0.3, 0.2, 0.4],
                "disgust": [0.1, 0.1, 0.1],
                "fear": [0.2, 0.3, 0.2],
                "joy": [0.05, 0.04, 0.1],
                "neutral": [0.2, 0.25, 0.1],
                "sadness": [0.1, 0.08, 0.05],
                "surprise": [0.05, 0.03, 0.05],
                "dominant_emotion": ["anger", "fear", "anger"],
            }
        )
        matrix = np.array(
            [
                [0.6, 0.2, 0.0],
                [0.5, 0.3, 0.0],
                [0.0, 0.0, 0.7],
            ]
        )
        vectorizer = FakeVectorizer(["covid", "claim", "climate"])

        records = build_term_context_records(
            df=df.iloc[:2],
            row_indices=np.array([0, 1]),
            term_entries=[("covid", 0.55)],
            vectorizer=vectorizer,
            tfidf_matrix=matrix,
            examples_per_term=2,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["term"], "covid")
        self.assertEqual(record["document_frequency"], 2)
        self.assertAlmostEqual(record["average_vader"]["vader_compound"], -0.25)
        self.assertEqual(len(record["representative_examples"]), 2)
        self.assertEqual(record["representative_examples"][0]["dataset"], "covid_a")


if __name__ == "__main__":
    unittest.main()
