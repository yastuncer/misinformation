import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


VADER_COLUMNS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]


def get_vader_scores(texts):
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for text in texts:
        clean_text = text if isinstance(text, str) else ""
        scores = analyzer.polarity_scores(clean_text)
        rows.append(
            {
                "vader_neg": scores["neg"],
                "vader_neu": scores["neu"],
                "vader_pos": scores["pos"],
                "vader_compound": scores["compound"],
            }
        )

    return pd.DataFrame(rows, columns=VADER_COLUMNS)


def avg_vader(texts):
    scores = get_vader_scores(texts)
    means = scores.mean()
    return tuple(float(means[column]) for column in VADER_COLUMNS)
