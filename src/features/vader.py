import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

VADER_COLUMNS = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']

# Calculate the average sentiment scores of a list of texts
# Input: texts = list of texts
# Output:
#   avg_neg = percentage of all texts with a negative sentiment
#   avg_neu = percentage of all texts with a neutral sentiment
#   avg_pos = percentage of all texts with a positive sentiment
#   avg_comp = compound sentiment of all texts
#       avg_comp > 0.05 is positive
#       avg_comp < -0.05 is negative
#       -0.05 < avg_comp < 0.05 is neutral 

def vader_series(texts):
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        rows.append(scores)
    return pd.DataFrame(rows)


def get_vader_scores(texts):
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for text in texts:
        clean_text = text if isinstance(text, str) else ''
        scores = analyzer.polarity_scores(clean_text)
        rows.append(
            {
                'vader_neg': scores['neg'],
                'vader_neu': scores['neu'],
                'vader_pos': scores['pos'],
                'vader_compound': scores['compound'],
            }
        )

    return pd.DataFrame(rows, columns=VADER_COLUMNS)


def avg_vader(texts):
    scores = get_vader_scores(texts)
    means = scores.mean()
    return tuple(float(means[column]) for column in VADER_COLUMNS)
