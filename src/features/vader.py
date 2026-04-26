from turtle import pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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


def avg_vader(texts):
    analyzer = SentimentIntensityAnalyzer()
    avg_neg, avg_neu, avg_pos, avg_comp = 0, 0, 0, 0

    for text in texts:
        scores = analyzer.polarity_scores(text)
        avg_neg += scores['neg']
        avg_neu += scores['neu']
        avg_pos += scores['pos']
        avg_comp += scores['compound']

    avg_neg /= len(texts)
    avg_neu /= len(texts)
    avg_pos /= len(texts)
    avg_comp /= len(texts)
    return avg_neg, avg_neu, avg_pos, avg_comp