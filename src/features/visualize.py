# src/features/visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_all(covid_path, climate_path, tfidf_path, vader_path, auth_urg_path, output_path):

    covid   = pd.read_csv(covid_path).dropna(subset=['anger'])
    climate = pd.read_csv(climate_path).dropna(subset=['anger'])
    with open(tfidf_path) as f:
        tfidf = json.load(f)
    with open(vader_path) as f:
        vader = json.load(f)
    with open(auth_urg_path) as f:
        auth_urg = json.load(f)

    emotions  = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    COV_COLOR = '#E05C5C'
    CLI_COLOR = '#4A90D9'

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('COVID vs Climate Misinformation — Linguistic & Emotional Analysis',
                 fontsize=15, fontweight='bold')

    # 1. mean emotion scores
    ax = axes[0, 0]
    x, w = np.arange(len(emotions)), 0.35
    ax.bar(x - w/2, [covid[e].mean()   for e in emotions], w, label='COVID',   color=COV_COLOR, alpha=0.85)
    ax.bar(x + w/2, [climate[e].mean() for e in emotions], w, label='Climate', color=CLI_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=30, ha='right')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Emotion Scores by Domain')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. dominant emotion stacked bar
    ax = axes[0, 1]
    covid_dom   = covid[emotions].idxmax(axis=1).value_counts(normalize=True).mul(100).reindex(emotions, fill_value=0)
    climate_dom = climate[emotions].idxmax(axis=1).value_counts(normalize=True).mul(100).reindex(emotions, fill_value=0)
    palette = sns.color_palette("Set2", len(emotions))
    bottom_c, bottom_cl = 0, 0
    for i, e in enumerate(emotions):
        ax.bar(0, covid_dom[e],   bottom=bottom_c,  color=palette[i], label=e, width=0.4)
        ax.bar(1, climate_dom[e], bottom=bottom_cl, color=palette[i], width=0.4)
        bottom_c  += covid_dom[e]
        bottom_cl += climate_dom[e]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['COVID', 'Climate'], fontsize=12)
    ax.set_ylabel('% of texts')
    ax.set_title('Dominant Emotion Distribution')
    ax.legend(loc='upper right', fontsize=8)

    # 3. VADER sentiment neg, neu, pos
    ax = axes[1, 0]
    categories = ['neg', 'neu', 'pos']
    labels     = ['Negative', 'Neutral', 'Positive']
    x2 = np.arange(len(categories))
    ax.bar(x2 - w/2, [vader['covid'][c]   for c in categories], w, label='COVID',   color=COV_COLOR, alpha=0.85)
    ax.bar(x2 + w/2, [vader['climate'][c] for c in categories], w, label='Climate', color=CLI_COLOR, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x2)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('VADER Sentiment Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. VADER sentiment compound
    ax = axes[1, 1]
    ax.bar(0, vader['covid']['compound'],   color=COV_COLOR, alpha=0.85, width=0.4, label='COVID')
    ax.bar(1, vader['climate']['compound'], color=CLI_COLOR, alpha=0.85, width=0.4, label='Climate')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['COVID', 'Climate'])
    ax.set_ylabel('Compound Score')
    ax.set_title('VADER Compound Sentiment')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 5. sadness & anger box plots
    ax = axes[2, 0]
    ax.boxplot([covid['sadness'].values],   positions=[1],   widths=0.35, patch_artist=True,
               boxprops=dict(facecolor=COV_COLOR, alpha=0.7), medianprops=dict(color='black', linewidth=2))
    ax.boxplot([climate['sadness'].values], positions=[1.5], widths=0.35, patch_artist=True,
               boxprops=dict(facecolor=CLI_COLOR, alpha=0.7), medianprops=dict(color='black', linewidth=2))
    ax.boxplot([covid['anger'].values],     positions=[2.5], widths=0.35, patch_artist=True,
               boxprops=dict(facecolor=COV_COLOR, alpha=0.7), medianprops=dict(color='black', linewidth=2))
    ax.boxplot([climate['anger'].values],   positions=[3],   widths=0.35, patch_artist=True,
               boxprops=dict(facecolor=CLI_COLOR, alpha=0.7), medianprops=dict(color='black', linewidth=2))
    ax.set_xticks([1.25, 2.75])
    ax.set_xticklabels(['Sadness', 'Anger'], fontsize=11)
    ax.set_ylabel('Emotion Score')
    ax.set_title('Key Finding: Sadness vs Anger (COVID vs Climate)')
    ax.legend([plt.Rectangle((0,0),1,1, fc=COV_COLOR, alpha=0.7),
               plt.Rectangle((0,0),1,1, fc=CLI_COLOR, alpha=0.7)], ['COVID', 'Climate'])
    ax.grid(axis='y', alpha=0.3)

    # 6. TF-IDF COVID
    ax = axes[2, 1]
    terms  = [t[0] for t in tfidf['covid'][:15]]
    scores = [t[1] for t in tfidf['covid'][:15]]
    ax.barh(terms[::-1], scores[::-1], color=COV_COLOR, alpha=0.85)
    ax.set_xlabel('Mean TF-IDF Score')
    ax.set_title('Top 15 TF-IDF Terms — COVID Misinformation')
    ax.grid(axis='x', alpha=0.3)

    # 7. TF-IDF Climate
    ax = axes[3, 0]
    terms  = [t[0] for t in tfidf['climate'][:15]]
    scores = [t[1] for t in tfidf['climate'][:15]]
    ax.barh(terms[::-1], scores[::-1], color=CLI_COLOR, alpha=0.85)
    ax.set_xlabel('Mean TF-IDF Score')
    ax.set_title('Top 15 TF-IDF Terms — Climate Misinformation')
    ax.grid(axis='x', alpha=0.3)

    # 8. Authoritative language and urgency scores
    ax = axes[3, 1]
    covid_auth = auth_urg['covid']['auth']
    covid_urg  = auth_urg['covid']['urg']
    climate_auth = auth_urg['climate']['auth']
    climate_urg  = auth_urg['climate']['urg']

    x3 = np.arange(2)
    ax.bar(x3 - w/2, [covid_auth, covid_urg], width=w, color=COV_COLOR, alpha=0.85, label='COVID')
    ax.bar(x3 + w/2, [climate_auth, climate_urg], width=w, color=CLI_COLOR, alpha=0.85, label='Climate')
    ax.set_xticks(x3)
    ax.set_xticklabels(['Authoritative language', 'Urgency'])
    ax.set_ylabel('Average Score (scaled 0 to 1)')
    ax.set_title('Authoritative Language & Urgency Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {output_path}")
    plt.show()




if __name__ == '__main__':
    plot_all(
        covid_path   = 'data/analysis/covid_emotions.csv',
        climate_path = 'data/analysis/climate_emotions.csv',
        tfidf_path   = 'data/analysis/top_tfidf_terms.json',
        vader_path   = 'data/analysis/vader_sentiment.json',
        auth_urg_path = 'data/analysis/auth_urg.json',
        output_path  = 'data/analysis/misinformation_analysis.png'
    )