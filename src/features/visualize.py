# src/features/visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

CONTEXT_EMOTIONS = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'joy']


def build_context_frame(term_context, domain, top_n=6):
    rows = []
    for entry in term_context.get(domain, {}).get('terms', [])[:top_n]:
        row = {
            'domain': domain,
            'term': entry['term'],
            'document_frequency': entry['document_frequency'],
            'share_of_domain_docs': entry['share_of_domain_docs'],
            'vader_compound': entry['average_vader']['vader_compound'],
        }
        for emotion in CONTEXT_EMOTIONS:
            row[emotion] = entry['average_emotions'][emotion]
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            'domain',
            'term',
            'document_frequency',
            'share_of_domain_docs',
            'vader_compound',
            *CONTEXT_EMOTIONS,
        ],
    )


def render_empty_context_axis(ax, title, message):
    ax.set_title(title)
    ax.axis('off')
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10)


def annotate_context_examples(ax, entries):
    lines = []
    for entry in entries:
        examples = entry.get('representative_examples', [])
        if not examples:
            continue
        example = examples[0]
        lines.append(f"{entry['term']}: \"{example['text']}\"")
    ax.axis('off')
    ax.text(
        0.0,
        1.0,
        '\n\n'.join(lines),
        va='top',
        ha='left',
        fontsize=9,
        wrap=True,
    )


def plot_tfidf_context(context_path, output_path, top_n=6, show=False):
    with open(context_path, encoding='utf-8') as handle:
        term_context = json.load(handle)

    covid_df = build_context_frame(term_context, 'covid', top_n=top_n)
    climate_df = build_context_frame(term_context, 'climate', top_n=top_n)
    cov_color = '#E05C5C'
    cli_color = '#4A90D9'

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.3, 1.0])
    fig.suptitle(
        'TF-IDF Context Differences Between COVID and Climate Misinformation',
        fontsize=15,
        fontweight='bold',
        y=0.99,
    )

    ax = fig.add_subplot(grid[0, 0])
    if covid_df.empty:
        render_empty_context_axis(ax, 'COVID Top-Term Context Sentiment', 'No TF-IDF context data available.')
    else:
        ax.barh(covid_df['term'][::-1], covid_df['vader_compound'][::-1], color=cov_color, alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Average VADER Compound in Matching Posts')
        ax.set_title('COVID Top-Term Context Sentiment')
        ax.grid(axis='x', alpha=0.3)

    ax = fig.add_subplot(grid[0, 1])
    if climate_df.empty:
        render_empty_context_axis(ax, 'Climate Top-Term Context Sentiment', 'No TF-IDF context data available.')
    else:
        ax.barh(
            climate_df['term'][::-1],
            climate_df['vader_compound'][::-1],
            color=cli_color,
            alpha=0.85,
        )
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Average VADER Compound in Matching Posts')
        ax.set_title('Climate Top-Term Context Sentiment')
        ax.grid(axis='x', alpha=0.3)

    ax = fig.add_subplot(grid[1, 0])
    if covid_df.empty:
        render_empty_context_axis(ax, 'COVID Term-Context Emotion Mix', 'No TF-IDF context data available.')
    else:
        sns.heatmap(
            covid_df.set_index('term')[CONTEXT_EMOTIONS],
            cmap='Reds',
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Mean emotion score'},
            ax=ax,
        )
        ax.set_title('COVID Term-Context Emotion Mix')
        ax.set_xlabel('Emotion')
        ax.set_ylabel('')

    ax = fig.add_subplot(grid[1, 1])
    if climate_df.empty:
        render_empty_context_axis(ax, 'Climate Term-Context Emotion Mix', 'No TF-IDF context data available.')
    else:
        sns.heatmap(
            climate_df.set_index('term')[CONTEXT_EMOTIONS],
            cmap='Blues',
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Mean emotion score'},
            ax=ax,
        )
        ax.set_title('Climate Term-Context Emotion Mix')
        ax.set_xlabel('Emotion')
        ax.set_ylabel('')

    ax = fig.add_subplot(grid[2, 0])
    ax.set_title('COVID Example Posts')
    annotate_context_examples(ax, term_context.get('covid', {}).get('terms', [])[:3])

    ax = fig.add_subplot(grid[2, 1])
    ax.set_title('Climate Example Posts')
    annotate_context_examples(ax, term_context.get('climate', {}).get('terms', [])[:3])

    fig.text(
        0.5,
        0.02,
        (
            'COVID top-term contexts skew more negative and threat-focused, while climate top-term '
            'contexts show more policy/science language with anger/fear but milder overall sentiment.'
        ),
        ha='center',
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_all(covid_path, climate_path, tfidf_path, vader_path, output_path):

    covid   = pd.read_csv(covid_path).dropna(subset=['anger'])
    climate = pd.read_csv(climate_path).dropna(subset=['anger'])
    with open(tfidf_path) as f:
        tfidf = json.load(f)
    with open(vader_path) as f:
        vader = json.load(f)

    emotions  = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    COV_COLOR = '#E05C5C'
    CLI_COLOR = '#4A90D9'

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
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

    # 3. VADER sentiment
    ax = axes[1, 0]
    categories = ['neg', 'neu', 'pos', 'compound']
    labels     = ['Negative', 'Neutral', 'Positive', 'Compound']
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

    # 4. sadness & anger box plots
    ax = axes[1, 1]
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

    # 5. TF-IDF COVID
    ax = axes[2, 0]
    terms  = [t[0] for t in tfidf['covid'][:15]]
    scores = [t[1] for t in tfidf['covid'][:15]]
    ax.barh(terms[::-1], scores[::-1], color=COV_COLOR, alpha=0.85)
    ax.set_xlabel('Mean TF-IDF Score')
    ax.set_title('Top 15 TF-IDF Terms — COVID Misinformation')
    ax.grid(axis='x', alpha=0.3)

    # 6. TF-IDF Climate
    ax = axes[2, 1]
    terms  = [t[0] for t in tfidf['climate'][:15]]
    scores = [t[1] for t in tfidf['climate'][:15]]
    ax.barh(terms[::-1], scores[::-1], color=CLI_COLOR, alpha=0.85)
    ax.set_xlabel('Mean TF-IDF Score')
    ax.set_title('Top 15 TF-IDF Terms — Climate Misinformation')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {output_path}")


if __name__ == '__main__':
    plot_all(
        covid_path   = 'data/analysis/covid_emotions.csv',
        climate_path = 'data/analysis/climate_emotions.csv',
        tfidf_path   = 'data/analysis/top_tfidf_terms.json',
        vader_path   = 'data/analysis/vader_sentiment.json',
        output_path  = 'data/analysis/misinformation_analysis.png'
    )
