import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def significance_label(p_value):
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_p_value(p_value):
    if p_value is None or pd.isna(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return f"p = {p_value:.1e}"
    return f"p = {p_value:.3f}"


def get_stat_lookup(stats_df):
    if stats_df is None:
        return {}
    return {
        (row["feature_group"], row["feature"]): row
        for _, row in stats_df.iterrows()
    }


def annotate_significance(ax, x_positions, left_values, right_values, rows, y_padding_ratio=0.03):
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-6)

    for x_pos, left_value, right_value, row in zip(x_positions, left_values, right_values, rows):
        if row is None:
            continue
        ax.text(
            x_pos,
            max(left_value, right_value) + y_span * y_padding_ratio,
            significance_label(row["adjusted_p_value"]),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_all(
    covid_path,
    climate_path,
    tfidf_path,
    vader_path,
    output_path,
    stats_path=None,
    summary_path=None,
):
    covid = pd.read_csv(covid_path).dropna(subset=["anger"])
    climate = pd.read_csv(climate_path).dropna(subset=["anger"])
    with open(tfidf_path, encoding="utf-8") as handle:
        tfidf = json.load(handle)
    with open(vader_path, encoding="utf-8") as handle:
        vader = json.load(handle)

    stats_df = pd.read_csv(stats_path) if stats_path else None
    stat_lookup = get_stat_lookup(stats_df)
    summary = None
    if summary_path:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)

    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    cov_color = "#E05C5C"
    cli_color = "#4A90D9"

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle(
        "COVID vs Climate Misinformation - Linguistic and Emotional Analysis",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    x, width = np.arange(len(emotions)), 0.35
    covid_emotion_means = [covid[column].mean() for column in emotions]
    climate_emotion_means = [climate[column].mean() for column in emotions]
    ax.bar(x - width / 2, covid_emotion_means, width, label="COVID", color=cov_color, alpha=0.85)
    ax.bar(x + width / 2, climate_emotion_means, width, label="Climate", color=cli_color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=30, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title("Mean Emotion Scores by Domain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    annotate_significance(
        ax,
        x,
        covid_emotion_means,
        climate_emotion_means,
        [stat_lookup.get(("emotion", emotion)) for emotion in emotions],
    )

    ax = axes[0, 1]
    covid_dom = covid[emotions].idxmax(axis=1).value_counts(normalize=True).mul(100).reindex(emotions, fill_value=0)
    climate_dom = climate[emotions].idxmax(axis=1).value_counts(normalize=True).mul(100).reindex(emotions, fill_value=0)
    palette = sns.color_palette("Set2", len(emotions))
    bottom_covid, bottom_climate = 0, 0
    for index, emotion in enumerate(emotions):
        ax.bar(0, covid_dom[emotion], bottom=bottom_covid, color=palette[index], label=emotion, width=0.4)
        ax.bar(1, climate_dom[emotion], bottom=bottom_climate, color=palette[index], width=0.4)
        bottom_covid += covid_dom[emotion]
        bottom_climate += climate_dom[emotion]
    dominant_row = stat_lookup.get(("emotion_distribution", "dominant_emotion"))
    dominant_title = "Dominant Emotion Distribution"
    if dominant_row is not None:
        dominant_title += f"\n{format_p_value(dominant_row['adjusted_p_value'])}, V = {dominant_row['effect_size']:.3f}"
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["COVID", "Climate"], fontsize=12)
    ax.set_ylabel("% of texts")
    ax.set_title(dominant_title)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 0]
    categories = ["neg", "neu", "pos", "compound"]
    labels = ["Negative", "Neutral", "Positive", "Compound"]
    stat_features = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
    x2 = np.arange(len(categories))
    covid_vader_values = [vader["covid"][category] for category in categories]
    climate_vader_values = [vader["climate"][category] for category in categories]
    ax.bar(x2 - width / 2, covid_vader_values, width, label="COVID", color=cov_color, alpha=0.85)
    ax.bar(x2 + width / 2, climate_vader_values, width, label="Climate", color=cli_color, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("VADER Sentiment Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    annotate_significance(
        ax,
        x2,
        covid_vader_values,
        climate_vader_values,
        [stat_lookup.get(("vader", feature)) for feature in stat_features],
        y_padding_ratio=0.05,
    )

    ax = axes[1, 1]
    ax.boxplot(
        [covid["sadness"].values],
        positions=[1],
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=cov_color, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )
    ax.boxplot(
        [climate["sadness"].values],
        positions=[1.5],
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=cli_color, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )
    ax.boxplot(
        [covid["anger"].values],
        positions=[2.5],
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=cov_color, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )
    ax.boxplot(
        [climate["anger"].values],
        positions=[3],
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=cli_color, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )
    sadness_row = stat_lookup.get(("emotion", "sadness"))
    anger_row = stat_lookup.get(("emotion", "anger"))
    ax.set_xticks([1.25, 2.75])
    ax.set_xticklabels(["Sadness", "Anger"], fontsize=11)
    ax.set_ylabel("Emotion Score")
    ax.set_title(
        "Key Emotion Differences\n"
        f"Sadness {significance_label(sadness_row['adjusted_p_value']) if sadness_row is not None else ''} | "
        f"Anger {significance_label(anger_row['adjusted_p_value']) if anger_row is not None else ''}"
    )
    ax.legend(
        [
            plt.Rectangle((0, 0), 1, 1, fc=cov_color, alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, fc=cli_color, alpha=0.7),
        ],
        ["COVID", "Climate"],
    )
    ax.grid(axis="y", alpha=0.3)
    summary_lines = []
    if sadness_row is not None:
        summary_lines.append(f"Sadness: {format_p_value(sadness_row['adjusted_p_value'])}")
    if anger_row is not None:
        summary_lines.append(f"Anger: {format_p_value(anger_row['adjusted_p_value'])}")
    if summary is not None and summary.get("missing_trust_cue_columns"):
        summary_lines.append("Trust cues pending Josh merge")
    if summary_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    ax = axes[2, 0]
    covid_terms = [term_score[0] for term_score in tfidf["covid"][:15]]
    covid_scores = [term_score[1] for term_score in tfidf["covid"][:15]]
    ax.barh(covid_terms[::-1], covid_scores[::-1], color=cov_color, alpha=0.85)
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title("Top 15 TF-IDF Terms - COVID Misinformation")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[2, 1]
    climate_terms = [term_score[0] for term_score in tfidf["climate"][:15]]
    climate_scores = [term_score[1] for term_score in tfidf["climate"][:15]]
    ax.barh(climate_terms[::-1], climate_scores[::-1], color=cli_color, alpha=0.85)
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title("Top 15 TF-IDF Terms - Climate Misinformation")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_all(
        covid_path="data/analysis/covid_emotions.csv",
        climate_path="data/analysis/climate_emotions.csv",
        tfidf_path="data/analysis/top_tfidf_terms.json",
        vader_path="data/analysis/vader_sentiment.json",
        output_path="data/analysis/misinformation_analysis.png",
        stats_path="data/analysis/statistical_tests.csv",
        summary_path="data/analysis/stats_summary.json",
    )
