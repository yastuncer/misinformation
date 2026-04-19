import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
CONTEXT_EMOTIONS = ["anger", "disgust", "fear", "sadness", "neutral", "joy"]


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


def numeric_test_label(summary):
    if not summary:
        return "adjusted significance"
    test_name = summary.get("numeric_test", "")
    if test_name == "mann_whitney_u":
        return "Mann-Whitney U significance"
    return "adjusted significance"


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


def build_context_frame(term_context, domain, top_n=6):
    rows = []
    for entry in term_context.get(domain, {}).get("terms", [])[:top_n]:
        row = {
            "domain": domain,
            "term": entry["term"],
            "document_frequency": entry["document_frequency"],
            "share_of_domain_docs": entry["share_of_domain_docs"],
            "vader_compound": entry["average_vader"]["vader_compound"],
        }
        for emotion in CONTEXT_EMOTIONS:
            row[emotion] = entry["average_emotions"][emotion]
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_context_examples(ax, entries):
    lines = []
    for entry in entries:
        examples = entry.get("representative_examples", [])
        if not examples:
            continue
        example = examples[0]
        lines.append(f"{entry['term']}: \"{example['text']}\"")
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "\n\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        wrap=True,
    )


def plot_tfidf_context(context_path, output_path, top_n=6, show=False):
    with open(context_path, encoding="utf-8") as handle:
        term_context = json.load(handle)

    covid_df = build_context_frame(term_context, "covid", top_n=top_n)
    climate_df = build_context_frame(term_context, "climate", top_n=top_n)
    cov_color = "#E05C5C"
    cli_color = "#4A90D9"

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.3, 1.0])
    fig.suptitle(
        "TF-IDF Context Differences Between COVID and Climate Misinformation",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.barh(covid_df["term"][::-1], covid_df["vader_compound"][::-1], color=cov_color, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average VADER Compound in Matching Posts")
    ax.set_title("COVID Top-Term Context Sentiment")
    ax.grid(axis="x", alpha=0.3)

    ax = fig.add_subplot(grid[0, 1])
    ax.barh(
        climate_df["term"][::-1],
        climate_df["vader_compound"][::-1],
        color=cli_color,
        alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average VADER Compound in Matching Posts")
    ax.set_title("Climate Top-Term Context Sentiment")
    ax.grid(axis="x", alpha=0.3)

    ax = fig.add_subplot(grid[1, 0])
    sns.heatmap(
        covid_df.set_index("term")[CONTEXT_EMOTIONS],
        cmap="Reds",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean emotion score"},
        ax=ax,
    )
    ax.set_title("COVID Term-Context Emotion Mix")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("")

    ax = fig.add_subplot(grid[1, 1])
    sns.heatmap(
        climate_df.set_index("term")[CONTEXT_EMOTIONS],
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean emotion score"},
        ax=ax,
    )
    ax.set_title("Climate Term-Context Emotion Mix")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("")

    ax = fig.add_subplot(grid[2, 0])
    ax.set_title("COVID Example Posts")
    annotate_context_examples(ax, term_context.get("covid", {}).get("terms", [])[:3])

    ax = fig.add_subplot(grid[2, 1])
    ax.set_title("Climate Example Posts")
    annotate_context_examples(ax, term_context.get("climate", {}).get("terms", [])[:3])

    fig.text(
        0.5,
        0.02,
        (
            "COVID top-term contexts skew more negative and threat-focused, while climate top-term "
            "contexts show more policy/science language with anger/fear but milder overall sentiment."
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_all(
    covid_path,
    climate_path,
    tfidf_path,
    vader_path,
    output_path,
    stats_path=None,
    summary_path=None,
    show=False,
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
    numeric_label = numeric_test_label(summary)

    cov_color = "#E05C5C"
    cli_color = "#4A90D9"

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle(
        "COVID vs Climate Misinformation - Linguistic and Emotional Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    ax = axes[0, 0]
    x, width = np.arange(len(EMOTIONS)), 0.35
    covid_emotion_means = [covid[column].mean() for column in EMOTIONS]
    climate_emotion_means = [climate[column].mean() for column in EMOTIONS]
    ax.bar(x - width / 2, covid_emotion_means, width, label="COVID", color=cov_color, alpha=0.85)
    ax.bar(x + width / 2, climate_emotion_means, width, label="Climate", color=cli_color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title(f"Mean Emotion Scores by Domain\n(bars show means; {numeric_label})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    annotate_significance(
        ax,
        x,
        covid_emotion_means,
        climate_emotion_means,
        [stat_lookup.get(("emotion", emotion)) for emotion in EMOTIONS],
    )

    ax = axes[0, 1]
    covid_dom = covid[EMOTIONS].idxmax(axis=1).value_counts(normalize=True).mul(100).reindex(EMOTIONS, fill_value=0)
    climate_dom = climate[EMOTIONS].idxmax(axis=1).value_counts(normalize=True).mul(100).reindex(EMOTIONS, fill_value=0)
    palette = sns.color_palette("Set2", len(EMOTIONS))
    bottom_covid, bottom_climate = 0, 0
    for index, emotion in enumerate(EMOTIONS):
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
    ax.set_title(f"VADER Sentiment Comparison\n(bars show means; {numeric_label})")
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
        f"Key Emotion Differences\n{numeric_label}\n"
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
        missing = ", ".join(summary["missing_trust_cue_columns"])
        summary_lines.append(f"Missing trust-cue columns: {missing}")
    if summary_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_all(
        covid_path="data/analysis/covid_emotions.csv",
        climate_path="data/analysis/climate_emotions.csv",
        tfidf_path="data/analysis/top_tfidf_terms.json",
        vader_path="data/analysis/vader_sentiment.json",
        output_path="data/analysis/misinformation_analysis.png",
        stats_path="data/analysis/statistical_tests_mwu.csv",
        summary_path="data/analysis/stats_summary_mwu.json",
        show=False,
    )
    plot_tfidf_context(
        context_path="data/analysis/tfidf_term_context.json",
        output_path="data/analysis/tfidf_context_comparison.png",
        show=False,
    )
