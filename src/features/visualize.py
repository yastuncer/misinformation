import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy import stats as scipy_stats


EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
CONTEXT_EMOTIONS = ["anger", "disgust", "fear", "sadness", "neutral", "joy"]
REFERENCE_BACKGROUND = "#f6f0e6"
REFERENCE_PANEL = "#fbf8f3"
REFERENCE_BORDER = "#c7b69d"
COVID_COLOR = "#f08b78"
CLIMATE_COLOR = "#78aeea"
SIGNIFICANT_COLOR = "#2fa66b"
TEXT_COLOR = "#2f2a26"


def holm_bonferroni(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid_mask = np.isfinite(p_values)
    valid = p_values[valid_mask]
    if valid.size == 0:
        return adjusted.tolist()

    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    adjusted_ranked = np.empty_like(ranked)
    running_max = 0.0

    for index, p_value in enumerate(ranked):
        candidate = min((m - index) * p_value, 1.0)
        running_max = max(running_max, candidate)
        adjusted_ranked[index] = running_max

    adjusted_valid = np.empty_like(valid)
    adjusted_valid[order] = adjusted_ranked
    adjusted[valid_mask] = adjusted_valid
    return adjusted.tolist()


def bootstrap_mean_ci(values, n_boot=2000, seed=42, ci=95, batch_size=250):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    mean_samples = []
    for start in range(0, n_boot, batch_size):
        draw_count = min(batch_size, n_boot - start)
        draw_indices = rng.integers(0, len(values), size=(draw_count, len(values)))
        mean_samples.append(values[draw_indices].mean(axis=1))

    boot_means = np.concatenate(mean_samples)
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_means, [alpha, 100 - alpha])
    return (float(lower), float(upper))


def format_mean(value):
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def format_compact_p_value(p_value):
    if p_value is None or pd.isna(p_value):
        return "n/a"
    if p_value == 0:
        return "0.0"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def key_finding_specifications():
    return [
        {
            "title": "1. Sadness / post",
            "feature": "sadness",
            "x_label": "Mean sadness score",
        },
        {
            "title": "2. Overall sentiment / post",
            "feature": "vader_compound",
            "x_label": "Mean compound score",
        },
        {
            "title": "3. Anger / post",
            "feature": "anger",
            "x_label": "Mean anger score",
        },
        {
            "title": "4. Authority cues / post",
            "feature": "auth_score",
            "x_label": "Mean authority score",
        },
    ]


def build_key_findings_rows(covid_df, climate_df, n_boot=2000, seed=42):
    rows = []
    specs = key_finding_specifications()

    for index, spec in enumerate(specs):
        covid_values = pd.to_numeric(covid_df[spec["feature"]], errors="coerce").dropna().to_numpy()
        climate_values = pd.to_numeric(climate_df[spec["feature"]], errors="coerce").dropna().to_numpy()
        try:
            statistic, p_value = scipy_stats.mannwhitneyu(
                covid_values,
                climate_values,
                alternative="two-sided",
                method="asymptotic",
            )
        except TypeError:
            statistic, p_value = scipy_stats.mannwhitneyu(
                covid_values,
                climate_values,
                alternative="two-sided",
            )
        rows.append(
            {
                **spec,
                "covid_mean": float(covid_values.mean()),
                "climate_mean": float(climate_values.mean()),
                "covid_ci": bootstrap_mean_ci(covid_values, n_boot=n_boot, seed=seed + index),
                "climate_ci": bootstrap_mean_ci(
                    climate_values,
                    n_boot=n_boot,
                    seed=seed + len(specs) + index,
                ),
                "p_value": float(p_value),
                "statistic": float(statistic),
            }
        )

    adjusted_p_values = holm_bonferroni([row["p_value"] for row in rows])
    for row, adjusted_p_value in zip(rows, adjusted_p_values):
        row["adjusted_p_value"] = float(adjusted_p_value)
        row["significant"] = bool(adjusted_p_value < 0.05)
    return rows


def style_reference_axis(ax):
    ax.set_facecolor(REFERENCE_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(REFERENCE_BORDER)
        spine.set_linewidth(0.9)
    ax.grid(axis="x", color=REFERENCE_BORDER, alpha=0.3, linewidth=0.8)
    ax.tick_params(colors="#5f564d")


def draw_key_finding_panel(ax, row, show_legend=False):
    covid_mean = row["covid_mean"]
    climate_mean = row["climate_mean"]
    covid_ci_low, covid_ci_high = row["covid_ci"]
    climate_ci_low, climate_ci_high = row["climate_ci"]

    style_reference_axis(ax)
    ax.errorbar(
        covid_mean,
        1,
        xerr=[[covid_mean - covid_ci_low], [covid_ci_high - covid_mean]],
        fmt="o",
        color=COVID_COLOR,
        ecolor=COVID_COLOR,
        markersize=7,
        markeredgewidth=1.8,
        elinewidth=2.2,
        capsize=6,
        capthick=2.2,
        label="COVID",
        zorder=3,
    )
    ax.errorbar(
        climate_mean,
        0,
        xerr=[[climate_mean - climate_ci_low], [climate_ci_high - climate_mean]],
        fmt="o",
        color=CLIMATE_COLOR,
        ecolor=CLIMATE_COLOR,
        markersize=7,
        markeredgewidth=1.8,
        elinewidth=2.2,
        capsize=6,
        capthick=2.2,
        label="Climate",
        zorder=3,
    )

    x_min = min(covid_ci_low, climate_ci_low)
    x_max = max(covid_ci_high, climate_ci_high)
    x_span = max(x_max - x_min, 0.02)
    ax.set_xlim(x_min - x_span * 0.45, x_max + x_span * 0.55)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["COVID", "Climate"], fontsize=11, color="#5f564d")
    ax.set_xlabel(row["x_label"], fontsize=11, color="#5f564d")
    ax.set_title(row["title"], loc="left", fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=12)

    status_label = "Significant" if row["significant"] else "Not significant"
    status_color = SIGNIFICANT_COLOR if row["significant"] else "#8b8176"
    ax.text(
        0.985,
        1.04,
        status_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=status_color,
    )

    ax.text(
        covid_mean,
        1.09,
        format_mean(covid_mean),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#6d6258",
    )
    ax.text(
        climate_mean,
        0.09,
        format_mean(climate_mean),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#6d6258",
    )
    if show_legend:
        legend = ax.legend(loc="upper right", frameon=False, fontsize=11)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)


def draw_summary_card(ax, index, row):
    ax.axis("off")
    card = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#e1d4c2",
        facecolor=REFERENCE_PANEL,
    )
    ax.add_patch(card)
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.12),
            0.025,
            0.72,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            linewidth=0,
            facecolor=SIGNIFICANT_COLOR if row["significant"] else "#8b8176",
        )
    )
    ax.text(0.1, 0.78, row["title"], fontsize=12.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(
        0.1,
        0.47,
        f"Climate={format_mean(row['climate_mean'])} | COVID={format_mean(row['covid_mean'])}",
        fontsize=12,
        color="#6d6258",
    )
    ax.text(
        0.1,
        0.2,
        f"p={format_compact_p_value(row['p_value'])} (Holm adj={format_compact_p_value(row['adjusted_p_value'])})",
        fontsize=12,
        color=SIGNIFICANT_COLOR if row["significant"] else "#8b8176",
    )


def plot_key_findings_results(
    covid_path,
    climate_path,
    output_path,
    show=False,
    n_boot=2000,
    seed=42,
):
    covid = pd.read_csv(covid_path)
    climate = pd.read_csv(climate_path)
    rows = build_key_findings_rows(covid, climate, n_boot=n_boot, seed=seed)

    fig = plt.figure(figsize=(16, 11), facecolor=REFERENCE_BACKGROUND)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.95], hspace=0.42, wspace=0.55)
    fig.suptitle(
        "Climate vs COVID Misinformation: Key Findings Results",
        fontsize=23,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "Four Mann-Whitney U tests with Holm-Bonferroni correction; points show descriptive means with 95% bootstrap CIs.",
        ha="center",
        fontsize=15,
        color="#6d6258",
    )

    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
    ]
    for index, (ax, row) in enumerate(zip(axes, rows)):
        draw_key_finding_panel(ax, row, show_legend=index == 1)

    header_ax = fig.add_subplot(grid[2, :])
    header_ax.axis("off")
    header_ax.text(
        0.0,
        1.02,
        "Statistical Summary",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="bottom",
    )
    card_grid = grid[2, :].subgridspec(1, 4, wspace=0.04)
    for index, row in enumerate(rows):
        card_ax = fig.add_subplot(card_grid[0, index])
        draw_summary_card(card_ax, index, row)

    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


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


def plot_rhetoric(rhetoric_path, output_path, show=False):
    rhetoric = pd.read_csv(rhetoric_path)
    cov_color = "#E05C5C"
    cli_color = "#4A90D9"

    score_features = ["urgency", "authority", "doubt"]
    rate_features = ["urgency_presence", "authority_presence", "doubt_presence"]
    score_labels = ["Urgency", "Authority", "Doubt"]
    rate_labels = ["Urgency Cue", "Authority Cue", "Doubt Cue"]

    score_rows = rhetoric[rhetoric["feature"].isin(score_features)].copy()
    score_rows["feature"] = pd.Categorical(score_rows["feature"], score_features, ordered=True)
    score_rows = score_rows.sort_values("feature")

    rate_rows = rhetoric[rhetoric["feature"].isin(rate_features)].copy()
    rate_rows["feature"] = pd.Categorical(rate_rows["feature"], rate_features, ordered=True)
    rate_rows = rate_rows.sort_values("feature")
    p_column = "adjusted_p_value" if "adjusted_p_value" in rhetoric.columns else "p_value"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Rhetoric Differences Between COVID and Climate Misinformation",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    x = np.arange(len(score_rows))
    width = 0.35
    covid_scores = score_rows["covid_mean"].astype(float).to_list()
    climate_scores = score_rows["climate_mean"].astype(float).to_list()
    ax.bar(x - width / 2, covid_scores, width, label="COVID", color=cov_color, alpha=0.85)
    ax.bar(x + width / 2, climate_scores, width, label="Climate", color=cli_color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(score_labels)
    ax.set_ylabel("Mean Score")
    ax.set_title("Mean Rhetoric Scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xpos, left_value, right_value, row in zip(x, covid_scores, climate_scores, score_rows.to_dict("records")):
        y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-6)
        ax.text(
            xpos,
            max(left_value, right_value) + y_span * 0.03,
            significance_label(row[p_column]),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[1]
    x2 = np.arange(len(rate_rows))
    covid_rates = rate_rows["covid_rate"].astype(float).to_list()
    climate_rates = rate_rows["climate_rate"].astype(float).to_list()
    ax.bar(x2 - width / 2, covid_rates, width, label="COVID", color=cov_color, alpha=0.85)
    ax.bar(x2 + width / 2, climate_rates, width, label="Climate", color=cli_color, alpha=0.85)
    ax.set_xticks(x2)
    ax.set_xticklabels(rate_labels)
    ax.set_ylabel("Share of Posts")
    ax.set_title("Rhetoric Cue Presence Rates")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xpos, left_value, right_value, row in zip(x2, covid_rates, climate_rates, rate_rows.to_dict("records")):
        y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-6)
        ax.text(
            xpos,
            max(left_value, right_value) + y_span * 0.03,
            significance_label(row[p_column]),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    direction_bits = []
    for row in score_rows.to_dict("records"):
        if row["climate_mean"] > row["covid_mean"]:
            direction_bits.append(f"higher climate {row['feature']}")
        elif row["covid_mean"] > row["climate_mean"]:
            direction_bits.append(f"higher COVID {row['feature']}")

    fig.text(
        0.5,
        0.02,
        "; ".join(direction_bits[:3]).capitalize() + ".",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_key_findings_results(
        covid_path="data/analysis/covid_emotions.csv",
        climate_path="data/analysis/climate_emotions.csv",
        output_path="data/analysis/key_findings_results.png",
        show=False,
    )
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
    plot_rhetoric(
        rhetoric_path="data/analysis/rhetoric_stats.csv",
        output_path="data/analysis/rhetoric_analysis.png",
        show=False,
    )
