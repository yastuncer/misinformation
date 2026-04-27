import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.features.vader import VADER_COLUMNS, get_vader_scores


ANALYSIS_DIR = Path("data/analysis")
COVID_PATH = ANALYSIS_DIR / "covid_emotions.csv"
CLIMATE_PATH = ANALYSIS_DIR / "climate_emotions.csv"
TEST_RESULTS_PATH = ANALYSIS_DIR / "statistical_tests_mwu.csv"
SUMMARY_PATH = ANALYSIS_DIR / "stats_summary_mwu.json"

EMOTION_COLUMNS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]
TRUST_CUE_COLUMNS = ["auth_score", "urg_score"]
CORPUS_COLUMNS = ["char_count", "word_count"]


def word_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def safe_mean(values):
    if len(values) == 0:
        return math.nan
    return float(np.mean(values))


def safe_median(values):
    if len(values) == 0:
        return math.nan
    return float(np.median(values))


def rank_biserial_correlation(u_statistic, sample_a_size, sample_b_size):
    if sample_a_size == 0 or sample_b_size == 0:
        return math.nan
    return float((2 * u_statistic) / (sample_a_size * sample_b_size) - 1)


def cramers_v(chi2_statistic, contingency_table):
    n = contingency_table.to_numpy().sum()
    if n == 0:
        return math.nan

    num_rows, num_cols = contingency_table.shape
    min_dim = min(num_rows - 1, num_cols - 1)
    if min_dim <= 0:
        return math.nan
    return math.sqrt(chi2_statistic / (n * min_dim))


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid_mask = np.isfinite(p_values)
    valid = p_values[valid_mask]
    if valid.size == 0:
        return adjusted.tolist()

    order = np.argsort(valid)
    ranked = valid[order]
    adjusted_ranked = np.empty_like(ranked)
    running_min = 1.0

    for index in range(len(ranked) - 1, -1, -1):
        rank = index + 1
        candidate = ranked[index] * len(ranked) / rank
        running_min = min(running_min, candidate)
        adjusted_ranked[index] = min(running_min, 1.0)

    adjusted_valid = np.empty_like(valid)
    adjusted_valid[order] = adjusted_ranked
    adjusted[valid_mask] = adjusted_valid
    return adjusted.tolist()


def to_python(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def safe_percentages(counts):
    total = counts.sum()
    if total == 0:
        return {emotion: 0.0 for emotion in counts.index}
    return {emotion: float(count / total) for emotion, count in counts.items()}


def prepare_domain_frame(path):
    df = pd.read_csv(path).copy()
    if "text" not in df.columns:
        raise ValueError(f"{path} must include a 'text' column.")

    missing_emotions = [column for column in EMOTION_COLUMNS if column not in df.columns]
    if missing_emotions:
        raise ValueError(
            f"{path} is missing expected emotion columns: {', '.join(missing_emotions)}"
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].map(word_count)
    df["dominant_emotion"] = df[EMOTION_COLUMNS].idxmax(axis=1)
    if "authority" in df.columns and "auth_score" not in df.columns:
        df["auth_score"] = pd.to_numeric(df["authority"], errors="coerce")
    if "urgency" in df.columns and "urg_score" not in df.columns:
        df["urg_score"] = pd.to_numeric(df["urgency"], errors="coerce")

    missing_vader_columns = [column for column in VADER_COLUMNS if column not in df.columns]
    if missing_vader_columns:
        vader_scores = get_vader_scores(df["text"].tolist())
        df = pd.concat([df.reset_index(drop=True), vader_scores], axis=1)
    return df


def numeric_test_result(feature_name, feature_group, covid_values, climate_values):
    covid_values = pd.to_numeric(pd.Series(covid_values), errors="coerce").dropna().to_numpy()
    climate_values = pd.to_numeric(pd.Series(climate_values), errors="coerce").dropna().to_numpy()

    covid_n = int(len(covid_values))
    climate_n = int(len(climate_values))
    covid_mean = safe_mean(covid_values)
    climate_mean = safe_mean(climate_values)
    covid_median = safe_median(covid_values)
    climate_median = safe_median(climate_values)

    if covid_n == 0 or climate_n == 0:
        statistic = math.nan
        p_value = math.nan
        effect_size = math.nan
        notes = "insufficient_data"
    else:
        try:
            statistic, p_value = stats.mannwhitneyu(
                covid_values,
                climate_values,
                alternative="two-sided",
                method="asymptotic",
            )
        except TypeError:
            statistic, p_value = stats.mannwhitneyu(
                covid_values,
                climate_values,
                alternative="two-sided",
            )
        effect_size = rank_biserial_correlation(statistic, covid_n, climate_n)
        notes = "means_are_descriptive_only"

    return {
        "feature_group": feature_group,
        "feature": feature_name,
        "test": "mann_whitney_u",
        "statistic_name": "u",
        "covid_n": covid_n,
        "climate_n": climate_n,
        "covid_mean": covid_mean,
        "climate_mean": climate_mean,
        "covid_median": covid_median,
        "climate_median": climate_median,
        "mean_difference": covid_mean - climate_mean,
        "median_difference": covid_median - climate_median,
        "statistic": float(statistic) if np.isfinite(statistic) else math.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else math.nan,
        "effect_size": float(effect_size),
        "effect_size_name": "rank_biserial_correlation",
        "notes": notes,
    }


def dominant_emotion_result(covid_df, climate_df):
    covid_counts = (
        covid_df["dominant_emotion"].value_counts().reindex(EMOTION_COLUMNS, fill_value=0)
    )
    climate_counts = (
        climate_df["dominant_emotion"].value_counts().reindex(EMOTION_COLUMNS, fill_value=0)
    )
    covid_total = int(covid_counts.sum())
    climate_total = int(climate_counts.sum())
    contingency = pd.DataFrame({"covid": covid_counts, "climate": climate_counts})
    contingency = contingency[contingency.sum(axis=1) > 0]

    if contingency.empty or covid_total == 0 or climate_total == 0:
        statistic = math.nan
        p_value = math.nan
        degrees_of_freedom = 0
        expected = np.empty((0, 0))
        effect_size = math.nan
        notes = "insufficient_data"
    else:
        statistic, p_value, degrees_of_freedom, expected = stats.chi2_contingency(contingency)
        effect_size = cramers_v(statistic, contingency)
        notes = f"dof={degrees_of_freedom}"

    result_row = {
        "feature_group": "emotion_distribution",
        "feature": "dominant_emotion",
        "test": "chi_square",
        "statistic_name": "chi2",
        "covid_n": covid_total,
        "climate_n": climate_total,
        "covid_mean": math.nan,
        "climate_mean": math.nan,
        "covid_median": math.nan,
        "climate_median": math.nan,
        "mean_difference": math.nan,
        "median_difference": math.nan,
        "statistic": float(statistic) if np.isfinite(statistic) else math.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else math.nan,
        "effect_size": float(effect_size) if np.isfinite(effect_size) else math.nan,
        "effect_size_name": "cramers_v",
        "notes": notes,
    }

    distribution_payload = {
        "covid": {
            "counts": {emotion: int(count) for emotion, count in covid_counts.items()},
            "percentages": safe_percentages(covid_counts),
        },
        "climate": {
            "counts": {emotion: int(count) for emotion, count in climate_counts.items()},
            "percentages": safe_percentages(climate_counts),
        },
        "expected_counts": {
            column: {
                emotion: float(expected[row_index, column_index])
                for row_index, emotion in enumerate(contingency.index)
            }
            for column_index, column in enumerate(contingency.columns)
        },
    }

    return result_row, distribution_payload


def summarize_domain(df):
    return {
        "rows": int(len(df)),
        "mean_char_count": float(df["char_count"].mean()),
        "median_char_count": float(df["char_count"].median()),
        "mean_word_count": float(df["word_count"].mean()),
        "median_word_count": float(df["word_count"].median()),
        "source_counts": {
            str(key): int(value)
            for key, value in df.get("source", pd.Series(dtype=object)).value_counts().items()
        },
        "dataset_counts": {
            str(key): int(value)
            for key, value in df.get("dataset", pd.Series(dtype=object)).value_counts().items()
        },
    }


def build_numeric_results(covid_df, climate_df):
    results = []
    feature_groups = [
        ("vader", VADER_COLUMNS),
        ("emotion", EMOTION_COLUMNS),
        ("corpus", CORPUS_COLUMNS),
        ("trust_cue", TRUST_CUE_COLUMNS),
    ]

    for feature_group, columns in feature_groups:
        for column in columns:
            if column not in covid_df.columns or column not in climate_df.columns:
                continue
            results.append(
                numeric_test_result(
                    feature_name=column,
                    feature_group=feature_group,
                    covid_values=covid_df[column],
                    climate_values=climate_df[column],
                )
            )
    return results


def add_adjusted_p_values(results):
    adjusted = benjamini_hochberg([row["p_value"] for row in results])
    for row, adjusted_p in zip(results, adjusted):
        row["adjusted_p_value"] = float(adjusted_p) if np.isfinite(adjusted_p) else math.nan
        row["significant_0_05"] = bool(row["p_value"] < 0.05)
        row["fdr_significant_0_05"] = bool(
            np.isfinite(row["adjusted_p_value"]) and row["adjusted_p_value"] < 0.05
        )
    return results


def run_stats(covid_path=COVID_PATH, climate_path=CLIMATE_PATH):
    covid_df = prepare_domain_frame(covid_path)
    climate_df = prepare_domain_frame(climate_path)

    results = build_numeric_results(covid_df, climate_df)
    dominant_row, dominant_payload = dominant_emotion_result(covid_df, climate_df)
    results.append(dominant_row)
    results = add_adjusted_p_values(results)

    results_df = pd.DataFrame(results).sort_values(
        by=["adjusted_p_value", "p_value", "feature_group", "feature"],
        na_position="last",
    )
    results_df.to_csv(TEST_RESULTS_PATH, index=False)

    summary_payload = {
        "numeric_test": "mann_whitney_u",
        "numeric_effect_size": "rank_biserial_correlation",
        "domains": {
            "covid": summarize_domain(covid_df),
            "climate": summarize_domain(climate_df),
        },
        "dominant_emotions": dominant_payload,
        "available_trust_cue_columns": [
            column
            for column in TRUST_CUE_COLUMNS
            if column in covid_df.columns and column in climate_df.columns
        ],
        "missing_trust_cue_columns": [
            column
            for column in TRUST_CUE_COLUMNS
            if column not in covid_df.columns or column not in climate_df.columns
        ],
        "top_findings": [
            {key: to_python(value) for key, value in row.items()}
            for row in results_df.head(10).to_dict(orient="records")
        ],
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved -> {TEST_RESULTS_PATH}")
    print(f"Saved -> {SUMMARY_PATH}")
    print("\nTop findings by adjusted p-value:")
    print(
        results_df[
            [
                "feature_group",
                "feature",
                "test",
                "covid_median",
                "climate_median",
                "p_value",
                "adjusted_p_value",
                "effect_size",
            ]
        ].head(10).to_string(index=False)
    )
    return results_df, summary_payload


if __name__ == "__main__":
    run_stats()
