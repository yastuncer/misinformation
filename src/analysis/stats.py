import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ANALYSIS_DIR = Path("data/analysis")
COVID_PATH = ANALYSIS_DIR / "covid_emotions.csv"
CLIMATE_PATH = ANALYSIS_DIR / "climate_emotions.csv"
TEST_RESULTS_PATH = ANALYSIS_DIR / "statistical_tests.csv"
SUMMARY_PATH = ANALYSIS_DIR / "stats_summary.json"

EMOTION_COLUMNS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]
VADER_COLUMNS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
TRUST_CUE_COLUMNS = ["auth_score", "urg_score"]
CORPUS_COLUMNS = ["char_count", "word_count"]


def compute_vader_scores(texts):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment is required to run statistical analysis. "
            "Install the project requirements first."
        ) from exc

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
    return pd.DataFrame(rows)


def word_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def cohens_d(sample_a, sample_b):
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.nan

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled_denom = len(a) + len(b) - 2
    if pooled_denom <= 0:
        return math.nan

    pooled_var = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / pooled_denom
    if pooled_var <= 0:
        return 0.0
    return (a.mean() - b.mean()) / math.sqrt(pooled_var)


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

    vader_scores = compute_vader_scores(df["text"])
    df = pd.concat([df.reset_index(drop=True), vader_scores], axis=1)
    return df


def numeric_test_result(feature_name, feature_group, covid_values, climate_values):
    covid_values = pd.to_numeric(pd.Series(covid_values), errors="coerce").dropna().to_numpy()
    climate_values = pd.to_numeric(pd.Series(climate_values), errors="coerce").dropna().to_numpy()

    statistic, p_value = stats.ttest_ind(
        covid_values,
        climate_values,
        equal_var=False,
        nan_policy="omit",
    )
    effect_size = cohens_d(covid_values, climate_values)

    return {
        "feature_group": feature_group,
        "feature": feature_name,
        "test": "welch_t_test",
        "covid_n": int(len(covid_values)),
        "climate_n": int(len(climate_values)),
        "covid_mean": float(np.mean(covid_values)),
        "climate_mean": float(np.mean(climate_values)),
        "covid_median": float(np.median(covid_values)),
        "climate_median": float(np.median(climate_values)),
        "mean_difference": float(np.mean(covid_values) - np.mean(climate_values)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "effect_size_name": "cohens_d",
        "notes": "",
    }


def dominant_emotion_result(covid_df, climate_df):
    covid_counts = (
        covid_df["dominant_emotion"].value_counts().reindex(EMOTION_COLUMNS, fill_value=0)
    )
    climate_counts = (
        climate_df["dominant_emotion"].value_counts().reindex(EMOTION_COLUMNS, fill_value=0)
    )
    contingency = pd.DataFrame({"covid": covid_counts, "climate": climate_counts})
    contingency = contingency[contingency.sum(axis=1) > 0]
    statistic, p_value, degrees_of_freedom, expected = stats.chi2_contingency(contingency)
    effect_size = cramers_v(statistic, contingency)

    result_row = {
        "feature_group": "emotion_distribution",
        "feature": "dominant_emotion",
        "test": "chi_square",
        "covid_n": int(covid_counts.sum()),
        "climate_n": int(climate_counts.sum()),
        "covid_mean": math.nan,
        "climate_mean": math.nan,
        "covid_median": math.nan,
        "climate_median": math.nan,
        "mean_difference": math.nan,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "effect_size_name": "cramers_v",
        "notes": f"dof={degrees_of_freedom}",
    }

    distribution_payload = {
        "covid": {
            "counts": {emotion: int(count) for emotion, count in covid_counts.items()},
            "percentages": {
                emotion: float(count / covid_counts.sum())
                for emotion, count in covid_counts.items()
            },
        },
        "climate": {
            "counts": {emotion: int(count) for emotion, count in climate_counts.items()},
            "percentages": {
                emotion: float(count / climate_counts.sum())
                for emotion, count in climate_counts.items()
            },
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
                "covid_mean",
                "climate_mean",
                "p_value",
                "adjusted_p_value",
                "effect_size",
            ]
        ].head(10).to_string(index=False)
    )
    return results_df, summary_payload


if __name__ == "__main__":
    run_stats()
