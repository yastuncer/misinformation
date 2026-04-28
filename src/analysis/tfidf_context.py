import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.vader import vader_series


ANALYSIS_DIR = Path("data/analysis")
TFIDF_CONTEXT_PATH = ANALYSIS_DIR / "tfidf_term_context.json"
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


def _column_scores(matrix, row_indices, column_index):
    values = matrix[row_indices, column_index]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float).reshape(-1)


def _ensure_vader_columns(df):
    missing = [column for column in VADER_COLUMNS if column not in df.columns]
    if not missing:
        return df

    vader_scores = vader_series([text if isinstance(text, str) else "" for text in df["text"].tolist()]).rename(
        columns={
            "neg": "vader_neg",
            "neu": "vader_neu",
            "pos": "vader_pos",
            "compound": "vader_compound",
        }
    )
    return pd.concat(
        [df.drop(columns=VADER_COLUMNS, errors="ignore").reset_index(drop=True), vader_scores],
        axis=1,
    )


def _ensure_dominant_emotion(df):
    if "dominant_emotion" in df.columns:
        return df
    df = df.copy()
    df["dominant_emotion"] = df[EMOTION_COLUMNS].idxmax(axis=1)
    return df


def _truncate_text(text, limit=240):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_term_context_records(
    df,
    row_indices,
    term_entries,
    vectorizer,
    tfidf_matrix,
    examples_per_term=3,
):
    feature_lookup = {
        feature: index for index, feature in enumerate(vectorizer.get_feature_names_out())
    }
    records = []

    for term, mean_tfidf_score in term_entries:
        if term not in feature_lookup:
            continue

        column_index = feature_lookup[term]
        term_scores = _column_scores(tfidf_matrix, row_indices, column_index)
        matched_positions = np.flatnonzero(term_scores > 0)

        if matched_positions.size == 0:
            continue

        matched_df = df.iloc[matched_positions].copy()
        matched_df["term_tfidf_score"] = term_scores[matched_positions]
        matched_df = matched_df.sort_values("term_tfidf_score", ascending=False)

        examples = []
        for _, row in matched_df.head(examples_per_term).iterrows():
            examples.append(
                {
                    "text": _truncate_text(row["text"]),
                    "dataset": str(row.get("dataset", "")),
                    "source": str(row.get("source", "")),
                    "term_tfidf_score": float(row["term_tfidf_score"]),
                    "vader_compound": float(row["vader_compound"]),
                    "dominant_emotion": str(row["dominant_emotion"]),
                }
            )

        records.append(
            {
                "term": term,
                "mean_tfidf_score": float(mean_tfidf_score),
                "document_frequency": int(matched_positions.size),
                "share_of_domain_docs": float(matched_positions.size / len(df)),
                "average_vader": {
                    column: float(matched_df[column].mean()) for column in VADER_COLUMNS
                },
                "average_emotions": {
                    column: float(matched_df[column].mean()) for column in EMOTION_COLUMNS
                },
                "representative_examples": examples,
            }
        )

    return records


def run_tfidf_context_analysis(
    covid_df,
    climate_df,
    vectorizer,
    tfidf_matrix,
    top_terms,
    output_path=TFIDF_CONTEXT_PATH,
    examples_per_term=3,
):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    covid_df = _ensure_dominant_emotion(_ensure_vader_columns(covid_df.copy()))
    climate_df = _ensure_dominant_emotion(_ensure_vader_columns(climate_df.copy()))

    covid_indices = np.arange(len(covid_df))
    climate_indices = np.arange(len(covid_df), len(covid_df) + len(climate_df))

    combined_df = np.concatenate([covid_df.index.to_numpy(), climate_df.index.to_numpy()])
    if combined_df.size != tfidf_matrix.shape[0]:
        raise ValueError("TF-IDF matrix row count does not match the combined domain data.")

    payload = {
        "covid": {
            "document_count": int(len(covid_df)),
            "terms": build_term_context_records(
                covid_df,
                covid_indices,
                top_terms.get("covid", []),
                vectorizer,
                tfidf_matrix,
                examples_per_term=examples_per_term,
            ),
        },
        "climate": {
            "document_count": int(len(climate_df)),
            "terms": build_term_context_records(
                climate_df,
                climate_indices,
                top_terms.get("climate", []),
                vectorizer,
                tfidf_matrix,
                examples_per_term=examples_per_term,
            ),
        },
    }

    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved -> {output_path}")
    return payload
