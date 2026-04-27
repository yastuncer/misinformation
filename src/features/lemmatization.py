# This module provides functions for lemmatizing text using spaCy.
# strip stopping words and punctuation, and lemmatizing the remaining tokens to their base forms.

import spacy
from tqdm.auto import tqdm
 
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # only need tagger for lemmas
 
 
def lemmatize_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    doc = nlp(text)
    return " ".join(
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    )


def _lemmatize_doc(doc):
    return " ".join(
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    )
 
 
def lemmatize_series(texts, batch_size=64, show_progress=True):
    """Apply lemmatization to a list or Series of texts."""
    normalized = [text if isinstance(text, str) else "" for text in texts]
    outputs = [""] * len(normalized)
    nonempty_positions = [index for index, text in enumerate(normalized) if text.strip()]
    nonempty_texts = [normalized[index] for index in nonempty_positions]

    doc_stream = nlp.pipe(nonempty_texts, batch_size=batch_size)
    if show_progress:
        doc_stream = tqdm(doc_stream, total=len(nonempty_texts), desc="  Lemmatizing", unit="doc")

    for position, doc in zip(nonempty_positions, doc_stream):
        outputs[position] = _lemmatize_doc(doc)

    return outputs
 
