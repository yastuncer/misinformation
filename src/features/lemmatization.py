# This module provides functions for lemmatizing text using spaCy.
# strip stopping words and punctuation, and lemmatizing the remaining tokens to their base forms.

import spacy
 
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # only need tagger for lemmas
 
 
def lemmatize_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    doc = nlp(text)
    return " ".join(
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    )
 
 
def lemmatize_series(texts):
    """Apply lemmatization to a list or Series of texts."""
    return [lemmatize_text(t) for t in texts]
 