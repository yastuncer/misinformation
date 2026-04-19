"""
Emotion analysis using j-hartmann/emotion-english-distilroberta-base.
Labels: anger, disgust, fear, joy, neutral, sadness, surprise.
Runs on CPU but is slow — batching is essential for large datasets.
"""
from transformers import pipeline
import pandas as pd

emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=None, 
    device=-1)

def get_emotions(text, max_length=512):
    """
    Get emotion scores for a given text using the j-hartmann/emotion-english-distilroberta-base model.
    Returns a dictionary with emotion labels as keys and their corresponding scores as values.
    """
    if not isinstance(text, str) or not text.strip():
        return {emotion: 0.0 for emotion in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']}
    
    try:
        results = emotion_classifier(text[:2000], truncation=True, max_length=max_length)
        # results is a list of [{'label': 'fear', 'score': 0.8}, ...]
        return {item['label']: item['score'] for item in results[0]}
    except Exception:
        return {emotion: 0.0 for emotion in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']}
    
def get_emotions_batch(texts, batch_size=64):
    truncated = [str(t)[:2000] if isinstance(t, str) else "" for t in texts]
    print(f"  Running emotion model on {len(truncated):,} texts in batches of 64...")
    results = emotion_classifier(truncated, truncation=True, max_length=512, batch_size=batch_size)

    rows = []
    for result in results:
        # handle both [[{...}]] and [{...}] output formats across transformers versions
        if isinstance(result, dict):
            result = [result]
        rows.append({item['label']: item['score'] for item in result})

    return pd.DataFrame(rows)



