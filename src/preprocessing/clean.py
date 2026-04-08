"""
Cleaning the dataset - stripping URLs, emojis, mentions, hashtags and rt prefix
"""

import re
import unicodedata

def clean_tweet(text):

    # guard to ensure no crashes occur
    if not isinstance(text, str):
        return ""

    text = re.sub(r'^RT\s+@\w+:\s+', '', text) # removing the prefix rt 
    text = re.sub(r'https?://\S+', 'URL', text) # remove URLs
    text = re.sub(r'@\w+', 'USER', text) # remove @mentions with token USER
    text = re.sub(r'#(\S+)', r'\1', text) # remove # from hashtags but keeping the word
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U00002702-\U000027B0]+', '', text) # removing all emojis
    text = text.lower() # lowercase letters

    return text

# putting clean tweets into a column 
def clean_series(texts):
    return [clean_tweet(t) for t in texts]