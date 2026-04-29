import requests
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
import re
import json

def classify_rhetoric_llm(text):
    prompt = f"""You are an expert linguistic analyst specializing in misinformation. 
Analyze the text below for three rhetorical strategies. Score each strategy as a float between 0.0 and 1.0 (where 0.0 is completely absent and 1.0 is highly prominent).

Definitions:
- "urgency": Language creating panic, artificial deadlines, or extreme pressure to act immediately.
- "authority": Language invoking pseudo-expertise, or aggressively attacking scientific/institutional consensus to establish rogue credibility.
- "doubt": Language explicitly designed to breed paranoia or undermine trust in mainstream sources and official data.

Text to analyze:
"{text}"

Return ONLY a raw JSON object. Do not include markdown formatting, code blocks, or conversational text.
Example expected output:
{{"urgency": 0.8, "authority": 0.0, "doubt": 0.5}}"""
    
    try:
        response = requests.post('http://localhost:11434/api/generate',
            json={'model': 'llama3.2', 'prompt': prompt, 'stream': False})
        
        llm_output = response.json().get('response', '')
        
        # Use regex to strip out conversational filler and grab the dictionary
        match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
        if match:
            clean_json = match.group(0)
        else:
            clean_json = llm_output
            
        return json.loads(clean_json)
        
    except Exception as e:
        # If the LLM fails or returns garbage, default to 0 to save the pipeline
        return {"urgency": 0.0, "authority": 0.0, "doubt": 0.0}


def extract_rhetoric_features(df):
    texts = df['text'].fillna('').astype(str)
    features = pd.DataFrame()
    features['domain'] = df['domain'].values

    print(f"  Running LLM rhetoric classification on {len(texts)} texts...")
    
    # Process the LLM ONCE per row, saving the dictionary
    llm_results = texts.apply(classify_rhetoric_llm)

    # Extract the values from the single dictionary to avoid making 3 API calls per row
    features['urgency'] = llm_results.apply(lambda x: float(x.get('urgency', 0.0)))
    features['authority'] = llm_results.apply(lambda x: float(x.get('authority', 0.0)))
    features['doubt'] = llm_results.apply(lambda x: float(x.get('doubt', 0.0)))

    for name in ['urgency', 'authority', 'doubt']:
        features[f'{name}_presence'] = (features[name] > 0).astype(int)

    features['authority_to_doubt_ratio'] = (features['authority'] / (features['doubt'] + 1e-6))
    return features

def run_rhetoric_tests(features_df):

    covid = features_df[features_df['domain'] == 'covid']
    climate = features_df[features_df['domain'] == 'climate']

    print(f"\n{'='*65}")
    print(f"Rhetoric feature comparison: COVID ({len(covid)}) vs Climate ({len(climate)})")
    print(f"{'='*65}")

    print(f"\n{'Feature':<28} {'COVID mean':>12} {'Climate mean':>14} {'p-value':>12} Sig")
    print('-' * 72)

    continuous_features = ['urgency', 'authority', 'doubt', 'authority_to_doubt_ratio']
    results = []

    for feature in continuous_features:
        stat, p = mannwhitneyu(covid[feature], climate[feature], alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{feature:<28} {covid[feature].mean():>12.4f} {climate[feature].mean():>14.4f} {p:>12.4e} {sig}")
        results.append({'feature': feature, 'covid_mean': covid[feature].mean(), 'climate_mean': climate[feature].mean(), 'p_value': p, 'test': 'mann-whitney'})
    
    for feature in ['urgency_presence', 'authority_presence', 'doubt_presence']:
        contingency = pd.crosstab(features_df['domain'], features_df[feature])
        chi2, p, _, _ = chi2_contingency(contingency)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        covid_rate = contingency.loc['covid'].get(1, 0) / contingency.loc['covid'].sum() if 'covid' in contingency.index else 0
        climate_rate = contingency.loc['climate'].get(1, 0) / contingency.loc['climate'].sum() if 'climate' in contingency.index else 0
        print(f"{feature:<28} {covid_rate:>12.4f} {climate_rate:>14.4f} {p:>12.4e} {sig}")
        results.append({'feature': feature, 'covid_rate': covid_rate, 'climate_rate': climate_rate, 'p_value': p, 'test': 'chi-squared'})

    return pd.DataFrame(results)
