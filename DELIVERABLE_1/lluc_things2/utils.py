import re
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

# --- Miki's contribution: Data cleaning, Simple Baseline, and Advanced TF-IDF ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_word_overlap(q1, q2):
    q1_words = set(clean_text(q1).split())
    q2_words = set(clean_text(q2).split())
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0.0
    return len(q1_words.intersection(q2_words)) / (len(q1_words) + len(q2_words))

def extract_simple_features(df):
    """
    Extract simple word overlap feature as a baseline.
    """
    features = []
    for _, row in df.iterrows():
        overlap = get_word_overlap(str(row['question1']), str(row['question2']))
        features.append([overlap])
    return np.array(features)

def extract_miki_features(df):
    """
    Extract Miki's features: Character n-gram TF-IDF Cosine Similarity.
    """
    q1_list = df['question1'].fillna("").astype(str).values
    q2_list = df['question2'].fillna("").astype(str).values
    
    # Fit a local character-level TF-IDF to capture robust structural similarity
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    tfidf.fit(np.concatenate((q1_list, q2_list)))
    
    vec1 = tfidf.transform(q1_list)
    vec2 = tfidf.transform(q2_list)
    
    cos_sim = 1 - paired_cosine_distances(vec1, vec2)
    return cos_sim.reshape(-1, 1)

# --- Jose's contribution: String distance features ---
def compute_jaccard_similarity(q1, q2):
    q1_words = set(clean_text(q1).split())
    q2_words = set(clean_text(q2).split())
    if len(q1_words) == 0 and len(q2_words) == 0:
        return 1.0
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0.0
    intersection = len(q1_words.intersection(q2_words))
    union = len(q1_words.union(q2_words))
    return intersection / union

def length_difference(q1, q2):
    q1_words = clean_text(q1).split()
    q2_words = clean_text(q2).split()
    return abs(len(q1_words) - len(q2_words))

def extract_jose_features(df):
    """
    Extract Jose's features: Normalized Levenshtein Edit Distance.
    """
    features = []
    for _, row in df.iterrows():
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        edit_dist = nltk.edit_distance(q1, q2)
        max_len = max(len(q1), len(q2))
        norm_dist = edit_dist / max_len if max_len > 0 else 0
        features.append([norm_dist])
    return np.array(features)

def extract_improved_features(df):
    """
    Extract improved features using Jaccard and length differences.
    """
    features = []
    for _, row in df.iterrows():
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        jaccard = compute_jaccard_similarity(q1, q2)
        len_diff = length_difference(q1, q2)
        overlap = get_word_overlap(q1, q2)
        features.append([overlap, jaccard, len_diff])
    return np.array(features)

# --- Lluc's contribution: First word feature ---
def get_first_word(text):
    text_clean = clean_text(text)
    words = text_clean.split()
    if len(words) > 0:
        return words[0]
    return ""

def extract_lluc_features(df):
    """
    Extract Lluc's features: first word match, WH-word presence.
    """
    features = []
    wh_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose'}
    for _, row in df.iterrows():
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        
        first_word_q1 = get_first_word(q1)
        first_word_q2 = get_first_word(q2)
        
        same_first_word = 1 if first_word_q1 == first_word_q2 and first_word_q1 != "" else 0
        q1_is_wh = 1 if first_word_q1 in wh_words else 0
        q2_is_wh = 1 if first_word_q2 in wh_words else 0
        
        features.append([same_first_word, q1_is_wh, q2_is_wh])
    return np.array(features)

# --- Final Model Contribution ---
def extract_final_features(df):
    """
    Combine all features for the Final Model.
    [overlap, jaccard, len_diff, same_first_word, q1_is_wh, q2_is_wh, miki_tfidf, jose_levenshtein]
    """
    improved = extract_improved_features(df)
    lluc = extract_lluc_features(df)
    miki = extract_miki_features(df)
    jose = extract_jose_features(df)
    
    return np.hstack((improved, lluc, miki, jose))
