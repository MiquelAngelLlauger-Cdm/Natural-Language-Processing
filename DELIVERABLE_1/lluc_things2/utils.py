import re
import numpy as np

# --- Miki's contribution: Data cleaning and Simple Baseline features ---
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
        overlap = get_word_overlap(row['question1'], row['question2'])
        features.append([overlap])
    return np.array(features)

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

# --- Lluc's contribution: Advanced feature extraction ---
def extract_improved_features(df):
    """
    Extract improved features using multiple distance metrics.
    """
    features = []
    for _, row in df.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        jaccard = compute_jaccard_similarity(q1, q2)
        len_diff = length_difference(q1, q2)
        overlap = get_word_overlap(q1, q2)
        features.append([overlap, jaccard, len_diff])
    return np.array(features)
