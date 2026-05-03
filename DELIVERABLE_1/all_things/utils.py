import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & TOKENIZERS
# ─────────────────────────────────────────────────────────────────────────────
ENGLISH_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'no', 'if',
    'then', 'so', 'up', 'out', 'about', 'into', 'than'
})

WH_WORDS = frozenset({'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose'})

def _safe_str(text):
    if text is None: return ""
    if isinstance(text, float) and np.isnan(text): return ""
    return str(text)

def whitespace_tokenizer(text):
    return _safe_str(text).lower().split()

def regex_tokenizer(text):
    return re.findall(r'\w+', _safe_str(text).lower())

def clean_tokenizer(text):
    return [t for t in regex_tokenizer(text) if t not in ENGLISH_STOPWORDS]

def char_ngram_tokenizer(text, n=3):
    t = re.sub(r'\s+', ' ', _safe_str(text).lower().strip())
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]

# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE FUNCTIONS (From Scratch)
# ─────────────────────────────────────────────────────────────────────────────
def jaccard_similarity(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def word_overlap_ratio(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2: return 1.0
    denom = min(len(s1), len(s2))
    return len(s1 & s2) / denom if denom > 0 else 0.0

def cosine_similarity_bow(tokens1, tokens2):
    cnt1, cnt2 = {}, {}
    for t in tokens1: cnt1[t] = cnt1.get(t, 0) + 1
    for t in tokens2: cnt2[t] = cnt2.get(t, 0) + 1
    dot = sum(cnt1[t] * cnt2.get(t, 0) for t in cnt1)
    norm1 = sum(v * v for v in cnt1.values()) ** 0.5
    norm2 = sum(v * v for v in cnt2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

def levenshtein_distance(s1, s2):
    s1, s2 = s1[:300], s2[:300]
    if len(s1) < len(s2): s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    row = list(range(n + 1))
    for i in range(1, m + 1):
        prev, row[0] = row[0], i
        for j in range(1, n + 1):
            tmp = row[j]
            row[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, row[j], row[j - 1])
            prev = tmp
    return row[n]

def normalized_levenshtein(s1, s2):
    a, b = _safe_str(s1).lower(), _safe_str(s2).lower()
    ml = max(len(a), len(b))
    return 0.0 if ml == 0 else levenshtein_distance(a, b) / ml

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE FEATURE
# ─────────────────────────────────────────────────────────────────────────────
def extract_simple_features(df):
    """Simple baseline feature: basic word overlap ratio."""
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values
    
    features = []
    for q1, q2 in zip(q1_raw, q2_raw):
        t1, t2 = regex_tokenizer(q1), regex_tokenizer(q2)
        features.append([word_overlap_ratio(t1, t2)])
    return np.array(features, dtype=np.float32)

def extract_improved_features(df):
    """Original improved features: basic overlap, jaccard, and length differences."""
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values
    features = []
    for q1, q2 in zip(q1_raw, q2_raw):
        t1, t2 = regex_tokenizer(q1), regex_tokenizer(q2)
        overlap = word_overlap_ratio(t1, t2)
        jaccard = jaccard_similarity(t1, t2)
        len_diff = abs(len(t1) - len(t2))
        features.append([overlap, jaccard, len_diff])
    return np.array(features, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# MIKI'S CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_miki_features(df, tfidf_word, tfidf_char):
    """
    Miki's features: Word and Character n-gram TF-IDF Cosine Similarities.
    Takes pre-fitted tfidf_word and tfidf_char vectorizers to prevent data leakage.
    Returns array of shape (n, 2)
    """
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values
    
    v1w, v2w = tfidf_word.transform(q1_raw), tfidf_word.transform(q2_raw)
    tfidf_w_sim = 1.0 - paired_cosine_distances(v1w, v2w)
    
    v1c, v2c = tfidf_char.transform(q1_raw), tfidf_char.transform(q2_raw)
    tfidf_c_sim = 1.0 - paired_cosine_distances(v1c, v2c)
    
    return np.column_stack((tfidf_w_sim, tfidf_c_sim))

# ─────────────────────────────────────────────────────────────────────────────
# JOSE'S CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def _tok_quad(t1, t2):
    l1, l2 = len(t1), len(t2)
    return [
        jaccard_similarity(t1, t2),
        word_overlap_ratio(t1, t2),
        cosine_similarity_bow(t1, t2),
        abs(l1 - l2) / (max(l1, l2) + 1),
    ]

def extract_jose_features(df):
    """
    Jose's features: Jaccard, Overlap, BoW-Cosine, Length differences across
    multiple tokenizers, plus the from-scratch Normalized Levenshtein distance.
    Returns array of shape (n, 13)
    """
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values
    
    tr1 = [regex_tokenizer(q) for q in q1_raw]
    tr2 = [regex_tokenizer(q) for q in q2_raw]
    tc1 = [clean_tokenizer(q) for q in q1_raw]
    tc2 = [clean_tokenizer(q) for q in q2_raw]
    tn1 = [char_ngram_tokenizer(q) for q in q1_raw]
    tn2 = [char_ngram_tokenizer(q) for q in q2_raw]
    
    rows = []
    for i, (q1, q2) in enumerate(zip(q1_raw, q2_raw)):
        f = (
            _tok_quad(tr1[i], tr2[i])   # 4
            + _tok_quad(tc1[i], tc2[i]) # 4
            + _tok_quad(tn1[i], tn2[i]) # 4
        )
        f.append(normalized_levenshtein(q1, q2)) # 1
        rows.append(f)
    return np.array(rows, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# LLUC'S CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_lluc_features(df):
    """
    Lluc's features: WH-word matching and First-word matching.
    Returns array of shape (n, 2)
    """
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values
    
    rows = []
    for q1, q2 in zip(q1_raw, q2_raw):
        t1, t2 = regex_tokenizer(q1), regex_tokenizer(q2)
        fw1 = t1[0] if t1 else ''
        fw2 = t2[0] if t2 else ''
        
        f = [
            int(fw1 == fw2 and fw1 in WH_WORDS),
            int(fw1 in WH_WORDS and fw2 in WH_WORDS)
        ]
        rows.append(f)
    return np.array(rows, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL MODEL FEATURE
# ─────────────────────────────────────────────────────────────────────────────
def extract_final_features(df, tfidf_word, tfidf_char):
    """
    Combine all features for the Final Model: Miki + Jose + Lluc + char length ratio.
    Returns an array with the complete 18-feature representation.
    """
    miki = extract_miki_features(df, tfidf_word, tfidf_char)
    jose = extract_jose_features(df)
    lluc = extract_lluc_features(df)
    
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values
    len_ratios = np.array([min(len(q1), len(q2)) / (max(len(q1), len(q2)) + 1) for q1, q2 in zip(q1_raw, q2_raw)]).reshape(-1, 1)
    
    return np.hstack((miki, jose, lluc, len_ratios))
