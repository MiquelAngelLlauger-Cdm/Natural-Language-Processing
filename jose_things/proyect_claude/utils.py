"""
NLP Deliverable 1 – Quora Duplicate Question Detection
utils.py: Feature extraction with four tokenizer approaches and from-scratch
          distance functions.
Author: Jose Calatayud
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: TOKENIZERS
# Four approaches that trade off speed, precision, and vocabulary coverage.
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

WH_WORDS = frozenset({'who', 'what', 'where', 'when', 'why', 'how',
                      'which', 'whom', 'whose'})


def _safe_str(text):
    """Convert NaN / None to empty string."""
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    return str(text)


def whitespace_tokenizer(text):
    """
    Tokenizer 1 – Split on whitespace only.
    Preserves punctuation attached to words (e.g. 'word,' stays 'word,').
    Fastest; useful as a rough baseline.
    """
    return _safe_str(text).lower().split()


def regex_tokenizer(text):
    """
    Tokenizer 2 – Extract \\w+ tokens via regex.
    Strips punctuation and splits contractions ('don't' → ['don', 't']).
    Standard for most NLP bag-of-words features.
    """
    return re.findall(r'\w+', _safe_str(text).lower())


def clean_tokenizer(text):
    """
    Tokenizer 3 – Regex tokens minus English stopwords.
    Keeps only content words; reduces noise from function words.
    Most informative for semantic similarity tasks.
    """
    return [t for t in regex_tokenizer(text) if t not in ENGLISH_STOPWORDS]


def char_ngram_tokenizer(text, n=3):
    """
    Tokenizer 4 – Character n-grams (default n=3).
    Robust to spelling variants, typos, and morphological inflections.
    Captures subword information not visible to word-level tokenizers.
    """
    t = re.sub(r'\s+', ' ', _safe_str(text).lower().strip())
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SIMILARITY / DISTANCE FUNCTIONS  (all implemented from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(tokens1, tokens2):
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|.

    Symmetric measure ranging in [0, 1].
    Returns 1.0 when both token sets are empty (identical empty texts).
    Returns 0.0 when one set is empty and the other is not.

    Works with any tokenizer output (word tokens or character n-grams).
    """
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def word_overlap_ratio(tokens1, tokens2):
    """
    Overlap coefficient: |A ∩ B| / min(|A|, |B|).

    Unlike Jaccard this is not penalised when one text is much shorter
    than the other, making it better for short-vs-long question pairs.
    """
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    denom = min(len(s1), len(s2))
    return len(s1 & s2) / denom if denom > 0 else 0.0


def cosine_similarity_bow(tokens1, tokens2):
    """
    Cosine similarity over bag-of-words count vectors.

    Builds count dicts from scratch (no sklearn / scipy).
    Handles frequency information: 'why why why' vs 'why' get different vectors.

        cos(A, B) = (A · B) / (||A|| · ||B||)
    """
    cnt1: dict = {}
    cnt2: dict = {}
    for t in tokens1:
        cnt1[t] = cnt1.get(t, 0) + 1
    for t in tokens2:
        cnt2[t] = cnt2.get(t, 0) + 1

    dot = sum(cnt1[t] * cnt2.get(t, 0) for t in cnt1)
    norm1 = sum(v * v for v in cnt1.values()) ** 0.5
    norm2 = sum(v * v for v in cnt2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def levenshtein_distance(s1, s2):
    """
    Levenshtein (edit) distance between two strings.

    Dynamic-programming implementation in O(min(m, n)) space.
    Truncated to first 300 characters to bound worst-case runtime.

    Cost model: each insertion, deletion, or substitution costs 1.
    """
    s1, s2 = s1[:300], s2[:300]
    if len(s1) < len(s2):
        s1, s2 = s2, s1          # keep s1 as the longer string
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
    """
    Levenshtein distance normalised to [0, 1] by dividing by max(|s1|, |s2|).
    0.0 = identical strings, 1.0 = completely different.
    """
    a = _safe_str(s1).lower()
    b = _safe_str(s2).lower()
    ml = max(len(a), len(b))
    return 0.0 if ml == 0 else levenshtein_distance(a, b) / ml


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _tok_quad(t1, t2):
    """
    Four similarity/difference statistics for a pre-tokenised pair.
    Returns: [jaccard, overlap_coeff, bow_cosine, normalised_len_diff]
    """
    l1, l2 = len(t1), len(t2)
    return [
        jaccard_similarity(t1, t2),
        word_overlap_ratio(t1, t2),
        cosine_similarity_bow(t1, t2),
        abs(l1 - l2) / (max(l1, l2) + 1),
    ]


def extract_simple_features(df, tfidf_vectorizer):
    """
    Simple model features: one TF-IDF word cosine similarity per pair.

    Fit tfidf_vectorizer on training questions beforehand; pass the same
    fitted object for validation and test.

    Returns a dense (n, 1) float array.
    """
    q1 = df['question1'].fillna('').astype(str).values
    q2 = df['question2'].fillna('').astype(str).values

    v1 = tfidf_vectorizer.transform(q1)
    v2 = tfidf_vectorizer.transform(q2)
    sim = 1.0 - paired_cosine_distances(v1, v2)
    return sim.reshape(-1, 1)


def extract_improved_features(df, tfidf_word, tfidf_char):
    """
    Improved model: 18-feature dense vector combining four tokeniser approaches.

    Feature layout
    ─────────────────────────────────────────────────────────────────
    Indices  Tokeniser          Metrics
    ───────  ─────────────────  ─────────────────────────────────────
     0– 3   regex_tokenizer    Jaccard, overlap-coeff, BoW-cosine,
                               normalised-length-diff
     4– 7   clean_tokenizer    (same four, stopwords removed)
     8–11   char_ngram (n=3)   (same four, at character level)
    12      TF-IDF word        cosine similarity (sklearn)
    13      TF-IDF char 2-gram cosine similarity (sklearn)
    14      Levenshtein        normalised edit distance (from scratch)
    15      char-length ratio  min(|q1|,|q2|) / max(|q1|,|q2|)
    16      same_wh_word       both questions start with same WH-word
    17      both_wh            both questions start with any WH-word
    ─────────────────────────────────────────────────────────────────

    Fit tfidf_word and tfidf_char on training data; reuse for val/test.
    Returns a dense (n, 18) float32 array.
    """
    q1_raw = df['question1'].fillna('').astype(str).values
    q2_raw = df['question2'].fillna('').astype(str).values

    # TF-IDF similarities (vectorised via sklearn sparse matrix ops)
    v1w, v2w = tfidf_word.transform(q1_raw), tfidf_word.transform(q2_raw)
    tfidf_w_sim = 1.0 - paired_cosine_distances(v1w, v2w)

    v1c, v2c = tfidf_char.transform(q1_raw), tfidf_char.transform(q2_raw)
    tfidf_c_sim = 1.0 - paired_cosine_distances(v1c, v2c)

    # Pre-tokenise all questions once (avoids redundant re-tokenisation)
    tr1 = [regex_tokenizer(q) for q in q1_raw]
    tr2 = [regex_tokenizer(q) for q in q2_raw]
    tc1 = [clean_tokenizer(q) for q in q1_raw]
    tc2 = [clean_tokenizer(q) for q in q2_raw]
    tn1 = [char_ngram_tokenizer(q) for q in q1_raw]
    tn2 = [char_ngram_tokenizer(q) for q in q2_raw]

    rows = []
    for i, (q1, q2) in enumerate(zip(q1_raw, q2_raw)):
        f = (
            _tok_quad(tr1[i], tr2[i])   # regex: 4 features
            + _tok_quad(tc1[i], tc2[i]) # clean: 4 features
            + _tok_quad(tn1[i], tn2[i]) # char-ngram: 4 features
        )

        f.append(tfidf_w_sim[i])                                         # 12
        f.append(tfidf_c_sim[i])                                         # 13
        f.append(normalized_levenshtein(q1, q2))                         # 14
        l1, l2 = len(q1), len(q2)
        f.append(min(l1, l2) / (max(l1, l2) + 1))                       # 15

        fw1 = tr1[i][0] if tr1[i] else ''
        fw2 = tr2[i][0] if tr2[i] else ''
        f.append(int(fw1 == fw2 and fw1 in WH_WORDS))                   # 16
        f.append(int(fw1 in WH_WORDS and fw2 in WH_WORDS))              # 17

        rows.append(f)

    return np.array(rows, dtype=np.float32)


def get_mistakes(clf, X, y):
    """
    Returns (wrong_indices, all_predictions) for a fitted classifier.
    Useful for error analysis in the explanation notebook.
    """
    preds = clf.predict(X)
    (idxs,) = np.where(preds != np.asarray(y))
    return idxs, preds
