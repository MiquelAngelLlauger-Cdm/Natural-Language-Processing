
from __future__ import annotations

import math
import os
import re
import string
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. COMMON HELPERS  (shared by the whole group)
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "Datasets", "QuoraQuestionPairs")
DATA_FILE = "quora_data.csv"

# Random seed reused across the project so every run is reproducible
RANDOM_STATE = 123


def load_quora_dataframe(path: str | None = None) -> pd.DataFrame:
    """Load the Quora CSV.

    The deliverable specifies that data must live in
    ``$HOME/Datasets/QuoraQuestionPairs``.  We default to that location
    but accept an override for testing.
    """
    if path is None:
        path = os.path.join(DATA_DIR, DATA_FILE)
    df = pd.read_csv(path)
    # Some Quora CSVs ship with a few NaN questions; replace by empty string
    df["question1"] = df["question1"].fillna("").astype(str)
    df["question2"] = df["question2"].fillna("").astype(str)
    return df


def split_quora(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the exact split described in the deliverable PDF.

    The same ``random_state`` and split sizes are reproduced verbatim so
    that *every* group obtains the same train/val/test partitions.
    """
    A_df, test_df = train_test_split(df, test_size=0.05, random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(A_df, test_size=0.05, random_state=RANDOM_STATE)
    return train_df, val_df, test_df


# ----- Tokenisation / preprocessing -----------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalise(text: str) -> str:
    """Lower-case and strip punctuation.

    We deliberately keep this very simple: removing punctuation and
    case is cheap and avoids most spurious mismatches between two
    paraphrases of the same question (e.g. "What's" vs "what is" we do
    not handle, but "Why?" and "why" become identical).
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """White-space tokeniser applied after :func:`normalise`."""
    return normalise(text).split()


# A small, hand-picked list of English stop words.  We avoid pulling in
# NLTK's list to keep the project dependency-light.
STOPWORDS = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "and", "or", "to", "in", "on", "for", "with", "as", "at",
        "by", "from", "that", "this", "these", "those", "it", "its",
        "i", "you", "he", "she", "we", "they", "them", "his", "her",
        "what", "which", "who", "whom", "where", "when", "how", "why",
        "do", "does", "did", "doing", "have", "has", "had", "having",
        "can", "could", "would", "should", "will", "shall", "may", "might",
        "not", "no", "but", "if", "than", "then", "so", "such",
    ]
)


def content_tokens(text: str) -> List[str]:
    """Tokens with stop words removed (used for some lexical features)."""
    return [w for w in tokenize(text) if w not in STOPWORDS]


# =============================================================================
# 2. MEMBER A  -  Lexical / set-based features
# =============================================================================
# Author : Member A
# Idea   : how much vocabulary do the two questions share?  A simple but
#          surprisingly strong signal for paraphrase detection.

def jaccard_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity of the two token sets.

    Implemented from scratch (no sklearn / nltk).  Returns 0.0 if both
    sentences are empty after normalisation, which keeps the feature
    well-defined for the Quora rows where one question is missing.
    """
    s1 = set(tokenize(text1))
    s2 = set(tokenize(text2))
    if not s1 and not s2:
        return 0.0
    inter = s1 & s2
    union = s1 | s2
    return len(inter) / len(union)


def jaccard_content(text1: str, text2: str) -> float:
    """Jaccard similarity *after* removing stop words.

    Stop words like "what", "is", "the" inflate the overlap of two
    unrelated questions, so we also expose the stop-word-free version.
    """
    s1 = set(content_tokens(text1))
    s2 = set(content_tokens(text2))
    if not s1 and not s2:
        return 0.0
    inter = s1 & s2
    union = s1 | s2
    return len(inter) / len(union) if union else 0.0


def word_overlap_ratio(text1: str, text2: str) -> float:
    """|q1 ∩ q2| / min(|q1|, |q2|).

    This complements Jaccard: it answers "what fraction of the shorter
    question is contained in the longer one?".  Useful when one
    question is a paraphrase of a *prefix* of the other.
    """
    s1, s2 = set(tokenize(text1)), set(tokenize(text2))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / min(len(s1), len(s2))


def length_features(text1: str, text2: str) -> List[float]:
    """Returns ``[len1, len2, abs_diff, len_ratio]`` in tokens.

    Two duplicates tend to have similar length; very different lengths
    are evidence of non-duplicates.  ``len_ratio`` is bounded in [0, 1].
    """
    t1, t2 = tokenize(text1), tokenize(text2)
    n1, n2 = len(t1), len(t2)
    diff = abs(n1 - n2)
    ratio = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 0.0
    return [float(n1), float(n2), float(diff), ratio]


def char_length_features(text1: str, text2: str) -> List[float]:
    """Same idea as :func:`length_features` but at the character level."""
    n1, n2 = len(text1), len(text2)
    diff = abs(n1 - n2)
    ratio = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 0.0
    return [float(n1), float(n2), float(diff), ratio]


def common_words_count(text1: str, text2: str) -> float:
    """Raw count of unique tokens shared between the two questions."""
    return float(len(set(tokenize(text1)) & set(tokenize(text2))))


# =============================================================================
# 3. MEMBER B  -  Edit-distance features
# =============================================================================
# Author : Member B
# Idea   : Levenshtein distance between the two questions, both at the
#          character and token level.  We code it from scratch using the
#          classic dynamic-programming recurrence, with a memory-efficient
#          two-row table so the algorithm stays O(min(|s|, |t|)) memory.

def levenshtein_distance(s: Sequence, t: Sequence) -> int:
    """Edit distance between two sequences (strings *or* token lists).

    Standard dynamic-programming algorithm with a two-row table.  We
    iterate so the rows have length ``min(|s|,|t|)+1``: this is
    important on long Quora questions where the naive (m+1)x(n+1)
    matrix can grow unpleasantly.

    Cost model: insertion = deletion = substitution = 1.
    """
    if len(s) < len(t):
        s, t = t, s
    # Now len(s) >= len(t)
    if not t:
        return len(s)

    previous = list(range(len(t) + 1))
    current = [0] * (len(t) + 1)
    for i, ch_s in enumerate(s, start=1):
        current[0] = i
        for j, ch_t in enumerate(t, start=1):
            cost = 0 if ch_s == ch_t else 1
            current[j] = min(
                previous[j] + 1,         # deletion
                current[j - 1] + 1,      # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous, current = current, previous
    return previous[len(t)]


def char_edit_distance(text1: str, text2: str) -> float:
    return float(levenshtein_distance(text1, text2))


def normalised_char_edit(text1: str, text2: str) -> float:
    """Edit distance divided by the longer string length, in [0, 1]."""
    m = max(len(text1), len(text2))
    if m == 0:
        return 0.0
    return levenshtein_distance(text1, text2) / m


def token_edit_distance(text1: str, text2: str) -> float:
    """Edit distance counted in *tokens* instead of characters."""
    return float(levenshtein_distance(tokenize(text1), tokenize(text2)))


def normalised_token_edit(text1: str, text2: str) -> float:
    t1, t2 = tokenize(text1), tokenize(text2)
    m = max(len(t1), len(t2))
    if m == 0:
        return 0.0
    return levenshtein_distance(t1, t2) / m


def longest_common_prefix(text1: str, text2: str) -> float:
    """Number of shared starting tokens.

    Many Quora duplicates share the first few words (e.g. both start
    with "How can I").  This cheap feature captures that pattern.
    """
    t1, t2 = tokenize(text1), tokenize(text2)
    n = 0
    for a, b in zip(t1, t2):
        if a != b:
            break
        n += 1
    return float(n)


# =============================================================================
# 4. MEMBER C  -  TF-IDF cosine similarity from scratch
# =============================================================================
# Author : Member C
# Idea   : reproduce TfidfVectorizer's behaviour by hand so we can show
#          we understand what is happening underneath.  The fitted
#          object exposes ``transform`` and ``cosine`` helpers used as
#          a feature in the final model.

class TfidfFromScratch:
    """A minimal TF-IDF vectoriser implemented from scratch.

    Mirrors the variant used in scikit-learn:

    * ``tf``  : raw term frequency (optional sublinear ``1 + log(tf)``)
    * ``idf`` : ``log((1 + N) / (1 + df)) + 1``  (smooth IDF, identical
       to sklearn's default)
    * Final vectors are L2-normalised, so cosine similarity reduces to
      a plain dot product.

    The class follows the standard ``fit`` / ``transform`` API so it is
    drop-in compatible with the rest of the project.
    """

    def __init__(self, sublinear_tf: bool = True, min_df: int = 1):
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    # -- fit ---------------------------------------------------------------
    def fit(self, documents: Iterable[str]) -> "TfidfFromScratch":
        # Count document frequencies
        df_counter: Counter = Counter()
        n_docs = 0
        for doc in documents:
            n_docs += 1
            for term in set(tokenize(doc)):
                df_counter[term] += 1

        # Build vocabulary respecting min_df
        vocab = {
            term: idx
            for idx, term in enumerate(
                sorted(t for t, c in df_counter.items() if c >= self.min_df)
            )
        }
        self.vocabulary_ = vocab

        # Smooth IDF, sklearn-style
        idf = np.zeros(len(vocab), dtype=np.float64)
        for term, idx in vocab.items():
            df = df_counter[term]
            idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf_ = idf
        self.n_docs_ = n_docs
        return self

    # -- transform ---------------------------------------------------------
    def _doc_vector(self, text: str) -> np.ndarray:
        """L2-normalised TF-IDF vector for a single document."""
        assert self.idf_ is not None, "Call fit() first."
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(len(self.vocabulary_), dtype=np.float64)

        counts: Counter = Counter(tokens)
        vec = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for term, c in counts.items():
            idx = self.vocabulary_.get(term)
            if idx is None:
                continue
            tf = 1.0 + math.log(c) if self.sublinear_tf else float(c)
            vec[idx] = tf * self.idf_[idx]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def transform(self, documents: Iterable[str]) -> np.ndarray:
        return np.vstack([self._doc_vector(d) for d in documents])

    # -- pairwise cosine ---------------------------------------------------
    def cosine(self, text1: str, text2: str) -> float:
        """Cosine similarity between two documents.

        Because both vectors are L2-normalised, this is just a dot
        product (see lecture 3, slide "With L2-normalised vectors,
        cosine reduces to a dot product").
        """
        v1 = self._doc_vector(text1)
        v2 = self._doc_vector(text2)
        return float(np.dot(v1, v2))


# =============================================================================
# 5. FEATURE ASSEMBLY + MODELS
# =============================================================================

# Names of the engineered features, in the order produced by
# ``build_feature_matrix``.  We expose them so the notebooks can show a
# pretty per-feature dataframe.
FEATURE_NAMES: List[str] = [
    "jaccard",
    "jaccard_content",
    "word_overlap_ratio",
    "common_words",
    "len1_tok",
    "len2_tok",
    "len_diff_tok",
    "len_ratio_tok",
    "len1_char",
    "len2_char",
    "len_diff_char",
    "len_ratio_char",
    "char_edit",
    "char_edit_norm",
    "tok_edit",
    "tok_edit_norm",
    "common_prefix",
    "tfidf_cosine_scratch",
    "tfidf_cosine_sklearn",
]


def _row_features(q1: str, q2: str, tfidf_scratch: TfidfFromScratch) -> List[float]:
    """All hand-crafted features for a single (q1, q2) row."""
    feats = [
        jaccard_similarity(q1, q2),
        jaccard_content(q1, q2),
        word_overlap_ratio(q1, q2),
        common_words_count(q1, q2),
    ]
    feats += length_features(q1, q2)
    feats += char_length_features(q1, q2)
    feats += [
        char_edit_distance(q1, q2),
        normalised_char_edit(q1, q2),
        token_edit_distance(q1, q2),
        normalised_token_edit(q1, q2),
        longest_common_prefix(q1, q2),
        tfidf_scratch.cosine(q1, q2),
    ]
    return feats


def build_feature_matrix(
    df: pd.DataFrame,
    tfidf_scratch: TfidfFromScratch,
    tfidf_sklearn: TfidfVectorizer,
) -> np.ndarray:
    """Compute the full feature matrix for a dataframe.

    Parameters
    ----------
    df :
        Dataframe with ``question1`` and ``question2`` columns.
    tfidf_scratch :
        A *fitted* :class:`TfidfFromScratch` (Member C's implementation).
    tfidf_sklearn :
        A *fitted* sklearn ``TfidfVectorizer`` used to compute a second,
        sparse cosine similarity feature.  Having the two side by side
        is a sanity check: their values must be very close.

    Returns
    -------
    X : ndarray of shape (n_rows, len(FEATURE_NAMES))
    """
    q1s = df["question1"].astype(str).tolist()
    q2s = df["question2"].astype(str).tolist()

    rows = [_row_features(q1, q2, tfidf_scratch) for q1, q2 in zip(q1s, q2s)]
    X_manual = np.asarray(rows, dtype=np.float64)

    # sklearn cosine similarity, computed via sparse matmul (very fast)
    M1 = tfidf_sklearn.transform(q1s)
    M2 = tfidf_sklearn.transform(q2s)
    cos = np.asarray(M1.multiply(M2).sum(axis=1)).ravel()
    X = np.hstack([X_manual, cos.reshape(-1, 1)])
    return X


# ----- Baseline model (TF-IDF on concatenated questions + LogReg) -----------

def fit_baseline(train_df: pd.DataFrame, max_features: int = 20000):
    """Fit the simple TF-IDF + Logistic-Regression baseline.

    The model concatenates ``q1`` and ``q2`` with a ``[SEP]`` token and
    feeds the resulting string to a standard sklearn pipeline (lecture
    3, "Baseline pipeline" slide).  Despite its simplicity it is a
    surprisingly strong starting point.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    texts = [f"{q1} [SEP] {q2}" for q1, q2 in zip(train_df["question1"], train_df["question2"])]
    X = vec.fit_transform(texts)
    y = train_df["is_duplicate"].astype(int).values
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf.fit(X, y)
    return vec, clf


def predict_baseline(vec: TfidfVectorizer, clf: LogisticRegression, df: pd.DataFrame):
    texts = [f"{q1} [SEP] {q2}" for q1, q2 in zip(df["question1"], df["question2"])]
    X = vec.transform(texts)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred


# ----- Improved model (engineered features + gradient boosting) -------------

def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Compute ROC-AUC, precision, recall as required by the deliverable."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }
