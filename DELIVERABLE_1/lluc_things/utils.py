"""
utils.py
Utility functions used by train_models.ipynb and reproduce_results.ipynb
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_quora(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_quora(quora_df: pd.DataFrame, seed: int = 123) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    A_df, test_df = train_test_split(quora_df, test_size=0.05, random_state=seed)
    train_df, val_df = train_test_split(A_df, test_size=0.05, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_pair_texts(df: pd.DataFrame) -> pd.Series:
    # combine question1 and question2 into a single representation
    q1 = df['question1'].fillna('')
    q2 = df['question2'].fillna('')
    return (q1 + ' [SEP] ' + q2)


def build_tfidf(train_texts, max_features: int = 50000) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), lowercase=True)
    X_train = vec.fit_transform(train_texts)
    return vec, X_train


def transform_tfidf(vec: TfidfVectorizer, texts) -> np.ndarray:
    return vec.transform(texts)


def train_logreg(X, y, seed: int = 123) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, solver='saga', random_state=seed)
    clf.fit(X, y)
    return clf


def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str):
    return joblib.load(path)


def set_seed(seed: int = 123):
    np.random.seed(seed)
