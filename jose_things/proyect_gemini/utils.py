import pandas as pd
import scipy
import sklearn
from sklearn import feature_extraction, linear_model, metrics
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))
    return mylist_of_strings

def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features built by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))
    
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1, X_q2))

    return X_q1q2

def jaccard_similarity_scratch(s1, s2):
    """
    Computes Jaccard Similarity between two strings s1 and s2 from scratch.
    """
    # Simple tokenization
    def tokenize(s):
        s = str(s).lower()
        return set(re.findall(r'\w+', s))
    
    set1 = tokenize(s1)
    set2 = tokenize(s2)
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union

def get_advanced_features(df, count_vectorizer):
    """
    Returns features combining CountVectorizer and Jaccard Similarity.
    """
    X_base = get_features_from_df(df, count_vectorizer)
    
    # Advanced features: Jaccard Similarity and Length Difference
    jaccard_sims = []
    len_diffs = []
    
    q1_list = list(df["question1"])
    q2_list = list(df["question2"])
    
    for q1, q2 in zip(q1_list, q2_list):
        jaccard_sims.append(jaccard_similarity_scratch(q1, q2))
        s1, s2 = str(q1), str(q2)
        len_diffs.append(abs(len(s1) - len(s2)) / (max(len(s1), len(s2)) + 1))
        
    # Combine features
    X_extra = np.column_stack((jaccard_sims, len_diffs))
    X_combined = scipy.sparse.hstack((X_base, scipy.sparse.csr_matrix(X_extra)))
    
    return X_combined

def get_mistakes(clf, X_q1q2, y):
    """
    Returns mistake indices and predictions.
    """
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices, = np.where(incorrect_predictions)
    
    return incorrect_indices, predictions

def get_sentence_embeddings(model, text_list):
    """
    Generates dense vector representations for the questions.
    """
    text_list_casted = cast_list_as_strings(text_list)
    return model.encode(text_list_casted, show_progress_bar=False)

def get_sbert_features(df, sbert_model):
    """
    Extracts features from the question1 and question2 columns by calculating the absolute difference between their embeddings, and optionally concatenating their cosine similarity.
    """
    embeddings1 = get_sentence_embeddings(sbert_model, list(df["question1"]))
    embeddings2 = get_sentence_embeddings(sbert_model, list(df["question2"]))
    
    abs_diff = np.abs(embeddings1 - embeddings2)
    
    from sklearn.metrics.pairwise import paired_cosine_distances
    cos_sim = 1 - paired_cosine_distances(embeddings1, embeddings2)
    
    X = scipy.sparse.csr_matrix(np.hstack((abs_diff, cos_sim.reshape(-1, 1))))
    return X
