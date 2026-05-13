import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Group Members: Miki, Jose, Lluc

def load_data(file_path):
    """
    Loads NER data from CSV and groups it by sentence_id.
    """
    df = pd.read_csv(file_path)
    # Ensure sentence_id is int
    df['sentence_id'] = df['sentence_id'].astype(int)
    
    sentences = []
    # Group by sentence_id
    grouped = df.groupby('sentence_id')
    for name, group in grouped:
        words = group['words'].values.tolist()
        tags = group['tags'].values.tolist()
        sentences.append(list(zip(words, tags)))
    
    return sentences

def word2features(sent, i):
    """
    Extracts features for a word at index i in sentence sent.
    """
    word = str(sent[i][0])
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = str(sent[i-1][0])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = str(sent[i+1][0])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluates the model and prints metrics as required by the guide.
    1. Accuracy on predicted non-O tags.
    2. Confusion matrix.
    3. F-score.
    """
    # Flatten the lists
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    
    # 1. Accuracy on non-O ground truth tags
    non_o_indices = [i for i, label in enumerate(y_true_flat) if label != 'O']
    y_true_non_o = [y_true_flat[i] for i in non_o_indices]
    y_pred_non_o = [y_pred_flat[i] for i in non_o_indices]
    
    acc_non_o = accuracy_score(y_true_non_o, y_pred_non_o)
    
    # 2. F-score (weighted)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy (excluding 'O'): {acc_non_o:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    
    # 3. Confusion Matrix
    labels = sorted(list(set(y_true_flat)))
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return acc_non_o, f1

def print_tiny_test_results(words, predicted_tags):
    """
    Prints results in format: w1/t1 w2/t2 ...
    """
    result = []
    for w, t in zip(words, predicted_tags):
        result.append(f"{w}/{t}")
    print(" ".join(result))
