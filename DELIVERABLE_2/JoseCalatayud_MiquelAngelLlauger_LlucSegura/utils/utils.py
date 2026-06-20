"""Shared data loading and evaluation helpers for the NER delivery."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def load_sentences(path):
    """Load a token-level CSV and return ordered sentences of (word, tag) pairs."""
    frame = pd.read_csv(Path(path))
    required = {"sentence_id", "words", "tags"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["words"] = frame["words"].astype(str)
    frame["tags"] = frame["tags"].astype(str)
    frame["sentence_id"] = frame["sentence_id"].astype(int)
    return [list(zip(group["words"], group["tags"]))
            for _, group in frame.groupby("sentence_id", sort=True)]


def flatten(sequences):
    return [item for sequence in sequences for item in sequence]


def non_o_accuracy(y_true, y_pred):
    """Token accuracy restricted to gold labels other than O, as required."""
    true_flat, pred_flat = flatten(y_true), flatten(y_pred)
    selected = [i for i, tag in enumerate(true_flat) if tag != "O"]
    return accuracy_score([true_flat[i] for i in selected],
                          [pred_flat[i] for i in selected])


def weighted_non_o_f1(y_true, y_pred):
    """Weighted token F1 over entity labels, excluding O and padding."""
    true_flat, pred_flat = flatten(y_true), flatten(y_pred)
    labels = sorted(set(true_flat).difference({"O", "<PAD>"}))
    return f1_score(true_flat, pred_flat, labels=labels,
                    average="weighted", zero_division=0)


def tagged_sentence(sentence, predictions):
    """Format one prediction in the guide's w1/t1 w2/t2 form."""
    return " ".join(f"{word}/{tag}" for (word, _), tag in zip(sentence, predictions))
