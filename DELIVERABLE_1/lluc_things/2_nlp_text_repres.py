"""
2_nlp_text_repres.py
Skeleton for text representation: bag-of-words, one-hot, basic vectorizers.
"""
from collections import Counter
from typing import List, Dict


def bow_vector(texts: List[str]) -> Dict[str, int]:
    c = Counter()
    for t in texts:
        c.update(t.split())
    return dict(c)


if __name__ == "__main__":
    docs = ["this is a doc", "this doc is short"]
    print(bow_vector(docs))
