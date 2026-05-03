"""
5_nlp_ngramlm.py
Skeleton for n-gram language model exercises.
"""
from collections import defaultdict
from typing import List


class NGramLM:
    def __init__(self, n: int = 2):
        self.n = n
        self.counts = defaultdict(int)

    def train(self, corpus: List[str]):
        for sent in corpus:
            tokens = sent.split()
            for i in range(len(tokens)-self.n+1):
                ngram = tuple(tokens[i:i+self.n])
                self.counts[ngram] += 1


if __name__ == "__main__":
    lm = NGramLM(2)
    lm.train(["this is a test", "this test is simple"])
    print(lm.counts)
