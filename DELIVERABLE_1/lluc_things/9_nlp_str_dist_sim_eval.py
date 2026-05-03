"""
9_nlp_str_dist_sim_eval.py
Skeleton for string distance and similarity evaluation experiments.
"""
from typing import List


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


if __name__ == "__main__":
    print(jaccard_similarity(["a","b"], ["b","c"]))
