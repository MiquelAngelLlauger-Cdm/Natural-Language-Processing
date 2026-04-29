"""
9_nlp_edit_dist.py
Skeleton for edit distance and string similarity exercises.
"""

from typing import List


def levenshtein(a: str, b: str) -> int:
    # simple DP implementation
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1]


if __name__ == "__main__":
    print(levenshtein("kitten", "sitting"))
