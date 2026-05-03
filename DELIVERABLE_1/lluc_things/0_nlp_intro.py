"""
0_nlp_intro.py
Skeleton for introductory exercises (tokenization, basic pipeline).

Fill in functions below to process sample text and demonstrate outputs.
"""

from typing import List


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer — replace with a better one if needed."""
    return text.split()


if __name__ == "__main__":
    sample = "This is a sample sentence for NLP intro."
    print("Tokens:", tokenize(sample))
