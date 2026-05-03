"""
1_nlp_regexp.py
Skeleton for regex and basic string processing exercises.
"""
import re
from typing import List


def find_emails(text: str) -> List[str]:
    pattern = r"[\w\.-]+@[\w\.-]+"
    return re.findall(pattern, text)


if __name__ == "__main__":
    text = "Contact us at help@example.com or admin@uni.edu"
    print(find_emails(text))
