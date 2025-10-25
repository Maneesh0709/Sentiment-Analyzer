import re
from typing import Literal

SentimentStr = Literal["negative", "neutral", "positive"]

LABEL_MAP_INT2STR = {0: "negative", 2: "neutral", 4: "positive"}
LABEL_MAP_STR2INT = {"negative": 0, "neutral": 2, "positive": 4}

_url_re = re.compile(r"https?://\S+|www\.\S+")
_user_re = re.compile(r"@\w+")
_hash_re = re.compile(r"#(\w+)")
_multi_space_re = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """
    Minimal, robust cleaning that plays nicely with TF-IDF:
    - lowercasing
    - remove URLs
    - drop @user mentions
    - convert #hashtag to 'hashtag' (keep the token)
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    x = text.lower()
    x = _url_re.sub(" ", x)
    x = _user_re.sub(" ", x)
    x = _hash_re.sub(r"\1", x)
    x = _multi_space_re.sub(" ", x).strip()
    return x
