import os
import joblib
from typing import Literal, Dict, Any
from app.utils import clean_text, LABEL_MAP_STR2INT, LABEL_MAP_INT2STR

SentimentStr = Literal["negative", "neutral", "positive"]

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_logreg.joblib")

def evaluate(text: str, neutral_margin: float = 0.15) -> Dict[str, Any]:
    """
    Load the saved model pipeline and return sentiment + confidence.
    - Returns 'neutral' if the model is uncertain (prob gap below neutral_margin).
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Train the model first.")

    pipe = joblib.load(MODEL_PATH)
    x = clean_text(text)
    proba = pipe.predict_proba([x])[0]  # [p(neg), p(pos)]
    neg_p, pos_p = float(proba[0]), float(proba[1])

    # neutral heuristic: if model isn't confident, call it neutral
    if abs(pos_p - neg_p) < neutral_margin:
        return {"sentiment": "neutral", "confidence": 1.0 - abs(pos_p - neg_p)}

    label = "positive" if pos_p >= neg_p else "negative"
    confidence = max(pos_p, neg_p)
    return {"sentiment": label, "confidence": confidence}
