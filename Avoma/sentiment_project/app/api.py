# app/api.py
import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.utils import clean_text

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_logreg.joblib")

app = FastAPI(
    title="Sentiment API",
    version="1.1.0",
    description=(
        "TF-IDF + Logistic Regression sentiment classifier.\n"
        "Neutral is inferred via calibrated uncertainty and light linguistic heuristics."
    ),
)

class EvaluateRequest(BaseModel):
    text: str = Field(..., description="Sentence to analyze")

class EvaluateResponse(BaseModel):
    sentiment: str
    confidence: float
    details: Optional[dict] = None

# --- Lightweight heuristic helpers ---
_url_re = re.compile(r"https?://\S+|www\.\S+")
_mention_re = re.compile(r"@\w+")
# A tiny set of obvious sentiment cue words (lowercased, post-cleaning)
SENTI_CUES = {
    "love","like","great","amazing","awesome","good","wonderful","happy",
    "hate","bad","awful","terrible","horrible","sad","angry","dislike","worst"
}

# Load model once at startup
pipe = None

@app.on_event("startup")
def load_model():
    global pipe
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train the model first.")
    pipe = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    req: EvaluateRequest,
    neutral_margin: float = 0.18,  # gap threshold: |pos - neg| < neutral_margin -> neutral
    max_confidence: float = 0.70   # max probability threshold: max(pos,neg) < max_confidence -> neutral
):
    """
    Returns sentiment ('negative'/'positive' or 'neutral') + confidence.

    Neutral is returned if ANY of these holds:
      A) Dual-threshold uncertainty:
         - Gap is small: |p(pos) - p(neg)| < neutral_margin
         - OR max probability < max_confidence
      B) Super-light heuristics:
         - Question-like text and not extremely confident
         - Very short URL/mention-dominant text and not extremely confident
         - No obvious sentiment cue words and not extremely confident
    Thresholds are tunable via query params.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty 'text' provided")

    # Original (raw) and cleaned text
    orig_text = req.text
    x = clean_text(orig_text)

    # Model probabilities
    try:
        proba = pipe.predict_proba([x])[0]  # [p(neg), p(pos)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    neg_p, pos_p = float(proba[0]), float(proba[1])
    gap = abs(pos_p - neg_p)
    maxp = max(pos_p, neg_p)

    # --- Super-light heuristics (no retraining needed) ---
    has_qmark = "?" in orig_text
    has_url = bool(_url_re.search(orig_text))
    has_mention = bool(_mention_re.search(orig_text))
    tokens = x.split()
    has_senti_word = any(t in SENTI_CUES for t in tokens)

    # Heuristic neutral triggers (kept conservative with high maxp caps)
    if (
        (has_qmark and maxp < 0.90) or
        ((has_url or has_mention) and len(tokens) < 5 and maxp < 0.90) or
        (not has_senti_word and maxp < 0.85)
    ):
        return EvaluateResponse(
            sentiment="neutral",
            confidence=1.0 - gap,  # simple proxy; optional to customize
            details={
                "neg": neg_p, "pos": pos_p, "gap": gap,
                "neutral_margin": neutral_margin,
                "max_confidence": max_confidence,
                "rule": "question/url/no-cue"
            },
        )

    # --- Dual neutral heuristic: low gap OR low confidence ---
    if (gap < neutral_margin) or (maxp < max_confidence):
        return EvaluateResponse(
            sentiment="neutral",
            confidence=1.0 - gap,
            details={
                "neg": neg_p, "pos": pos_p, "gap": gap,
                "neutral_margin": neutral_margin,
                "max_confidence": max_confidence,
                "rule": "uncertainty"
            },
        )

    # Otherwise return the majority class
    label = "positive" if pos_p >= neg_p else "negative"
    return EvaluateResponse(
        sentiment=label,
        confidence=maxp,
        details={
            "neg": neg_p, "pos": pos_p, "gap": gap,
            "neutral_margin": neutral_margin,
            "max_confidence": max_confidence
        },
    )
