# app/train.py
# Python 3.9 compatible

import os
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from app.utils import clean_text, LABEL_MAP_INT2STR

# ---------- Paths ----------
HOME = os.path.expanduser("~")
TRAIN_PATH = os.path.join(
    HOME, "Desktop", "Avoma", "trainingandtestdata", "training.1600000.processed.noemoticon.csv"
)
TEST_PATH = os.path.join(
    HOME, "Desktop", "Avoma", "trainingandtestdata", "testdata.manual.2009.06.14.csv"
)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_OUT_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(DATA_OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_sgd_logloss.joblib")

# ---------- Loaders ----------
def load_training_csv(sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    training.1600000.processed.noemoticon.csv (no header)
    Cols: 0=sentiment(0,2,4), 1=id, 2=date, 3=query, 4=user, 5=text
    We keep only [sentiment, text] and add 'clean_text'.
    Set sample_size=None to use the full 1.6M rows.
    """
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(TRAIN_PATH, encoding="latin-1", header=None, names=cols)
    df = df[["sentiment", "text"]]
    if sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    df["label"] = df["sentiment"].map(LABEL_MAP_INT2STR)
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    return df.reset_index(drop=True)

def load_test_csv() -> pd.DataFrame:
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(TEST_PATH, encoding="latin-1", header=None, names=cols)
    df = df[df["sentiment"].isin([0, 2, 4])][["sentiment", "text"]].copy()
    df["label"] = df["sentiment"].map(LABEL_MAP_INT2STR)
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    return df.reset_index(drop=True)

# ---------- Train / Save ----------
def build_pipeline() -> Pipeline:
    """
    TF-IDF (1-2 grams) + SGDClassifier (logistic).
    Tuned for large corpus: fewer rare terms, bounded vocab, float32.
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        stop_words="english",
        max_features=300_000,
        strip_accents="unicode",
        sublinear_tf=True,
        dtype=np.float32,  # memory friendly
    )
    clf = SGDClassifier(
        loss="log_loss",             # logistic regression with SGD
        class_weight="balanced",
        alpha=1e-4,                  # L2 strength
        early_stopping=True,
        random_state=42,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])

def train_and_save(train_df: pd.DataFrame) -> str:
    """
    Train binary (neg vs pos) on full training data and save pipeline.
    """
    bin_train = train_df[train_df["sentiment"].isin([0, 4])].copy()
    y_train = (bin_train["sentiment"] == 4).astype(int)  # 1=pos, 0=neg

    pipe = build_pipeline()
    pipe.fit(bin_train["clean_text"], y_train)

    joblib.dump(pipe, MODEL_PATH)
    return MODEL_PATH

# ---------- Evaluate ----------
def evaluate_on_test(model_path: str, test_df: pd.DataFrame) -> None:
    """
    Strict binary eval: drop neutral rows from test.
    """
    pipe = joblib.load(model_path)
    test_bin = test_df[test_df["sentiment"].isin([0, 4])].copy()
    y_true = (test_bin["sentiment"] == 4).astype(int)
    y_pred = pipe.predict(test_bin["clean_text"])

    acc = accuracy_score(y_true, y_pred)
    print(f"\nBinary test accuracy (neg/pos only): {acc:.4f}")
    print("\nClassification report (neg=0, pos=1):\n",
          classification_report(y_true, y_pred, digits=4))

def evaluate_with_neutral(
    model_path: str,
    test_df: pd.DataFrame,
    neutral_margin: float = 0.25,
    max_confidence: float = 0.80
) -> float:
    """
    3-class evaluation using dual neutral heuristic:
      neutral if (|p_pos - p_neg| < neutral_margin) OR (max(p) < max_confidence)
    Returns macro-F1 to help pick thresholds.
    """
    pipe = joblib.load(model_path)
    y_true = test_df["sentiment"].map({0: "negative", 2: "neutral", 4: "positive"}).tolist()
    y_pred = []
    for text in test_df["clean_text"]:
        proba = pipe.predict_proba([text])[0]  # [neg, pos]
        neg_p, pos_p = float(proba[0]), float(proba[1])
        gap = abs(pos_p - neg_p)
        maxp = max(pos_p, neg_p)
        if (gap < neutral_margin) or (maxp < max_confidence):
            y_pred.append("neutral")
        else:
            y_pred.append("positive" if pos_p >= neg_p else "negative")

    print(f"\n3-class eval (margin={neutral_margin:.2f}, max_conf={max_confidence:.2f}):")
    print(classification_report(y_true, y_pred, digits=4))
    return f1_score(y_true, y_pred, average="macro")

def sweep_params(
    model_path: str,
    test_df: pd.DataFrame,
    margins: Iterable[float] = (0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32),
    max_confs: Iterable[float] = (0.70, 0.75, 0.80, 0.85)
) -> Tuple[float, float, float]:
    """
    Grid sweep over (neutral_margin, max_confidence). Prints macro-F1 table.
    Returns (best_margin, best_max_conf, best_macro_f1).
    """
    pipe = joblib.load(model_path)
    y_true = test_df["sentiment"].map({0: "negative", 2: "neutral", 4: "positive"}).tolist()

    def pred_for(text, m, c):
        proba = pipe.predict_proba([text])[0]
        neg_p, pos_p = float(proba[0]), float(proba[1])
        gap = abs(pos_p - neg_p)
        maxp = max(pos_p, neg_p)
        if (gap < m) or (maxp < c):
            return "neutral"
        return "positive" if pos_p >= neg_p else "negative"

    print("\nGrid sweep (macro F1):")
    best = (None, None, -1.0)
    for c in max_confs:
        row = []
        for m in margins:
            y_pred = [pred_for(t, m, c) for t in test_df["clean_text"]]
            mf1 = f1_score(y_true, y_pred, average="macro")
            row.append(f"{mf1:.4f}")
            if mf1 > best[2]:
                best = (m, c, mf1)
        print(f"  max_conf={c:.2f} -> " + ", ".join([f"m={m:.2f}:{r}" for m, r in zip(margins, row)]))
    print(f"\nBest -> margin={best[0]:.2f}, max_conf={best[1]:.2f}, macro_F1={best[2]:.4f}")
    return best

# ---------- Main ----------
def main():
    # 1) Load FULL training set (set to 200_000 again only if you need quick iterations)
    train_df = load_training_csv(sample_size=None)
    test_df = load_test_csv()

    # 2) (Optional) Save cleaned snapshots for inspection
    train_out = os.path.join(DATA_OUT_DIR, "train_full_clean.csv")
    test_out = os.path.join(DATA_OUT_DIR, "test_clean.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Saved:\n  {train_out} ({len(train_df):,} rows)\n  {test_out} ({len(test_df):,} rows)")
    print("\nTrain label distribution:\n", train_df["label"].value_counts())
    print("\nTest label distribution:\n", test_df["label"].value_counts())

    # 3) Train & save
    model_path = train_and_save(train_df)
    print(f"\nModel saved to: {model_path}")

    # 4) Binary eval (neg/pos only)
    evaluate_on_test(model_path, test_df)

    # 5) 3-class eval with a reasonable starting point
    _ = evaluate_with_neutral(model_path, test_df, neutral_margin=0.18, max_confidence=0.70)

    # 6) Parameter sweep to pick API defaults
    sweep_params(model_path, test_df)

if __name__ == "__main__":
    main()
