# Sentiment Analyzer

A simple NLP pipeline to classify text sentiment (positive/negative/neutral).

## Features
- Preprocessing (tokenization, cleaning)
- Train/evaluate model (`train.py`, `model.py`)
- Reusable utilities (`utils.py`)
- Minimal API skeleton (`app/api.py`)

## Setup
```bash
# optional: create and activate a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r Avoma/sentiment_project/requirements.txt
python Avoma/sentiment_project/app/train.py
python Avoma/sentiment_project/app/model.py "I love this movie!"
