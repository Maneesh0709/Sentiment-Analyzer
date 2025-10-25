# Twitter Sentiment Classifier + REST API

A simple end-to-end sentiment analysis project using **TF-IDF + Linear Classifier**, exposed through a **FastAPI REST API**.

We are not aiming for a state-of-the-art model. The focus is on:
- Clean, maintainable, well-documented code
- RESTful API design
- Explainable methods and trade-offs

---

## 🚀 Features
- Trainable on full **1.6M tweet dataset** (binary: positive vs negative).
- Inference API with **neutral handling** via calibrated probability thresholds.
- RESTful endpoints with Pydantic schemas and error handling.
- Swagger auto-docs available at `/docs`.
- Lightweight heuristics for clearly neutral phrases (questions, URL/mention tweets, hedging words like *“okay / nothing special”*).

---

## 📂 Project Structure
```
sentiment_project/
├── app/
│   ├── __init__.py
│   ├── utils.py          # text cleaning + label map
│   ├── train.py          # training, evaluation, sweep
│   ├── api.py            # FastAPI app with inference
├── models/               # trained models (.joblib)
├── data/                 # optional: cleaned train/test csvs
├── requirements.txt
├── README.md

```

---

## ⚙️ Environment Setup

### Prerequisites
- Python **3.9+** (works with 3.10 / 3.11 too)
- macOS/Linux/Windows

### Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### requirements.txt
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
fastapi==0.114.2
uvicorn[standard]==0.30.6
joblib==1.4.2
nltk==3.9.1
```

---

## 📊 Datasets

Expected dataset location:

```
~/Desktop/Avoma/trainingandtestdata/training.1600000.processed.noemoticon.csv
~/Desktop/Avoma/trainingandtestdata/testdata.manual.2009.06.14.csv
```

*(Adjust paths inside `app/train.py` if needed.)*

---

## 🏋️ Training

Run training on the **full dataset** (1.6M rows):
```bash
python -m app.train
```

What it does:
- Cleans data → saves snapshots in `data/`.
- Trains binary model (neg/pos).
- Saves model in `models/tfidf_sgd_logloss.joblib`.
- Prints binary metrics, 3-class metrics, and a grid sweep to suggest best thresholds.

> For quick iterations, change `sample_size=200_000` inside `load_training_csv()`.

---

## 🌐 Run the API

Ensure `MODEL_PATH` in `app/api.py` points to your trained model:
```python
MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_sgd_logloss.joblib")
```

Start the API server:
```bash
python -m uvicorn app.api:app --reload
```

- Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🧪 Example Request

```bash
curl -s -X POST "http://127.0.0.1:8000/evaluate"   -H "Content-Type: application/json"   -d '{"text":"It was okay, nothing special."}'
```

Example response:
```json
{
  "sentiment": "neutral",
  "confidence": 0.81,
  "details": {
    "neg": 0.22,
    "pos": 0.19,
    "gap": 0.03,
    "neutral_margin": 0.18,
    "max_confidence": 0.70
  }
}
```

---

## 🛠️ API Endpoints

- `GET /health` → service status  
- `POST /evaluate` → classify one text  
  - Query params: `neutral_margin`, `max_confidence`  
- (Optional) `POST /evaluate/batch` → classify multiple texts

---

## 📈 Design Choices
- **Model**: TF-IDF (1–2 grams, pruned vocab) + linear classifier (scalable to 1.6M).
- **Neutral**: Added via uncertainty (gap + confidence thresholds). Defaults chosen via grid sweep.
- **REST**: Clean endpoints, proper status codes, auto-docs.
- **Not SOTA**: But explainable, fast, and maintainable.




---

## 📌 Next Steps
- Swap classifier to **LinearSVC + Platt scaling** for better calibrated probabilities.
- Add Docker + CI pipeline.
- Extend with a true 3-class dataset if available.
