<h1 align="center">🧠 Sentiment Analyzer — NLP Text Classification Project</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/NLP-scikit--learn%2C%20NLTK-green" alt="NLP Badge"/>
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status Badge"/>
</p>

---

## 📘 Overview
**Sentiment Analyzer** is a Natural Language Processing (NLP) project that classifies user text into **Positive**, **Negative**, or **Neutral** sentiments.  
It demonstrates data preprocessing, model training, and evaluation on real-world text datasets — built entirely from scratch in Python.

---

## ⚙️ Tech Stack
| Component | Description |
|------------|-------------|
| 🐍 **Language** | Python 3.8+ |
| 📦 **Libraries** | scikit-learn, pandas, numpy, nltk, seaborn |
| 🧠 **Model** | Logistic Regression / Random Forest for sentiment classification |
| 🧪 **Tools** | Jupyter Notebook / VS Code |
| 💾 **Dataset** | `training.1600000.processed.noemoticon.csv` (large-scale sentiment dataset) |

---

## 🚀 Features
✅ Clean text preprocessing (tokenization, stopword removal, stemming)  
✅ Train and evaluate multiple machine learning models  
✅ Predict user sentiment dynamically  
✅ Modular code structure for scalability  
✅ Visualize performance metrics (accuracy, confusion matrix, etc.)

---

## 🏗️ Project Structure

---

## 🧩 How It Works
1. **Data Preprocessing** → Clean, tokenize, and vectorize raw text data.  
2. **Model Training** → Train sentiment classifiers using scikit-learn ML algorithms.  
3. **Prediction** → Input a sentence and get predicted sentiment instantly.  
4. **Evaluation** → Compare performance via metrics and confusion matrix.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Maneesh0709/Sentiment-Analyzer.git
cd Sentiment-Analyzer
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate   # macOS/Linux
pip install -r Avoma/sentiment_project/requirements.txt
python Avoma/sentiment_project/app/train.py
python Avoma/sentiment_project/app/model.py "I absolutely love this product!"
User: "This movie was fantastic!"
Bot: "Predicted Sentiment → Positive 😄"
