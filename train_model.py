"""
train_model.py
--------------
Run this once to train the chatbot model and generate the .pkl file.

Usage:
    python train_model.py

Outputs:
    - job_portal_model.pkl   (the trained model, used by app.py)
    - training_report.txt    (accuracy + full classification report)
"""

import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(BASE_DIR, "JobPortalDataset.csv")
MODEL_OUT   = os.path.join(BASE_DIR, "job_portal_model.pkl")
REPORT_OUT  = os.path.join(BASE_DIR, "training_report.txt")

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET)
print(f"  Rows loaded     : {len(df)}")
print(f"  Unique responses: {df['Response'].nunique()}")

X = df["User Input"]
y = df["Response"]

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples    : {len(X_test)}")

# ── Build pipeline ───────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,
        min_df=1,
    )),
    ("clf", LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=10,
    )),
])

# ── Train ────────────────────────────────────────────────────────────────────
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred   = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, zero_division=0)

print(f"  Accuracy: {accuracy:.2%}")

# ── Save .pkl ────────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\nModel saved   → {MODEL_OUT}")

# ── Write training_report.txt ────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

lines = [
    "=" * 60,
    "  Job Dekho Chatbot — Training Report",
    f"  Generated : {timestamp}",
    "=" * 60,
    "",
    f"Dataset        : {DATASET}",
    f"Total rows     : {len(df)}",
    f"Unique classes : {df['Response'].nunique()}",
    f"Train samples  : {len(X_train)}",
    f"Test samples   : {len(X_test)}",
    "",
    "-" * 60,
    f"  Accuracy : {accuracy:.2%}",
    "-" * 60,
    "",
    "Classification Report:",
    "",
    report,
    "=" * 60,
    "  Quick Smoke Test",
    "=" * 60,
    "",
]

# ── Smoke test ───────────────────────────────────────────────────────────────
test_queries = [
    "How do I apply for a job?",
    "I forgot my password",
    "How do I post a job?",
    "What is the Interview Dashboard?",
    "How do real time notifications work?",
    "How do I accept an application?",
    "What job categories are available?",
]

for q in test_queries:
    answer = pipeline.predict([q])[0]
    lines.append(f"Q: {q}")
    lines.append(f"A: {answer}")
    lines.append("")

with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Report saved  → {REPORT_OUT}")
print("\nAll done. You can now run:  python app.py")
