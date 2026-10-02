"""Train category + priority models and write an honest evaluation report.

Usage:
  python train.py                                   # synthetic demo data
  python train.py --csv data/customer_support_tickets.csv \
      --text-col "Ticket Description" --cat-col "Ticket Type" --prio-col "Ticket Priority"
"""
import argparse, json
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from common import clean_text

ap = argparse.ArgumentParser()
ap.add_argument("--csv"); ap.add_argument("--text-col", default="text")
ap.add_argument("--cat-col", default="category"); ap.add_argument("--prio-col", default="priority")
a = ap.parse_args()

if a.csv:
    df = pd.read_csv(a.csv)
    df = df.rename(columns={a.text_col: "text", a.cat_col: "category", a.prio_col: "priority"})
    source = a.csv
else:
    from make_synthetic import build
    df = build(); source = "synthetic (make_synthetic.py)"

df = df[["text", "category", "priority"]].dropna()
df["priority"] = df["priority"].astype(str).str.lower()
df["clean"] = df["text"].map(clean_text)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])


def make():
    # LogisticRegression gives calibrated-ish predict_proba for the confidence score
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=2000, C=5, class_weight="balanced"))])


report = {"dataset": source, "n_train": len(train), "n_test": len(test)}
for name, col in [("category", "category"), ("priority", "priority")]:
    m = make().fit(train["clean"], train[col])
    pred = m.predict(test["clean"])
    labels = sorted(df[col].unique())
    report[name] = {
        "accuracy": round(accuracy_score(test[col], pred), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(test[col], pred, labels=labels).tolist(),
        "per_class": classification_report(test[col], pred, output_dict=True, zero_division=0),
    }
    if name == "category":
        wrong = test.assign(predicted=pred)[pred != test[col]].head(50)
        report["misclassified"] = [
            {"text": r.text, "actual": r.category, "predicted": r.predicted} for r in wrong.itertuples()]
    joblib.dump(m, f"models/{name}.joblib")
    print(f"{name}: accuracy = {report[name]['accuracy']:.3f}")

json.dump(report, open("models/metrics.json", "w"), indent=2)
print("saved models/ + metrics.json")
