"""AutoTriage AI backend (PS 16). Run: uvicorn app:app --host 0.0.0.0 --port 8000"""
import io, json, sqlite3, time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import joblib, pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from common import CONF_THRESHOLD, DEFAULT_TEAM, TEAM_MAP, clean_text

STATE = {}
DB = "tickets.db"
SARCASM = ["thanks for nothing", "great, another", "oh wonderful", "oh great", "yeah right",
           "love waiting", "real helpful", "fantastic job", "brilliant, it broke", "just perfect"]
ANGRY = {"useless": -2.5, "unacceptable": -2.5, "ridiculous": -2.2, "furious": -3.0,
         "worst": -2.5, "scam": -3.0, "incompetent": -3.0, "garbage": -2.5}


@asynccontextmanager
async def lifespan(app):
    STATE["cat"] = joblib.load("models/category.joblib")      # load once at startup
    STATE["prio"] = joblib.load("models/priority.joblib")
    STATE["vader"] = SentimentIntensityAnalyzer()
    STATE["vader"].lexicon.update(ANGRY)
    try:
        STATE["metrics"] = json.load(open("models/metrics.json"))
    except FileNotFoundError:
        STATE["metrics"] = {}
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tickets(
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, text TEXT, category TEXT,
            priority TEXT, sentiment TEXT, sentiment_score REAL, suggested_team TEXT,
            confidence REAL, escalate INTEGER, routing TEXT)""")
    yield


app = FastAPI(title="AutoTriage AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Ticket(BaseModel):
    text: str


def sentiment_of(text: str):
    low = text.lower()
    score = STATE["vader"].polarity_scores(text)["compound"]
    if any(p in low for p in SARCASM):                         # VADER misses sarcasm
        score = min(score, -0.5)
    label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
    return label, round(score, 3)


def triage_one(text: str, save: bool = True) -> dict:
    t0 = time.perf_counter()
    clean = clean_text(text)
    if not clean:
        raise HTTPException(422, "Ticket text is empty")
    cat_p = STATE["cat"].predict_proba([clean])[0]
    pri_p = STATE["prio"].predict_proba([clean])[0]
    category = STATE["cat"].classes_[cat_p.argmax()]
    priority = STATE["prio"].classes_[pri_p.argmax()]
    confidence = float(cat_p.max())                            # max predict_proba
    sentiment, score = sentiment_of(text)
    manual = confidence < CONF_THRESHOLD
    team = "Manual Queue" if manual else TEAM_MAP.get(category, DEFAULT_TEAM)
    out = {
        "category": category, "priority": priority, "sentiment": sentiment,
        "sentiment_score": score, "suggested_team": team,
        "confidence": round(confidence, 4),
        "routing": "manual_queue" if manual else "auto_routed",
        "escalate": sentiment == "negative" and priority == "critical",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    if save:
        with sqlite3.connect(DB) as c:
            cur = c.execute(
                "INSERT INTO tickets(created_at,text,category,priority,sentiment,sentiment_score,"
                "suggested_team,confidence,escalate,routing) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), text, category, priority, sentiment, score,
                 team, out["confidence"], int(out["escalate"]), out["routing"]))
            out["id"] = cur.lastrowid
    return out


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "cat" in STATE, "threshold": CONF_THRESHOLD}


@app.post("/triage")
def triage(t: Ticket):
    return triage_one(t.text)


@app.post("/triage/batch")
def triage_batch(items: list[Ticket]):
    return [{"text": i.text, **triage_one(i.text)} for i in items]


@app.post("/triage/csv")
async def triage_csv(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    col = next((c for c in df.columns if c.lower() in
                ("text", "ticket", "description", "ticket description", "body", "message")), df.columns[0])
    rows = [{"text": str(x), **triage_one(str(x))} for x in df[col].dropna()]
    return {"count": len(rows), "text_column": col, "results": rows}


@app.get("/tickets")
def tickets(limit: int = 50):
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "escalate": bool(r["escalate"])} for r in rows]


@app.delete("/tickets")
def clear_tickets():
    with sqlite3.connect(DB) as c:
        c.execute("DELETE FROM tickets")
    return {"cleared": True}


@app.get("/stats")
def stats():
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute("SELECT * FROM tickets")]
    n = len(rows)
    today = datetime.now(timezone.utc).date()
    days = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    per_day = Counter(r["created_at"][:10] for r in rows)
    m = STATE["metrics"]
    cross = Counter(f'{r["category"]}|{r["priority"]}' for r in rows)
    return {
        "total": n,
        "category_accuracy": m.get("category", {}).get("accuracy"),
        "priority_accuracy": m.get("priority", {}).get("accuracy"),
        "high_critical": sum(r["priority"] in ("high", "critical") for r in rows),
        "escalations": sum(r["escalate"] for r in rows),
        "manual_queue": sum(r["routing"] == "manual_queue" for r in rows),
        "avg_confidence": round(sum(r["confidence"] for r in rows) / n, 4) if n else None,
        "by_category": dict(Counter(r["category"] for r in rows)),
        "by_priority": dict(Counter(r["priority"] for r in rows)),
        "by_sentiment": dict(Counter(r["sentiment"] for r in rows)),
        "category_x_priority": [{"category": k.split("|")[0], "priority": k.split("|")[1], "count": v}
                                for k, v in cross.items()],
        "over_time": [{"date": d, "count": per_day.get(d, 0)} for d in days],
    }


@app.get("/metrics")
def metrics():
    m = STATE["metrics"]
    if not m:
        raise HTTPException(404, "Run train.py first")
    return {"dataset": m["dataset"], "n_train": m["n_train"], "n_test": m["n_test"],
            "category": {k: m["category"][k] for k in ("accuracy", "labels", "confusion_matrix")},
            "priority": {k: m["priority"][k] for k in ("accuracy", "labels", "confusion_matrix")}}


@app.get("/misclassified")
def misclassified():
    return STATE["metrics"].get("misclassified", [])
