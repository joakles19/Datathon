import sqlite3, threading, time
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hearhere_audio.features import AudioProcessingError
from hearhere_audio.inference import HearHereClassifier

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Fail fast: if the model, its hash, or the fold-4 thresholds don't match,
# the server refuses to start instead of silently using a fake prediction.
classifier = HearHereClassifier()

lock = threading.Lock()
db = sqlite3.connect("events.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.execute("""CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, label TEXT, confidence REAL, action TEXT,
    priority TEXT, message TEXT, acknowledged INTEGER DEFAULT 0, acked_at REAL)""")

# Default thresholds come from models/thresholds.json (selected on fold 4).
FOLD4 = classifier.thresholds["minimum_scores"]
PREFS = {
    "siren":           {"enabled": True, "threshold": FOLD4["siren"],           "cooldown": 30, "priority": "high",   "message": "Siren detected"},
    "glass_breaking":  {"enabled": True, "threshold": FOLD4["glass_breaking"],  "cooldown": 30, "priority": "high",   "message": "Glass may have broken"},
    "door_wood_knock": {"enabled": True, "threshold": FOLD4["door_wood_knock"], "cooldown": 15, "priority": "normal", "message": "Someone may be at the door"},
}
last_alert: dict[str, float] = {}
snoozed_until: dict[str, float] = {}


def pick_label(top_label: str, scores: dict[str, float]):
    """Pick the alert class whose score clears its own threshold (highest wins).
    If none qualifies, fall back to the model's top class."""
    candidates = [
        (score, label) for label, score in scores.items()
        if label in PREFS and PREFS[label]["enabled"] and score >= PREFS[label]["threshold"]
    ]
    if candidates:
        score, label = max(candidates)
        return label, score
    return top_label, scores[top_label]


def decide(label: str, conf: float, now: float):
    p = PREFS.get(label)
    if p is None or not p["enabled"] or conf < p["threshold"]:
        return "ignore", None, None
    if now < snoozed_until.get(label, 0):
        return "suppressed", p["priority"], "snoozed"
    if now - last_alert.get(label, 0) < p["cooldown"]:
        return "suppressed", p["priority"], "repeat within cooldown"
    last_alert[label] = now
    return "alert", p["priority"], p["message"]


@app.post("/classify")
def classify(file: UploadFile = File(...)):
    audio = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not audio:
        raise HTTPException(400, "empty file")
    if len(audio) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")

    started = time.perf_counter()
    try:
        result = classifier.predict_bytes(audio)
    except AudioProcessingError as exc:
        raise HTTPException(422, str(exc))
    processing_ms = round((time.perf_counter() - started) * 1000, 1)

    scores = result["scores"]
    label, conf = pick_label(result["label"], scores)
    now = time.time()
    with lock:
        action, priority, message = decide(label, conf, now)
        event_id = None
        if action != "ignore":
            cur = db.execute(
                "INSERT INTO events (ts,label,confidence,action,priority,message) VALUES (?,?,?,?,?,?)",
                (now, label, conf, action, priority, message))
            db.commit()
            event_id = cur.lastrowid
    return {"label": label, "predicted_label": result["label"], "confidence": conf,
            "scores": scores, "model_version": result["model_version"],
            "action": action, "priority": priority, "message": message,
            "event_id": event_id, "processing_ms": processing_ms}


@app.get("/events")
def events(limit: int = 50):
    with lock:
        rows = db.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.post("/events/{event_id}/ack")
def ack(event_id: int):
    with lock:
        cur = db.execute("UPDATE events SET acknowledged=1, acked_at=? WHERE id=?",
                         (time.time(), event_id))
        db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "event not found")
    return {"ok": True}


@app.get("/prefs")
def get_prefs():
    return PREFS


class PrefUpdate(BaseModel):
    enabled: bool | None = None
    threshold: float | None = Field(default=None, ge=0, le=1)
    cooldown: int | None = Field(default=None, ge=0)
    priority: Literal["high", "normal"] | None = None


@app.put("/prefs/{label}")
def put_prefs(label: str, upd: PrefUpdate):
    if label not in PREFS:
        raise HTTPException(404, "unknown sound")
    PREFS[label].update(upd.model_dump(exclude_none=True))
    return PREFS[label]


class Snooze(BaseModel):
    label: str
    minutes: int = Field(ge=1, le=1440)


@app.post("/snooze")
def snooze(s: Snooze):
    if s.label not in PREFS:
        raise HTTPException(404, "unknown sound")
    snoozed_until[s.label] = time.time() + s.minutes * 60
    return {"label": s.label, "until": snoozed_until[s.label]}