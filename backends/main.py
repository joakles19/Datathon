import sqlite3, threading, time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hearhere_audio.config import ALERT_CLASSES
from hearhere_audio.features import AudioProcessingError
from hearhere_audio.inference import HearHereClassifier

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MONITOR_WINDOW_HISTORY = 3
MONITOR_STATE_TTL_SECONDS = 60
MAX_MONITOR_STREAMS = 256

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Fail fast: if the model, its hash, or the fold-4 thresholds don't match,
# the server refuses to start instead of silently using a fake prediction.
classifier = HearHereClassifier()

lock = threading.Lock()
DATABASE_PATH = Path(__file__).resolve().parents[1] / "events.db"
db = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db.execute("""CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, label TEXT, confidence REAL, action TEXT,
    priority TEXT, message TEXT, acknowledged INTEGER DEFAULT 0, acked_at REAL)""")

# Default thresholds come from models/thresholds.json (selected on fold 4).
FOLD4 = classifier.thresholds["minimum_scores"]
ALERT_SETTINGS = {
    "siren": {"cooldown": 30, "priority": "high", "message": "Siren detected"},
    "glass_breaking": {"cooldown": 30, "priority": "high", "message": "Glass may have broken"},
    "door_wood_knock": {"cooldown": 15, "priority": "normal", "message": "Someone may be at the door"},
    "clock_alarm": {"cooldown": 30, "priority": "high", "message": "Clock alarm detected"},
    "crying_baby": {"cooldown": 30, "priority": "high", "message": "Crying baby detected"},
    "car_horn": {"cooldown": 30, "priority": "high", "message": "Car horn detected"},
}
PREFS = {
    label: {
        "enabled": label in classifier.thresholds["recommended_alert_classes"],
        "threshold": FOLD4.get(label, 1.0),
        **ALERT_SETTINGS[label],
    }
    for label in ALERT_CLASSES
}
last_alert: dict[str, float] = {}
snoozed_until: dict[str, float] = {}
monitor_windows: OrderedDict[str, dict] = OrderedDict()


@app.get("/health")
def health():
    """Report that the backend and validated model are ready."""
    return {"status": "ok", "model_version": classifier.metadata["model_version"]}


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


def record_monitor_window(stream_id: str, window_index: int, candidate: str | None, now: float) -> int:
    """Keep a bounded per-stream history of recent threshold-clearing candidates."""
    expired = [
        key for key, state in monitor_windows.items()
        if now - state["updated_at"] > MONITOR_STATE_TTL_SECONDS
    ]
    for key in expired:
        del monitor_windows[key]

    state = monitor_windows.get(stream_id)
    if state is None:
        state = {"updated_at": now, "last_index": -1, "candidates": deque(maxlen=MONITOR_WINDOW_HISTORY)}
        monitor_windows[stream_id] = state
    elif window_index <= state["last_index"]:
        raise HTTPException(409, "monitor windows must arrive in increasing order")

    state["candidates"].append(candidate)
    state["last_index"] = window_index
    state["updated_at"] = now
    monitor_windows.move_to_end(stream_id)
    while len(monitor_windows) > MAX_MONITOR_STREAMS:
        monitor_windows.popitem(last=False)
    return sum(item == candidate for item in state["candidates"]) if candidate else 0


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
def classify(
    file: UploadFile = File(...),
    stream_id: str | None = Form(default=None, min_length=1, max_length=64),
    window_index: int | None = Form(default=None, ge=0),
):
    if (stream_id is None) != (window_index is None):
        raise HTTPException(400, "stream_id and window_index must be provided together")
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
        confirmation = "not_required"
        if stream_id is not None and window_index is not None:
            candidate = (
                label
                if label in PREFS and PREFS[label]["enabled"] and conf >= PREFS[label]["threshold"]
                else None
            )
            matching_windows = record_monitor_window(stream_id, window_index, candidate, now)
            if candidate is not None and matching_windows < 2:
                return {
                    "label": candidate,
                    "predicted_label": result["label"],
                    "confidence": scores[candidate],
                    "scores": scores,
                    "model_version": result["model_version"],
                    "action": "ignore",
                    "priority": PREFS[candidate]["priority"],
                    "message": "Waiting for a second match in the recent monitoring windows",
                    "event_id": None,
                    "processing_ms": processing_ms,
                    "confirmation": "pending",
                }
            if candidate is not None:
                label = candidate
                conf = scores[candidate]
                confirmation = "confirmed"
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
            "event_id": event_id, "processing_ms": processing_ms,
            "confirmation": confirmation}


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
