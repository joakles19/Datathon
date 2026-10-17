import sqlite3, time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hearhere_audio.features import AudioProcessingError
from hearhere_audio.inference import HearHereClassifier

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

db = sqlite3.connect("events.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.execute("""CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, label TEXT, confidence REAL, action TEXT,
    priority TEXT, message TEXT, acknowledged INTEGER DEFAULT 0, acked_at REAL)""")

classifier = HearHereClassifier()
thresholds = classifier.thresholds["minimum_scores"]
PREFS = {
    "siren":           {"enabled": True, "threshold": thresholds["siren"], "cooldown": 30, "priority": "high",   "message": "Siren detected"},
    "glass_breaking":  {"enabled": True, "threshold": thresholds["glass_breaking"], "cooldown": 30, "priority": "high",   "message": "Glass may have broken"},
    "door_wood_knock": {"enabled": True, "threshold": thresholds["door_wood_knock"], "cooldown": 15, "priority": "normal", "message": "Someone may be at the door"},
}
last_alert: dict[str, float] = {}
snoozed_until: dict[str, float] = {}

def predict(audio_bytes: bytes):
    return classifier.predict_bytes(audio_bytes)

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
async def classify(file: UploadFile = File(...)):
    audio = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(audio) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio upload exceeds the 10 MiB limit")
    try:
        prediction = predict(audio)
    except AudioProcessingError as exc:
        raise HTTPException(422, str(exc)) from exc
    scores = prediction["scores"]
    eligible = [
        label
        for label, preference in PREFS.items()
        if preference["enabled"] and scores[label] >= preference["threshold"]
    ]
    if eligible:
        label = max(eligible, key=scores.get)
        confidence = scores[label]
    else:
        label = prediction["label"]
        confidence = prediction["score"]
    now = time.time()
    if eligible:
        action, priority, message = decide(label, confidence, now)
    else:
        action, priority, message = "ignore", None, None
    event_id = None
    if action != "ignore":
        cur = db.execute(
            "INSERT INTO events (ts,label,confidence,action,priority,message) VALUES (?,?,?,?,?,?)",
            (now, label, confidence, action, priority, message))
        db.commit()
        event_id = cur.lastrowid
    return {
        "label": label,
        "predicted_label": prediction["label"],
        "confidence": confidence,
        "action": action,
        "priority": priority,
        "message": message,
        "event_id": event_id,
        "scores": scores,
        "model_version": prediction["model_version"],
    }

@app.get("/events")
def events(limit: int = 50):
    rows = db.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

@app.post("/events/{event_id}/ack")
def ack(event_id: int):
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
    threshold: float | None = None
    cooldown: int | None = None
    priority: str | None = None

@app.put("/prefs/{label}")
def put_prefs(label: str, upd: PrefUpdate):
    if label not in PREFS:
        raise HTTPException(404, "unknown sound")
    PREFS[label].update(upd.model_dump(exclude_none=True))
    return PREFS[label]

class Snooze(BaseModel):
    label: str
    minutes: int

@app.post("/snooze")
def snooze(s: Snooze):
    snoozed_until[s.label] = time.time() + s.minutes * 60
    return {"label": s.label, "until": snoozed_until[s.label]}
