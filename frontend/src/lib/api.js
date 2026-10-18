/*
 * BACKEND CONTRACT (HearHere FastAPI backend, backends/main.py)
 *
 *   POST {api}/classify        multipart/form-data, file field "file", WAV of at most 5 seconds
 *     -> { label, predicted_label, confidence, scores{10 classes}, model_version,
 *          action: "alert" | "suppressed" | "ignore", confirmation, priority, message, event_id, processing_ms }
 *   GET  {api}/events          recent events, newest first
 *   POST {api}/events/{id}/ack acknowledge one event
 *   GET  {api}/prefs           per-sound settings (enabled, threshold, cooldown, priority, message)
 *   PUT  {api}/prefs/{label}   change one setting
 *   POST {api}/snooze          { label, minutes }
 *
 * The backend decides whether to alert. The frontend only displays `action`; monitoring candidates can be pending confirmation.
 * `confidence` is an uncalibrated model score, not a probability of a real event.
 * Every clip is converted in the browser to 16 kHz mono 16-bit WAV and cut into windows of at most 5 s.
 */

import { toWavWindows } from './wav.js';

export const DEFAULT_API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const EXPECTED_MODEL_VERSION = 'hearhere-rf-v2';
export const WAV_OPTIONS = { sampleRate: 16000, channels: 1 };

const base = api => api.replace(/\/+$/, '');
const JSON_HEADERS = { 'Content-Type': 'application/json' };

/** True when the backend decided this clip should notify the user. */
export const isAlert = result => result.action === 'alert';
const monitorQueues = new Map();

function serializeMonitorRequest(streamId, requestFn) {
  const previous = monitorQueues.get(streamId) || Promise.resolve();
  const current = previous.catch(() => {}).then(requestFn);
  monitorQueues.set(streamId, current);
  return current.finally(() => {
    if (monitorQueues.get(streamId) === current) monitorQueues.delete(streamId);
  });
}

async function request(api, path, options) {
  const res = await fetch(`${base(api)}${path}`, options);
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = Array.isArray(body.detail)
        ? body.detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
        : typeof body.detail === 'string' ? body.detail : '';
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail || `Server returned ${res.status}`);
  }
  return res.json();
}

/**
 * Convert to WAV windows and classify them in order, calling onResult(result) for each window
 * as soon as it is ready. Windows are sent one at a time so cooldowns apply in time order.
 */
export function analyzeAudio(blob, filename, options) {
  const monitorMatch = filename.match(/^monitor-([a-f\d]{16,64})-(\d+)\./i);
  if (monitorMatch && !options.demo) {
    const monitor = { streamId: monitorMatch[1], firstWindowIndex: Number(monitorMatch[2]) };
    return serializeMonitorRequest(monitor.streamId, () => analyzeAudioNow(blob, filename, options, monitor));
  }
  return analyzeAudioNow(blob, filename, options, null);
}

async function analyzeAudioNow(blob, filename, { api, demo, onResult }, monitor) {
  if (demo) return onResult(await demoAnalyze());

  let windows;
  try {
    windows = await toWavWindows(blob, WAV_OPTIONS);
  } catch (err) {
    throw new Error(`WAV conversion failed (${err.message})`);
  }
  if (!windows.length) throw new Error('no audio found in this file');

  const stem = filename.replace(/\.[^./\\]+$/, '');
  for (const [i, w] of windows.entries()) {
    const body = new FormData();
    body.append('file', w.blob, `${stem}-${i + 1}.wav`);
    if (monitor) {
      body.append('stream_id', monitor.streamId);
      body.append('window_index', String(monitor.firstWindowIndex + i));
    }
    const data = await request(api, '/classify', { method: 'POST', body });
    if (data.model_version !== EXPECTED_MODEL_VERSION) {
      throw new Error(`Backend model mismatch: expected ${EXPECTED_MODEL_VERSION}, received ${data.model_version || 'unknown'}`);
    }
    if (typeof data.confidence !== 'number' || !data.scores || !data.action) {
      throw new Error('Unexpected response format');
    }
    const span = windows.length > 1 ? `${+w.start.toFixed(1)}–${+w.end.toFixed(1)}s` : null;
    onResult({ ...data, window: span || (monitor ? `live window ${monitor.firstWindowIndex + i + 1}` : null) });
  }
}

export const fetchEvents = api => request(api, '/events?limit=50');
export const ackEvent = (api, id) => request(api, `/events/${id}/ack`, { method: 'POST' });
export const fetchPrefs = api => request(api, '/prefs');
export const updatePref = (api, label, patch) =>
  request(api, `/prefs/${label}`, { method: 'PUT', headers: JSON_HEADERS, body: JSON.stringify(patch) });
export const snoozeSound = (api, label, minutes) =>
  request(api, '/snooze', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ label, minutes }) });

export async function checkHealth(api) {
  const health = await request(api, '/health');
  if (health.model_version !== EXPECTED_MODEL_VERSION) {
    throw new Error(`Backend model mismatch: expected ${EXPECTED_MODEL_VERSION}, received ${health.model_version || 'unknown'}`);
  }
  return health;
}

/* ---------- demo mode: simulated results in the same shape, so the UI works without a backend ---------- */
const ALERTS = {
  siren: ['high', 'Siren detected'],
  glass_breaking: ['high', 'Glass may have broken'],
  door_wood_knock: ['normal', 'Someone may be at the door'],
  clock_alarm: ['high', 'Clock alarm detected'],
  crying_baby: ['high', 'Crying baby detected'],
  car_horn: ['high', 'Car horn detected'],
};
const OTHERS = ['engine', 'rain', 'vacuum_cleaner', 'wind'];
const CLASSES = [...Object.keys(ALERTS), ...OTHERS];
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const demoLastAlert = {};

function demoScores(top, p) {
  const rest = CLASSES.filter(c => c !== top);
  const w = rest.map(() => Math.random());
  const total = w.reduce((a, b) => a + b, 0);
  return Object.fromEntries([[top, p], ...rest.map((c, i) => [c, ((1 - p) * w[i]) / total])]);
}

async function demoAnalyze() {
  await new Promise(r => setTimeout(r, 500 + Math.random() * 500));
  const isImportant = Math.random() < 0.4;
  const label = pick(isImportant ? Object.keys(ALERTS) : OTHERS);
  const confidence = isImportant ? 0.55 + Math.random() * 0.4 : 0.4 + Math.random() * 0.4;
  const base = { label, predicted_label: label, confidence, scores: demoScores(label, confidence),
    model_version: 'demo', priority: null, message: null, event_id: null, processing_ms: 0 };
  if (!isImportant) return { ...base, action: 'ignore' };
  const [priority, message] = ALERTS[label];
  const now = Date.now();
  const repeat = now - (demoLastAlert[label] || 0) < 30000;
  if (!repeat) demoLastAlert[label] = now;
  return { ...base, priority, action: repeat ? 'suppressed' : 'alert', message: repeat ? 'repeat within cooldown' : message };
}
