/*
 * BACKEND CONTRACT — adjust analyzeAudio() / isAlert() if your API differs.
 *
 *   POST {api}/analyze   multipart/form-data, file field "audio"
 *   -> 200 JSON {
 *        "break_in": true,             // overall verdict (optional)
 *        "label": "glass_breaking",    // top class
 *        "confidence": 0.93,           // 0..1
 *        "events": [{ "label": "glass_breaking", "confidence": 0.93, "start": 1.2, "end": 2.0 }]
 *      }
 *   GET  {api}/health    -> 200 when the service is up (optional)
 */

export const DEFAULT_API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const BREAK_IN_LABELS = /break|glass|forced|smash|tamper|kick|crowbar/i;

export function isAlert(result, thresholdPct) {
  const flagged = result.break_in ?? BREAK_IN_LABELS.test(result.label || '');
  return Boolean(flagged) && result.confidence * 100 >= thresholdPct;
}

export async function analyzeAudio(blob, filename, { api, demo }) {
  if (demo) return demoAnalyze();
  const body = new FormData();
  body.append('audio', blob, filename);
  const res = await fetch(`${api.replace(/\/+$/, '')}/analyze`, { method: 'POST', body });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  const data = await res.json();
  if (typeof data.confidence !== 'number') throw new Error('Unexpected response format');
  return data;
}

export async function checkHealth(api) {
  const res = await fetch(`${api.replace(/\/+$/, '')}/health`);
  return res.ok;
}

/* ---------- demo mode: fake detections so the UI works without a backend ---------- */
const THREAT = ['glass_breaking', 'door_forced', 'window_smash', 'alarm_tamper'];
const BENIGN = ['speech', 'footsteps', 'background_noise', 'dog_bark'];
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

async function demoAnalyze() {
  await new Promise(r => setTimeout(r, 700 + Math.random() * 600));
  if (Math.random() < 0.35) {
    const label = pick(THREAT);
    const confidence = 0.75 + Math.random() * 0.24;
    const start = +(Math.random() * 2).toFixed(1);
    return { break_in: true, label, confidence, events: [{ label, confidence, start, end: +(start + 1 + Math.random()).toFixed(1) }] };
  }
  return { break_in: false, label: pick(BENIGN), confidence: 0.5 + Math.random() * 0.4, events: [] };
}
